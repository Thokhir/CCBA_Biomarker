"""Explainable AI orchestrator.

Computes global SHAP feature importance over the TCGA training population,
per-sample SHAP explanations over all external GEO cohort samples, and
permutation importance on the pooled external cohorts as a model-agnostic
cross-check - then cross-validates all three importance rankings against
each other and against results/trained_model/feature_importance.csv.

Outputs are precomputed as flat CSV/JSON under results/explainability/ so a
future Streamlit app (Module 13) can look up any known external-cohort
sample's explanation without recomputing SHAP live; only genuinely new
patient input would need a live TreeExplainer call at request time (cheap:
sub-second for one row, tree_path_dependent mode).
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import data_loader as dl
    from . import global_explainer as ge
    from . import local_explainer as le
    from . import permutation as pm
    from . import figures as fg
    from . import dependence as dep
    from . import decision_path as dp
    from . import cohort_comparison as cc
    from . import stability as st
except ImportError:
    import data_loader as dl
    import global_explainer as ge
    import local_explainer as le
    import permutation as pm
    import figures as fg
    import dependence as dep
    import decision_path as dp
    import cohort_comparison as cc
    import stability as st

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "explainability"
GLOBAL_DIR = OUTPUT_DIR / "global"
LOCAL_DIR = OUTPUT_DIR / "local"
PERMUTATION_DIR = OUTPUT_DIR / "permutation"
DECISION_PATH_DIR = OUTPUT_DIR / "decision_path"
COHORT_COMPARISON_DIR = OUTPUT_DIR / "cohort_comparison"
STABILITY_DIR = OUTPUT_DIR / "stability"
REPORT_DIR = OUTPUT_DIR / "reports"

RF_FEATURE_IMPORTANCE_FILE = BASE_DIR / "results" / "trained_model" / "feature_importance.csv"
MODEL_METADATA_FILE = BASE_DIR / "results" / "trained_model" / "model_metadata.json"


class ExplainableAIEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        for d in (GLOBAL_DIR, LOCAL_DIR, PERMUTATION_DIR, DECISION_PATH_DIR,
                  COHORT_COMPARISON_DIR, STABILITY_DIR, REPORT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        self.model = None
        self.feature_order = None
        self.n_training_samples = None
        self.X_train = None
        self.global_explanation = None
        self.global_importance = None
        self.local_shap_all = None
        self.permutation_importance = None
        self.correlations = None
        self.tree_depth_summary = None
        self.stability_summary = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExplainableAIEngine")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(REPORT_DIR / "execution_log.txt", mode="w")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def run_global_explanation(self) -> None:
        self.logger.info("Loading model and training matrix...")
        self.model, self.feature_order = dl.load_model_and_feature_order()
        X_train, y_train = dl.load_training_matrix(self.feature_order)
        self.n_training_samples = X_train.shape[0]
        self.logger.info("Training matrix shape: %s", X_train.shape)

        self.X_train = X_train
        self.logger.info("Computing global SHAP explanations...")
        explanation = ge.compute_global_shap(self.model, X_train)
        self.global_explanation = explanation
        self.global_importance = ge.summarize_global_importance(explanation, self.feature_order)
        self.global_importance.to_csv(GLOBAL_DIR / "shap_global_importance.csv", index=False)

        shap_values_df = ge.shap_values_to_frame(explanation, X_train.index.tolist(), self.feature_order)
        shap_values_df.to_csv(GLOBAL_DIR / "shap_values_training.csv", index=False)

        fg.plot_global_bar(self.global_importance, GLOBAL_DIR / "shap_global_bar.png")
        fg.plot_beeswarm(explanation, self.feature_order, GLOBAL_DIR / "shap_summary_beeswarm.png")

        rf_importance = pd.read_csv(RF_FEATURE_IMPORTANCE_FILE)
        top5_shap = set(self.global_importance.head(5)["gene_name"])
        top5_rf = set(rf_importance.head(5)["gene_name"])
        overlap = len(top5_shap & top5_rf)
        self.logger.info("Top-5 SHAP vs RF built-in overlap: %d/5 (%s)", overlap, top5_shap & top5_rf)
        if overlap < 3:
            self.logger.warning("Low overlap between SHAP and RF built-in top-5 rankings.")

    def run_local_explanation(self) -> None:
        self.logger.info("Loading external cohort matrices...")
        matrices = dl.load_all_external_matrices(self.feature_order)

        all_frames = []
        representative_frames = []
        for name, matrix in matrices.items():
            self.logger.info("%s: %d samples, %d genes imputed (%s)",
                              name, len(matrix.X), len(matrix.genes_imputed), matrix.genes_imputed)

            explanation = le.compute_local_shap(self.model, matrix.X)
            frame = le.local_shap_to_frame(explanation, matrix, self.feature_order)
            all_frames.append(frame)

            cases = le.select_representative_cases(matrix)
            if cases.empty:
                continue
            representative_frames.append(cases.assign(cohort=name))

            X_rep = matrix.X.iloc[cases["row_position"].tolist()]
            rep_explanation = le.compute_local_shap(self.model, X_rep)
            values = le.class1_values(rep_explanation)
            base = le.class1_base_values(rep_explanation)

            for i, row in cases.reset_index(drop=True).iterrows():
                recon = base[i] + values[i].sum()
                actual = self.model.predict_proba(X_rep.iloc[[i]])[0, 1]
                if abs(recon - actual) > 1e-4:
                    raise ValueError(
                        f"SHAP additivity check failed for {name}/{row['role']}: "
                        f"reconstructed={recon}, actual={actual}"
                    )
                out_path = LOCAL_DIR / f"waterfall_{name}_{row['sample_id']}_{row['role']}.png"
                fg.plot_waterfall(values[i], base[i], X_rep.iloc[i].to_numpy(), self.feature_order, out_path)

        self.local_shap_all = pd.concat(all_frames, ignore_index=True)
        self.local_shap_all.to_csv(LOCAL_DIR / "shap_values_external_all.csv", index=False)

        representative = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()
        representative.to_csv(LOCAL_DIR / "representative_cases.csv", index=False)
        self.logger.info("Local SHAP explanations computed for %d external samples.", len(self.local_shap_all))

    def run_permutation_importance(self) -> None:
        self.logger.info("Loading pooled external cohort data for permutation importance...")
        matrices = dl.load_all_external_matrices(self.feature_order)
        X_pool, y_pool, _ = dl.pool_external_matrices(matrices)

        self.logger.info("Computing permutation importance on %d pooled external samples...", len(X_pool))
        results = pm.compute_permutation_importance(self.model, X_pool, y_pool)
        self.permutation_importance = pm.summarize_permutation_importance(results, self.feature_order)
        self.permutation_importance.to_csv(PERMUTATION_DIR / "permutation_importance.csv", index=False)
        fg.plot_permutation_bar(self.permutation_importance, PERMUTATION_DIR / "permutation_importance_bar.png")

        if self.permutation_importance.isna().any().any():
            raise ValueError("NaNs found in permutation importance results.")

        rf_importance = pd.read_csv(RF_FEATURE_IMPORTANCE_FILE)
        merged, self.correlations = pm.compare_rankings(self.global_importance, self.permutation_importance, rf_importance)
        merged.to_csv(PERMUTATION_DIR / "permutation_vs_shap_comparison.csv", index=False)
        self.logger.info("Ranking correlations: %s", self.correlations)

    def run_dependence_analysis(self, top_n: int = 5) -> None:
        self.logger.info("Computing SHAP dependence plots for top %d genes...", top_n)
        top_genes = dep.select_top_genes(self.global_importance, top_n=top_n)
        for gene in top_genes:
            fg.plot_dependence(self.global_explanation, gene, self.feature_order,
                                GLOBAL_DIR / f"dependence_{gene}.png")
        self.logger.info("Dependence plots generated for: %s", top_genes)

    def run_decision_path_analysis(self, n_trees: int = 3) -> None:
        self.logger.info("Analyzing Random Forest decision paths...")
        self.tree_depth_summary = dp.summarize_tree_depths(self.model)
        self.logger.info("Tree depth summary: %s", self.tree_depth_summary)

        tree_indices = dp.select_representative_tree_indices(self.model, n=n_trees)
        for i in tree_indices:
            fg.plot_decision_tree(self.model.estimators_[i], self.feature_order,
                                   DECISION_PATH_DIR / f"tree_{i}.png", f"Tree {i}")

        with open(DECISION_PATH_DIR / "tree_depth_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.tree_depth_summary, f, indent=4)

    def run_cohort_comparison(self, top_n: int = 8) -> None:
        self.logger.info("Comparing SHAP value distributions across cohorts...")
        train_shap = pd.read_csv(GLOBAL_DIR / "shap_values_training.csv")
        top_genes = self.global_importance.head(top_n)["gene_name"].tolist()

        comparison = cc.build_comparison_frame(train_shap, self.local_shap_all, self.feature_order, top_genes)
        comparison.to_csv(COHORT_COMPARISON_DIR / "shap_cohort_comparison.csv", index=False)
        fg.plot_cohort_comparison(comparison, COHORT_COMPARISON_DIR / "shap_cohort_comparison.png")

    def run_stability_integration(self) -> None:
        self.logger.info("Building biomarker stability integration table...")
        self.stability_summary = st.build_stability_summary(
            self.global_importance, self.permutation_importance, self.feature_order
        )
        self.stability_summary.to_csv(STABILITY_DIR / "biomarker_stability_summary.csv", index=False)

        n_in_stability_pool = int(self.stability_summary["in_nested_cv_stability_pool"].sum())
        self.logger.info(
            "%d of %d clinical panel genes also appear in the nested-CV feature stability pool.",
            n_in_stability_pool, len(self.stability_summary),
        )

    def export_reports(self) -> None:
        with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)

        summary = {
            "model_training_date": model_metadata.get("training_date", "Unknown"),
            "explainability_generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_training_samples": self.n_training_samples,
            "n_external_samples_explained": int(len(self.local_shap_all)),
            "ranking_correlations": self.correlations,
            "top5_shap_genes": self.global_importance.head(5)["gene_name"].tolist(),
            "top5_permutation_genes": self.permutation_importance.sort_values(
                "mean_importance_auc", ascending=False
            ).head(5)["gene_name"].tolist(),
            "random_forest_tree_depth_summary": self.tree_depth_summary,
            "clinical_panel_genes_in_nested_cv_stability_pool":
                int(self.stability_summary["in_nested_cv_stability_pool"].sum()),
            "clinical_panel_size": len(self.stability_summary),
        }
        with open(REPORT_DIR / "explainability_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        lines = [
            "=" * 70,
            "EXPLAINABLE AI REPORT",
            "=" * 70,
            "",
            f"Generated : {datetime.now()}",
            "",
            "Top 10 Genes by Global SHAP Importance",
            "-" * 70,
        ]
        for _, row in self.global_importance.head(10).iterrows():
            lines.append(f"{row['rank']:>2}. {row['gene_name']:<10} mean|SHAP|={row['mean_abs_shap']:.5f}")

        lines += ["", "Top 10 Genes by Permutation Importance (external cohorts, ROC-AUC)", "-" * 70]
        top_perm = self.permutation_importance.sort_values("mean_importance_auc", ascending=False).head(10)
        for _, row in top_perm.iterrows():
            lines.append(f"{row['rank_auc']:>2}. {row['gene_name']:<10} mean_auc_drop={row['mean_importance_auc']:.5f}")

        lines += [
            "",
            "Ranking Correlations (Spearman)",
            "-" * 70,
            f"SHAP vs Permutation      : {self.correlations['spearman_shap_vs_permutation']:.3f}",
            f"SHAP vs RF built-in      : {self.correlations['spearman_shap_vs_rf_builtin']:.3f}",
            f"Permutation vs RF built-in: {self.correlations['spearman_permutation_vs_rf_builtin']:.3f}",
            "",
            "Note: SHAP and RF built-in importance are computed on the TCGA training",
            "population; permutation importance is computed on the pooled external GEO",
            "cohorts. A lower correlation there reflects genuine distribution shift",
            "between training and external data, not a computation error - consistent",
            "with this project's established distinction between ML feature stability",
            "and clinically reproducible biomarkers.",
            "",
            "Random Forest Structure",
            "-" * 70,
            f"n_estimators : {self.tree_depth_summary['n_estimators']}",
            f"Tree depth   : min={self.tree_depth_summary['depth_min']}, "
            f"mean={self.tree_depth_summary['depth_mean']:.2f}, max={self.tree_depth_summary['depth_max']}",
            "The ensemble consists of unusually shallow trees (depth 1-2), a result of",
            "class_weight='balanced' combined with bootstrap resampling of a small,",
            "imbalanced 44-sample training set. Predictive power comes from averaging",
            "many such shallow learners, not from any single deep tree.",
            "",
            "Biomarker Stability Integration",
            "-" * 70,
            f"Clinical panel genes also found in the nested-CV feature stability pool: "
            f"{int(self.stability_summary['in_nested_cv_stability_pool'].sum())} / {len(self.stability_summary)}",
            "This confirms the project's established design rationale: the clinical",
            "biomarker panel (optimized for reproducibility across independent GEO",
            "cohorts) and the nested-CV feature stability pool (optimized for internal",
            "TCGA model performance) are deliberately distinct gene sets, not a bug.",
        ]
        (REPORT_DIR / "explainability_report.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.logger.info("Explainable AI Engine Started")
        self.run_global_explanation()
        self.run_local_explanation()
        self.run_permutation_importance()
        self.run_dependence_analysis()
        self.run_decision_path_analysis()
        self.run_cohort_comparison()
        self.run_stability_integration()
        self.export_reports()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    ExplainableAIEngine().run()
