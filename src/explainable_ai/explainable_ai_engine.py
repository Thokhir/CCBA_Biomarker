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
except ImportError:
    import data_loader as dl
    import global_explainer as ge
    import local_explainer as le
    import permutation as pm
    import figures as fg

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "explainability"
GLOBAL_DIR = OUTPUT_DIR / "global"
LOCAL_DIR = OUTPUT_DIR / "local"
PERMUTATION_DIR = OUTPUT_DIR / "permutation"
REPORT_DIR = OUTPUT_DIR / "reports"

RF_FEATURE_IMPORTANCE_FILE = BASE_DIR / "results" / "trained_model" / "feature_importance.csv"
MODEL_METADATA_FILE = BASE_DIR / "results" / "trained_model" / "model_metadata.json"


class ExplainableAIEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        for d in (GLOBAL_DIR, LOCAL_DIR, PERMUTATION_DIR, REPORT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        self.model = None
        self.feature_order = None
        self.n_training_samples = None
        self.global_importance = None
        self.local_shap_all = None
        self.permutation_importance = None
        self.correlations = None

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

        self.logger.info("Computing global SHAP explanations...")
        explanation = ge.compute_global_shap(self.model, X_train)
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
        ]
        (REPORT_DIR / "explainability_report.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.logger.info("Explainable AI Engine Started")
        self.run_global_explanation()
        self.run_local_explanation()
        self.run_permutation_importance()
        self.export_reports()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    ExplainableAIEngine().run()
