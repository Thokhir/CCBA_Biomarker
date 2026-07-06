"""Survival & prognostic analysis orchestrator.

Evaluates whether the clinical biomarker panel predicts patient outcomes,
not just tumor/normal status. Restricted to TCGA-CHOL tumor samples with
usable follow-up data (34 patients, 18 observed deaths) - a small cohort
that this module treats as exploratory/hypothesis-generating throughout,
not confirmatory: KM/logrank and continuous Cox regression can and do
disagree here (expected with n=34, not a bug), only 1-2 covariates are
combined into any multivariate model (events-per-variable heuristic), and
time-dependent AUC uses a simplified landmark-time method since
scikit-survival's IPCW-based estimator could not be installed on this
machine (no C++ build toolchain available).
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from . import survival_data_loader as sdl
    from . import kaplan_meier as km
    from . import cox_regression as cr
    from . import risk_score as rs
    from . import time_dependent_roc as tdr
    from . import nomogram as nm
    from . import figures as fg
except ImportError:
    import survival_data_loader as sdl
    import kaplan_meier as km
    import cox_regression as cr
    import risk_score as rs
    import time_dependent_roc as tdr
    import nomogram as nm
    import figures as fg

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "survival"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

TOP_N_KM_FIGURES = 3


class SurvivalAnalysisEngine:
    def __init__(self):
        self.execution_start = datetime.now()
        for d in (OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        self.genes = None
        self.cohort = None
        self.km_results = None
        self.cox_results = None
        self.risk_genes = None
        self.multivariate_cph = None
        self.roc_df = None
        self.nomogram_scales = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("SurvivalAnalysisEngine")
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

    def load_cohort(self) -> None:
        self.logger.info("Loading survival cohort...")
        self.genes = sdl.load_panel_genes()
        cohort = sdl.load_survival_cohort()
        self.cohort = sdl.add_expression_groups(cohort, self.genes)
        self.logger.info("Cohort: %d tumor samples, %d deaths, %d censored",
                          len(self.cohort), self.cohort["OS_status"].sum(),
                          len(self.cohort) - self.cohort["OS_status"].sum())

    def run_kaplan_meier(self) -> None:
        self.logger.info("Running Kaplan-Meier + logrank tests per gene...")
        self.km_results = km.run_km_per_gene(self.cohort, self.genes)
        self.km_results.to_csv(OUTPUT_DIR / "KM_statistics.csv", index=False)
        n_sig = int((self.km_results["logrank_p_value"] < 0.05).sum())
        self.logger.info("KM: %d / %d genes significant (logrank p<0.05)", n_sig, len(self.genes))

        for _, row in self.km_results.head(TOP_N_KM_FIGURES).iterrows():
            kmf_high, kmf_low = km.fit_km_curves(self.cohort, row["gene_name"])
            fg.plot_km_curve(kmf_high, kmf_low, row["gene_name"], row["logrank_p_value"],
                              FIGURE_DIR / f"KM_{row['gene_name']}.png")

    def run_cox_regression(self) -> None:
        self.logger.info("Running univariate Cox regression per gene...")
        self.cox_results = cr.run_univariate_cox(self.cohort, self.genes)
        self.cox_results.to_csv(OUTPUT_DIR / "Hazard_Ratios.csv", index=False)
        n_sig = int((self.cox_results["p_value"] < 0.05).sum())
        self.logger.info("Cox: %d / %d genes significant (p<0.05)", n_sig, len(self.genes))
        fg.plot_hazard_forest(self.cox_results, FIGURE_DIR / "Hazard_Forest.png")

    def run_risk_score(self) -> None:
        self.logger.info("Building composite risk score...")
        self.risk_genes = rs.select_risk_genes(self.cox_results)
        self.logger.info("Risk score genes (Cox p < %.2f): %s", rs.RISK_GENE_PVALUE_THRESHOLD, self.risk_genes)

        self.cohort["risk_score"] = rs.compute_risk_score(self.cohort, self.cox_results, self.risk_genes)
        self.multivariate_cph = rs.fit_risk_score_cox(self.cohort, extra_covariates=["age_at_index"])

        risk_table = self.cohort[["case_id", "OS_time", "OS_status", "age_at_index", "risk_score"]].copy()
        risk_table.to_csv(OUTPUT_DIR / "Risk_Score.csv", index=False)

        cox_model_summary = self.multivariate_cph.summary.reset_index()
        cox_model_summary.to_csv(OUTPUT_DIR / "Cox_Model.csv", index=False)
        self.logger.info("Multivariate model (risk_score + age):\n%s",
                          self.multivariate_cph.summary[["coef", "exp(coef)", "p"]])

    def run_time_dependent_roc(self) -> None:
        self.logger.info("Computing time-dependent (landmark) ROC-AUC...")
        horizons = tdr.default_horizons(self.cohort)
        self.roc_df = tdr.compute_landmark_auc(self.cohort, "risk_score", horizons)
        self.roc_df.to_csv(OUTPUT_DIR / "TimeROC.csv", index=False)
        self.logger.info("Landmark AUC by horizon:\n%s", self.roc_df)
        fg.plot_time_roc(self.roc_df, FIGURE_DIR / "TimeROC.png")

    def run_nomogram(self) -> None:
        self.logger.info("Building simplified nomogram...")
        self.nomogram_scales = nm.build_nomogram_points(
            self.multivariate_cph, self.cohort, ["risk_score", "age_at_index"]
        )
        points_df = nm.total_points_table(self.cohort, self.nomogram_scales)
        points_df[["case_id", "risk_score", "age_at_index", "total_points"]].to_csv(
            OUTPUT_DIR / "Nomogram.csv", index=False
        )
        fg.plot_nomogram(self.nomogram_scales, FIGURE_DIR / "Nomogram.png")

    def export_reports(self) -> None:
        summary = {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cohort_size": len(self.cohort),
            "n_deaths": int(self.cohort["OS_status"].sum()),
            "n_censored": int(len(self.cohort) - self.cohort["OS_status"].sum()),
            "km_significant_genes": self.km_results.loc[
                self.km_results["logrank_p_value"] < 0.05, "gene_name"].tolist(),
            "cox_top5_genes": self.cox_results.head(5)[["gene_name", "hazard_ratio", "p_value"]].to_dict("records"),
            "risk_score_genes": self.risk_genes,
            "multivariate_model": self.multivariate_cph.summary[["coef", "exp(coef)", "p"]].reset_index().to_dict("records"),
            "landmark_auc_by_horizon": self.roc_df.to_dict("records"),
        }
        with open(REPORT_DIR / "survival_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        lines = [
            "=" * 70,
            "SURVIVAL & PROGNOSTIC ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Generated : {datetime.now()}",
            f"Cohort : {len(self.cohort)} TCGA-CHOL tumor samples "
            f"({int(self.cohort['OS_status'].sum())} deaths, "
            f"{len(self.cohort) - int(self.cohort['OS_status'].sum())} censored)",
            "",
            "IMPORTANT: This is an exploratory, hypothesis-generating analysis, not a",
            "confirmatory one. TCGA-CHOL is a small cohort; with only 18 observed",
            "deaths, statistical power is limited and results should be interpreted",
            "accordingly, not as validated prognostic claims.",
            "",
            "Kaplan-Meier (median-split high vs. low expression, log-rank test)",
            "-" * 70,
        ]
        sig_km = self.km_results[self.km_results["logrank_p_value"] < 0.05]
        if len(sig_km):
            for _, row in sig_km.iterrows():
                lines.append(f"{row['gene_name']}: logrank p={row['logrank_p_value']:.4f}")
        else:
            lines.append("No gene reached logrank p<0.05.")

        lines += [
            "",
            "Univariate Cox Regression (continuous, standardized expression)",
            "-" * 70,
        ]
        sig_cox = self.cox_results[self.cox_results["p_value"] < 0.05]
        if len(sig_cox):
            for _, row in sig_cox.iterrows():
                lines.append(f"{row['gene_name']}: HR={row['hazard_ratio']:.3f}, p={row['p_value']:.4f}")
        else:
            top_cox = self.cox_results.iloc[0]
            lines.append(f"No gene reached Cox p<0.05. Closest: {top_cox['gene_name']} "
                          f"(HR={top_cox['hazard_ratio']:.3f}, p={top_cox['p_value']:.4f})")
        lines.append(
            "Note: KM (median-split) and continuous Cox regression test different "
            "assumptions and can disagree at this sample size - this is expected, not an error."
        )

        lines += [
            "",
            f"Composite Risk Score (genes with Cox p < {rs.RISK_GENE_PVALUE_THRESHOLD}: {', '.join(self.risk_genes) or 'none'})",
            "-" * 70,
        ]
        for _, row in self.multivariate_cph.summary.reset_index().iterrows():
            lines.append(f"{row['covariate']}: HR={row['exp(coef)']:.3f}, p={row['p']:.4f}")

        lines += [
            "",
            "Time-Dependent Discrimination (landmark-time ROC-AUC, risk score)",
            "-" * 70,
            "Note: uses the landmark-time method (not IPCW-weighted dynamic AUC via",
            "scikit-survival, which could not be installed on this machine - no C++",
            "build toolchain available). Patients censored before a given horizon are",
            "excluded from that horizon's AUC.",
        ]
        for _, row in self.roc_df.iterrows():
            auc_str = f"{row['auc']:.3f}" if pd.notna(row["auc"]) else "N/A (insufficient events)"
            lines.append(f"Day {int(row['horizon_days'])}: AUC={auc_str} (n={int(row['n_evaluable'])})")

        (REPORT_DIR / "survival_report.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.logger.info("Survival Analysis Engine Started")
        self.load_cohort()
        self.run_kaplan_meier()
        self.run_cox_regression()
        self.run_risk_score()
        self.run_time_dependent_roc()
        self.run_nomogram()
        self.export_reports()
        self.logger.info("Execution Time: %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    SurvivalAnalysisEngine().run()
