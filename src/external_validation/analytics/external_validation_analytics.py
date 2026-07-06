"""External validation analytics orchestrator.

Evaluates the trained model's diagnostic performance against independent
GEO cohorts: per-cohort and pooled metrics, bootstrap confidence intervals,
ROC/PR/calibration curves, figures, and summary reports.
"""
import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from . import config, data_contract, metrics, curves, bootstrap, figures, reporting
except ImportError:
    import config
    import data_contract
    import metrics
    import curves
    import bootstrap
    import figures
    import reporting


class ExternalValidationAnalytics:
    def __init__(self):
        self.paths = config.AnalyticsPaths.from_module_file(__file__)
        self.paths.ensure_dirs()
        self.logger = self._setup_logger()
        self.metadata: dict = {}
        self.cohorts: dict[str, data_contract.Cohort] = {}
        self.performance: pd.DataFrame | None = None
        self.roc_tables: dict[str, pd.DataFrame] = {}
        self.pr_tables: dict[str, pd.DataFrame] = {}
        self.calibration_tables: dict[str, pd.DataFrame] = {}
        self.cohort_arrays: dict[str, tuple] = {}
        self.bootstrap_results: dict = {}
        self.execution_start = datetime.now()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ExternalValidationAnalytics")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.paths.log_dir / config.LOG_FILE_NAME, mode="w")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def load_metadata(self) -> None:
        self.logger.info("Loading model metadata...")
        with open(self.paths.model_metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def load_and_validate_cohorts(self) -> None:
        self.logger.info("Loading prediction files...")
        self.cohorts = data_contract.load_cohorts(self.paths.prediction_dir)
        for name, cohort in self.cohorts.items():
            self.logger.info("%s : %d samples", name, cohort.n_samples)
        self.logger.info("Running quality control...")
        data_contract.validate_all_cohorts(self.cohorts)
        self.logger.info("Quality control passed for all %d cohorts.", len(self.cohorts))

    def evaluate_cohorts(self) -> None:
        rows = []
        for name, cohort in self.cohorts.items():
            y_true = data_contract.standardize_labels(cohort.data)
            y_pred, y_prob = data_contract.extract_predictions(cohort.data)
            self.cohort_arrays[name] = (y_true, y_pred, y_prob)

            m = metrics.compute_diagnostic_metrics(y_true, y_pred, y_prob)
            rows.append({"Cohort": name, **m.to_dict()})

            self.roc_tables[name] = curves.compute_roc_table(y_true, y_prob)
            self.pr_tables[name] = curves.compute_pr_table(y_true, y_prob)
            self.calibration_tables[name] = curves.compute_calibration_table(y_true, y_prob)

            self.logger.info("%s -> Accuracy=%.4f ROC_AUC=%.4f", name, m.Accuracy, m.ROC_AUC)

        self.performance = pd.DataFrame(rows)
        self.performance.to_csv(self.paths.output_dir / "cohort_metrics.csv", index=False)

    def evaluate_overall(self) -> pd.DataFrame:
        pooled = pd.concat([c.data for c in self.cohorts.values()], ignore_index=True)
        y_true = data_contract.standardize_labels(pooled)
        y_pred, y_prob = data_contract.extract_predictions(pooled)
        m = metrics.compute_diagnostic_metrics(y_true, y_pred, y_prob)
        row = m.to_dict()
        row["Samples"] = len(pooled)
        row["Tumor"] = int(np.sum(y_true == 1))
        row["Normal"] = int(np.sum(y_true == 0))
        overall_df = pd.DataFrame([row])
        overall_df.to_csv(self.paths.output_dir / "overall_metrics.csv", index=False)
        return overall_df

    def run_bootstrap(self) -> None:
        ci_df, self.bootstrap_results = bootstrap.bootstrap_all_cohorts(self.cohort_arrays)
        ci_df.to_csv(self.paths.output_dir / "bootstrap_confidence_intervals.csv", index=False)

    def export_curve_tables(self) -> None:
        for name, df in self.roc_tables.items():
            df.to_csv(self.paths.roc_data_dir / f"{name}_roc.csv", index=False)
        for name, df in self.pr_tables.items():
            df.to_csv(self.paths.pr_data_dir / f"{name}_precision_recall.csv", index=False)
        for name, df in self.calibration_tables.items():
            df.to_csv(self.paths.calibration_data_dir / f"{name}_calibration.csv", index=False)

    def generate_figures(self) -> None:
        auc_by_cohort = dict(zip(self.performance["Cohort"], self.performance["ROC_AUC"]))
        figures.plot_roc_curves(self.roc_tables, auc_by_cohort, self.paths.figure_dir / "roc_curves.png")
        figures.plot_pr_curves(self.pr_tables, self.paths.figure_dir / "precision_recall_curves.png")
        figures.plot_calibration_curves(self.calibration_tables, self.paths.figure_dir / "calibration_curves.png")
        for _, row in self.performance.iterrows():
            figures.plot_confusion_matrix(
                row["Cohort"], row["TN"], row["FP"], row["FN"], row["TP"],
                self.paths.figure_dir / f"{row['Cohort']}_confusion_matrix.png",
            )
        for name, result in self.bootstrap_results.items():
            figures.plot_bootstrap_auc_histogram(
                name, result.distributions["ROC_AUC"],
                self.paths.figure_dir / f"{name}_bootstrap_auc.png",
            )

    def export_reports(self) -> None:
        execution_seconds = (datetime.now() - self.execution_start).total_seconds()
        total_samples = sum(c.n_samples for c in self.cohorts.values())
        reporting.write_json_summary(
            self.metadata, len(self.cohorts), total_samples, execution_seconds,
            self.paths.report_dir / "validation_summary.json",
        )
        reporting.write_text_report(
            self.performance, len(self.cohorts),
            self.paths.report_dir / "validation_report.txt",
        )

    def run(self) -> None:
        self.logger.info("External Validation Analytics Started")
        self.load_metadata()
        self.load_and_validate_cohorts()
        self.evaluate_cohorts()
        self.run_bootstrap()
        self.evaluate_overall()
        self.export_curve_tables()
        self.generate_figures()
        self.export_reports()
        self.logger.info("Execution Time : %s", datetime.now() - self.execution_start)


if __name__ == "__main__":
    ExternalValidationAnalytics().run()
