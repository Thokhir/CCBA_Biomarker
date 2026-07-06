"""Paths and constants for the external validation analytics module."""
from dataclasses import dataclass, field
from pathlib import Path

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2000

CONFIDENCE_LEVEL = 0.95
CI_LOWER_PERCENTILE = (1 - CONFIDENCE_LEVEL) / 2 * 100
CI_UPPER_PERCENTILE = 100 - CI_LOWER_PERCENTILE

FIGURE_DPI = 300
CALIBRATION_N_BINS = 10
CALIBRATION_STRATEGY = "uniform"

LOG_FILE_NAME = "execution_log.txt"
REPORT_BASENAME = "validation_report"

FIGSIZE_ROC = (8, 8)
FIGSIZE_PR = (8, 8)
FIGSIZE_CALIBRATION = (8, 8)
FIGSIZE_CONFUSION = (5, 5)
FIGSIZE_BOOTSTRAP_HIST = (7, 5)
BOOTSTRAP_HIST_BINS = 40

# The single confirmed column contract - no legacy aliases.
PREDICTED_LABEL_COL = "predicted_label"
PROBABILITY_COL = "tumor_probability"
LABEL_COL_PRIMARY = "final_label"
LABEL_COL_FALLBACK = "phenotype"
REQUIRED_ANNOTATION_COLS = ("sample_id", "phenotype")

LABEL_MAPPING = {
    "tumor": 1, "primary tumor": 1, "cancer": 1,
    "cholangiocarcinoma": 1, "cca": 1, "1": 1, "true": 1,
    "normal": 0, "healthy": 0, "control": 0, "adjacent": 0,
    "0": 0, "false": 0,
}

ORDERED_METRIC_NAMES = [
    "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity",
    "Precision", "Recall", "F1", "ROC_AUC", "MCC", "Kappa",
    "PPV", "NPV", "FPR", "FNR", "LR_Positive", "LR_Negative",
    "Diagnostic_Odds_Ratio", "Brier_Score",
]


@dataclass(frozen=True)
class AnalyticsPaths:
    base: Path
    prediction_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    model_metadata_file: Path = field(init=False)
    output_dir: Path = field(init=False)
    figure_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    roc_data_dir: Path = field(init=False)
    pr_data_dir: Path = field(init=False)
    calibration_data_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "prediction_dir", self.base / "results" / "external_predictions")
        object.__setattr__(self, "model_dir", self.base / "results" / "trained_model")
        object.__setattr__(self, "model_metadata_file", self.model_dir / "model_metadata.json")
        object.__setattr__(self, "output_dir", self.base / "results" / "external_validation")
        object.__setattr__(self, "figure_dir", self.output_dir / "figures")
        object.__setattr__(self, "report_dir", self.output_dir / "reports")
        object.__setattr__(self, "log_dir", self.output_dir / "logs")
        object.__setattr__(self, "roc_data_dir", self.output_dir / "roc_data")
        object.__setattr__(self, "pr_data_dir", self.output_dir / "pr_data")
        object.__setattr__(self, "calibration_data_dir", self.output_dir / "calibration_data")

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.figure_dir, self.report_dir,
                  self.log_dir, self.roc_data_dir, self.pr_data_dir,
                  self.calibration_data_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_module_file(cls, module_file: str) -> "AnalyticsPaths":
        # src/external_validation/analytics/config.py -> project root is parents[3]
        return cls(base=Path(module_file).resolve().parents[3])
