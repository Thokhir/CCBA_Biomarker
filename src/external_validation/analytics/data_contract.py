"""Loading, per-cohort QC, and label standardization for external validation cohorts."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


@dataclass
class Cohort:
    name: str
    data: pd.DataFrame
    source_file: str

    @property
    def n_samples(self) -> int:
        return len(self.data)


def discover_prediction_files(prediction_dir: Path) -> list[Path]:
    files = sorted(prediction_dir.glob("*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")
    return files


def load_cohorts(prediction_dir: Path) -> dict[str, Cohort]:
    cohorts: dict[str, Cohort] = {}
    for file in discover_prediction_files(prediction_dir):
        name = file.stem.replace("_predictions", "")
        df = pd.read_csv(file)
        cohorts[name] = Cohort(name=name, data=df, source_file=file.name)
    return cohorts


def validate_cohort(cohort: Cohort) -> None:
    """Validate a single cohort's dataframe against the required column contract.

    Called once per cohort by validate_all_cohorts, so every cohort is checked
    (as opposed to only the last one processed by a loop).
    """
    df = cohort.data
    missing = [c for c in config.REQUIRED_ANNOTATION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{cohort.name}: missing required columns: {missing}")

    if config.PREDICTED_LABEL_COL not in df.columns:
        raise ValueError(f"{cohort.name}: missing '{config.PREDICTED_LABEL_COL}' column")

    if config.PROBABILITY_COL not in df.columns:
        raise ValueError(f"{cohort.name}: missing '{config.PROBABILITY_COL}' column")

    if config.LABEL_COL_PRIMARY not in df.columns and config.LABEL_COL_FALLBACK not in df.columns:
        raise ValueError(
            f"{cohort.name}: missing both '{config.LABEL_COL_PRIMARY}' and "
            f"'{config.LABEL_COL_FALLBACK}' columns"
        )


def validate_all_cohorts(cohorts: dict[str, Cohort]) -> None:
    for cohort in cohorts.values():
        validate_cohort(cohort)


def standardize_labels(df: pd.DataFrame) -> np.ndarray:
    """Binary ground-truth label array. Tumor=1, Normal=0."""
    if config.LABEL_COL_PRIMARY in df.columns:
        raw = df[config.LABEL_COL_PRIMARY]
    elif config.LABEL_COL_FALLBACK in df.columns:
        raw = df[config.LABEL_COL_FALLBACK]
    else:
        raise ValueError(
            f"No label column found (expected '{config.LABEL_COL_PRIMARY}' "
            f"or '{config.LABEL_COL_FALLBACK}')"
        )

    normalized = raw.astype(str).str.strip().str.lower()
    y = normalized.map(config.LABEL_MAPPING)

    if y.isna().any():
        unknown = sorted(normalized[y.isna()].unique())
        raise ValueError(f"Unknown label values encountered: {unknown}")

    return y.astype(int).to_numpy()


def extract_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, y_prob) using the single confirmed column contract."""
    y_pred = df[config.PREDICTED_LABEL_COL].astype(int).to_numpy()
    y_prob = df[config.PROBABILITY_COL].astype(float).to_numpy()
    return y_pred, y_prob


def count_classes(df: pd.DataFrame) -> tuple[int, int]:
    """Tumor/Normal counts from the phenotype column."""
    if config.LABEL_COL_FALLBACK not in df.columns:
        return 0, 0
    phenotype = df[config.LABEL_COL_FALLBACK].astype(str).str.lower()
    tumor = int((phenotype == "tumor").sum())
    normal = int((phenotype == "normal").sum())
    return tumor, normal
