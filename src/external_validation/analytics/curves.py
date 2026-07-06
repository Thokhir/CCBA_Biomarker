"""ROC / PR / calibration curve table computation."""
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

try:
    from . import config
except ImportError:
    import config


def compute_roc_table(y_true, y_prob) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({
        "False_Positive_Rate": fpr,
        "True_Positive_Rate": tpr,
        "Threshold": thresholds,
    })


def compute_pr_table(y_true, y_prob) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # sklearn returns thresholds with len = len(precision) - 1
    return pd.DataFrame({
        "Recall": recall[:-1],
        "Precision": precision[:-1],
        "Threshold": thresholds,
    })


def compute_calibration_table(y_true, y_prob) -> pd.DataFrame:
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob,
        n_bins=config.CALIBRATION_N_BINS,
        strategy=config.CALIBRATION_STRATEGY,
    )
    return pd.DataFrame({
        "Mean_Predicted_Probability": mean_predicted,
        "Fraction_of_Positives": fraction_positive,
    })
