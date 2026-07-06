"""Single source of truth for diagnostic performance metrics.

Both per-cohort evaluation and pooled/overall evaluation call
compute_diagnostic_metrics() - the same function, not two parallel
implementations that can drift out of sync.
"""
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, brier_score_loss,
)


@dataclass
class DiagnosticMetrics:
    Accuracy: float
    Balanced_Accuracy: float
    Sensitivity: float
    Specificity: float
    Precision: float
    Recall: float
    F1: float
    ROC_AUC: float
    MCC: float
    Kappa: float
    PPV: float
    NPV: float
    FPR: float
    FNR: float
    LR_Positive: float
    LR_Negative: float
    Diagnostic_Odds_Ratio: float
    Brier_Score: float
    TP: int
    TN: int
    FP: int
    FN: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_diagnostic_metrics(y_true, y_pred, y_prob) -> DiagnosticMetrics:
    tn, fp, fn, tp = compute_confusion_counts(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    ppv = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    if specificity == 1:
        lr_positive = np.inf
    else:
        lr_positive = sensitivity / (1 - specificity)

    if specificity == 0:
        lr_negative = np.inf
    else:
        lr_negative = (1 - sensitivity) / specificity

    if np.isinf(lr_positive) or lr_negative == 0:
        dor = np.inf
    else:
        dor = lr_positive / lr_negative

    return DiagnosticMetrics(
        Accuracy=accuracy, Balanced_Accuracy=balanced_accuracy,
        Sensitivity=sensitivity, Specificity=specificity,
        Precision=precision, Recall=recall, F1=f1, ROC_AUC=roc_auc,
        MCC=mcc, Kappa=kappa, PPV=ppv, NPV=npv, FPR=fpr, FNR=fnr,
        LR_Positive=lr_positive, LR_Negative=lr_negative,
        Diagnostic_Odds_Ratio=dor, Brier_Score=brier,
        TP=tp, TN=tn, FP=fp, FN=fn,
    )
