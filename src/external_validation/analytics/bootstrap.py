"""Bootstrap resampling confidence-interval engine."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from . import config, metrics
except ImportError:
    import config
    import metrics

BOOTSTRAP_METRIC_KEYS = ("Accuracy", "ROC_AUC", "F1", "Sensitivity", "Specificity")


@dataclass
class BootstrapResult:
    cohort: str
    distributions: dict  # metric_name -> list[float]
    ci_row: dict          # flat row for the summary CSV


def bootstrap_cohort(cohort_name: str, y_true, y_pred, y_prob, rng: np.random.RandomState) -> BootstrapResult:
    n = len(y_true)
    samples = {k: [] for k in BOOTSTRAP_METRIC_KEYS}

    for _ in range(config.BOOTSTRAP_ITERATIONS):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt, yp, pr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = metrics.compute_diagnostic_metrics(yt, yp, pr)
        for key in BOOTSTRAP_METRIC_KEYS:
            samples[key].append(getattr(m, key))

    def ci(values):
        return (
            float(np.percentile(values, config.CI_LOWER_PERCENTILE)),
            float(np.percentile(values, config.CI_UPPER_PERCENTILE)),
        )

    ci_row = {"Cohort": cohort_name}
    for key in BOOTSTRAP_METRIC_KEYS:
        low, high = ci(samples[key])
        ci_row[f"{key}_Lower"] = low
        ci_row[f"{key}_Upper"] = high

    return BootstrapResult(cohort=cohort_name, distributions=samples, ci_row=ci_row)


def bootstrap_all_cohorts(cohort_arrays: dict[str, tuple]) -> tuple[pd.DataFrame, dict[str, BootstrapResult]]:
    """cohort_arrays: {name: (y_true, y_pred, y_prob)}, iterated in insertion order."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    results = {}
    rows = []
    for name, (y_true, y_pred, y_prob) in cohort_arrays.items():
        result = bootstrap_cohort(name, y_true, y_pred, y_prob, rng)
        results[name] = result
        rows.append(result.ci_row)
    return pd.DataFrame(rows), results
