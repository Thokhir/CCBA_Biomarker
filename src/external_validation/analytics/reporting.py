"""JSON summary and human-readable text report writers."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def write_json_summary(model_metadata: dict, n_cohorts: int, total_samples: int,
                        execution_seconds: float, out_path: Path) -> None:
    summary = {
        "model": model_metadata,
        "number_of_external_cohorts": n_cohorts,
        "total_samples": total_samples,
        "execution_time": str(timedelta(seconds=execution_seconds)),
    }
    out_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")


def write_text_report(performance: pd.DataFrame, n_cohorts: int, out_path: Path) -> None:
    lines = [
        "=" * 70,
        "EXTERNAL VALIDATION REPORT",
        "=" * 70,
        "",
        f"Generated : {datetime.now()}",
        f"External Cohorts : {n_cohorts}",
        "",
        "Performance Summary",
        "-" * 70,
        "",
    ]
    for _, row in performance.iterrows():
        lines += [
            f"Cohort : {row['Cohort']}",
            f"Accuracy : {row['Accuracy']:.4f}",
            f"ROC AUC : {row['ROC_AUC']:.4f}",
            f"Sensitivity : {row['Sensitivity']:.4f}",
            f"Specificity : {row['Specificity']:.4f}",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
