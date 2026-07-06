"""Uploaded CSV shape/column validation with friendly, actionable messages."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    genes_present: list = field(default_factory=list)
    genes_missing: list = field(default_factory=list)


def validate_upload(df: pd.DataFrame, feature_order: list) -> ValidationResult:
    errors = []
    warnings = []

    if len(df) == 0:
        errors.append("The uploaded file has no data rows.")
        return ValidationResult(is_valid=False, errors=errors)

    if len(df) > 1:
        errors.append(
            f"This tool scores one patient at a time; the uploaded file has {len(df)} rows. "
            "Please upload a single-row file (one patient)."
        )
        return ValidationResult(is_valid=False, errors=errors)

    genes_present = [g for g in feature_order if g in df.columns]
    genes_missing = [g for g in feature_order if g not in df.columns]

    if not genes_present:
        errors.append(
            "None of the 20 clinical biomarker panel genes were found as columns in this file. "
            "Download the CSV template below for the exact expected format."
        )
        return ValidationResult(is_valid=False, errors=errors,
                                 genes_present=genes_present, genes_missing=genes_missing)

    if genes_missing:
        warnings.append(
            f"{len(genes_missing)} of 20 panel genes were not found in this file and will be "
            f"treated as missing (imputed as 0): {', '.join(genes_missing)}. "
            "Predictions and explanations based on a partial panel are weaker evidence."
        )

    return ValidationResult(is_valid=True, errors=errors, warnings=warnings,
                             genes_present=genes_present, genes_missing=genes_missing)
