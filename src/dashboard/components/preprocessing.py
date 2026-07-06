"""Builds a single-patient feature matrix from an uploaded row: 0-fills any
missing clinical-panel genes and reorders columns to match the model's
feature_order, tracking which genes were imputed.

Adapted from src/explainable_ai/data_loader.py::build_external_feature_matrix
(same 0-fill + reorder behavior), rather than re-deriving that logic, since
it's already correct and verified (genes_imputed tracking, exact feature
ordering) for the batch external-cohort case.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class PatientMatrix:
    sample_id: str
    X: pd.DataFrame
    genes_provided: list
    genes_imputed: list


def build_patient_matrix(raw_row: pd.Series, feature_order: list, sample_id: str) -> PatientMatrix:
    genes_provided = [g for g in feature_order if g in raw_row.index]
    genes_imputed = [g for g in feature_order if g not in raw_row.index]

    values = {}
    for gene in genes_provided:
        values[gene] = float(raw_row[gene])
    for gene in genes_imputed:
        values[gene] = 0.0

    X = pd.DataFrame([values], columns=feature_order)
    return PatientMatrix(sample_id=sample_id, X=X, genes_provided=genes_provided, genes_imputed=genes_imputed)
