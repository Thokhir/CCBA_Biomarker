"""Runs the trained RF model on a single patient's aligned feature matrix
and buckets the probability into a confidence label.

Confidence thresholds are copied verbatim from
src/external_validation/prediction/external_prediction_engine.py so the
dashboard's confidence badge matches the convention already used in every
Module 8 cohort report. Note this convention buckets on the raw tumor
probability, not distance from 0.5 - so a confidently-Normal prediction
(e.g. probability=0.05) is labeled "Low" confidence, not "Very High". This
asymmetry is inherited from the existing platform convention, not
introduced here; kept as-is for consistency rather than silently changed.
"""
from dataclasses import dataclass


@dataclass
class PredictionResult:
    predicted_label: int
    tumor_probability: float
    confidence: str


def _confidence_bucket(tumor_probability: float) -> str:
    if tumor_probability >= 0.90:
        return "Very High"
    elif tumor_probability >= 0.75:
        return "High"
    elif tumor_probability >= 0.60:
        return "Moderate"
    return "Low"


def predict_patient(model, patient_x) -> PredictionResult:
    predicted_label = int(model.predict(patient_x)[0])
    tumor_probability = float(model.predict_proba(patient_x)[0, 1])
    return PredictionResult(
        predicted_label=predicted_label,
        tumor_probability=tumor_probability,
        confidence=_confidence_bucket(tumor_probability),
    )
