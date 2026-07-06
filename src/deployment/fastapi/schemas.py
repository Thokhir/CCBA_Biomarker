"""Pydantic request/response models for the CCA-BDP prediction API."""
from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sample_id: str = Field(default="patient_1", description="Patient/sample identifier")
    gene_expression: dict = Field(
        ..., description="Gene symbol -> log-CPM expression value. Missing panel genes are imputed as 0.",
        examples=[{"OTC": 3.2, "USH2A": 6.5, "RAD51": 7.9}],
    )


class PredictionResponse(BaseModel):
    sample_id: str
    predicted_label: int
    predicted_class: str
    tumor_probability: float
    confidence: str
    genes_provided: list
    genes_imputed: list


class GeneContribution(BaseModel):
    gene_name: str
    shap_value: float
    is_imputed: bool


class ExplanationResponse(BaseModel):
    sample_id: str
    base_value: float
    predicted_probability: float
    contributions: list[GeneContribution]


class HealthResponse(BaseModel):
    status: str
    model_training_date: Optional[str] = None
    n_panel_genes: int
