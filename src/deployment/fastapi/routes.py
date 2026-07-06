"""API route handlers for the CCA-BDP prediction service."""
from fastapi import APIRouter, HTTPException

try:
    from .schemas import PredictionRequest, PredictionResponse, ExplanationResponse, HealthResponse
    from .prediction_service import PredictionService
except ImportError:
    from schemas import PredictionRequest, PredictionResponse, ExplanationResponse, HealthResponse
    from prediction_service import PredictionService

router = APIRouter()
_service: PredictionService = None


def get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@router.get("/health", response_model=HealthResponse)
def health():
    service = get_service()
    return HealthResponse(status="ok", n_panel_genes=len(service.feature_order))


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    service = get_service()
    try:
        result = service.predict(request.sample_id, request.gene_expression)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PredictionResponse(**result)


@router.post("/explain", response_model=ExplanationResponse)
def explain(request: PredictionRequest):
    service = get_service()
    try:
        result = service.explain(request.sample_id, request.gene_expression)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ExplanationResponse(**result)
