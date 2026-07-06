"""CCA-BDP prediction REST API entry point.

Run with: uvicorn src.deployment.fastapi.api:app --reload
(or via Docker - see src/deployment/docker/Dockerfile)

Wraps the same trained model and SHAP explanation pipeline the Streamlit
dashboard (Module 13) uses, for programmatic/system-to-system integration
rather than interactive browser use. Research use only, not a diagnostic
device - same disclaimer as the dashboard's PDF report.
"""
from fastapi import FastAPI

try:
    from .routes import router
except ImportError:
    from routes import router

app = FastAPI(
    title="CCA-BDP Prediction API",
    description="Cholangiocarcinoma biomarker discovery platform - diagnostic prediction and SHAP "
                "explanation API. Research use only, not a diagnostic device.",
    version="1.0.0",
)
app.include_router(router)
