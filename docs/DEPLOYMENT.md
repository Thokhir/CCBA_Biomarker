# CCA-BDP Deployment Guide

Four deployment targets, in order of setup effort.

## 1. Streamlit Community Cloud (least effort)

No new code needed - `requirements.txt` and `.streamlit/config.toml` already exist at the project root.

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at this repo with main file path `src/dashboard/app.py`.
3. Streamlit Cloud installs `requirements.txt` and applies `.streamlit/config.toml` automatically.

Note: `results/` (model artifacts, precomputed CSVs/figures) must be committed to the repo (already is) since Streamlit Cloud only has what's in the git repository - there is no separate data volume.

## 2. FastAPI REST API (local or any host)

```bash
pip install -r requirements.txt
uvicorn src.deployment.fastapi.api:app --host 0.0.0.0 --port 8000
```

Verified this session: `/health`, `/predict`, `/explain` all return correct results matching the Streamlit dashboard's predictions exactly (same underlying model/SHAP pipeline, reused not reimplemented). Interactive API docs at `http://localhost:8000/docs` (FastAPI's built-in Swagger UI).

## 3. Docker (both services, containerized)

```bash
docker compose -f src/deployment/docker/docker-compose.yml up --build
```

Starts the FastAPI service on `:8000` and the Streamlit dashboard on `:8501` from the same image. **Not built/tested in this environment** - the Docker CLI is installed but the daemon (Docker Desktop) was not running when this was written. Build and smoke-test both services before relying on this in production; if package installs fail on the `python:3.12-slim` base image, this platform was developed against Python 3.14, so bumping the base image tag may be needed.

## 4. Desktop executable (Windows, most effort)

```bash
python src/deployment/desktop/build_executable.py
```

Bundles the Streamlit dashboard into a standalone `.exe` via PyInstaller (`src/deployment/desktop/launcher.py` starts the app and opens a browser tab automatically). **Not executed in this session** - the dependency footprint (scikit-learn, shap, streamlit, gseapy, xgboost, lightgbm, reportlab, python-docx, kaleido, matplotlib, lifelines) is large, and PyInstaller bundles this size commonly need iterative `--hidden-import` fixes on the first attempt. Budget time for a debug cycle; `streamlit` and `shap` are the most common causes of missing-module errors in bundles like this.

## What's actually verified vs. not

| Target | Verified this session |
|---|---|
| Streamlit Cloud | Config files confirmed present and correct; actual cloud deploy not attempted (requires a GitHub push + Streamlit Cloud account) |
| FastAPI | **Yes** - ran locally, tested `/health`, `/predict`, `/explain` with real requests, confirmed results match the dashboard exactly |
| Docker | No - Docker daemon unavailable in this environment |
| Desktop .exe | No - not run, given the build/debug time cost for this dependency footprint |
