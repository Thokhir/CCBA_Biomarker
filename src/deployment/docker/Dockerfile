# CCA-BDP container image - runs either the FastAPI prediction service or
# the Streamlit dashboard, selected via the command in docker-compose.yml.
#
# NOTE: this Dockerfile's build has NOT been verified in this environment -
# the Docker CLI is installed but the daemon was not running (Docker
# Desktop not started) when this was written. Build and test with:
#   docker compose -f src/deployment/docker/docker-compose.yml build
# before relying on it. This platform was developed against Python 3.14
# (very recent); if package installs fail on 3.12 below, try bumping the
# base image tag to match the development environment.

FROM python:3.12-slim

WORKDIR /app

# System dependencies for scientific/plotting libraries (matplotlib, lxml
# for python-docx, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default command runs the FastAPI service; docker-compose.yml overrides
# this for the dashboard service.
CMD ["uvicorn", "src.deployment.fastapi.api:app", "--host", "0.0.0.0", "--port", "8000"]
