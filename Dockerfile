FROM python:3.10-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MLFLOW_TRACKING_URI=http://mlflow:5000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY apps /app/apps
COPY docs /app/docs
COPY README.md /app/README.md


FROM runtime-base AS mlflow-server
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:/app/mlruns", "--default-artifact-root", "/app/mlruns"]


FROM runtime-base AS data-pipeline
CMD ["python", "scripts/package_data.py", "--data", "Data/Base Final1.csv", "--output-dir", "artifacts/data/base_final_package"]


FROM runtime-base AS trainer
CMD ["python", "scripts/train.py", "--data-package-dir", "artifacts/data/base_final_package", "--model", "xgboost", "--registered-model-name", "colombia-tourism-forecasting"]


FROM runtime-base AS drift-monitor
CMD ["python", "scripts/check_drift.py", "--reference-dir", "artifacts/data/base_final_package", "--current-data", "Data/Base Final1.csv", "--output-dir", "artifacts/drift/latest"]


FROM runtime-base AS api-server
EXPOSE 8000
CMD ["uvicorn", "colombia_tourism.api.server:app", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime-base AS streamlit-app
EXPOSE 8501
CMD ["streamlit", "run", "apps/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
