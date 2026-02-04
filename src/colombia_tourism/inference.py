"""Inference utilities for loading models and predicting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def _is_mlflow_dir(path: Path) -> bool:
    return path.is_dir() and (path / "MLmodel").exists()


def load_model(model_uri: str):
    if model_uri is None:
        raise ValueError("model_uri is required")

    if model_uri.startswith("runs:") or model_uri.startswith("models:"):
        return mlflow.pyfunc.load_model(model_uri)

    path = Path(model_uri)
    if _is_mlflow_dir(path):
        return mlflow.pyfunc.load_model(str(path))

    if path.suffix in {".pkl", ".joblib"}:
        return joblib.load(path)

    raise ValueError("Unsupported model_uri. Use MLflow URI or .pkl/.joblib path.")


def predict_dataframe(model, df: pd.DataFrame):
    preds = model.predict(df)
    return preds


def parse_run_id(model_uri: str) -> str | None:
    if not model_uri.startswith("runs:/"):
        return None
    parts = model_uri.split("/")
    if len(parts) < 2:
        return None
    return parts[1]


def download_run_artifact(run_id: str, artifact_path: str) -> Path:
    client = MlflowClient()
    tmpdir = tempfile.mkdtemp(prefix="mlflow_artifact_")
    local_path = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
    return Path(local_path)


def load_feature_names_from_run(run_id: str) -> list[str] | None:
    try:
        path = download_run_artifact(run_id, "metadata/feature_names.json")
    except Exception:
        return None

    try:
        payload = json.loads(path.read_text())
        return payload.get("feature_names")
    except Exception:
        return None


def align_features(df: pd.DataFrame, feature_names: Iterable[str]):
    feature_names = list(feature_names)
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]

    aligned = df.copy()
    for col in missing:
        aligned[col] = 0

    aligned = aligned[feature_names]
    return aligned, missing, extra
