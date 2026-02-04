"""MLflow utilities for logging models and data."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd


def set_experiment(experiment_name: str | None) -> None:
    if experiment_name:
        mlflow.set_experiment(experiment_name)


def log_feature_names(feature_names: Iterable[str], artifact_path: str = "metadata"):
    payload = {"feature_names": list(feature_names)}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "feature_names.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_target_name(target: str, artifact_path: str = "metadata"):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "target_name.txt"
        path.write_text(target)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _fingerprint_df(df: pd.DataFrame) -> str:
    data_bytes = df.head(1000).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()


def log_dataset(
    df: pd.DataFrame,
    name: str = "dataset",
    artifact_path: str = "data",
    sample_size: int | None = None,
):
    """Log a dataset snapshot to MLflow.

    If sample_size is provided, logs a sample instead of full dataset.
    """
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("dataset_cols", df.shape[1])
    mlflow.log_param("dataset_fingerprint", _fingerprint_df(df))
