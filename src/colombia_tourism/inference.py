"""Inference utilities for loading, resolving and serving models."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd


def _require_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        raise ImportError(
            "mlflow is required for MLflow URIs and registry-backed inference."
        ) from exc
    return mlflow, MlflowClient


def _is_mlflow_dir(path: Path) -> bool:
    return path.is_dir() and (path / "MLmodel").exists()


@dataclass(frozen=True)
class ResolvedModelBundle:
    """Resolved model plus serving metadata."""

    model_uri: str
    source: str
    artifact_path: str
    registered_model_name: str | None
    model_alias: str | None
    model_version: str | None
    run_id: str | None
    feature_names: list[str]
    target_name: str | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_count"] = len(self.feature_names)
        return payload


def parse_run_id(model_uri: str) -> str | None:
    """Extract the run ID from ``runs:/<RUN_ID>/...`` URIs."""
    if not model_uri.startswith("runs:/"):
        return None
    parts = model_uri.split("/")
    if len(parts) < 2:
        return None
    return parts[1]


def _resolve_mlflow_models_uri(
    registered_model_name: str,
    *,
    model_alias: str | None = None,
    model_version: str | None = None,
) -> str:
    if model_alias and model_version:
        raise ValueError("Specify either model_alias or model_version, not both")
    if model_version:
        return f"models:/{registered_model_name}/{model_version}"
    alias = model_alias or "champion"
    return f"models:/{registered_model_name}@{alias}"


def resolve_model_uri(
    *,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    model_alias: str | None = None,
    model_version: str | None = None,
    artifact_path: str = "model",
) -> str:
    """Resolve a final MLflow or filesystem model URI."""
    if model_uri:
        return model_uri
    if registered_model_name:
        return _resolve_mlflow_models_uri(
            registered_model_name,
            model_alias=model_alias,
            model_version=model_version,
        )
    raise ValueError("Either model_uri or registered_model_name must be provided")


def _canonical_model_uri(model_uri: str) -> str:
    if not model_uri.startswith("models:/"):
        return model_uri
    spec = model_uri.removeprefix("models:/")
    if "@" not in spec:
        return model_uri
    name, alias = spec.split("@", 1)
    _, MlflowClient = _require_mlflow()
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(name, alias)
    return f"models:/{name}/{version_info.version}"


@lru_cache(maxsize=16)
def _load_model_cached(resolved_uri: str):
    if resolved_uri.startswith("runs:") or resolved_uri.startswith("models:"):
        mlflow, _ = _require_mlflow()
        return mlflow.pyfunc.load_model(resolved_uri)

    path = Path(resolved_uri)
    if _is_mlflow_dir(path):
        mlflow, _ = _require_mlflow()
        return mlflow.pyfunc.load_model(str(path))

    if path.suffix in {".pkl", ".joblib"}:
        return joblib.load(path)

    raise ValueError("Unsupported model_uri. Use MLflow URI or .pkl/.joblib path.")


def load_model(model_uri: str):
    """Load a model from MLflow or a local serialized artifact."""
    if model_uri is None:
        raise ValueError("model_uri is required")
    resolved_uri = _canonical_model_uri(model_uri)
    return _load_model_cached(resolved_uri)


def predict_dataframe(model, df: pd.DataFrame):
    preds = model.predict(df)
    return preds


def download_run_artifact(run_id: str, artifact_path: str) -> Path:
    _, MlflowClient = _require_mlflow()
    client = MlflowClient()
    tmpdir = tempfile.mkdtemp(prefix="mlflow_artifact_")
    local_path = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
    return Path(local_path)


def _json_from_path(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_run_from_registered_uri(model_uri: str) -> tuple[str | None, str | None, str | None]:
    if not model_uri.startswith("models:/"):
        return None, None, None

    spec = model_uri.removeprefix("models:/")
    _, MlflowClient = _require_mlflow()
    client = MlflowClient()
    try:
        if "@" in spec:
            name, alias = spec.split("@", 1)
            version_info = client.get_model_version_by_alias(name, alias)
            return version_info.run_id, name, str(version_info.version)
        parts = spec.split("/", 1)
        if len(parts) == 2:
            name, version = parts
            version_info = client.get_model_version(name, version)
            return version_info.run_id, name, str(version_info.version)
    except Exception:
        return None, None, None
    return None, None, None


def resolve_run_id(model_uri: str) -> str | None:
    """Resolve a run ID from runs:/ or models:/ URIs."""
    run_id = parse_run_id(model_uri)
    if run_id:
        return run_id
    run_id, _, _ = _resolve_run_from_registered_uri(model_uri)
    return run_id


def _load_artifact_payload_from_run(
    run_id: str,
    artifact_candidates: Iterable[str],
) -> tuple[Any, str | None]:
    for artifact in artifact_candidates:
        try:
            path = download_run_artifact(run_id, artifact)
        except Exception:
            continue

        if path.suffix == ".json":
            payload = _json_from_path(path)
        else:
            try:
                payload = path.read_text(encoding="utf-8").strip()
            except Exception:
                payload = None

        if payload is not None:
            return payload, artifact
    return None, None


def _load_artifact_payload_from_local(
    model_uri: str,
    artifact_candidates: Iterable[str],
) -> tuple[Any, str | None]:
    path = Path(model_uri)
    candidate_paths: list[Path] = []

    def _variants(base: Path, artifact: str) -> list[Path]:
        artifact_path = Path(artifact)
        variants = [base / artifact_path]
        if len(artifact_path.parts) > 1:
            variants.append(base / Path(*artifact_path.parts[1:]))
        variants.append(base / artifact_path.name)
        return variants

    if path.is_dir():
        for artifact in artifact_candidates:
            candidate_paths.extend(_variants(path, artifact))
    else:
        for artifact in artifact_candidates:
            candidate_paths.extend(_variants(path.parent, artifact))
        candidate_paths.append(path.with_suffix(".metadata.json"))

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            payload = _json_from_path(candidate)
        else:
            try:
                payload = candidate.read_text(encoding="utf-8").strip()
            except Exception:
                payload = None
        if payload is not None:
            return payload, str(candidate)
    return None, None


def load_model_metadata(model_uri: str) -> dict[str, Any]:
    """Load logged model metadata from MLflow artifacts or local files."""
    artifact_candidates = (
        "metadata/model_metadata.json",
        "model/model_metadata.json",
        "model_metadata.json",
    )
    run_id = resolve_run_id(model_uri)
    if run_id:
        payload, _ = _load_artifact_payload_from_run(run_id, artifact_candidates)
        if isinstance(payload, dict):
            return payload

    payload, _ = _load_artifact_payload_from_local(model_uri, artifact_candidates)
    if isinstance(payload, dict):
        return payload
    return {}


def load_feature_names_from_run(run_id: str) -> list[str] | None:
    payload, _ = _load_artifact_payload_from_run(
        run_id,
        ("metadata/feature_names.json", "model/feature_names.json", "feature_names.json"),
    )
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        feature_names = payload.get("feature_names")
        if isinstance(feature_names, list):
            return feature_names
    return None


def load_feature_names(model_uri: str) -> list[str] | None:
    run_id = resolve_run_id(model_uri)
    if run_id:
        feature_names = load_feature_names_from_run(run_id)
        if feature_names:
            return feature_names

    payload, _ = _load_artifact_payload_from_local(
        model_uri,
        (
            "metadata/feature_names.json",
            "model/feature_names.json",
            "feature_names.json",
        ),
    )
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        feature_names = payload.get("feature_names")
        if isinstance(feature_names, list):
            return feature_names
    return None


def load_target_name(model_uri: str) -> str | None:
    """Load the target name logged alongside the model."""
    run_id = resolve_run_id(model_uri)
    artifact_candidates = (
        "metadata/target_name.txt",
        "model/target_name.txt",
        "target_name.txt",
    )
    if run_id:
        payload, _ = _load_artifact_payload_from_run(run_id, artifact_candidates)
        if isinstance(payload, str) and payload:
            return payload

    payload, _ = _load_artifact_payload_from_local(model_uri, artifact_candidates)
    if isinstance(payload, str) and payload:
        return payload
    return None


def align_features(
    df: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    fill_value: float = 0.0,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Align input data to the model feature order."""
    feature_names = list(feature_names)
    missing = [column for column in feature_names if column not in df.columns]
    extra = [column for column in df.columns if column not in feature_names]

    if strict and (missing or extra):
        raise ValueError(
            f"Feature mismatch. Missing: {missing or '[]'} | Extra: {extra or '[]'}"
        )

    aligned = df.copy()
    for column in missing:
        aligned[column] = fill_value

    aligned = aligned[feature_names]
    return aligned, missing, extra


def resolve_model_bundle(
    *,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    model_alias: str | None = None,
    model_version: str | None = None,
    artifact_path: str = "model",
) -> ResolvedModelBundle:
    """Resolve a model reference into a serving bundle with metadata."""
    requested_uri = resolve_model_uri(
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        model_alias=model_alias,
        model_version=model_version,
        artifact_path=artifact_path,
    )
    resolved_uri = _canonical_model_uri(requested_uri)
    metadata = load_model_metadata(resolved_uri)
    feature_names = load_feature_names(resolved_uri) or []
    target_name = load_target_name(resolved_uri)
    run_id = resolve_run_id(resolved_uri)
    registry_run_id, registry_name, registry_version = _resolve_run_from_registered_uri(
        resolved_uri
    )
    registry_alias = None
    if requested_uri.startswith("models:/"):
        spec = requested_uri.removeprefix("models:/")
        if "@" in spec:
            _, registry_alias = spec.split("@", 1)
    source = "mlflow_registry" if resolved_uri.startswith("models:/") else "mlflow_run"
    if not resolved_uri.startswith(("models:/", "runs:/")):
        source = "local_artifact"

    return ResolvedModelBundle(
        model_uri=resolved_uri,
        source=source,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name or registry_name,
        model_alias=(model_alias or registry_alias) if resolved_uri.startswith("models:/") else None,
        model_version=model_version or registry_version,
        run_id=run_id or registry_run_id,
        feature_names=feature_names,
        target_name=target_name,
        metadata=metadata,
    )


def load_model_bundle(
    *,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    model_alias: str | None = None,
    model_version: str | None = None,
    artifact_path: str = "model",
) -> tuple[Any, ResolvedModelBundle]:
    """Load a model together with its metadata bundle."""
    bundle = resolve_model_bundle(
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        model_alias=model_alias,
        model_version=model_version,
        artifact_path=artifact_path,
    )
    model = load_model(bundle.model_uri)
    return model, bundle


def list_registered_model_versions(registered_model_name: str) -> list[dict[str, Any]]:
    """List versions for one registered model in MLflow."""
    _, MlflowClient = _require_mlflow()
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{registered_model_name}'")
    rows: list[dict[str, Any]] = []
    for version in versions:
        rows.append(
            {
                "name": version.name,
                "version": str(version.version),
                "run_id": version.run_id,
                "current_stage": getattr(version, "current_stage", None),
                "status": getattr(version, "status", None),
                "source": getattr(version, "source", None),
                "description": getattr(version, "description", None),
            }
        )
    return rows
