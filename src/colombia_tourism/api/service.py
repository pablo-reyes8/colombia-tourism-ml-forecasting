"""Service layer backing the FastAPI application."""

from __future__ import annotations

import io
import os
from dataclasses import asdict
from typing import Any

import pandas as pd

from colombia_tourism.inference import (
    align_features,
    list_registered_model_versions,
    load_model_bundle,
    predict_dataframe,
)


DEFAULT_MODEL_URI = os.getenv("CTF_MODEL_URI")
DEFAULT_REGISTERED_MODEL_NAME = os.getenv("CTF_REGISTERED_MODEL_NAME")
DEFAULT_MODEL_ALIAS = os.getenv("CTF_MODEL_ALIAS", "champion")
DEFAULT_MODEL_VERSION = os.getenv("CTF_MODEL_VERSION")
DEFAULT_ARTIFACT_PATH = os.getenv("CTF_MODEL_ARTIFACT_PATH", "model")


def _bundle_payload(bundle) -> dict[str, Any]:
    payload = asdict(bundle)
    payload["feature_count"] = len(bundle.feature_names)
    return payload


class PredictionService:
    """Reusable prediction/explanation service."""

    def __init__(
        self,
        *,
        default_model_uri: str | None = DEFAULT_MODEL_URI,
        default_registered_model_name: str | None = DEFAULT_REGISTERED_MODEL_NAME,
        default_model_alias: str | None = DEFAULT_MODEL_ALIAS,
        default_model_version: str | None = DEFAULT_MODEL_VERSION,
        default_artifact_path: str = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        self.default_model_uri = default_model_uri
        self.default_registered_model_name = default_registered_model_name
        self.default_model_alias = default_model_alias
        self.default_model_version = default_model_version
        self.default_artifact_path = default_artifact_path

    def _selector_kwargs(self, model: dict[str, Any] | None = None) -> dict[str, Any]:
        model = model or {}
        return {
            "model_uri": model.get("model_uri") or self.default_model_uri,
            "registered_model_name": model.get("registered_model_name")
            or self.default_registered_model_name,
            "model_alias": model.get("model_alias") or self.default_model_alias,
            "model_version": model.get("model_version") or self.default_model_version,
            "artifact_path": model.get("artifact_path") or self.default_artifact_path,
        }

    def _load_cached_bundle(
        self,
        model_uri: str | None,
        registered_model_name: str | None,
        model_alias: str | None,
        model_version: str | None,
        artifact_path: str,
    ):
        return load_model_bundle(
            model_uri=model_uri,
            registered_model_name=registered_model_name,
            model_alias=model_alias,
            model_version=model_version,
            artifact_path=artifact_path,
        )

    def resolve(self, model: dict[str, Any] | None = None):
        selector = self._selector_kwargs(model)
        return self._load_cached_bundle(
            selector["model_uri"],
            selector["registered_model_name"],
            selector["model_alias"],
            selector["model_version"],
            selector["artifact_path"],
        )

    def default_model_info(self) -> dict[str, Any] | None:
        try:
            _, bundle = self.resolve()
        except Exception:
            return None
        return _bundle_payload(bundle)

    def readiness(self) -> bool:
        try:
            self.resolve()
            return True
        except Exception:
            return False

    def feature_schema(self, model: dict[str, Any] | None = None) -> dict[str, Any]:
        _, bundle = self.resolve(model)
        return {
            "model": _bundle_payload(bundle),
            "feature_names": bundle.feature_names,
            "target_name": bundle.target_name,
            "feature_count": len(bundle.feature_names),
        }

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        *,
        bundle,
        target: str | None = None,
        strict_features: bool = False,
        fill_missing_value: float = 0.0,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        prepared = df.copy()
        if target and target in prepared.columns:
            prepared = prepared.drop(columns=[target])

        missing: list[str] = []
        extra: list[str] = []
        if bundle.feature_names:
            prepared, missing, extra = align_features(
                prepared,
                bundle.feature_names,
                fill_value=fill_missing_value,
                strict=strict_features,
            )
        return prepared, missing, extra

    def predict_records(
        self,
        records: list[dict[str, Any]],
        *,
        model: dict[str, Any] | None = None,
        target: str | None = None,
        strict_features: bool = False,
        fill_missing_value: float = 0.0,
        prediction_column: str = "prediction",
        include_input_records: bool = False,
    ) -> dict[str, Any]:
        estimator, bundle = self.resolve(model)
        prepared, missing, extra = self._prepare_dataframe(
            pd.DataFrame(records),
            bundle=bundle,
            target=target,
            strict_features=strict_features,
            fill_missing_value=fill_missing_value,
        )
        predictions = predict_dataframe(estimator, prepared)
        return {
            "model": _bundle_payload(bundle),
            "n_records": len(prepared),
            "prediction_column": prediction_column,
            "predictions": [float(value) for value in predictions],
            "missing_features": missing,
            "extra_features": extra,
            "aligned_records": prepared.to_dict(orient="records")
            if include_input_records
            else None,
        }

    def predict_single_record(self, record: dict[str, Any], **kwargs) -> dict[str, Any]:
        response = self.predict_records([record], **kwargs)
        response["prediction"] = response["predictions"][0]
        return response

    def predict_csv_bytes(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        model: dict[str, Any] | None = None,
        target: str | None = None,
        strict_features: bool = False,
        fill_missing_value: float = 0.0,
        prediction_column: str = "prediction",
        include_input_records: bool = False,
    ) -> dict[str, Any]:
        df = pd.read_csv(io.BytesIO(content))
        response = self.predict_records(
            df.to_dict(orient="records"),
            model=model,
            target=target,
            strict_features=strict_features,
            fill_missing_value=fill_missing_value,
            prediction_column=prediction_column,
            include_input_records=include_input_records,
        )
        response["filename"] = filename
        return response

    def explain_records(
        self,
        records: list[dict[str, Any]],
        *,
        model: dict[str, Any] | None = None,
        target: str | None = None,
        strict_features: bool = False,
        fill_missing_value: float = 0.0,
        max_samples: int = 200,
    ) -> dict[str, Any]:
        estimator, bundle = self.resolve(model)
        prepared, missing, extra = self._prepare_dataframe(
            pd.DataFrame(records),
            bundle=bundle,
            target=target,
            strict_features=strict_features,
            fill_missing_value=fill_missing_value,
        )
        from colombia_tourism.interpretation import shap_summary

        summary, _, _ = shap_summary(estimator, prepared, max_samples=max_samples)
        return {
            "model": _bundle_payload(bundle),
            "n_records": len(prepared),
            "mean_abs_shap": summary.to_dict(orient="records"),
            "missing_features": missing,
            "extra_features": extra,
        }

    def list_registered_versions(
        self,
        registered_model_name: str | None = None,
    ) -> dict[str, Any]:
        name = registered_model_name or self.default_registered_model_name
        if not name:
            raise ValueError("registered_model_name is required")
        return {
            "registered_model_name": name,
            "versions": list_registered_model_versions(name),
        }
