"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSelector(BaseModel):
    """Reference to the model that should serve the request."""

    model_config = ConfigDict(extra="forbid")

    model_uri: str | None = Field(
        default=None,
        description="Explicit MLflow URI or local model path.",
    )
    registered_model_name: str | None = Field(
        default=None,
        description="MLflow registered model name. Used when model_uri is not provided.",
    )
    model_alias: str | None = Field(
        default="champion",
        description="MLflow model alias, typically 'champion'.",
    )
    model_version: str | None = Field(
        default=None,
        description="Specific MLflow model version. Overrides alias if provided.",
    )
    artifact_path: str = Field(default="model", description="Artifact path inside the MLflow run.")


class PredictionOptions(BaseModel):
    """Serving-time feature handling options."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(
        default=None,
        description="Optional target column to drop from incoming records.",
    )
    strict_features: bool = Field(
        default=False,
        description="Reject requests when extra or missing features are detected.",
    )
    fill_missing_value: float = Field(
        default=0.0,
        description="Value injected for missing model features when strict_features is false.",
    )
    prediction_column: str = Field(
        default="prediction",
        description="Output column name used in file responses and metadata.",
    )
    include_input_records: bool = Field(
        default=False,
        description="Echo the aligned input records in the response.",
    )


class PredictRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of city-feature records to score.",
        examples=[
            [
                {
                    "Ciudad": "medellin",
                    "Temperatura": 24.5,
                    "Pib Ponderado": 1500.0,
                    "Inflacion": 8.3,
                    "Eventos": 3,
                    "Area Urbana": 120.0,
                    "Area Rural": 80.0,
                    "Area Agua": 4.0,
                    "N Camas": 25000,
                }
            ]
        ],
    )
    model_uri: str | None = Field(
        default=None,
        description="Deprecated top-level model URI kept for backward compatibility.",
    )
    target: str | None = Field(
        default=None,
        description="Deprecated top-level target kept for backward compatibility.",
    )
    model: ModelSelector = Field(default_factory=ModelSelector)
    options: PredictionOptions = Field(default_factory=PredictionOptions)


class PredictSingleRequest(BaseModel):
    """Single-record prediction request."""

    model_config = ConfigDict(extra="forbid")

    record: dict[str, Any]
    model_uri: str | None = None
    target: str | None = None
    model: ModelSelector = Field(default_factory=ModelSelector)
    options: PredictionOptions = Field(default_factory=PredictionOptions)


class ExplainRequest(BaseModel):
    """Global explanation request over a batch of records."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1)
    model_uri: str | None = Field(default=None)
    target: str | None = Field(default=None)
    model: ModelSelector = Field(default_factory=ModelSelector)
    options: PredictionOptions = Field(default_factory=PredictionOptions)
    max_samples: int = Field(default=200, ge=10, le=2000)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_ready: bool
    default_model: dict[str, Any] | None = None


class ModelInfoResponse(BaseModel):
    model: dict[str, Any]


class FeatureSchemaResponse(BaseModel):
    model: dict[str, Any]
    feature_names: list[str]
    target_name: str | None
    feature_count: int


class PredictionResponse(BaseModel):
    model: dict[str, Any]
    n_records: int
    prediction_column: str
    predictions: list[float]
    missing_features: list[str]
    extra_features: list[str]
    aligned_records: list[dict[str, Any]] | None = None


class PredictionFileResponse(PredictionResponse):
    filename: str | None = None


class ExplanationResponse(BaseModel):
    model: dict[str, Any]
    n_records: int
    mean_abs_shap: list[dict[str, Any]]
    missing_features: list[str]
    extra_features: list[str]


class RegisteredModelsResponse(BaseModel):
    registered_model_name: str
    versions: list[dict[str, Any]]
