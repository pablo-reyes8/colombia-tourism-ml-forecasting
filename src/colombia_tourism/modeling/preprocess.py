"""Preprocessing builders for ML pipelines."""

from __future__ import annotations

from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)


def _resolve_scaler(scaler):
    if scaler is None or scaler == "none":
        return None
    if hasattr(scaler, "fit"):
        return scaler
    if scaler == "standard":
        return StandardScaler()
    if scaler == "minmax":
        return MinMaxScaler()
    if scaler == "robust":
        return RobustScaler()
    if scaler == "maxabs":
        return MaxAbsScaler()
    raise ValueError("Unknown scaler option")


def _numeric_pipeline(
    scaler: str | None = "standard",
    poly_degree: int | None = None,
    pca_components: int | None = None,
    pca_variance: float | None = None,
):
    steps = []
    scaler_obj = _resolve_scaler(scaler)
    if scaler_obj is not None:
        steps.append(("scaler", scaler_obj))

    if poly_degree is not None and poly_degree > 1:
        steps.append(
            (
                "poly",
                PolynomialFeatures(degree=poly_degree, include_bias=False),
            )
        )

    if pca_components is not None or pca_variance is not None:
        n_components = pca_components if pca_components is not None else pca_variance
        steps.append(("pca", PCA(n_components=n_components)))

    if not steps:
        return "passthrough"
    return Pipeline(steps)


def make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    scaler: str | None = "standard",
    poly_degree: int | None = None,
    pca_components: int | None = None,
    pca_variance: float | None = None,
    remainder: str = "drop",
) -> ColumnTransformer:
    """Create a ColumnTransformer with optional polynomial features and PCA."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features or [])

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                _numeric_pipeline(
                    scaler=scaler,
                    poly_degree=poly_degree,
                    pca_components=pca_components,
                    pca_variance=pca_variance,
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=remainder)
