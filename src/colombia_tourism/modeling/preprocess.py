"""Preprocessing builders and panel-data utilities.

This module centralizes the preprocessing logic used in the notebooks:
- sklearn-compatible feature preprocessing (scalers, polynomial features, PCA)
- LOESS pattern smoothing and annual-to-monthly disaggregation
- KNN imputation for mixed categorical/numeric panel data
- Spatial kriging interpolation for missing territorial values
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:  # pragma: no cover - optional dependency
    lowess = None


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _resolve_scaler(scaler: str | Any | None):
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
    raise ValueError(f"Unknown scaler option: {scaler}")


def _resolve_numeric_imputer(
    numeric_imputer: str | Any | None,
    knn_neighbors: int = 5,
):
    if numeric_imputer is None or numeric_imputer == "none":
        return None
    if hasattr(numeric_imputer, "fit"):
        return numeric_imputer

    option = str(numeric_imputer).lower()
    if option in {"mean", "median", "most_frequent"}:
        return SimpleImputer(strategy=option)
    if option == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    raise ValueError(f"Unknown numeric imputer option: {numeric_imputer}")


def _numeric_pipeline(
    scaler: str | Any | None = "standard",
    numeric_imputer: str | Any | None = None,
    knn_neighbors: int = 5,
    poly_degree: int | None = None,
    pca_components: int | None = None,
    pca_variance: float | None = None,
):
    steps: list[tuple[str, Any]] = []

    imputer_obj = _resolve_numeric_imputer(
        numeric_imputer=numeric_imputer,
        knn_neighbors=knn_neighbors,
    )
    if imputer_obj is not None:
        steps.append(("imputer", imputer_obj))

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
    return Pipeline(steps=steps)


def make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    scaler: str | Any | None = "standard",
    numeric_imputer: str | Any | None = None,
    knn_neighbors: int = 5,
    poly_degree: int | None = None,
    pca_components: int | None = None,
    pca_variance: float | None = None,
    remainder: str = "drop",
) -> ColumnTransformer:
    """Create a ColumnTransformer for mixed tabular data."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features or [])

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                _numeric_pipeline(
                    scaler=scaler,
                    numeric_imputer=numeric_imputer,
                    knn_neighbors=knn_neighbors,
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
                _make_one_hot_encoder(),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=remainder)


def _normalize_monthly_pattern(monthly_pattern: Sequence[float]) -> np.ndarray:
    pattern = np.asarray(monthly_pattern, dtype=float)
    if pattern.ndim != 1:
        raise ValueError("monthly_pattern must be one-dimensional")
    if len(pattern) != 12:
        raise ValueError("monthly_pattern must contain 12 values")
    total = pattern.sum()
    if total <= 0:
        raise ValueError("monthly_pattern must have positive sum")
    return pattern / total


def loess_smooth_pattern(
    monthly_pattern: Sequence[float],
    frac: float = 0.4,
    normalize: bool = True,
) -> np.ndarray:
    """Smooth a monthly pattern using LOWESS/LOESS."""
    if lowess is None:
        raise ImportError("statsmodels is required for loess_smooth_pattern")

    pattern = np.asarray(monthly_pattern, dtype=float)
    months = np.arange(1, len(pattern) + 1)
    smoothed = lowess(pattern, months, frac=frac, return_sorted=False)

    if not normalize:
        return smoothed

    total = smoothed.sum()
    if total <= 0:
        raise ValueError("Smoothed pattern has non-positive sum")
    return smoothed / total


def build_noisy_loess_pattern(
    base_pattern: Sequence[float],
    noise_std: float = 0.005,
    floor: float = 0.01,
    random_state: int = 42,
    frac: float = 0.4,
) -> np.ndarray:
    """Build a notebook-style LOESS pattern from a base monthly profile."""
    base_pattern = _normalize_monthly_pattern(base_pattern)
    rng = np.random.default_rng(random_state)
    noisy = base_pattern + rng.normal(0.0, noise_std, 12)
    noisy = np.clip(noisy, floor, None)
    noisy = noisy / noisy.sum()
    return loess_smooth_pattern(noisy, frac=frac, normalize=True)


def decompose_annual_values(
    annual_value: float,
    monthly_pattern: Sequence[float],
) -> np.ndarray:
    """Split an annual scalar into 12 monthly values using a normalized pattern."""
    pattern = _normalize_monthly_pattern(monthly_pattern)
    return pattern * float(annual_value)


def decompose_annual_dataframe(
    df_annual: pd.DataFrame,
    value_columns: Sequence[str],
    monthly_pattern: Sequence[float],
    entity_columns: Sequence[str] | None = None,
    month_col: str = "Mes",
) -> pd.DataFrame:
    """Expand annual rows into 12 monthly rows using a LOESS-smoothed pattern."""
    pattern = _normalize_monthly_pattern(monthly_pattern)
    value_columns = list(value_columns)
    entity_columns = list(entity_columns or [])

    missing = [col for col in (*value_columns, *entity_columns) if col not in df_annual]
    if missing:
        raise KeyError(f"Missing columns in df_annual: {missing}")

    frames: list[pd.DataFrame] = []
    for _, row in df_annual.iterrows():
        monthly = pd.DataFrame({month_col: np.arange(1, 13)})
        for col in entity_columns:
            monthly[col] = row[col]
        for col in value_columns:
            monthly[col] = pattern * float(row[col])
        frames.append(monthly)

    if not frames:
        return pd.DataFrame(columns=[*entity_columns, month_col, *value_columns])

    order = [*entity_columns, month_col, *value_columns]
    return pd.concat(frames, ignore_index=True)[order]


def repeat_rows_for_12_months(
    df: pd.DataFrame,
    month_col: str = "Mes",
) -> pd.DataFrame:
    """Repeat each row 12 times and assign month numbers 1..12."""
    repeated = df.loc[df.index.repeat(12)].reset_index(drop=True)
    repeated[month_col] = (repeated.index % 12) + 1
    return repeated


def _require_pykrige():
    try:
        from pykrige.ok import OrdinaryKriging
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pykrige is required for kriging interpolation") from exc
    return OrdinaryKriging


def _extract_xy_coordinates(
    df: pd.DataFrame,
    geometry_col: str = "geometry",
    lon_col: str = "lng",
    lat_col: str = "lat",
) -> tuple[pd.Series, pd.Series]:
    if geometry_col in df.columns:
        def get_coord(value, coord: str) -> float:
            if value is None:
                return np.nan
            if hasattr(value, coord):
                return float(getattr(value, coord))
            return np.nan

        x = df[geometry_col].map(lambda geom: get_coord(geom, "x"))
        y = df[geometry_col].map(lambda geom: get_coord(geom, "y"))
        return x, y

    if lon_col not in df.columns or lat_col not in df.columns:
        raise KeyError(
            f"Need either '{geometry_col}' or both '{lon_col}' and '{lat_col}' columns"
        )
    return df[lon_col].astype(float), df[lat_col].astype(float)


def kriging_impute(
    df: pd.DataFrame,
    target_column: str,
    group_column: str | None = None,
    geometry_col: str = "geometry",
    lon_col: str = "lng",
    lat_col: str = "lat",
    variogram_model: str = "spherical",
    min_known_points: int = 3,
    fail_silently: bool = True,
) -> pd.DataFrame:
    """Impute missing values using Ordinary Kriging.

    Parameters:
        group_column:
            If provided, kriging is performed independently per group (e.g. month).
    """
    OrdinaryKriging = _require_pykrige()
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found")

    out = df.copy()
    x, y = _extract_xy_coordinates(
        out,
        geometry_col=geometry_col,
        lon_col=lon_col,
        lat_col=lat_col,
    )
    out["_x"] = x
    out["_y"] = y

    if group_column is None:
        groups = [(None, out.index)]
    else:
        if group_column not in out.columns:
            raise KeyError(f"Group column '{group_column}' not found")
        groups = list(out.groupby(group_column, dropna=False).groups.items())

    for _, idx in groups:
        block = out.loc[idx]
        known = block[
            block[target_column].notna() & block["_x"].notna() & block["_y"].notna()
        ]
        missing_idx = block.index[
            block[target_column].isna() & block["_x"].notna() & block["_y"].notna()
        ]

        if len(missing_idx) == 0:
            continue
        if len(known) < min_known_points:
            continue

        try:
            model = OrdinaryKriging(
                known["_x"].to_numpy(),
                known["_y"].to_numpy(),
                known[target_column].astype(float).to_numpy(),
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )
            pred, _ = model.execute(
                "points",
                out.loc[missing_idx, "_x"].to_numpy(),
                out.loc[missing_idx, "_y"].to_numpy(),
            )
            out.loc[missing_idx, target_column] = np.asarray(pred, dtype=float)
        except Exception:
            if not fail_silently:
                raise

    return out.drop(columns=["_x", "_y"])


def kriging_impute_many(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    group_column: str | None = None,
    geometry_col: str = "geometry",
    lon_col: str = "lng",
    lat_col: str = "lat",
    variogram_model: str = "spherical",
    min_known_points: int = 3,
    fail_silently: bool = True,
) -> pd.DataFrame:
    """Apply kriging imputation sequentially to multiple columns."""
    result = df.copy()
    for target in target_columns:
        result = kriging_impute(
            result,
            target_column=target,
            group_column=group_column,
            geometry_col=geometry_col,
            lon_col=lon_col,
            lat_col=lat_col,
            variogram_model=variogram_model,
            min_known_points=min_known_points,
            fail_silently=fail_silently,
        )
    return result


def knn_impute_mixed_panel(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    columns_to_impute: Sequence[str] | None = None,
    n_neighbors: int = 5,
    weights: str = "uniform",
    return_artifacts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """KNN imputation workflow used in notebook 3.

    The function reproduces the same approach:
    1) one-hot encode categorical columns
    2) standardize numeric columns
    3) KNN-impute the transformed matrix
    4) back-transform numeric columns to the original scale
    5) replace selected columns in the original dataframe
    """
    categorical_columns = list(categorical_columns)
    numeric_columns = list(numeric_columns)
    columns_to_impute = list(columns_to_impute or numeric_columns)

    missing = [
        col
        for col in [*categorical_columns, *numeric_columns, *columns_to_impute]
        if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing columns in input dataframe: {missing}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _make_one_hot_encoder(), categorical_columns),
            ("num", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("imputer", KNNImputer(n_neighbors=n_neighbors, weights=weights)),
        ]
    )

    transformed = pipeline.fit_transform(df)

    encoder = preprocessor.named_transformers_["cat"]
    encoded_columns = list(encoder.get_feature_names_out(categorical_columns))
    transformed_columns = encoded_columns + numeric_columns
    transformed_df = pd.DataFrame(
        transformed,
        columns=transformed_columns,
        index=df.index,
    )

    scaler = preprocessor.named_transformers_["num"]
    transformed_df[numeric_columns] = (
        transformed_df[numeric_columns] * scaler.scale_
    ) + scaler.mean_

    out = df.copy()
    out[columns_to_impute] = transformed_df[columns_to_impute]

    if return_artifacts:
        return out, pipeline, transformed_df
    return out


def merge_satellite_features_by_year(
    yearly_bases: Mapping[int, pd.DataFrame],
    satellite_features: pd.DataFrame,
    city_col: str = "Ciudad",
    year_col: str = "año",
) -> dict[int, pd.DataFrame]:
    """Merge annual satellite features into each yearly base."""
    if year_col not in satellite_features.columns:
        raise KeyError(f"Satellite data must include '{year_col}' column")
    if city_col not in satellite_features.columns:
        raise KeyError(f"Satellite data must include '{city_col}' column")

    merged: dict[int, pd.DataFrame] = {}
    for year, base in yearly_bases.items():
        if city_col not in base.columns:
            raise KeyError(f"Base for year {year} does not include '{city_col}'")
        sat_year = satellite_features[satellite_features[year_col] == year]
        merged[year] = pd.merge(base, sat_year, on=city_col, how="left")
    return merged


def concat_yearly_bases(
    yearly_bases: Mapping[int, pd.DataFrame],
    month_col: str = "Mes",
    append_year_to_month: bool = True,
) -> pd.DataFrame:
    """Concatenate yearly panel tables with optional `Mes` -> `Mes-año` formatting."""
    frames: list[pd.DataFrame] = []
    for year, frame in sorted(yearly_bases.items()):
        tmp = frame.copy()
        if append_year_to_month and month_col in tmp.columns:
            tmp[month_col] = tmp[month_col].astype(str) + f"-{year}"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
