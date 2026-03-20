"""Reusable feature engineering helpers.

This module consolidates the strongest reusable patterns spread across
notebooks 1, 4 and 5:
- weighted aggregation helpers used during dataset construction
- panel/date features for the monthly city dataset
- ratio/composition features for land cover, accessibility and tourism supply
- lag/rolling features for forecasting experiments
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from colombia_tourism.data import DEFAULT_TARGET, parse_panel_months


def _safe_ratio(
    numerator,
    denominator,
    *,
    fill_value: float = np.nan,
):
    numerator = pd.Series(numerator, copy=False)
    denominator = pd.Series(denominator, copy=False)
    denominator = denominator.replace({0: np.nan})
    result = numerator / denominator
    if not np.isnan(fill_value):
        result = result.fillna(fill_value)
    return result


def weighted_mean(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    weight_col: str,
    output_col: str,
) -> pd.DataFrame:
    """Compute weighted mean per group and return a compact DataFrame."""
    tmp = df.copy()
    tmp["_weighted"] = tmp[value_col] * tmp[weight_col]
    grouped = tmp.groupby(group_col).agg(
        weighted_sum=("_weighted", "sum"),
        weight_sum=(weight_col, "sum"),
    )
    grouped[output_col] = grouped["weighted_sum"] / grouped["weight_sum"]
    return grouped[[output_col]].reset_index()


def normalize_minmax(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Min-max normalize selected columns and return a copy."""
    scaler = MinMaxScaler()
    result = df.copy()
    result[list(columns)] = scaler.fit_transform(result[list(columns)])
    return result


def minmax_weighted_score(
    df: pd.DataFrame,
    columns: Iterable[str],
    weights: Iterable[float],
    output_col: str,
) -> pd.DataFrame:
    """Build a weighted score after min-max scaling.

    Example: importance of access points = 0.3 * puntos + 0.7 * total_personas.
    """
    columns = list(columns)
    weights = list(weights)
    if len(columns) != len(weights):
        raise ValueError("columns and weights must have the same length")

    scaled = normalize_minmax(df, columns)
    score = 0
    for col, weight in zip(columns, weights):
        score += weight * scaled[col]
    scaled[output_col] = score
    return scaled


def add_proxy_poverty(
    df: pd.DataFrame,
    gdp_col: str = "Pib Ponderado",
    population_col: str = "population",
    output_col: str = "Proxy Pobreza",
) -> pd.DataFrame:
    """Compute a GDP-per-capita proxy."""
    result = df.copy()
    result[output_col] = result[gdp_col] / result[population_col]
    return result


def add_calendar_features(
    df: pd.DataFrame,
    month_col: str = "Mes",
    *,
    datetime_col: str = "fecha",
    add_cyclical: bool = True,
    drop_datetime_source: bool = False,
) -> pd.DataFrame:
    """Add year/month/quarter features from the project month column."""
    result = df.copy()
    result[datetime_col] = parse_panel_months(result[month_col], strict=False)
    result["year"] = result[datetime_col].dt.year
    result["month"] = result[datetime_col].dt.month
    result["quarter"] = result[datetime_col].dt.quarter
    result["semester"] = ((result["month"] - 1) // 6) + 1

    if add_cyclical:
        radians = 2 * np.pi * result["month"] / 12.0
        result["month_sin"] = np.sin(radians)
        result["month_cos"] = np.cos(radians)

    if drop_datetime_source:
        result = result.drop(columns=[month_col])
    return result


def add_landcover_features(
    df: pd.DataFrame,
    urban_col: str = "Area Urbana",
    rural_col: str = "Area Rural",
    water_col: str = "Area Agua",
) -> pd.DataFrame:
    """Derive land-cover totals, shares and ratios."""
    result = df.copy()
    area_total = result[[urban_col, rural_col, water_col]].sum(axis=1)
    result["area_total"] = area_total
    result["urban_share"] = _safe_ratio(result[urban_col], area_total)
    result["rural_share"] = _safe_ratio(result[rural_col], area_total)
    result["water_share"] = _safe_ratio(result[water_col], area_total)
    result["urban_rural_ratio"] = _safe_ratio(result[urban_col], result[rural_col])
    result["water_to_urban_ratio"] = _safe_ratio(result[water_col], result[urban_col])
    return result


def add_capacity_features(
    df: pd.DataFrame,
    establishments_col: str = "Establecimientos de turismo",
    rooms_col: str = "N Habitaciones",
    beds_col: str = "N Camas",
    tourist_col: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Build tourism-capacity ratios used repeatedly in EDA and modeling."""
    result = df.copy()
    result["rooms_per_establishment"] = _safe_ratio(
        result[rooms_col],
        result[establishments_col],
    )
    result["beds_per_establishment"] = _safe_ratio(
        result[beds_col],
        result[establishments_col],
    )
    result["beds_per_room"] = _safe_ratio(result[beds_col], result[rooms_col])
    if tourist_col in result.columns:
        result["tourists_per_establishment"] = _safe_ratio(
            result[tourist_col],
            result[establishments_col],
        )
        result["tourists_per_room"] = _safe_ratio(result[tourist_col], result[rooms_col])
        result["tourists_per_bed"] = _safe_ratio(result[tourist_col], result[beds_col])
    return result


def add_accessibility_features(
    df: pd.DataFrame,
    *,
    access_distance_col: str = "Distancia a accseos",
    top_distance_col: str = "Distancia al TOP",
    importance_col: str = "importancia accesos",
    roads_col: str = "Nmero Vias",
    entry_col: str = "Entradas Extranjeros Zona",
) -> pd.DataFrame:
    """Create interpretable connectivity features inspired by notebook 1."""
    result = df.copy()
    epsilon = 1e-6
    result["inverse_access_distance"] = 1.0 / (result[access_distance_col] + epsilon)
    result["inverse_top_distance"] = 1.0 / (result[top_distance_col] + epsilon)
    result["roads_x_access_importance"] = result[roads_col] * result[importance_col]
    result["entry_flow_x_access_importance"] = result[entry_col] * result[importance_col]
    result["entry_flow_per_road"] = _safe_ratio(result[entry_col], result[roads_col])
    return result


def add_security_features(
    df: pd.DataFrame,
    *,
    tourist_col: str = DEFAULT_TARGET,
    homicide_col: str = "Homicidios",
    theft_col: str = "Hurtos",
    sexual_col: str = "Delitos Sexuales",
) -> pd.DataFrame:
    """Build aggregate crime pressure features."""
    result = df.copy()
    result["total_crime"] = result[[homicide_col, theft_col, sexual_col]].sum(axis=1)
    result["violent_crime"] = result[homicide_col] + result[sexual_col]
    result["property_crime_share"] = _safe_ratio(result[theft_col], result["total_crime"])
    result["violent_crime_share"] = _safe_ratio(
        result["violent_crime"],
        result["total_crime"],
    )
    if tourist_col in result.columns:
        tourists = result[tourist_col].replace({0: np.nan})
        result["crime_per_1k_tourists"] = result["total_crime"] / tourists * 1000
        result["homicides_per_1k_tourists"] = result[homicide_col] / tourists * 1000
    return result


def add_spend_features(
    df: pd.DataFrame,
    *,
    daily_total_col: str = "Gasto Promedio Diario",
    trip_total_col: str = "Gasto Promedio Viaje",
) -> pd.DataFrame:
    """Build spending composition features for daily and trip-level costs."""
    result = df.copy()

    daily_components = {
        "lodging_daily_share": "Gasto Alojamiento Diario",
        "transport_daily_share": "Gasto Transporte Diario",
        "food_daily_share": "Gasto alimetos Diario",
        "other_daily_share": "Otros Gastos Diario",
    }
    trip_components = {
        "lodging_trip_share": "Gasto  Alojamiento Viaje",
        "transport_trip_share": "Gasto Transporte Viaje",
        "food_trip_share": "Gasto alimetos Viaje",
        "other_trip_share": "Otros Gastos Viaje",
    }

    for output_col, input_col in daily_components.items():
        result[output_col] = _safe_ratio(result[input_col], result[daily_total_col])
    for output_col, input_col in trip_components.items():
        result[output_col] = _safe_ratio(result[input_col], result[trip_total_col])

    result["trip_to_daily_spend_ratio"] = _safe_ratio(
        result[trip_total_col],
        result[daily_total_col],
    )
    return result


def add_density_features(
    df: pd.DataFrame,
    *,
    tourist_col: str = DEFAULT_TARGET,
    urban_col: str = "Area Urbana",
    rural_col: str = "Area Rural",
    water_col: str = "Area Agua",
) -> pd.DataFrame:
    """Create city-density features tying flows to spatial extent."""
    result = df.copy()
    result["tourists_per_urban_area"] = _safe_ratio(result[tourist_col], result[urban_col])
    result["tourists_per_rural_area"] = _safe_ratio(result[tourist_col], result[rural_col])
    result["tourists_per_water_area"] = _safe_ratio(result[tourist_col], result[water_col])
    return result


def add_panel_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_col: str = "Ciudad",
    time_col: str = "Mes",
    lags: Sequence[int] = (1, 3, 6, 12),
    sort: bool = True,
) -> pd.DataFrame:
    """Add city-level lag features for selected columns."""
    result = df.copy()
    result["_panel_date"] = parse_panel_months(result[time_col], strict=False)
    if sort:
        result = result.sort_values([group_col, "_panel_date"]).reset_index(drop=True)

    grouped = result.groupby(group_col, dropna=False)
    for column in columns:
        if column not in result.columns:
            continue
        for lag in lags:
            result[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    return result.drop(columns=["_panel_date"])


def add_panel_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_col: str = "Ciudad",
    time_col: str = "Mes",
    windows: Sequence[int] = (3, 6, 12),
    stats: Sequence[str] = ("mean", "std"),
    min_periods: int = 1,
    sort: bool = True,
) -> pd.DataFrame:
    """Add rolling-window panel features computed within each city."""
    result = df.copy()
    result["_panel_date"] = parse_panel_months(result[time_col], strict=False)
    if sort:
        result = result.sort_values([group_col, "_panel_date"]).reset_index(drop=True)

    valid_stats = {"mean", "sum", "std", "median", "max", "min"}
    unknown = [stat for stat in stats if stat not in valid_stats]
    if unknown:
        raise ValueError(f"Unknown rolling stats: {unknown}")

    for column in columns:
        if column not in result.columns:
            continue
        grouped = result.groupby(group_col, dropna=False)[column]
        shifted = grouped.shift(1)
        for window in windows:
            rolling = shifted.groupby(result[group_col]).rolling(
                window=window,
                min_periods=min_periods,
            )
            for stat in stats:
                feature_name = f"{column}_roll_{stat}_{window}"
                result[feature_name] = getattr(rolling, stat)().reset_index(
                    level=0,
                    drop=True,
                )

    return result.drop(columns=["_panel_date"])


def build_connectivity_index(
    df: pd.DataFrame,
    *,
    components: Sequence[str] = (
        "importancia accesos",
        "Nmero Vias",
        "inverse_access_distance",
        "inverse_top_distance",
    ),
    weights: Sequence[float] | None = None,
    output_col: str = "connectivity_index",
) -> pd.DataFrame:
    """Build a min-max weighted connectivity index."""
    result = df.copy()
    usable_components = [column for column in components if column in result.columns]
    if not usable_components:
        raise ValueError("None of the requested connectivity components are present")

    if weights is None:
        weights = [1 / len(usable_components)] * len(usable_components)
    else:
        weights = list(weights)[: len(usable_components)]
        if len(weights) != len(usable_components):
            raise ValueError("weights must match the number of usable components")

    scored = minmax_weighted_score(
        result,
        columns=usable_components,
        weights=weights,
        output_col=output_col,
    )
    result[output_col] = scored[output_col]
    return result


def build_modeling_features(
    df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET,
    group_col: str = "Ciudad",
    time_col: str = "Mes",
    lag_columns: Sequence[str] | None = None,
    rolling_columns: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 3, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
    rolling_stats: Sequence[str] = ("mean", "std"),
    include_target_history: bool = True,
    include_density_features: bool = False,
) -> pd.DataFrame:
    """Create a model-ready feature table from the final panel dataset."""
    result = df.copy()
    result = add_calendar_features(result, month_col=time_col)
    result = add_landcover_features(result)
    result = add_capacity_features(result, tourist_col=target_col)
    result = add_accessibility_features(result)
    result = add_security_features(result, tourist_col=target_col)
    result = add_spend_features(result)
    if include_density_features and target_col in result.columns:
        result = add_density_features(result, tourist_col=target_col)
    result = build_connectivity_index(result)

    lag_columns = list(lag_columns or [])
    rolling_columns = list(rolling_columns or [])
    if include_target_history and target_col in result.columns:
        if target_col not in lag_columns:
            lag_columns.insert(0, target_col)
        if target_col not in rolling_columns:
            rolling_columns.insert(0, target_col)

    if lag_columns:
        result = add_panel_lag_features(
            result,
            lag_columns,
            group_col=group_col,
            time_col=time_col,
            lags=lags,
        )
    if rolling_columns:
        result = add_panel_rolling_features(
            result,
            rolling_columns,
            group_col=group_col,
            time_col=time_col,
            windows=rolling_windows,
            stats=rolling_stats,
        )

    return result
