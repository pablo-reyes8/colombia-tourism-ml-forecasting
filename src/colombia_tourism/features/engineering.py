"""Reusable feature engineering helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
