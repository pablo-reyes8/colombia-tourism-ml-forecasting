"""Crime dataset shaping helpers."""

from __future__ import annotations

import pandas as pd

from .text import normalize_text_column


def build_crime_base(
    df: pd.DataFrame,
    date_col: str,
    count_col: str,
    output_col: str,
    city_col: str = "MUNICIPIO",
) -> pd.DataFrame:
    """Normalize and aggregate crime data to city-month level."""
    df = df.copy()

    # Some raw sheets include headers in the first row
    if df.columns.tolist() != list(df.iloc[0]):
        pass
    else:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    df[date_col] = pd.to_datetime(df[date_col])
    df["Mes"] = df[date_col].dt.month_name()
    df = normalize_text_column(df, city_col, "Ciudad")

    grouped = df.groupby(["Ciudad", "Mes"])[count_col].sum().reset_index()
    return grouped.rename(columns={count_col: output_col})
