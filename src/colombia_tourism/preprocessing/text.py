"""Text normalization helpers used across datasets."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping

import pandas as pd

_CITY_REPLACEMENTS: Mapping[str, str] = {
    "sanandresdetumaco": "tumaco",
    "bogotadc": "bogota",
}


def remove_accents(value: str) -> str:
    if value is None:
        return value
    if pd.isna(value):
        return value
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", str(value))
        if unicodedata.category(ch) != "Mn"
    )


def normalize_text_column(
    df: pd.DataFrame,
    column: str,
    new_name: str | None = None,
) -> pd.DataFrame:
    """Standardize city names and similar text columns.

    Steps:
    - lowercase
    - remove spaces and punctuation
    - replace known variants
    - remove accents
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    series = df[column].astype(str).str.lower()
    series = series.str.replace(" ", "", regex=False)
    series = series.str.replace(r"[^\w\s]", "", regex=True)
    series = series.str.replace(r"ct$", "", regex=True)
    for old, new in _CITY_REPLACEMENTS.items():
        series = series.str.replace(old, new, regex=False)
    series = series.apply(remove_accents)

    df = df.copy()
    df[column] = series
    if new_name and new_name != column:
        df = df.rename(columns={column: new_name})
    return df
