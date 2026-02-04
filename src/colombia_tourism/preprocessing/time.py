"""Time and calendar utilities."""

from __future__ import annotations

import pandas as pd

from colombia_tourism.config import MONTHS_EN, MONTHS_ES_TO_EN


def translate_months_es_to_en(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].map(MONTHS_ES_TO_EN)
    return df


def month_name_to_number(df: pd.DataFrame, column: str) -> pd.DataFrame:
    months_inverted = {name: num for num, name in MONTHS_EN.items()}
    df = df.copy()
    df[column] = df[column].map(months_inverted)
    return df


def complete_months(
    df: pd.DataFrame,
    entity_col: str,
    month_col: str,
    value_col: str,
    aggregate: str = "median",
) -> pd.DataFrame:
    """Ensure each entity has the 12 months present.

    aggregate:
        - 'median' or 'sum'
    """
    if aggregate not in {"median", "sum"}:
        raise ValueError("aggregate must be 'median' or 'sum'")

    if aggregate == "median":
        grouped = df.groupby([entity_col, month_col])[value_col].median().reset_index()
    else:
        grouped = df.groupby([entity_col, month_col])[value_col].sum().reset_index()

    entities = grouped[entity_col].unique()
    months = range(1, 13)

    combinations = pd.MultiIndex.from_product(
        [entities, months], names=[entity_col, month_col]
    ).to_frame(index=False)
    combinations[month_col] = combinations[month_col].map(MONTHS_EN)

    result = pd.merge(combinations, grouped, on=[entity_col, month_col], how="left")
    if aggregate == "sum":
        result[value_col] = result[value_col].fillna(0)
    return result
