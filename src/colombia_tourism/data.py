"""Dataset I/O helpers and feature metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from .config import DEFAULT_BASE_FINAL

ENTITY_COLUMN = "Ciudad"
TIME_COLUMN = "Mes"
DEFAULT_FEATURES = [
    "Homicidios",
    "Hurtos",
    "Delitos Sexuales",
    "Temperatura",
    "Dolar",
    "Pib Ponderado",
    "Distancia a accseos",
    "importancia accesos",
    "Establecimientos de turismo",
    "N Habitaciones",
    "N Camas",
    "Distancia al TOP",
    "Proxy Pobreza",
    "Gasto Promedio Diario",
    "Gasto Alojamiento Diario",
    "Gasto Transporte Diario",
    "Gasto alimetos Diario",
    "Otros Gastos Diario",
    "Gasto Promedio Viaje",
    "Gasto  Alojamiento Viaje",
    "Gasto Transporte Viaje",
    "Gasto alimetos Viaje",
    "Otros Gastos Viaje",
    "Inflacion",
    "Eventos",
    "Area Urbana",
    "Area Rural",
    "Area Agua",
    "Nmero Vias",
]

DEFAULT_TARGET = "Nmero Extranjeros"

FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "target": (DEFAULT_TARGET,),
    "security": ("Homicidios", "Hurtos", "Delitos Sexuales"),
    "climate": ("Temperatura",),
    "economic": ("Dolar", "Pib Ponderado", "Inflacion", "Proxy Pobreza"),
    "accessibility": (
        "Entradas Extranjeros Zona",
        "Distancia a accseos",
        "importancia accesos",
        "Distancia al TOP",
        "Nmero Vias",
    ),
    "tourism_supply": (
        "Establecimientos de turismo",
        "N Habitaciones",
        "N Camas",
        "Eventos",
    ),
    "satellite": ("Area Urbana", "Area Rural", "Area Agua"),
    "daily_spend": (
        "Gasto Promedio Diario",
        "Gasto Alojamiento Diario",
        "Gasto Transporte Diario",
        "Gasto alimetos Diario",
        "Otros Gastos Diario",
    ),
    "trip_spend": (
        "Gasto Promedio Viaje",
        "Gasto  Alojamiento Viaje",
        "Gasto Transporte Viaje",
        "Gasto alimetos Viaje",
        "Otros Gastos Viaje",
    ),
}

GROUPED_DEFAULT_FEATURES = {
    name: tuple(col for col in columns if col != DEFAULT_TARGET)
    for name, columns in FEATURE_GROUPS.items()
    if name != "target"
}


def parse_panel_months(
    values,
    *,
    day: int = 1,
    strict: bool = False,
) -> pd.Series:
    """Parse the project's month column into pandas timestamps.

    The repository uses multiple month representations across notebooks:
    - ``1-2018`` from the final panel table
    - ``2018-01`` in some plotting cells
    - native datetime values
    """
    series = pd.Series(values).copy()
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    text = series.astype(str).str.strip()

    for fmt in ("%m-%Y", "%Y-%m", "%m/%Y", "%Y/%m"):
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(
            text.loc[mask],
            format=fmt,
            errors="coerce",
        )

    if parsed.isna().any():
        mask = parsed.isna()
        parsed.loc[mask] = pd.to_datetime(text.loc[mask], errors="coerce")

    if strict and parsed.isna().any():
        sample = text.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Unable to parse month values. Sample: {sample}")

    if day != 1:
        parsed = parsed + pd.to_timedelta(day - 1, unit="D")
    return parsed


def ensure_panel_datetime(
    df: pd.DataFrame,
    month_col: str = TIME_COLUMN,
    output_col: str = "fecha",
    *,
    dropna: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Attach a normalized datetime column to the panel dataset."""
    if month_col not in df.columns:
        raise KeyError(f"Column '{month_col}' not found in dataframe")

    out = df.copy()
    out[output_col] = parse_panel_months(out[month_col], strict=strict)
    if dropna:
        out = out.dropna(subset=[output_col])
    return out


def available_feature_groups(df_or_columns) -> dict[str, list[str]]:
    """Return the project feature groups intersected with available columns."""
    if isinstance(df_or_columns, pd.DataFrame):
        columns = set(df_or_columns.columns)
    else:
        columns = set(df_or_columns)

    return {
        group: [column for column in group_columns if column in columns]
        for group, group_columns in FEATURE_GROUPS.items()
        if any(column in columns for column in group_columns)
    }


def infer_numeric_features(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] = (ENTITY_COLUMN, TIME_COLUMN, DEFAULT_TARGET),
) -> list[str]:
    """Infer numeric features excluding entity/time columns."""
    excluded = set(exclude)
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def load_base_final(
    path=DEFAULT_BASE_FINAL,
    columns: Iterable[str] | None = None,
    *,
    parse_dates: bool = False,
    month_col: str = TIME_COLUMN,
    datetime_col: str = "fecha",
    sort_panel: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if columns:
        df = df[list(columns)]
    if parse_dates and month_col in df.columns:
        df = ensure_panel_datetime(df, month_col=month_col, output_col=datetime_col)
    if sort_panel and {ENTITY_COLUMN, datetime_col}.issubset(df.columns):
        df = df.sort_values([ENTITY_COLUMN, datetime_col]).reset_index(drop=True)
    return df


def split_features_target(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    features: Iterable[str] | None = None,
):
    if features is None:
        features = [col for col in df.columns if col != target]
    X = df[list(features)]
    y = df[target]
    return X, y
