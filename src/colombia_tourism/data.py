"""Dataset I/O helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import DEFAULT_BASE_FINAL

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


def load_base_final(
    path=DEFAULT_BASE_FINAL,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if columns:
        df = df[list(columns)]
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
