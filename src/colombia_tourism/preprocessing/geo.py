"""Geospatial helpers."""

from __future__ import annotations

import pandas as pd


def add_point_geometry(
    df: pd.DataFrame,
    lon_col: str = "lng",
    lat_col: str = "lat",
):
    """Add a shapely Point geometry column.

    This keeps geopandas as an optional dependency by importing lazily.
    """
    from shapely.geometry import Point
    import geopandas as gpd

    df = df.copy()
    df["geometry"] = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    return gpd.GeoDataFrame(df, geometry="geometry")


def convert_location_to_geometry(df: pd.DataFrame, column: str):
    """Convert '(lat, lon)' string columns into geometry.

    Drops intermediate helper columns.
    """
    from shapely.geometry import Point

    df = df.copy()
    df["_ubicacion"] = df[column].str.strip("()")
    df[["latitud", "longitud"]] = df["_ubicacion"].str.split(",", expand=True)
    df["latitud"] = df["latitud"].astype(float)
    df["longitud"] = df["longitud"].astype(float)
    df["geometry"] = df.apply(
        lambda row: Point(row["longitud"], row["latitud"]), axis=1
    )
    df = df.drop([column, "_ubicacion", "latitud", "longitud"], axis=1)
    return df
