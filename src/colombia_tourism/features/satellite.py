"""Sentinel-2 feature extraction helpers (Google Earth Engine).

This module encapsulates notebook 2 logic for:
- cloud masking
- spectral index computation
- Otsu thresholding
- city-year urban/rural/water area extraction
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _require_ee():
    try:
        import ee
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "earthengine-api is required for Sentinel-2 extraction. "
            "Install it with `pip install earthengine-api`."
        ) from exc
    return ee


def initialize_earth_engine(
    project: str | None = None,
    authenticate: bool = True,
) -> None:
    """Initialize the Earth Engine client."""
    ee = _require_ee()
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return
    except Exception:
        if not authenticate:
            raise
    ee.Authenticate()
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def geojson_to_ee_geometry(geometry: dict[str, Any]):
    """Convert GeoJSON-like geometry dict into an ee.Geometry object."""
    ee = _require_ee()
    geometry_type = geometry["type"]
    coords = geometry.get("coordinates")

    if geometry_type == "Polygon":
        return ee.Geometry.Polygon(coords)
    if geometry_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords)
    if geometry_type == "LineString":
        return ee.Geometry.LineString(coords)
    if geometry_type == "MultiLineString":
        return ee.Geometry.MultiLineString(coords)
    if geometry_type == "Point":
        return ee.Geometry.Point(coords)
    if geometry_type == "MultiPoint":
        return ee.Geometry.MultiPoint(coords)
    if geometry_type == "GeometryCollection":
        return ee.Geometry(geometry)
    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def _to_ee_geometry(geometry: Any):
    if isinstance(geometry, dict):
        return geojson_to_ee_geometry(geometry)
    if hasattr(geometry, "__geo_interface__"):
        return geojson_to_ee_geometry(geometry.__geo_interface__)
    return geometry


def otsu_threshold(histogram):
    """Compute Otsu threshold from an Earth Engine histogram dictionary."""
    ee = _require_ee()
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get("histogram"))
    bins = ee.Array(histogram.get("bucketMeans"))

    total = counts.reduce("sum", [0]).get([0])
    sum_total = counts.multiply(bins).reduce("sum", [0]).get([0])
    size = counts.length().get([0])

    def compute_between_var(index):
        index = ee.Number(index).toInt()
        w_b = counts.slice(0, 0, index).reduce("sum", [0]).get([0])
        sum_b = (
            counts.slice(0, 0, index)
            .multiply(bins.slice(0, 0, index))
            .reduce("sum", [0])
            .get([0])
        )
        w_f = total.subtract(w_b)
        sum_f = sum_total.subtract(sum_b)
        m_b = ee.Number(sum_b).divide(w_b)
        m_f = ee.Number(sum_f).divide(w_f)
        return ee.Number(w_b).multiply(w_f).multiply(m_b.subtract(m_f).pow(2))

    indices = ee.List.sequence(1, ee.Number(size).subtract(1))
    between_vars = indices.map(compute_between_var)
    max_var = ee.Array(between_vars).reduce("max", [0]).get([0])
    threshold_index = between_vars.indexOf(max_var)
    return bins.get([threshold_index])


def calculate_percentile(image, band: str, percentile: float, study_area):
    """Compute band percentile over study area."""
    ee = _require_ee()
    value = image.select(band).reduceRegion(
        reducer=ee.Reducer.percentile([percentile]),
        geometry=study_area,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).get(band)
    return ee.Number(value)


def mask_s2_clouds(image, cloud_probability_threshold: int = 40):
    """Apply S2 cloud probability mask to one Sentinel-2 image."""
    ee = _require_ee()
    img_id = image.get("system:index")
    cloud_prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(ee.Filter.eq("system:index", img_id))
        .first()
    )

    cloud_mask = ee.Image(
        ee.Algorithms.If(
            cloud_prob,
            ee.Image(cloud_prob).select("probability").lt(cloud_probability_threshold),
            ee.Image.constant(1),
        )
    )
    return image.updateMask(cloud_mask).copyProperties(image, ["system:time_start"])


def _build_composite(
    study_area,
    year: int,
    max_cloud_percentage: int = 50,
    cloud_probability_threshold: int = 40,
    max_images: int = 55,
):
    ee = _require_ee()
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(study_area)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percentage))
        .map(lambda img: ee.Image(mask_s2_clouds(img, cloud_probability_threshold)))
        .limit(max_images)
    )
    n_images = int(collection.size().getInfo())
    if n_images == 0:
        return None, n_images

    composite = (
        collection.select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .median()
        .clip(study_area)
        .multiply(0.0001)
    )
    return composite, n_images


def _compute_indices(composite):
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    gndvi = composite.normalizedDifference(["B8", "B3"]).rename("GNDVI")
    mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    evi = composite.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": composite.select("B8"),
            "RED": composite.select("B4"),
            "BLUE": composite.select("B2"),
        },
    ).rename("EVI")
    aweish = composite.expression(
        "B3 + 2.5 * B8 - 1.5 * (B11 + B12) - 0.25 * B2",
        {
            "B2": composite.select("B2"),
            "B3": composite.select("B3"),
            "B8": composite.select("B8"),
            "B11": composite.select("B11"),
            "B12": composite.select("B12"),
        },
    ).rename("AWEIsh")
    return composite.addBands([ndwi, ndbi, ndvi, gndvi, mndwi, evi, aweish])


def _classify_masks(composite, study_area):
    ee = _require_ee()

    ndbi_hist = composite.select("NDBI").reduceRegion(
        reducer=ee.Reducer.histogram(255, 2),
        geometry=study_area,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).get("NDBI")
    ndbi_threshold = ee.Number(0.2)
    if ndbi_hist is not None:
        ndbi_threshold = ee.Number(otsu_threshold(ndbi_hist))

    ndvi_threshold = calculate_percentile(composite, "NDVI", 55, study_area)
    evi_threshold = calculate_percentile(composite, "EVI", 45, study_area)
    urban = (
        composite.select("NDBI")
        .gt(ndbi_threshold)
        .And(composite.select("EVI").lt(evi_threshold))
        .And(composite.select("NDVI").lt(ndvi_threshold))
        .And(composite.select("NDWI").lt(ee.Number(0.0)))
        .rename("urban")
    )

    ndvi_threshold_r = calculate_percentile(composite, "NDVI", 86, study_area)
    evi_threshold_r = calculate_percentile(composite, "EVI", 81, study_area)
    gndvi_threshold = calculate_percentile(composite, "GNDVI", 86, study_area)
    rural = (
        composite.select("NDVI")
        .gt(ndvi_threshold_r)
        .And(composite.select("EVI").gt(evi_threshold_r))
        .And(composite.select("GNDVI").gt(gndvi_threshold))
        .rename("rural")
    )

    mndwi_threshold = calculate_percentile(composite, "MNDWI", 91, study_area)
    aweish_threshold = calculate_percentile(composite, "AWEIsh", 92, study_area)
    ndwi_threshold_w = calculate_percentile(composite, "NDWI", 91, study_area)
    water = (
        composite.select("NDWI")
        .gt(ndwi_threshold_w)
        .And(composite.select("MNDWI").gt(mndwi_threshold))
        .And(composite.select("AWEIsh").gt(aweish_threshold))
        .rename("water")
    )
    return {"urban": urban, "rural": rural, "water": water}


def _mask_area_km2(mask, band_name: str, study_area, scale: int = 10):
    ee = _require_ee()
    pixel_count = mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=study_area,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).get(band_name)
    pixel_count = ee.Number(ee.Algorithms.If(pixel_count, pixel_count, 0))
    return pixel_count.multiply(scale * scale).divide(1e6)


def _safe_get_info(value) -> float | None:
    try:
        return float(value.getInfo())
    except Exception:
        return None


def compute_city_year_landcover_metrics(
    geometry: Any,
    year: int,
    max_cloud_percentage: int = 50,
    cloud_probability_threshold: int = 40,
    max_images: int = 55,
    scale: int = 10,
    return_masks: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Compute urban/rural/water areas (km²) for one city-year."""
    study_area = _to_ee_geometry(geometry)
    composite, n_images = _build_composite(
        study_area=study_area,
        year=year,
        max_cloud_percentage=max_cloud_percentage,
        cloud_probability_threshold=cloud_probability_threshold,
        max_images=max_images,
    )

    metrics = {
        "año": int(year),
        "n_imagenes": n_images,
        "Area Urbana": None,
        "Area Rural": None,
        "Area Agua": None,
    }
    masks: dict[str, Any] = {}
    if composite is None:
        if return_masks:
            return metrics, masks
        return metrics

    composite = _compute_indices(composite)
    masks = _classify_masks(composite, study_area)

    area_urban = _mask_area_km2(masks["urban"], "urban", study_area, scale=scale)
    area_rural = _mask_area_km2(masks["rural"], "rural", study_area, scale=scale)
    area_water = _mask_area_km2(masks["water"], "water", study_area, scale=scale)

    metrics["Area Urbana"] = _safe_get_info(area_urban)
    metrics["Area Rural"] = _safe_get_info(area_rural)
    metrics["Area Agua"] = _safe_get_info(area_water)

    if return_masks:
        return metrics, masks
    return metrics


def extract_sentinel2_landcover_features(
    areas_df: pd.DataFrame,
    city_col: str = "Ciudad",
    geometry_col: str = "geometry",
    years: Iterable[int] = (2018, 2019, 2021, 2022, 2023),
    sleep_seconds: float = 0.0,
    return_masks: bool = False,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[int, dict[str, Any]]]]:
    """Extract city-year Sentinel-2 features for all rows in `areas_df`."""
    if city_col not in areas_df.columns:
        raise KeyError(f"Missing '{city_col}' in areas_df")
    if geometry_col not in areas_df.columns:
        raise KeyError(f"Missing '{geometry_col}' in areas_df")

    records: list[dict[str, Any]] = []
    mask_store: dict[str, dict[int, dict[str, Any]]] = {}
    for _, row in areas_df.iterrows():
        city = row[city_col]
        geometry = row[geometry_col]
        for year in years:
            if return_masks:
                metrics, masks = compute_city_year_landcover_metrics(
                    geometry=geometry,
                    year=year,
                    return_masks=True,
                    **kwargs,
                )
            else:
                metrics = compute_city_year_landcover_metrics(
                    geometry=geometry,
                    year=year,
                    return_masks=False,
                    **kwargs,
                )
                masks = {}

            metrics[city_col] = city
            records.append(metrics)

            if return_masks:
                mask_store.setdefault(city, {})[int(year)] = masks

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    features_df = pd.DataFrame.from_records(records)
    if return_masks:
        return features_df, mask_store
    return features_df
