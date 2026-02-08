"""Feature engineering utilities."""

from .engineering import (
    add_proxy_poverty,
    minmax_weighted_score,
    weighted_mean,
    normalize_minmax,
)
from .satellite import (
    initialize_earth_engine,
    geojson_to_ee_geometry,
    compute_city_year_landcover_metrics,
    extract_sentinel2_landcover_features,
)

__all__ = [
    "add_proxy_poverty",
    "minmax_weighted_score",
    "weighted_mean",
    "normalize_minmax",
    "initialize_earth_engine",
    "geojson_to_ee_geometry",
    "compute_city_year_landcover_metrics",
    "extract_sentinel2_landcover_features",
]
