"""Modeling utilities and reusable pipelines."""

from .preprocess import (
    make_preprocessor,
    loess_smooth_pattern,
    build_noisy_loess_pattern,
    decompose_annual_values,
    decompose_annual_dataframe,
    repeat_rows_for_12_months,
    kriging_impute,
    kriging_impute_many,
    knn_impute_mixed_panel,
    merge_satellite_features_by_year,
    concat_yearly_bases,
)
from .models import (
    build_model,
    build_catboost,
    fit_ols,
    fit_random_effects_panel,
    build_keras_regressor,
    build_lightgbm,
    model_search_space,
    build_sklearn_model,
    build_xgboost,
)
from .pipelines import build_pipeline
from .evaluation import (
    train_test_split_xy,
    evaluate_regressor,
    cross_validate_regressor,
    fit_and_evaluate,
)
from .selection import tune_model

__all__ = [
    "make_preprocessor",
    "loess_smooth_pattern",
    "build_noisy_loess_pattern",
    "decompose_annual_values",
    "decompose_annual_dataframe",
    "repeat_rows_for_12_months",
    "kriging_impute",
    "kriging_impute_many",
    "knn_impute_mixed_panel",
    "merge_satellite_features_by_year",
    "concat_yearly_bases",
    "build_pipeline",
    "train_test_split_xy",
    "evaluate_regressor",
    "cross_validate_regressor",
    "fit_and_evaluate",
    "tune_model",
    "build_model",
    "build_sklearn_model",
    "build_catboost",
    "build_lightgbm",
    "build_xgboost",
    "build_keras_regressor",
    "model_search_space",
    "fit_ols",
    "fit_random_effects_panel",
]
