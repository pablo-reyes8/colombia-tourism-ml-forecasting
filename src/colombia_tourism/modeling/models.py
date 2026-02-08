"""Model factories for ML and econometric comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from scikeras.wrappers import KerasRegressor
    from tensorflow import keras
except Exception:  # pragma: no cover
    KerasRegressor = None
    keras = None


def _normalize_model_name(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "regresion_lineal": "linear",
        "linear_regression": "linear",
        "regresion_ridge": "ridge",
        "regresion_lasso": "lasso",
        "regresion_elastic_net": "elasticnet",
        "elastic_net": "elasticnet",
        "knn_regressor": "knn",
        "decision_tree_regressor": "decision_tree",
        "arbol_de_decision": "decision_tree",
        "randomforest": "random_forest",
        "gradientboosting": "gradient_boosting",
        "gradient_boostig": "gradient_boosting",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
    }
    return aliases.get(normalized, normalized)


def build_sklearn_model(name: str, **kwargs: Any):
    """Build scikit-learn regressors used in notebook 5."""
    name = _normalize_model_name(name)

    constructors = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
        "knn": KNeighborsRegressor,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "svr": SVR,
    }
    defaults = {
        "ridge": {"alpha": 2.0},
        "lasso": {"alpha": 3.0, "max_iter": 5000},
        "elasticnet": {"alpha": 10.0, "l1_ratio": 0.020408, "max_iter": 5000},
        "knn": {"n_neighbors": 3, "weights": "distance", "metric": "manhattan"},
        "decision_tree": {
            "max_depth": 13,
            "min_samples_split": 2,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 38,
            "max_depth": 45,
            "min_samples_split": 8,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 58,
            "max_depth": 9,
            "random_state": 42,
        },
        "svr": {"kernel": "linear", "epsilon": 0.05},
    }

    if name not in constructors:
        raise ValueError(f"Unknown sklearn model: {name}")

    params = dict(defaults.get(name, {}))
    params.update(kwargs)
    return constructors[name](**params)


def build_catboost(**kwargs: Any):
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed")
    defaults = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "verbose": False,
        "random_state": 42,
    }
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)


def build_lightgbm(**kwargs: Any):
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed")
    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": -1,
        "random_state": 42,
    }
    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


def build_xgboost(preset: str | None = None, **kwargs: Any):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    if preset == "notebook_best":
        defaults.update(
            {
                "n_estimators": 266,
                "max_depth": 17,
                "gamma": 0.21052631578947367,
            }
        )

    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def _build_keras_model(
    n_features_in_: int,
    hidden_units: int = 128,
    hidden_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(hidden_units, activation="relu"))
        if dropout and dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_keras_regressor(
    hidden_units: int = 128,
    hidden_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
):
    if KerasRegressor is None or keras is None:
        raise ImportError("tensorflow / scikeras not installed")

    return KerasRegressor(
        model=_build_keras_model,
        model__hidden_units=hidden_units,
        model__hidden_layers=hidden_layers,
        model__dropout=dropout,
        model__learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )


def build_model(name: str, **kwargs: Any):
    """Unified model factory across sklearn/boosting/keras families."""
    normalized = _normalize_model_name(name)
    if normalized in {
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "knn",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svr",
    }:
        return build_sklearn_model(normalized, **kwargs)
    if normalized == "xgboost":
        return build_xgboost(**kwargs)
    if normalized == "lightgbm":
        return build_lightgbm(**kwargs)
    if normalized == "catboost":
        return build_catboost(**kwargs)
    if normalized == "keras":
        return build_keras_regressor(**kwargs)
    raise ValueError(f"Unknown model name: {name}")


def model_search_space(
    name: str,
    pipeline_prefix: str | None = None,
) -> dict[str, Any]:
    """Return RandomizedSearchCV-ready parameter spaces.

    If `pipeline_prefix` is provided (e.g. ``regressor``), keys are prefixed as
    ``regressor__param``.
    """
    normalized = _normalize_model_name(name)

    def key(param: str) -> str:
        if not pipeline_prefix:
            return param
        return f"{pipeline_prefix}__{param}"

    spaces = {
        "lasso": {
            key("alpha"): randint(1, 122),
        },
        "ridge": {
            key("alpha"): randint(1, 122),
        },
        "elasticnet": {
            key("alpha"): randint(1, 150),
            key("l1_ratio"): uniform(0, 1),
        },
        "knn": {
            key("n_neighbors"): randint(2, 30),
            key("weights"): ["uniform", "distance"],
            key("metric"): ["euclidean", "manhattan", "minkowski"],
        },
        "decision_tree": {
            key("max_depth"): randint(2, 55),
            key("min_samples_split"): randint(2, 15),
            key("min_samples_leaf"): randint(2, 15),
        },
        "random_forest": {
            key("n_estimators"): randint(2, 100),
            key("max_depth"): randint(2, 55),
            key("min_samples_split"): randint(2, 15),
            key("min_samples_leaf"): randint(2, 15),
        },
        "gradient_boosting": {
            key("n_estimators"): randint(2, 100),
            key("max_depth"): randint(2, 25),
        },
        "xgboost": {
            key("n_estimators"): randint(2, 300),
            key("max_depth"): randint(2, 25),
            key("gamma"): uniform(0, 0.5),
        },
        "svr": {
            key("C"): uniform(0.1, 1000),
            key("epsilon"): uniform(0.01, 1),
            key("kernel"): ["linear", "rbf", "poly"],
            key("gamma"): ["scale", "auto"],
        },
    }

    if normalized not in spaces:
        raise ValueError(f"No predefined search space for model: {name}")
    return spaces[normalized]


def fit_ols(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    add_constant: bool = True,
):
    """Fit a statsmodels OLS regression."""
    import statsmodels.api as sm

    design = X.copy()
    if add_constant:
        design = sm.add_constant(design, has_constant="add")
    return sm.OLS(y, design).fit()


def fit_random_effects_panel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    entity_col: str,
    time_col: str,
    time_format: str | None = None,
    add_constant: bool = True,
):
    """Fit a random-effects panel model (linearmodels)."""
    import statsmodels.api as sm
    from linearmodels.panel import RandomEffects

    feature_cols = list(feature_cols)
    required_cols = [target_col, entity_col, time_col, *feature_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in panel dataframe: {missing}")

    panel = df.copy()
    panel = panel.dropna(subset=required_cols)
    if not pd.api.types.is_datetime64_any_dtype(panel[time_col]):
        panel[time_col] = pd.to_datetime(panel[time_col], format=time_format, errors="coerce")
    panel = panel.dropna(subset=[time_col]).set_index([entity_col, time_col]).sort_index()

    y = panel[target_col]
    X = panel[feature_cols]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    model = RandomEffects(y, X)
    return model.fit()
