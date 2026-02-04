"""Model factory helpers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

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


def build_sklearn_model(name: str, **kwargs: Any):
    name = name.lower()
    if name == "linear":
        return LinearRegression(**kwargs)
    if name == "ridge":
        return Ridge(**kwargs)
    if name == "lasso":
        return Lasso(**kwargs)
    if name == "elasticnet":
        return ElasticNet(**kwargs)
    if name == "random_forest":
        return RandomForestRegressor(**kwargs)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(**kwargs)
    raise ValueError(f"Unknown sklearn model: {name}")


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


def build_xgboost(**kwargs: Any):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
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
