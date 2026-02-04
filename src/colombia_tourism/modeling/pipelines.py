"""Pipeline builders."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def build_pipeline(model, preprocessor=None, target_scaler=None):
    """Build a model pipeline with optional target scaling."""
    if preprocessor is not None:
        estimator = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
    else:
        estimator = model

    if target_scaler is not None:
        return TransformedTargetRegressor(
            regressor=estimator, transformer=target_scaler
        )

    return estimator
