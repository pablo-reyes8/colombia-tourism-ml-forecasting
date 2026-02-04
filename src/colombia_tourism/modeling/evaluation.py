"""Evaluation helpers for regressors."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split


def train_test_split_xy(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_regressor(estimator, X_test, y_test) -> Dict[str, float]:
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
    }


def cross_validate_regressor(
    estimator,
    X,
    y,
    cv: int = 5,
    scoring: Dict[str, str] | None = None,
    n_jobs: int | None = None,
):
    scoring = scoring or {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }
    return cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
    )


def fit_and_evaluate(
    estimator,
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int | None = None,
    scoring: Dict[str, str] | None = None,
    n_jobs: int | None = None,
):
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X, y, test_size=test_size, random_state=random_state
    )
    estimator.fit(X_train, y_train)
    metrics = evaluate_regressor(estimator, X_test, y_test)
    cv_results = None
    if cv:
        cv_results = cross_validate_regressor(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    return metrics, cv_results
