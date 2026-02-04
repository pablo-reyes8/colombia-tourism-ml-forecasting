"""Permutation importance wrappers."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_df(
    estimator,
    X,
    y,
    scoring: str = "r2",
    n_repeats: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    feature_names = getattr(X, "columns", None)
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(result.importances_mean.shape[0])]

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
