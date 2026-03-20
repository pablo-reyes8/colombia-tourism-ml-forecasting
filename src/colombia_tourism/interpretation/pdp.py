"""Partial dependence plots (PDPs)."""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay


def plot_partial_dependence(
    estimator,
    X,
    features,
    grid_resolution: int = 50,
    ax=None,
):
    return PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=features,
        grid_resolution=grid_resolution,
        ax=ax,
    )


def partial_dependence_sweep(
    estimator,
    X,
    feature: str,
    *,
    values=None,
    start: float | None = None,
    stop: float | None = None,
    num: int = 25,
) -> pd.DataFrame:
    """Compute a manual PDP sweep by replacing one feature at a time."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not found in X")

    if values is None:
        if start is None:
            start = float(X[feature].min())
        if stop is None:
            stop = float(X[feature].max())
        values = np.linspace(start, stop, num)

    rows = []
    for value in values:
        X_temp = X.copy()
        X_temp[feature] = value
        y_pred = estimator.predict(X_temp)
        rows.append(
            {
                "feature": feature,
                "feature_value": value,
                "prediction_mean": float(np.mean(y_pred)),
                "prediction_std": float(np.std(y_pred)),
            }
        )
    return pd.DataFrame(rows)


def plot_partial_dependence_sweep(
    estimator,
    X,
    feature: str,
    *,
    values=None,
    start: float | None = None,
    stop: float | None = None,
    num: int = 25,
    ax=None,
    color: str = "black",
):
    """Plot the manual PDP sweep used in notebook 6."""
    sweep = partial_dependence_sweep(
        estimator,
        X,
        feature,
        values=values,
        start=start,
        stop=stop,
        num=num,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(
        sweep["feature_value"],
        sweep["prediction_mean"],
        label=feature,
        linewidth=2,
        color=color,
    )
    tick_step = max(len(sweep) // 10, 1)
    ax.set_xticks(sweep["feature_value"].iloc[::tick_step])
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Average prediction")
    ax.set_title(f"Partial dependence: {feature}")
    ax.legend()
    fig.tight_layout()
    return fig, sweep


def plot_pdp_grid(
    estimator,
    X,
    features,
    n_cols: int = 3,
    grid_resolution: int = 50,
    figsize: tuple[int, int] | None = None,
):
    features = list(features)
    n_rows = ceil(len(features) / n_cols)
    figsize = figsize or (6 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, feature in zip(axes, features):
        plot_partial_dependence(
            estimator,
            X,
            features=[feature],
            grid_resolution=grid_resolution,
            ax=ax,
        )
        ax.set_title(str(feature))

    for ax in axes[len(features) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig
