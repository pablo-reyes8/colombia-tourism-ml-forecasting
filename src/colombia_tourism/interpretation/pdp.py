"""Partial dependence plots (PDPs)."""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
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
