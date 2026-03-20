"""EDA utilities extracted from notebook 4 and generalized for reuse."""

from __future__ import annotations

from math import ceil
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from colombia_tourism.data import DEFAULT_TARGET, FEATURE_GROUPS, parse_panel_months


def _resolve_variables(df: pd.DataFrame, variables) -> list[str]:
    if variables is None:
        return df.select_dtypes(include="number").columns.tolist()
    return [column for column in variables if column in df.columns]


def _with_panel_time(df: pd.DataFrame, time_col: str = "Mes") -> pd.DataFrame:
    out = df.copy()
    out["_panel_date"] = parse_panel_months(out[time_col], strict=False)
    out["year"] = out["_panel_date"].dt.year
    out["month"] = out["_panel_date"].dt.month
    return out


def summarize_variables(
    df: pd.DataFrame,
    variables=None,
    *,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Return a rich descriptive-statistics table for numeric variables."""
    rows = []
    for variable in _resolve_variables(df, variables):
        series = df[variable].dropna()
        if series.empty:
            continue

        modes = series.mode()
        row = {
            "variable": variable,
            "count": int(series.count()),
            "missing": int(df[variable].isna().sum()),
            "missing_pct": df[variable].isna().mean() * 100,
            "mean": series.mean(),
            "median": series.median(),
            "mode": modes.iloc[0] if not modes.empty else np.nan,
            "std": series.std(ddof=0),
            "variance": series.var(ddof=0),
            "min": series.min(),
            "q25": series.quantile(0.25),
            "q75": series.quantile(0.75),
            "max": series.max(),
            "range": series.max() - series.min(),
            "iqr": series.quantile(0.75) - series.quantile(0.25),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if round_digits is not None and not summary.empty:
        numeric = summary.select_dtypes(include="number").columns
        summary[numeric] = summary[numeric].round(round_digits)
    return summary


def summarize_feature_groups(
    df: pd.DataFrame,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    *,
    round_digits: int | None = 4,
) -> dict[str, pd.DataFrame]:
    """Summarize the project's predefined feature groups."""
    groups = groups or FEATURE_GROUPS
    return {
        group: summarize_variables(df, variables, round_digits=round_digits)
        for group, variables in groups.items()
        if any(variable in df.columns for variable in variables)
    }


def correlation_with_target(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    variables=None,
    *,
    method: str = "pearson",
    absolute: bool = True,
) -> pd.DataFrame:
    """Return feature-target correlations sorted by magnitude."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    variables = [
        column
        for column in _resolve_variables(df, variables)
        if column != target_col
    ]
    corr = (
        df[variables + [target_col]]
        .corr(method=method)[target_col]
        .drop(labels=[target_col])
        .rename("correlation")
        .to_frame()
    )
    corr["abs_correlation"] = corr["correlation"].abs()
    corr = corr.sort_values(
        "abs_correlation" if absolute else "correlation",
        ascending=False,
    ).reset_index(names="feature")
    return corr


def plot_correlation_heatmap(
    df: pd.DataFrame,
    variables=None,
    *,
    method: str = "pearson",
    figsize: tuple[int, int] = (12, 10),
    annot: bool = False,
    cmap: str = "coolwarm",
    center: float = 0.0,
    ax=None,
):
    """Plot a correlation heatmap for the selected variables."""
    variables = _resolve_variables(df, variables)
    corr = df[variables].corr(method=method)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sns.heatmap(corr, cmap=cmap, center=center, annot=annot, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig, corr


def plot_group_pairplot(
    df: pd.DataFrame,
    variables,
    *,
    title: str | None = None,
    sample: int | None = None,
    diag_kind: str = "kde",
    plot_color: str = "#DA70D6",
    diag_color: str = "#DB7093",
):
    """Render a pairplot with the styling used in notebook 4."""
    variables = _resolve_variables(df, variables)
    if sample is not None and len(df) > sample:
        plot_df = df[variables].sample(sample, random_state=42)
    else:
        plot_df = df[variables]

    with sns.axes_style("white"):
        grid = sns.pairplot(
            plot_df,
            diag_kind=diag_kind,
            corner=False,
            plot_kws={"alpha": 0.6, "color": plot_color, "s": 70},
            diag_kws={"color": diag_color, "fill": True, "alpha": 0.6},
        )
    if title:
        grid.fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
        grid.fig.tight_layout()
    return grid


def aggregate_monthly_panel(
    df: pd.DataFrame,
    value_col: str,
    *,
    time_col: str = "Mes",
    agg: str = "sum",
) -> pd.DataFrame:
    """Aggregate a panel variable by year-month."""
    panel = _with_panel_time(df, time_col=time_col)
    grouped = (
        panel.groupby(["year", "month"], as_index=False)[value_col]
        .agg(agg)
        .sort_values(["year", "month"])
    )
    return grouped


def plot_year_profiles(
    df: pd.DataFrame,
    value_col: str,
    *,
    years=None,
    time_col: str = "Mes",
    agg: str = "sum",
    palette: str = "rocket",
    ax=None,
):
    """Plot the monthly profile of a variable across multiple years."""
    monthly = aggregate_monthly_panel(df, value_col, time_col=time_col, agg=agg)
    if years is not None:
        monthly = monthly[monthly["year"].isin(list(years))]
    years = sorted(monthly["year"].dropna().unique().tolist())
    colors = sns.color_palette(palette, n_colors=max(len(years), 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    for color, year in zip(colors, years):
        block = monthly[monthly["year"] == year]
        ax.plot(
            block["month"],
            block[value_col],
            label=str(year),
            linewidth=2.5,
            marker="o",
            color=color,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel(value_col)
    ax.set_title(f"Monthly {value_col} Profiles")
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Year")
    fig.tight_layout()
    return fig, monthly


def _collapse_entities_for_share_plot(
    data: pd.DataFrame,
    entity_col: str,
    value_col: str,
    *,
    threshold: float | None = None,
    top_n: int | None = None,
) -> pd.Series:
    grouped = data.groupby(entity_col)[value_col].sum().sort_values(ascending=False)
    if top_n is not None:
        top = grouped.head(top_n)
        remainder = grouped.iloc[top_n:].sum()
        if remainder > 0:
            top.loc["Otros"] = remainder
        return top

    if threshold is None:
        return grouped

    above = grouped[grouped >= threshold].copy()
    remainder = grouped[grouped < threshold].sum()
    if remainder > 0:
        above.loc["Otros"] = remainder
    return above


def plot_entity_share_comparison(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    years: list[int],
    *,
    time_col: str = "Mes",
    threshold: float | None = None,
    top_n: int | None = None,
    label_threshold_pct: float = 1.0,
):
    """Compare entity contribution shares across years using pie charts."""
    panel = _with_panel_time(df, time_col=time_col)
    fig, axes = plt.subplots(1, len(years), figsize=(8 * len(years), 9))
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        year_df = panel[panel["year"] == year]
        grouped = _collapse_entities_for_share_plot(
            year_df,
            entity_col=entity_col,
            value_col=value_col,
            threshold=threshold,
            top_n=top_n,
        )
        colors = plt.get_cmap("tab20")(range(len(grouped)))
        wedges, _ = ax.pie(grouped, startangle=90, colors=colors, autopct="")
        percentages = grouped / grouped.sum() * 100
        for wedge, (entity, pct) in zip(wedges, percentages.items()):
            if pct < label_threshold_pct:
                continue
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle)) * 1.04
            y = np.sin(np.deg2rad(angle)) * 1.04
            ax.text(x, y, f"{pct:.1f}%", ha="center", va="center", fontsize=10, weight="bold")

        legend_labels = [f"{entity}: {pct:.1f}%" for entity, pct in percentages.items()]
        ax.legend(wedges, legend_labels, title=f"{entity_col} {year}", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{value_col} Share by {entity_col} ({year})", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_dual_axis_yearly_panels(
    df: pd.DataFrame,
    *,
    bar_col: str,
    line_col: str,
    years: list[int],
    time_col: str = "Mes",
    bar_agg: str = "sum",
    line_agg: str = "mean",
    bar_palette: Sequence[str] | None = None,
):
    """Replicate the notebook's tourists-vs-events/establishments panels."""
    panel = _with_panel_time(df, time_col=time_col)
    colors = bar_palette or ["#FFB6C1", "#87CEFA", "#90EE90", "#FFDAB9", "#E6E6FA"]

    fig, axes = plt.subplots(nrows=len(years), figsize=(15, 4 * len(years)), sharex=True)
    if len(years) == 1:
        axes = [axes]

    for idx, (ax, year) in enumerate(zip(axes, years)):
        block = panel[panel["year"] == year]
        grouped = block.groupby("month").agg({bar_col: bar_agg, line_col: line_agg})
        ax.bar(
            grouped.index,
            grouped[bar_col],
            color=colors[idx % len(colors)],
            alpha=0.8,
            width=0.7,
            label=f"{bar_col} {year}",
        )
        ax.set_ylabel(bar_col)
        ax.set_xticks(range(1, 13))
        twin = ax.twinx()
        twin.plot(
            grouped.index,
            grouped[line_col],
            color="black",
            marker="o",
            linewidth=3,
            label=f"{line_col} {year}",
        )
        twin.set_ylabel(line_col)
        for month, value in grouped[line_col].items():
            twin.annotate(
                f"{value:.1f}",
                (month, value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )
        ax.legend(loc="upper left")
        twin.legend(loc="upper right")
        ax.set_title(f"{bar_col} vs {line_col} ({year})", fontsize=13, fontweight="bold")

    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    return fig


def plot_bubble_panels(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: str,
    hue: str,
    years: list[int],
    time_col: str = "Mes",
    n_cols: int = 2,
    size_range: tuple[int, int] = (50, 230),
    palette: str = "magma",
):
    """Plot the multi-year bubble comparison used for crime exploration."""
    panel = _with_panel_time(df, time_col=time_col)
    n_rows = ceil(len(years) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, year in zip(axes, years):
        block = panel[panel["year"] == year]
        sns.scatterplot(
            data=block,
            x=x,
            y=y,
            size=size,
            hue=hue,
            sizes=size_range,
            alpha=0.7,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{year}: {x} vs {y}", fontsize=14, fontweight="bold")

    for ax in axes[len(years) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_3d_feature_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    z: str,
    color: str | None = None,
    cmap: str = "viridis",
    ax=None,
    title: str | None = None,
):
    """Create a 3D scatter plot similar to the notebook's exploratory views."""
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    values = df[color] if color else None
    scatter = ax.scatter(df[x], df[y], df[z], c=values, cmap=cmap, s=50)
    ax.set_xlabel(x, fontsize=12, fontweight="bold")
    ax.set_ylabel(y, fontsize=12, fontweight="bold")
    ax.set_zlabel(z, fontsize=12, fontweight="bold")
    ax.set_title(title or f"{x} vs {y} vs {z}", fontsize=14, fontweight="bold")

    if color:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=12)
        cbar.set_label(color, fontsize=11)

    fig.tight_layout()
    return fig
