"""Exploratory analysis helpers."""

from .eda import (
    aggregate_monthly_panel,
    correlation_with_target,
    plot_3d_feature_scatter,
    plot_bubble_panels,
    plot_correlation_heatmap,
    plot_dual_axis_yearly_panels,
    plot_entity_share_comparison,
    plot_group_pairplot,
    plot_year_profiles,
    summarize_feature_groups,
    summarize_variables,
)

__all__ = [
    "aggregate_monthly_panel",
    "correlation_with_target",
    "plot_3d_feature_scatter",
    "plot_bubble_panels",
    "plot_correlation_heatmap",
    "plot_dual_axis_yearly_panels",
    "plot_entity_share_comparison",
    "plot_group_pairplot",
    "plot_year_profiles",
    "summarize_feature_groups",
    "summarize_variables",
]
