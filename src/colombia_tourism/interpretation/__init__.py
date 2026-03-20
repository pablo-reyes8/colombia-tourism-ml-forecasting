"""Model interpretation utilities."""

from .permutation import permutation_importance_df
from .lime import (
    lime_explain_instance,
    lime_explain_instances,
    lime_explanation_to_frame,
)
from .pdp import (
    partial_dependence_sweep,
    plot_partial_dependence,
    plot_partial_dependence_sweep,
    plot_pdp_grid,
)
from .shap_utils import shap_summary

__all__ = [
    "permutation_importance_df",
    "lime_explain_instance",
    "lime_explain_instances",
    "lime_explanation_to_frame",
    "partial_dependence_sweep",
    "plot_partial_dependence",
    "plot_partial_dependence_sweep",
    "plot_pdp_grid",
    "shap_summary",
]
