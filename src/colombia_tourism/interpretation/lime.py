"""LIME helpers for tabular models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def lime_explanation_to_frame(explanation) -> pd.DataFrame:
    """Convert a LIME explanation into a tidy dataframe."""
    return pd.DataFrame(
        explanation.as_list(),
        columns=["feature_interval", "contribution"],
    )


def lime_explain_instance(
    estimator,
    X_train,
    instance_index: int,
    num_features: int = 10,
    random_state: int | None = None,
    title: str | None = None,
    colors: tuple[str, str] = ("#B03060", "#FF7F50"),
):
    """Return a LIME explanation and matplotlib figure.

    Keeps LIME as an optional dependency.
    """
    from lime import lime_tabular

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    feature_names = X_train.columns.tolist()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode="regression",
        random_state=random_state,
    )

    instance = X_train.iloc[instance_index].values

    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        return estimator.predict(X_df)

    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
    )

    fig = exp.as_pyplot_figure()
    ax = fig.gca()
    for idx, bar in enumerate(ax.patches):
        bar.set_color(colors[idx % len(colors)])
    if title:
        ax.set_title(title, fontsize=15)
    ax.set_xlabel("Contribution")
    ax.set_ylabel("Features")
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    return exp, fig


def lime_explain_instances(
    estimator,
    X_train,
    instance_indices,
    *,
    num_features: int = 10,
    random_state: int | None = None,
    title_template: str = "LIME explanation for row {index}",
) -> list[tuple[int, pd.DataFrame, plt.Figure]]:
    """Explain multiple rows and return tidy tables plus figures."""
    outputs = []
    for index in instance_indices:
        exp, fig = lime_explain_instance(
            estimator,
            X_train,
            instance_index=index,
            num_features=num_features,
            random_state=random_state,
            title=title_template.format(index=index),
        )
        outputs.append((index, lime_explanation_to_frame(exp), fig))
    return outputs
