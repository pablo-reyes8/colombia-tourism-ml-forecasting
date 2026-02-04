"""LIME helpers for tabular models."""

from __future__ import annotations

import pandas as pd


def lime_explain_instance(
    estimator,
    X_train,
    instance_index: int,
    num_features: int = 10,
    random_state: int | None = None,
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
    return exp, fig
