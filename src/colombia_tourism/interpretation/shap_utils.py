"""SHAP helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_summary(
    model,
    X: pd.DataFrame,
    max_samples: int = 200,
    random_state: int = 42,
):
    import shap

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    sample = X.sample(min(len(X), max_samples), random_state=random_state)

    def predict_fn(data):
        data_df = pd.DataFrame(data, columns=sample.columns)
        return model.predict(data_df)

    explainer = shap.Explainer(predict_fn, sample)
    shap_values = explainer(sample)

    values = shap_values.values
    mean_abs = np.abs(values).mean(axis=0)

    summary = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return summary, shap_values, sample
