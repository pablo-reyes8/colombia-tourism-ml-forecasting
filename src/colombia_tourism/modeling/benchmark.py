"""Benchmark utilities for ML and econometric model comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .evaluation import cross_validate_regressor, train_test_split_xy
from .models import build_model, fit_ols, fit_random_effects_panel
from .pipelines import build_pipeline


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def _resolve_models(model_specs: Mapping[str, Any] | Sequence[str]) -> dict[str, Any]:
    if isinstance(model_specs, Mapping):
        return dict(model_specs)
    return {name: build_model(name) for name in model_specs}


def benchmark_regressors(
    model_specs: Mapping[str, Any] | Sequence[str],
    X,
    y,
    *,
    preprocessor=None,
    target_scaler=None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int | None = 5,
    scoring: dict[str, str] | None = None,
    n_jobs: int | None = None,
    sort_by: str = "test_r2",
    return_estimators: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark several regressors on a common split."""
    models = _resolve_models(model_specs)
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        estimator = clone(model) if hasattr(model, "get_params") else model
        estimator = build_pipeline(
            estimator,
            preprocessor=clone(preprocessor) if hasattr(preprocessor, "get_params") else preprocessor,
            target_scaler=clone(target_scaler) if hasattr(target_scaler, "get_params") else target_scaler,
        )

        start = perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        train_pred = estimator.predict(X_train)
        test_pred = estimator.predict(X_test)
        train_metrics = _regression_metrics(y_train, train_pred)
        test_metrics = _regression_metrics(y_test, test_pred)

        row = {
            "model": name,
            "family": "machine_learning",
            "n_features": getattr(X_train, "shape", [None, None])[1],
            "fit_seconds": fit_seconds,
            "train_r2": train_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "overfit_gap_r2": train_metrics["r2"] - test_metrics["r2"],
        }

        if cv:
            cv_results = cross_validate_regressor(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
            )
            row["cv_r2_mean"] = float(np.mean(cv_results["test_r2"]))
            row["cv_r2_std"] = float(np.std(cv_results["test_r2"]))
            row["cv_mae_mean"] = float(-np.mean(cv_results["test_mae"]))
            row["cv_rmse_mean"] = float(np.sqrt(-np.mean(cv_results["test_mse"])))

        rows.append(row)
        fitted[name] = estimator

    results = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
    if return_estimators:
        return results, fitted
    return results


def benchmark_model_names(
    model_names: Sequence[str],
    X,
    y,
    **kwargs,
):
    """Thin wrapper around :func:`benchmark_regressors` for model-name lists."""
    return benchmark_regressors(model_names, X, y, **kwargs)


def benchmark_ols_specs(
    df: pd.DataFrame,
    *,
    target_col: str,
    specs: Mapping[str, Sequence[str]],
    test_size: float = 0.2,
    random_state: int = 42,
    add_constant: bool = True,
) -> pd.DataFrame:
    """Benchmark several OLS specifications on a common train/test split."""
    import statsmodels.api as sm

    rows: list[dict[str, Any]] = []
    for name, features in specs.items():
        required = [target_col, *features]
        subset = df.dropna(subset=required).copy()
        X_train, X_test, y_train, y_test = train_test_split_xy(
            subset[list(features)],
            subset[target_col],
            test_size=test_size,
            random_state=random_state,
        )

        start = perf_counter()
        model = fit_ols(X_train, y_train, add_constant=add_constant)
        fit_seconds = perf_counter() - start

        X_train_design = sm.add_constant(X_train, has_constant="add") if add_constant else X_train
        X_test_design = sm.add_constant(X_test, has_constant="add") if add_constant else X_test
        train_pred = model.predict(X_train_design)
        test_pred = model.predict(X_test_design)

        train_metrics = _regression_metrics(y_train, train_pred)
        test_metrics = _regression_metrics(y_test, test_pred)
        rows.append(
            {
                "model": name,
                "family": "ols",
                "n_features": len(features),
                "fit_seconds": fit_seconds,
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_rmse": test_metrics["rmse"],
                "aic": float(model.aic),
                "bic": float(model.bic),
                "adj_r2": float(model.rsquared_adj),
            }
        )

    return pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)


def benchmark_random_effects_specs(
    df: pd.DataFrame,
    *,
    target_col: str,
    specs: Mapping[str, Sequence[str]],
    entity_col: str,
    time_col: str,
    time_format: str | None = None,
    add_constant: bool = True,
) -> pd.DataFrame:
    """Fit one or more random-effects panel specifications."""
    rows: list[dict[str, Any]] = []
    for name, features in specs.items():
        start = perf_counter()
        result = fit_random_effects_panel(
            df,
            target_col=target_col,
            feature_cols=features,
            entity_col=entity_col,
            time_col=time_col,
            time_format=time_format,
            add_constant=add_constant,
        )
        fit_seconds = perf_counter() - start
        rows.append(
            {
                "model": name,
                "family": "random_effects",
                "n_features": len(features),
                "fit_seconds": fit_seconds,
                "rsquared_overall": float(getattr(result, "rsquared_overall", np.nan)),
                "rsquared_between": float(getattr(result, "rsquared_between", np.nan)),
                "rsquared_within": float(getattr(result, "rsquared_within", np.nan)),
                "loglik": float(getattr(result, "loglik", np.nan)),
                "nobs": int(getattr(result, "nobs", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("rsquared_overall", ascending=False).reset_index(drop=True)


def benchmark_all(
    *,
    X,
    y,
    ml_models: Mapping[str, Any] | Sequence[str],
    ml_preprocessor=None,
    ml_target_scaler=None,
    panel_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    ols_specs: Mapping[str, Sequence[str]] | None = None,
    random_effects_specs: Mapping[str, Sequence[str]] | None = None,
    entity_col: str = "Ciudad",
    time_col: str = "Mes",
    time_format: str | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Run the full comparison stack used in notebook 5."""
    results = {
        "machine_learning": benchmark_regressors(
            ml_models,
            X,
            y,
            preprocessor=ml_preprocessor,
            target_scaler=ml_target_scaler,
            **kwargs,
        )
    }
    if panel_df is not None and target_col and ols_specs:
        results["ols"] = benchmark_ols_specs(
            panel_df,
            target_col=target_col,
            specs=ols_specs,
            test_size=kwargs.get("test_size", 0.2),
            random_state=kwargs.get("random_state", 42),
        )
    if panel_df is not None and target_col and random_effects_specs:
        results["random_effects"] = benchmark_random_effects_specs(
            panel_df,
            target_col=target_col,
            specs=random_effects_specs,
            entity_col=entity_col,
            time_col=time_col,
            time_format=time_format,
        )
    return results
