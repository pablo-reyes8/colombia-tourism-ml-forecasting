"""MLflow utilities and orchestration wrappers for training workflows."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone

from colombia_tourism.modeling import (
    build_model,
    build_pipeline,
    cross_validate_regressor,
    model_search_space,
    tune_model,
    train_test_split_xy,
)
from colombia_tourism.modeling.evaluation import evaluate_regressor
from colombia_tourism.modeling.preprocess import make_preprocessor


def set_tracking_uri(tracking_uri: str | None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def set_experiment(experiment_name: str | None) -> None:
    if experiment_name:
        mlflow.set_experiment(experiment_name)


def log_json_artifact(
    payload: Any,
    filename: str,
    *,
    artifact_path: str = "metadata",
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_feature_names(feature_names: Iterable[str], artifact_path: str = "metadata"):
    payload = {"feature_names": list(feature_names)}
    log_json_artifact(payload, "feature_names.json", artifact_path=artifact_path)


def log_target_name(target: str, artifact_path: str = "metadata"):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "target_name.txt"
        path.write_text(target, encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model_metadata(
    metadata: Mapping[str, Any],
    *,
    artifact_path: str = "metadata",
) -> None:
    log_json_artifact(dict(metadata), "model_metadata.json", artifact_path=artifact_path)


def _fingerprint_df(df: pd.DataFrame) -> str:
    data_bytes = df.head(1000).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()


def log_dataset(
    df: pd.DataFrame,
    name: str = "dataset",
    artifact_path: str = "data",
    sample_size: int | None = None,
):
    """Log a dataset snapshot to MLflow."""
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("dataset_cols", df.shape[1])
    mlflow.log_param("dataset_fingerprint", _fingerprint_df(df))


def _flatten_for_params(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        param_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_for_params(param_key, value))
        elif isinstance(value, (list, tuple, set)):
            flat[param_key] = json.dumps(list(value), ensure_ascii=False)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[param_key] = value
        else:
            flat[param_key] = str(value)
    return flat


def _extract_cv_metrics(cv_results) -> dict[str, float]:
    if cv_results is None:
        return {}
    metrics: dict[str, float] = {}
    if "test_r2" in cv_results:
        metrics["cv_r2_mean"] = float(cv_results["test_r2"].mean())
        metrics["cv_r2_std"] = float(cv_results["test_r2"].std())
    if "test_mae" in cv_results:
        metrics["cv_mae_mean"] = float(-cv_results["test_mae"].mean())
    if "test_mse" in cv_results:
        metrics["cv_mse_mean"] = float(-cv_results["test_mse"].mean())
        metrics["cv_rmse_mean"] = float((-cv_results["test_mse"].mean()) ** 0.5)
    return metrics


def _safe_input_example(X: pd.DataFrame, rows: int = 50) -> pd.DataFrame:
    return X.head(min(rows, len(X))).copy()


@dataclass
class TrainingResult:
    model_name: str
    run_id: str
    model_uri: str
    metrics: dict[str, float]
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    model_alias: str | None = None
    best_params: dict[str, Any] | None = None

    def as_summary_row(self) -> dict[str, Any]:
        row = {
            "model": self.model_name,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "registered_model_name": self.registered_model_name,
            "registered_model_version": self.registered_model_version,
            "model_alias": self.model_alias,
        }
        row.update(self.metrics)
        if self.best_params:
            row["best_params"] = json.dumps(self.best_params, ensure_ascii=False)
        return row


class MLflowTrainingOrchestrator:
    """Train, tune, log and register candidate models in MLflow."""

    def __init__(
        self,
        *,
        experiment_name: str = "colombia-tourism",
        tracking_uri: str | None = None,
        registered_model_name: str | None = None,
        model_alias: str | None = "champion",
        artifact_path: str = "model",
    ) -> None:
        set_tracking_uri(tracking_uri)
        set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        self.model_alias = model_alias
        self.artifact_path = artifact_path
        self.client = MlflowClient()

    def _build_preprocessor(
        self,
        feature_names: Sequence[str],
        *,
        scaler: str = "standard",
        numeric_imputer: str | None = None,
        knn_neighbors: int = 5,
        poly_degree: int | None = None,
        pca_components: int | None = None,
        pca_variance: float | None = None,
    ):
        return make_preprocessor(
            numeric_features=feature_names,
            scaler=scaler,
            numeric_imputer=numeric_imputer,
            knn_neighbors=knn_neighbors,
            poly_degree=poly_degree,
            pca_components=pca_components,
            pca_variance=pca_variance,
            remainder="drop",
        )

    def _register_model(
        self,
        *,
        run_id: str,
        registered_model_name: str | None,
        artifact_path: str,
        model_alias: str | None,
    ) -> tuple[str | None, str | None]:
        if not registered_model_name:
            return None, None

        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )
        version = str(model_version.version)
        if model_alias:
            self.client.set_registered_model_alias(
                registered_model_name,
                model_alias,
                version,
            )
        return version, model_alias

    def train_candidate(
        self,
        *,
        df: pd.DataFrame,
        target: str,
        feature_names: Sequence[str],
        model_name: str = "xgboost",
        model_params: Mapping[str, Any] | None = None,
        registered_model_name: str | None = None,
        model_alias: str | None = None,
        run_name: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int | None = 5,
        scoring: Mapping[str, str] | None = None,
        n_jobs: int | None = None,
        scaler: str = "standard",
        numeric_imputer: str | None = None,
        knn_neighbors: int = 5,
        poly_degree: int | None = None,
        pca_components: int | None = None,
        pca_variance: float | None = None,
        log_data_flag: bool = True,
        data_sample: int = 1000,
        tune_hyperparameters: bool = False,
        param_distributions: Mapping[str, Any] | None = None,
        n_iter: int = 30,
        search_scoring: str = "r2",
    ) -> TrainingResult:
        feature_names = list(feature_names)
        model_params = dict(model_params or {})
        registered_model_name = registered_model_name or self.registered_model_name
        model_alias = model_alias or self.model_alias

        X = df[feature_names]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split_xy(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        preprocessor = self._build_preprocessor(
            feature_names,
            scaler=scaler,
            numeric_imputer=numeric_imputer,
            knn_neighbors=knn_neighbors,
            poly_degree=poly_degree,
            pca_components=pca_components,
            pca_variance=pca_variance,
        )

        base_model = build_model(model_name, **model_params)
        base_estimator = build_pipeline(base_model, preprocessor=preprocessor)
        estimator = base_estimator
        best_params: dict[str, Any] | None = None
        tuning_payload: dict[str, Any] | None = None

        run_name = run_name or model_name
        with mlflow.start_run(
            run_name=run_name,
            nested=mlflow.active_run() is not None,
        ) as run:
            fit_start = perf_counter()

            if tune_hyperparameters:
                search_space = dict(
                    param_distributions
                    or model_search_space(model_name, pipeline_prefix="model")
                )
                search = tune_model(
                    estimator,
                    search_space,
                    X_train,
                    y_train,
                    cv=cv or 5,
                    scoring=search_scoring,
                    n_iter=n_iter,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                estimator = search.best_estimator_
                best_params = dict(search.best_params_)
                tuning_payload = {
                    "search_scoring": search_scoring,
                    "n_iter": n_iter,
                    "best_score": float(search.best_score_),
                    "best_params": best_params,
                }
            else:
                estimator = clone(base_estimator)
                estimator.fit(X_train, y_train)

            fit_seconds = perf_counter() - fit_start

            train_metrics = {
                f"train_{key}": value
                for key, value in evaluate_regressor(estimator, X_train, y_train).items()
            }
            test_metrics = {
                f"test_{key}": value
                for key, value in evaluate_regressor(estimator, X_test, y_test).items()
            }
            metrics = {
                **train_metrics,
                **test_metrics,
                "fit_seconds": fit_seconds,
                "overfit_gap_r2": train_metrics["train_r2"] - test_metrics["test_r2"],
            }

            cv_metrics: dict[str, float] = {}
            if cv:
                cv_results = cross_validate_regressor(
                    clone(estimator),
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                )
                cv_metrics = _extract_cv_metrics(cv_results)
                metrics.update(cv_metrics)

            final_estimator = clone(estimator)
            final_estimator.fit(X, y)

            mlflow.log_params(
                {
                    "model": model_name,
                    "test_size": test_size,
                    "random_state": random_state,
                    "features": len(feature_names),
                    "scaler": scaler,
                    "numeric_imputer": numeric_imputer,
                    "knn_neighbors": knn_neighbors,
                    "poly_degree": poly_degree,
                    "pca_components": pca_components,
                    "pca_variance": pca_variance,
                    "tune_hyperparameters": tune_hyperparameters,
                    "registered_model_name": registered_model_name,
                }
            )
            mlflow.log_params(_flatten_for_params("model_params", model_params))
            if best_params:
                mlflow.log_params(_flatten_for_params("best_params", best_params))
            mlflow.log_metrics(metrics)

            if log_data_flag:
                log_dataset(df, sample_size=data_sample)

            log_feature_names(feature_names)
            log_target_name(target)
            log_feature_names(feature_names, artifact_path=self.artifact_path)
            log_target_name(target, artifact_path=self.artifact_path)

            metadata = {
                "model_name": model_name,
                "artifact_path": self.artifact_path,
                "registered_model_name": registered_model_name,
                "model_alias": model_alias,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "target_name": target,
                "metrics": metrics,
                "best_params": best_params,
                "experiment_name": self.experiment_name,
            }
            log_model_metadata(metadata)
            log_model_metadata(metadata, artifact_path=self.artifact_path)

            if tuning_payload:
                log_json_artifact(tuning_payload, "tuning_summary.json", artifact_path="metadata")

            sample = _safe_input_example(X)
            signature = infer_signature(sample, final_estimator.predict(sample))
            mlflow.sklearn.log_model(
                final_estimator,
                self.artifact_path,
                signature=signature,
                input_example=sample,
            )

            registered_version, promoted_alias = self._register_model(
                run_id=run.info.run_id,
                registered_model_name=registered_model_name,
                artifact_path=self.artifact_path,
                model_alias=model_alias,
            )

            return TrainingResult(
                model_name=model_name,
                run_id=run.info.run_id,
                model_uri=f"runs:/{run.info.run_id}/{self.artifact_path}",
                metrics=metrics,
                registered_model_name=registered_model_name,
                registered_model_version=registered_version,
                model_alias=promoted_alias,
                best_params=best_params,
            )

    def train_sequence(
        self,
        *,
        df: pd.DataFrame,
        target: str,
        feature_names: Sequence[str],
        model_names: Sequence[str],
        run_name: str = "training-sequence",
        primary_metric: str = "test_r2",
        promote_best: bool = True,
        **train_kwargs,
    ) -> tuple[TrainingResult, pd.DataFrame]:
        """Train multiple candidates and optionally promote the best one."""
        feature_names = list(feature_names)
        parent_needed = len(model_names) > 1
        results: list[TrainingResult] = []

        if parent_needed:
            with mlflow.start_run(run_name=run_name) as parent_run:
                mlflow.log_params(
                    {
                        "sequence_models": json.dumps(list(model_names), ensure_ascii=False),
                        "primary_metric": primary_metric,
                        "registered_model_name": train_kwargs.get("registered_model_name")
                        or self.registered_model_name,
                    }
                )
                for model_name in model_names:
                    candidate_kwargs = dict(train_kwargs)
                    if promote_best:
                        candidate_kwargs["model_alias"] = None
                    candidate = self.train_candidate(
                        df=df,
                        target=target,
                        feature_names=feature_names,
                        model_name=model_name,
                        run_name=model_name,
                        **candidate_kwargs,
                    )
                    results.append(candidate)

                summary = pd.DataFrame([result.as_summary_row() for result in results])
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "sequence_summary.csv"
                    summary.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="metadata")
                mlflow.log_param("parent_run_id", parent_run.info.run_id)
        else:
            results.append(
                self.train_candidate(
                    df=df,
                    target=target,
                    feature_names=feature_names,
                    model_name=model_names[0],
                    run_name=run_name,
                    **train_kwargs,
                )
            )

        summary = pd.DataFrame([result.as_summary_row() for result in results])
        best_idx = summary[primary_metric].astype(float).idxmax()
        best_result = results[int(best_idx)]

        if promote_best and best_result.registered_model_name and best_result.registered_model_version:
            alias = train_kwargs.get("model_alias") or self.model_alias
            if alias:
                self.client.set_registered_model_alias(
                    best_result.registered_model_name,
                    alias,
                    best_result.registered_model_version,
                )
                best_result.model_alias = alias

        return best_result, summary
