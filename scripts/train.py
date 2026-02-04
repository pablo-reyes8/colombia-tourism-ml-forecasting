"""Train a model, evaluate, and log to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

from colombia_tourism.data import DEFAULT_FEATURES, DEFAULT_TARGET, load_base_final
from colombia_tourism.mlflow_utils import log_dataset, log_feature_names, log_target_name, set_experiment
from colombia_tourism.modeling import (
    build_catboost,
    build_keras_regressor,
    build_lightgbm,
    build_pipeline,
    build_sklearn_model,
    build_xgboost,
    fit_and_evaluate,
    make_preprocessor,
)


def load_feature_list(path: str | None):
    if not path:
        return None
    path = Path(path)
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_model(name: str):
    name = name.lower()
    if name in {"linear", "ridge", "lasso", "elasticnet", "random_forest", "gradient_boosting"}:
        return build_sklearn_model(name)
    if name == "xgboost":
        return build_xgboost()
    if name == "lightgbm":
        return build_lightgbm()
    if name == "catboost":
        return build_catboost()
    if name == "keras":
        return build_keras_regressor()
    raise ValueError(f"Modelo no soportado: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", default=None, help="Path to features list (json/txt)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--scaler", default="standard")
    parser.add_argument("--poly-degree", type=int, default=None)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-variance", type=float, default=None)
    parser.add_argument("--experiment", default="colombia-tourism")
    parser.add_argument("--log-data", action="store_true", default=True)
    parser.add_argument("--no-log-data", action="store_false", dest="log_data")
    parser.add_argument("--data-sample", type=int, default=1000)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    df = load_base_final(args.data)

    feature_list = load_feature_list(args.features)
    if feature_list is None:
        feature_list = [f for f in DEFAULT_FEATURES if f in df.columns]

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in dataframe")

    X = df[feature_list]
    y = df[args.target]

    preprocessor = make_preprocessor(
        numeric_features=feature_list,
        scaler=args.scaler,
        poly_degree=args.poly_degree,
        pca_components=args.pca_components,
        pca_variance=args.pca_variance,
        remainder="drop",
    )

    model = build_model(args.model)
    pipeline = build_pipeline(model, preprocessor=preprocessor)

    set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.model}"):
        metrics, cv_results = fit_and_evaluate(
            pipeline,
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            cv=args.cv,
        )

        mlflow.log_params(
            {
                "model": args.model,
                "test_size": args.test_size,
                "scaler": args.scaler,
                "poly_degree": args.poly_degree,
                "pca_components": args.pca_components,
                "pca_variance": args.pca_variance,
                "features": len(feature_list),
            }
        )
        mlflow.log_metrics(metrics)

        if cv_results is not None:
            mlflow.log_metric("cv_r2_mean", cv_results["test_r2"].mean())
            mlflow.log_metric("cv_mae_mean", -cv_results["test_mae"].mean())
            mlflow.log_metric("cv_mse_mean", -cv_results["test_mse"].mean())

        if args.log_data:
            log_dataset(df, sample_size=args.data_sample)

        log_feature_names(feature_list)
        log_target_name(args.target)

        sample = X.head(50)
        signature = infer_signature(sample, pipeline.predict(sample))
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,
            input_example=sample,
        )


if __name__ == "__main__":
    main()
