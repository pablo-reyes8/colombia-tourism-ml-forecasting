"""Train, tune and register tourism models with MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from colombia_tourism.data import DEFAULT_FEATURES, DEFAULT_TARGET, load_base_final
from colombia_tourism.mlops import (
    load_dataset_package_manifest,
    load_feature_list,
    load_packaged_feature_names,
    load_packaged_modeling_dataset,
)
from colombia_tourism.mlflow_utils import MLflowTrainingOrchestrator


def load_model_params(payload: str | None) -> dict:
    if not payload:
        return {}
    return json.loads(payload)


def load_training_frame(args):
    if args.data_package_dir:
        manifest = load_dataset_package_manifest(args.data_package_dir)
        df = load_packaged_modeling_dataset(args.data_package_dir)
        if args.features:
            feature_list = load_feature_list(args.features)
        else:
            feature_list = load_packaged_feature_names(args.data_package_dir)
        target = manifest.get("target", args.target)
        dataset_artifact_dir = args.data_package_dir
        return df, target, feature_list, dataset_artifact_dir

    df = load_base_final(args.data)
    feature_list = load_feature_list(args.features)
    if feature_list is None:
        feature_list = [feature for feature in DEFAULT_FEATURES if feature in df.columns]
    return df, args.target, feature_list, None


def main():
    parser = argparse.ArgumentParser(description="Train and register models with MLflow.")
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument(
        "--data-package-dir",
        default=None,
        help="Path to a packaged dataset directory created by scripts/package_data.py",
    )
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--candidate-models", nargs="+", default=None)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", default=None, help="Path to features list (json/txt)")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="colombia-tourism")
    parser.add_argument("--registered-model-name", default="colombia-tourism-forecasting")
    parser.add_argument("--model-alias", default="champion")
    parser.add_argument("--artifact-path", default="model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--scaler", default="standard")
    parser.add_argument("--numeric-imputer", default="mean")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--poly-degree", type=int, default=None)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-variance", type=float, default=None)
    parser.add_argument("--log-data", action="store_true", default=True)
    parser.add_argument("--no-log-data", action="store_false", dest="log_data")
    parser.add_argument("--data-sample", type=int, default=1000)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--search-scoring", default="r2")
    parser.add_argument("--model-params", default=None, help="JSON string with model-specific parameters")
    args = parser.parse_args()

    df, target, feature_list, dataset_artifact_dir = load_training_frame(args)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataframe")

    model_params = load_model_params(args.model_params)
    candidate_models = args.candidate_models or [args.model]

    orchestrator = MLflowTrainingOrchestrator(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model_name,
        model_alias=args.model_alias,
        artifact_path=args.artifact_path,
    )

    best_result, summary = orchestrator.train_sequence(
        df=df,
        target=target,
        feature_names=feature_list,
        model_names=candidate_models,
        run_name=candidate_models[0] if len(candidate_models) == 1 else "model-selection-sequence",
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs,
        scaler=args.scaler,
        numeric_imputer=args.numeric_imputer,
        knn_neighbors=args.knn_neighbors,
        poly_degree=args.poly_degree,
        pca_components=args.pca_components,
        pca_variance=args.pca_variance,
        log_data_flag=args.log_data,
        data_sample=args.data_sample,
        tune_hyperparameters=args.tune,
        n_iter=args.n_iter,
        search_scoring=args.search_scoring,
        model_params=model_params,
        dataset_artifact_dir=dataset_artifact_dir,
    )

    print("Best model:")
    print(json.dumps(best_result.as_summary_row(), indent=2, ensure_ascii=False))
    print("\nCandidate summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
