"""Benchmark several regressors on the tourism panel dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from colombia_tourism.data import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    infer_numeric_features,
    load_base_final,
)
from colombia_tourism.features import build_modeling_features
from colombia_tourism.modeling import (
    benchmark_model_names,
    make_preprocessor,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark several models on a shared split.")
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument("--output", default=str(Path("outputs") / "benchmark_results.csv"))
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "ridge", "random_forest", "xgboost"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--engineer-features", action="store_true")
    args = parser.parse_args()

    df = load_base_final(args.data, parse_dates=True, sort_panel=True)
    if args.engineer_features:
        df = build_modeling_features(
            df,
            lag_columns=["Pib Ponderado", "Eventos", "Temperatura"],
            rolling_columns=["Pib Ponderado", "Eventos"],
        )

    base_features = [feature for feature in DEFAULT_FEATURES if feature in df.columns]
    if args.engineer_features:
        candidate_features = infer_numeric_features(df, exclude=("Ciudad", "Mes", args.target))
        features = [feature for feature in candidate_features if feature != "year"]
    else:
        features = base_features

    X = df[features].fillna(0)
    y = df[args.target]
    preprocessor = make_preprocessor(
        numeric_features=features,
        scaler="standard",
        numeric_imputer="mean",
        remainder="drop",
    )

    results = benchmark_model_names(
        args.models,
        X,
        y,
        preprocessor=preprocessor,
        test_size=args.test_size,
        cv=args.cv,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)


if __name__ == "__main__":
    main()
