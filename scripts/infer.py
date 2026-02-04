"""CLI for batch inference."""

from __future__ import annotations

import argparse

import pandas as pd

from colombia_tourism.inference import (
    align_features,
    load_feature_names,
    load_model,
    predict_dataframe,
)


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV.")
    parser.add_argument("--model-uri", required=True, help="MLflow URI or .pkl/.joblib path")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV with predictions")
    parser.add_argument("--target", default=None, help="Target column to drop if present")
    parser.add_argument("--prediction-col", default="prediction")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])

    feature_names = load_feature_names(args.model_uri)
    missing = []
    extra = []
    if feature_names:
        df, missing, extra = align_features(df, feature_names)

    model = load_model(args.model_uri)
    preds = predict_dataframe(model, df)

    result = pd.read_csv(args.input)
    result[args.prediction_col] = preds
    result.to_csv(args.output, index=False)

    if missing:
        print(f"Missing features added with 0: {missing}")
    if extra:
        print(f"Extra columns ignored: {extra}")


if __name__ == "__main__":
    main()
