"""CLI for SHAP interpretation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from colombia_tourism.inference import (
    align_features,
    load_feature_names,
    load_model,
)
from colombia_tourism.interpretation.shap_utils import shap_summary


def main():
    parser = argparse.ArgumentParser(description="Run SHAP interpretation on a CSV.")
    parser.add_argument("--model-uri", required=True, help="MLflow URI or .pkl/.joblib path")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV for SHAP summary")
    parser.add_argument("--target", default=None, help="Target column to drop if present")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--plot-dir", default=None, help="Directory to save SHAP plots")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])

    feature_names = load_feature_names(args.model_uri)
    if feature_names:
        df, _, _ = align_features(df, feature_names)

    model = load_model(args.model_uri)
    summary, shap_values, sample = shap_summary(
        model, df, max_samples=args.max_samples
    )
    summary.to_csv(args.output, index=False)

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        fig1.savefig(plot_dir / "shap_summary.png", bbox_inches="tight")
        plt.close(fig1)

        fig2 = plt.figure()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        fig2.savefig(plot_dir / "shap_summary_bar.png", bbox_inches="tight")
        plt.close(fig2)


if __name__ == "__main__":
    main()
