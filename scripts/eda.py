"""Run reusable EDA on the final tourism panel dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from colombia_tourism.analysis import (
    correlation_with_target,
    plot_correlation_heatmap,
    plot_year_profiles,
    summarize_feature_groups,
    summarize_variables,
)
from colombia_tourism.data import load_base_final


def main():
    parser = argparse.ArgumentParser(description="Generate reusable EDA outputs.")
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument("--output-dir", default=str(Path("outputs") / "eda"))
    parser.add_argument("--target", default="Nmero Extranjeros")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_base_final(args.data, parse_dates=True, sort_panel=True)

    summarize_variables(df).to_csv(output_dir / "summary_all_numeric.csv", index=False)

    for group_name, summary in summarize_feature_groups(df).items():
        summary.to_csv(output_dir / f"summary_{group_name}.csv", index=False)

    correlation_with_target(df, target_col=args.target).to_csv(
        output_dir / "target_correlations.csv",
        index=False,
    )

    fig, _ = plot_correlation_heatmap(df)
    fig.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
    fig, _ = plot_year_profiles(df, args.target)
    fig.savefig(output_dir / "target_year_profiles.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
