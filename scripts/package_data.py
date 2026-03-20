"""Package dataset, feature contracts and metadata artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from colombia_tourism.data import DEFAULT_TARGET
from colombia_tourism.mlops import (
    DEFAULT_DATA_ARTIFACT_DIR,
    FeatureBuildConfig,
    prepare_dataset_bundle,
    write_dataset_package,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create a reproducible dataset package with schema, feature contracts and ingestion artifacts.",
    )
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_ARTIFACT_DIR))
    parser.add_argument("--dataset-name", default="base_final")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", default=None, help="Path to feature list (txt/json).")
    parser.add_argument("--engineer-features", action="store_true")
    parser.add_argument("--include-density-features", action="store_true")
    parser.add_argument(
        "--lag-columns",
        nargs="+",
        default=["Pib Ponderado", "Eventos", "Temperatura"],
    )
    parser.add_argument(
        "--rolling-columns",
        nargs="+",
        default=["Pib Ponderado", "Eventos"],
    )
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3, 12])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument(
        "--rolling-stats",
        nargs="+",
        default=["mean", "std"],
    )
    parser.add_argument(
        "--no-target-history",
        action="store_false",
        dest="include_target_history",
    )
    parser.set_defaults(include_target_history=True)
    args = parser.parse_args()

    config = FeatureBuildConfig(
        engineer_features=args.engineer_features,
        include_density_features=args.include_density_features,
        include_target_history=args.include_target_history,
        lag_columns=tuple(args.lag_columns),
        rolling_columns=tuple(args.rolling_columns),
        lags=tuple(args.lags),
        rolling_windows=tuple(args.rolling_windows),
        rolling_stats=tuple(args.rolling_stats),
    )

    bundle = prepare_dataset_bundle(
        args.data,
        target=args.target,
        features_path=args.features,
        feature_build_config=config,
    )
    artifacts = write_dataset_package(
        bundle,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )

    print("Dataset package created:")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "dataset_name": args.dataset_name,
                "target": args.target,
                "feature_count": len(bundle.feature_names),
                "raw_shape": list(bundle.raw_df.shape),
                "modeling_shape": list(bundle.modeling_df.shape),
                "engineer_features": config.engineer_features,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\nArtifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
