"""Run a reusable feature-selection suite on the tourism panel dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from colombia_tourism.data import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    ENTITY_COLUMN,
    TIME_COLUMN,
    infer_numeric_features,
    load_base_final,
)
from colombia_tourism.features.engineering import build_modeling_features
from colombia_tourism.modeling.selection import (
    DEFAULT_SELECTION_METHODS,
    aggregate_feature_selection_results,
    feature_selection_summary_frame,
    run_feature_selection_suite,
    save_feature_selection_report,
    select_consensus_features,
)


def load_feature_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        return list(json.loads(source.read_text(encoding="utf-8")))
    return [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_feature_pool(
    df,
    *,
    target: str,
    features_path: str | None,
    engineer_features: bool,
) -> list[str]:
    feature_list = load_feature_list(features_path)
    if feature_list is not None:
        return [feature for feature in feature_list if feature in df.columns]

    if engineer_features:
        candidate_features = infer_numeric_features(
            df,
            exclude=(ENTITY_COLUMN, TIME_COLUMN, "fecha", target),
        )
        return [feature for feature in candidate_features if feature != "year"]

    return [feature for feature in DEFAULT_FEATURES if feature in df.columns]


def main():
    parser = argparse.ArgumentParser(
        description="Select important tourism features with Lasso, sparse-group and latent-factor methods.",
    )
    parser.add_argument("--data", default=str(Path("Data") / "Base Final1.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "feature_selection"),
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", default=None, help="Path to feature list (txt/json).")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=DEFAULT_SELECTION_METHODS,
        help="Selection methods to run. Defaults to the full suite.",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=15)
    parser.add_argument("--consensus-min-votes", type=int, default=1)
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
    parser.add_argument(
        "--no-target-history",
        action="store_false",
        dest="include_target_history",
    )
    parser.set_defaults(include_target_history=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_base_final(
        args.data,
        parse_dates=args.engineer_features,
        sort_panel=args.engineer_features,
    )
    if args.engineer_features:
        df = build_modeling_features(
            df,
            target_col=args.target,
            lag_columns=args.lag_columns,
            rolling_columns=args.rolling_columns,
            include_target_history=args.include_target_history,
            include_density_features=args.include_density_features,
        )

    feature_pool = resolve_feature_pool(
        df,
        target=args.target,
        features_path=args.features,
        engineer_features=args.engineer_features,
    )
    if not feature_pool:
        raise ValueError("No candidate features were resolved for selection.")

    results = run_feature_selection_suite(
        df,
        target=args.target,
        feature_names=feature_pool,
        methods=args.methods,
        cv=args.cv,
        random_state=args.random_state,
        poly_degree=args.poly_degree,
        max_features=args.max_features,
    )
    artifacts = save_feature_selection_report(
        results,
        output_dir,
        consensus_top_k=args.max_features,
        consensus_min_votes=args.consensus_min_votes,
    )
    consensus = aggregate_feature_selection_results(results)
    selected_features = select_consensus_features(
        results,
        top_k=args.max_features,
        min_votes=args.consensus_min_votes,
    )

    (output_dir / "feature_pool.txt").write_text(
        "\n".join(feature_pool),
        encoding="utf-8",
    )
    (output_dir / "selection_config.json").write_text(
        json.dumps(
            {
                "data": args.data,
                "target": args.target,
                "methods": args.methods or DEFAULT_SELECTION_METHODS,
                "cv": args.cv,
                "random_state": args.random_state,
                "poly_degree": args.poly_degree,
                "max_features": args.max_features,
                "consensus_min_votes": args.consensus_min_votes,
                "engineer_features": args.engineer_features,
                "include_density_features": args.include_density_features,
                "include_target_history": args.include_target_history,
                "lag_columns": args.lag_columns,
                "rolling_columns": args.rolling_columns,
                "feature_pool_size": len(feature_pool),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = feature_selection_summary_frame(results)[
        ["method", "n_selected_original_features", "model_score"]
    ]

    print("Feature selection summary:")
    print(summary.to_string(index=False))
    print("\nConsensus top features:")
    for feature in selected_features:
        row = consensus.loc[consensus["feature"] == feature].iloc[0]
        print(
            f"- {feature} | votes={int(row['selection_votes'])} | "
            f"mean_score={row['mean_normalized_score']:.4f}"
        )

    print("\nArtifacts:")
    print(f"- feature_pool: {output_dir / 'feature_pool.txt'}")
    print(f"- config: {output_dir / 'selection_config.json'}")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
