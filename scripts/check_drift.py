"""Check dataset drift against a packaged baseline and optionally retrain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from colombia_tourism.config import PROJECT_ROOT
from colombia_tourism.mlops import (
    FeatureBuildConfig,
    analyze_dataset_drift,
    load_dataset_package_manifest,
    load_packaged_feature_names,
    load_packaged_modeling_dataset,
    prepare_dataset_bundle,
    should_retrain,
    write_dataset_package,
    write_drift_report,
)


def build_default_retrain_command(args, current_package_dir: Path) -> list[str]:
    command = [
        args.python_executable,
        "scripts/train.py",
        "--data-package-dir",
        str(current_package_dir),
        "--registered-model-name",
        args.train_registered_model_name,
        "--experiment",
        args.train_experiment,
    ]
    if args.train_candidate_models:
        command.extend(["--candidate-models", *args.train_candidate_models])
    else:
        command.extend(["--model", args.train_model])
    if args.train_tracking_uri:
        command.extend(["--tracking-uri", args.train_tracking_uri])
    if args.train_model_params:
        command.extend(["--model-params", args.train_model_params])
    if args.train_tune:
        command.append("--tune")
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Detect dataset drift against a reference package and optionally retrain the champion model.",
    )
    parser.add_argument("--reference-dir", required=True, help="Path to packaged baseline data directory.")
    parser.add_argument("--current-data", required=True, help="Current dataset CSV to compare against baseline.")
    parser.add_argument("--output-dir", default=str(Path("artifacts") / "drift" / "latest"))
    parser.add_argument("--psi-warning", type=float, default=0.1)
    parser.add_argument("--psi-critical", type=float, default=0.2)
    parser.add_argument("--ks-alpha", type=float, default=0.05)
    parser.add_argument("--missing-rate-delta-threshold", type=float, default=0.05)
    parser.add_argument("--mean-shift-threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-drifted-features", type=int, default=3)
    parser.add_argument("--critical-feature-count", type=int, default=1)
    parser.add_argument("--mean-psi-threshold", type=float, default=0.15)
    parser.add_argument("--retrain-on-drift", action="store_true")
    parser.add_argument("--retrain-command", default=None, help="Optional shell command to execute on drift.")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--train-model", default="xgboost")
    parser.add_argument("--train-candidate-models", nargs="+", default=None)
    parser.add_argument("--train-experiment", default="colombia-tourism")
    parser.add_argument("--train-registered-model-name", default="colombia-tourism-forecasting")
    parser.add_argument("--train-tracking-uri", default=None)
    parser.add_argument("--train-model-params", default=None)
    parser.add_argument("--train-tune", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = load_dataset_package_manifest(args.reference_dir)
    reference_df = load_packaged_modeling_dataset(args.reference_dir)
    reference_features = load_packaged_feature_names(args.reference_dir)
    feature_build_config = FeatureBuildConfig.from_dict(
        reference_manifest.get("feature_build_config")
    )
    target = reference_manifest["target"]

    current_bundle = prepare_dataset_bundle(
        args.current_data,
        target=target,
        feature_names=reference_features,
        feature_build_config=feature_build_config,
    )
    current_package_dir = output_dir / "current_package"
    current_artifacts = write_dataset_package(
        current_bundle,
        output_dir=current_package_dir,
        dataset_name="current_candidate",
    )

    drift_frame, drift_summary = analyze_dataset_drift(
        reference_df,
        current_bundle.modeling_df,
        feature_names=reference_features,
        psi_warning=args.psi_warning,
        psi_critical=args.psi_critical,
        ks_alpha=args.ks_alpha,
        missing_rate_delta_threshold=args.missing_rate_delta_threshold,
        mean_shift_threshold=args.mean_shift_threshold,
        bins=args.bins,
    )
    retrain_required, retrain_reason = should_retrain(
        drift_summary,
        min_drifted_features=args.min_drifted_features,
        critical_feature_count=args.critical_feature_count,
        mean_psi_threshold=args.mean_psi_threshold,
    )

    decision = {
        "generated_at": drift_summary.get("generated_at"),
        "reference_dir": str(args.reference_dir),
        "current_data": args.current_data,
        "current_package_dir": str(current_package_dir),
        "retrain_required": retrain_required,
        "retrain_reason": retrain_reason,
        "retrain_requested": args.retrain_on_drift,
        "retrain_executed": False,
        "retrain_command": None,
        "retrain_return_code": None,
    }

    if retrain_required and args.retrain_on_drift:
        if args.retrain_command:
            decision["retrain_command"] = args.retrain_command
            completed = subprocess.run(
                args.retrain_command,
                cwd=PROJECT_ROOT,
                shell=True,
                check=False,
            )
        else:
            command = build_default_retrain_command(args, current_package_dir)
            decision["retrain_command"] = command
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
            )
        decision["retrain_executed"] = True
        decision["retrain_return_code"] = completed.returncode
        if completed.returncode != 0:
            write_drift_report(
                drift_frame,
                drift_summary,
                output_dir,
                retrain_decision=decision,
            )
            raise SystemExit(completed.returncode)

    artifacts = write_drift_report(
        drift_frame,
        drift_summary,
        output_dir,
        retrain_decision=decision,
    )

    print("Drift summary:")
    print(json.dumps(drift_summary, indent=2, ensure_ascii=False))
    print("\nDecision:")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print("\nArtifacts:")
    print(f"- current_package: {current_package_dir}")
    for name, path in current_artifacts.items():
        print(f"- current_{name}: {path}")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")

    if retrain_required and args.fail_on_drift:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
