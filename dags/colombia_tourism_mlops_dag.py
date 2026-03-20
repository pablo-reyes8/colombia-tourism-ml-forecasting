"""Airflow DAG orchestrating the end-to-end Colombia Tourism MLOps pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import request

import pendulum
from airflow.sdk import dag, task


PROJECT_ROOT = Path(os.getenv("CTF_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PYTHON_BIN = os.getenv("CTF_PYTHON_BIN", sys.executable)
DATA_PATH = PROJECT_ROOT / "Data" / "Base Final1.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BASELINE_PACKAGE_DIR = ARTIFACTS_DIR / "data" / "production_baseline"
CURRENT_PACKAGE_DIR = ARTIFACTS_DIR / "data" / "current_candidate"
DRIFT_OUTPUT_DIR = ARTIFACTS_DIR / "drift" / "latest"
INFERENCE_OUTPUT_DIR = ARTIFACTS_DIR / "inference" / "latest"
API_SMOKE_OUTPUT = ARTIFACTS_DIR / "api" / "smoke" / "latest.json"
REGISTERED_MODEL_NAME = os.getenv("CTF_REGISTERED_MODEL_NAME", "colombia-tourism-forecasting")
MODEL_ALIAS = os.getenv("CTF_MODEL_ALIAS", "champion")
API_BASE_URL = os.getenv("CTF_API_BASE_URL", "http://api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TRAIN_EXPERIMENT = os.getenv("CTF_TRAIN_EXPERIMENT", "colombia-tourism")
TRAIN_MODEL = os.getenv("CTF_TRAIN_MODEL", "xgboost")


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    env["CTF_REGISTERED_MODEL_NAME"] = REGISTERED_MODEL_NAME
    env["CTF_MODEL_ALIAS"] = MODEL_ALIAS
    env["CTF_PROJECT_ROOT"] = str(PROJECT_ROOT)
    return env


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_project_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed.stdout


@dag(
    dag_id="colombia_tourism_end_to_end_mlops",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "tourism", "xgboost", "airflow"],
    doc_md="""
    ## Colombia Tourism End-to-End MLOps

    This DAG orchestrates:
    1. Dataset packaging and feature/data contract materialization
    2. Bootstrap training when no production baseline exists
    3. Drift monitoring against the production baseline
    4. Conditional retraining and baseline promotion
    5. Batch inference artifact generation
    6. API smoke testing against the deployed FastAPI service
    """,
)
def colombia_tourism_end_to_end_mlops():
    @task
    def package_current_data() -> dict[str, str]:
        CURRENT_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                PYTHON_BIN,
                "scripts/package_data.py",
                "--data",
                str(DATA_PATH),
                "--output-dir",
                str(CURRENT_PACKAGE_DIR),
                "--dataset-name",
                "current_candidate",
            ]
        )
        return {
            "package_dir": str(CURRENT_PACKAGE_DIR),
            "manifest": str(CURRENT_PACKAGE_DIR / "metadata" / "manifest.json"),
            "raw_snapshot": str(CURRENT_PACKAGE_DIR / "raw" / "current_candidate_snapshot.csv"),
            "modeling_dataset": str(CURRENT_PACKAGE_DIR / "processed" / "modeling_dataset.csv"),
            "feature_list": str(CURRENT_PACKAGE_DIR / "metadata" / "feature_list.txt"),
        }

    @task
    def inspect_pipeline_state() -> dict[str, str | bool]:
        baseline_manifest = BASELINE_PACKAGE_DIR / "metadata" / "manifest.json"
        return {
            "baseline_exists": baseline_manifest.exists(),
            "baseline_package_dir": str(BASELINE_PACKAGE_DIR),
            "baseline_manifest": str(baseline_manifest),
        }

    @task
    def bootstrap_train_if_needed(
        current_package: dict[str, str],
        pipeline_state: dict[str, str | bool],
    ) -> dict[str, str | bool]:
        if pipeline_state["baseline_exists"]:
            return {
                "bootstrapped": False,
                "trained": False,
                "reason": "baseline_already_exists",
            }

        _run_command(
            [
                PYTHON_BIN,
                "scripts/train.py",
                "--data-package-dir",
                current_package["package_dir"],
                "--model",
                TRAIN_MODEL,
                "--experiment",
                TRAIN_EXPERIMENT,
                "--registered-model-name",
                REGISTERED_MODEL_NAME,
                "--tracking-uri",
                MLFLOW_TRACKING_URI,
            ]
        )
        _run_command(
            [
                PYTHON_BIN,
                "scripts/promote_data_package.py",
                "--source-dir",
                current_package["package_dir"],
                "--target-dir",
                str(BASELINE_PACKAGE_DIR),
                "--reason",
                "bootstrap_training",
            ]
        )
        return {
            "bootstrapped": True,
            "trained": True,
            "reason": "bootstrap_training",
        }

    @task
    def run_drift_monitoring(
        current_package: dict[str, str],
        pipeline_state: dict[str, str | bool],
    ) -> dict[str, str | bool]:
        if not pipeline_state["baseline_exists"]:
            return {
                "drift_checked": False,
                "retrain_required": False,
                "retrain_reason": "bootstrap_mode",
            }

        DRIFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                PYTHON_BIN,
                "scripts/check_drift.py",
                "--reference-dir",
                str(BASELINE_PACKAGE_DIR),
                "--current-data",
                current_package["raw_snapshot"],
                "--output-dir",
                str(DRIFT_OUTPUT_DIR),
            ]
        )
        decision_path = DRIFT_OUTPUT_DIR / "retrain_decision.json"
        summary_path = DRIFT_OUTPUT_DIR / "drift_summary.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        decision["drift_summary"] = summary
        return decision

    @task
    def retrain_if_needed(
        current_package: dict[str, str],
        bootstrap_state: dict[str, str | bool],
        drift_state: dict[str, str | bool],
    ) -> dict[str, str | bool]:
        if bootstrap_state.get("trained"):
            return {
                "trained": False,
                "reason": "already_bootstrapped_this_run",
            }

        if not drift_state.get("retrain_required"):
            return {
                "trained": False,
                "reason": str(drift_state.get("retrain_reason", "drift_thresholds_not_met")),
            }

        _run_command(
            [
                PYTHON_BIN,
                "scripts/train.py",
                "--data-package-dir",
                current_package["package_dir"],
                "--model",
                TRAIN_MODEL,
                "--experiment",
                TRAIN_EXPERIMENT,
                "--registered-model-name",
                REGISTERED_MODEL_NAME,
                "--tracking-uri",
                MLFLOW_TRACKING_URI,
            ]
        )
        _run_command(
            [
                PYTHON_BIN,
                "scripts/promote_data_package.py",
                "--source-dir",
                current_package["package_dir"],
                "--target-dir",
                str(BASELINE_PACKAGE_DIR),
                "--reason",
                "drift_retraining",
            ]
        )
        return {
            "trained": True,
            "reason": str(drift_state.get("retrain_reason", "drift_retraining")),
        }

    @task
    def run_batch_inference(
        current_package: dict[str, str],
        retrain_state: dict[str, str | bool],
    ) -> dict[str, str]:
        INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        predictions_path = INFERENCE_OUTPUT_DIR / "predictions.csv"
        _run_command(
            [
                PYTHON_BIN,
                "scripts/infer.py",
                "--model-uri",
                f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
                "--input",
                current_package["modeling_dataset"],
                "--output",
                str(predictions_path),
                "--target",
                "Nmero Extranjeros",
            ]
        )
        return {
            "predictions_path": str(predictions_path),
            "model_uri": f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
            "retrain_reason": str(retrain_state.get("reason", "n/a")),
        }

    @task
    def wait_for_api_ready() -> str:
        health_url = f"{API_BASE_URL.rstrip('/')}/health/ready"
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=30) as response:
            payload = response.read().decode("utf-8")
            if response.status >= 400:
                raise RuntimeError(f"API healthcheck failed with status {response.status}")
        print(payload)
        return health_url

    @task
    def run_api_smoke_test(current_package: dict[str, str], _: str) -> dict[str, str]:
        API_SMOKE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                PYTHON_BIN,
                "scripts/api_smoke_test.py",
                "--api-base-url",
                API_BASE_URL,
                "--input",
                current_package["modeling_dataset"],
                "--target",
                "Nmero Extranjeros",
                "--registered-model-name",
                REGISTERED_MODEL_NAME,
                "--model-alias",
                MODEL_ALIAS,
                "--output",
                str(API_SMOKE_OUTPUT),
            ]
        )
        return {
            "api_smoke_output": str(API_SMOKE_OUTPUT),
            "api_base_url": API_BASE_URL,
        }

    packaged = package_current_data()
    pipeline_state = inspect_pipeline_state()
    bootstrap_state = bootstrap_train_if_needed(packaged, pipeline_state)
    drift_state = run_drift_monitoring(packaged, pipeline_state)
    retrain_state = retrain_if_needed(packaged, bootstrap_state, drift_state)
    inference_state = run_batch_inference(packaged, retrain_state)
    api_health = wait_for_api_ready()
    api_smoke_state = run_api_smoke_test(packaged, api_health)
    inference_state >> api_smoke_state


colombia_tourism_end_to_end_mlops()
