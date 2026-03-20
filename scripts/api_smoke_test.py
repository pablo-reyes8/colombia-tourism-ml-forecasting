"""Run an HTTP smoke test against the FastAPI prediction service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run a smoke prediction request against the API.")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--input", required=True, help="CSV used to extract one sample record")
    parser.add_argument("--target", default=None, help="Optional target column to drop from sample payload")
    parser.add_argument("--registered-model-name", default="colombia-tourism-forecasting")
    parser.add_argument("--model-alias", default="champion")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", default=None, help="Optional path to save API response JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input CSV is empty")

    record = df.head(1).copy()
    if args.target and args.target in record.columns:
        record = record.drop(columns=[args.target])

    payload = {
        "record": record.iloc[0].to_dict(),
        "model": {
            "registered_model_name": args.registered_model_name,
            "model_alias": args.model_alias,
        },
        "options": {
            "strict_features": False,
            "fill_missing_value": 0.0,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{args.api_base_url.rstrip('/')}/api/v1/predict/single",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=args.timeout) as response:
        body = response.read().decode("utf-8")
        status_code = response.status

    response_payload = json.loads(body)
    result = {
        "status_code": status_code,
        "prediction": response_payload.get("prediction"),
        "model": response_payload.get("model"),
        "missing_features": response_payload.get("missing_features"),
        "extra_features": response_payload.get("extra_features"),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
