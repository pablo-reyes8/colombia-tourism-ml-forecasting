"""Promote a packaged dataset directory to the production baseline location."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from colombia_tourism.mlops import load_dataset_package_manifest


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def main():
    parser = argparse.ArgumentParser(
        description="Copy a packaged dataset directory into the production baseline path.",
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--reason", default="manual_promotion")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source package not found: {source_dir}")

    manifest = load_dataset_package_manifest(source_dir)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)

    promotion_payload = {
        "promoted_at": utc_now(),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "reason": args.reason,
        "dataset_name": manifest.get("dataset_name"),
        "processed_fingerprint": manifest.get("processed_fingerprint"),
    }
    metadata_dir = target_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "promotion_metadata.json").write_text(
        json.dumps(promotion_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(promotion_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
