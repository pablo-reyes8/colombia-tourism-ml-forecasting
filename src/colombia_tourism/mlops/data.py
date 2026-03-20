"""Data packaging and contract utilities for MLOps workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from colombia_tourism.config import PROJECT_ROOT
from colombia_tourism.data import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    ENTITY_COLUMN,
    TIME_COLUMN,
    available_feature_groups,
    infer_numeric_features,
    load_base_final,
    parse_panel_months,
)
from colombia_tourism.features.engineering import build_modeling_features

DEFAULT_DATA_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "data" / "base_final_package"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_feature_list(path: str | Path | None) -> list[str] | None:
    if not path:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON feature lists must contain a list of feature names")
        return [str(value) for value in payload]
    return [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    normalized = df.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype(str)
    values = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


@dataclass(frozen=True)
class FeatureBuildConfig:
    engineer_features: bool = False
    include_density_features: bool = False
    include_target_history: bool = True
    lag_columns: tuple[str, ...] = ("Pib Ponderado", "Eventos", "Temperatura")
    rolling_columns: tuple[str, ...] = ("Pib Ponderado", "Eventos")
    lags: tuple[int, ...] = (1, 3, 12)
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    rolling_stats: tuple[str, ...] = ("mean", "std")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None):
        if payload is None:
            return cls()
        normalized = dict(payload)
        for key in ("lag_columns", "rolling_columns", "lags", "rolling_windows", "rolling_stats"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


@dataclass
class DatasetBundle:
    raw_df: pd.DataFrame
    modeling_df: pd.DataFrame
    feature_names: list[str]
    target: str
    source_path: str | None = None
    feature_build_config: FeatureBuildConfig = field(default_factory=FeatureBuildConfig)


def resolve_feature_names(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    features_path: str | Path | None = None,
    engineer_features: bool = False,
) -> list[str]:
    explicit = feature_names or load_feature_list(features_path)
    if explicit:
        resolved = [feature for feature in explicit if feature in df.columns]
        if not resolved:
            raise ValueError("Resolved feature list is empty after intersecting with dataframe columns")
        return resolved

    if engineer_features:
        return infer_numeric_features(
            df,
            exclude=(ENTITY_COLUMN, TIME_COLUMN, "fecha", target),
        )

    resolved = [feature for feature in DEFAULT_FEATURES if feature in df.columns]
    if not resolved:
        raise ValueError("No default features were found in the dataframe")
    return resolved


def apply_feature_build_config(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_build_config: FeatureBuildConfig | None = None,
) -> pd.DataFrame:
    config = feature_build_config or FeatureBuildConfig()
    if not config.engineer_features:
        return df.copy()
    return build_modeling_features(
        df,
        target_col=target,
        lag_columns=config.lag_columns,
        rolling_columns=config.rolling_columns,
        lags=config.lags,
        rolling_windows=config.rolling_windows,
        rolling_stats=config.rolling_stats,
        include_target_history=config.include_target_history,
        include_density_features=config.include_density_features,
    )


def prepare_dataset_bundle(
    data: str | Path | pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    features_path: str | Path | None = None,
    feature_build_config: FeatureBuildConfig | None = None,
) -> DatasetBundle:
    if isinstance(data, pd.DataFrame):
        raw_df = data.copy()
        source_path = None
    else:
        source_path = str(data)
        raw_df = load_base_final(data)

    config = feature_build_config or FeatureBuildConfig()
    modeling_df = apply_feature_build_config(
        raw_df,
        target=target,
        feature_build_config=config,
    )
    resolved_features = resolve_feature_names(
        modeling_df,
        target=target,
        feature_names=feature_names,
        features_path=features_path,
        engineer_features=config.engineer_features,
    )

    if target not in modeling_df.columns:
        raise KeyError(f"Target column '{target}' not found in modeling dataframe")

    return DatasetBundle(
        raw_df=raw_df,
        modeling_df=modeling_df,
        feature_names=resolved_features,
        target=target,
        source_path=source_path,
        feature_build_config=config,
    )


def _numeric_column_profile(series: pd.Series) -> dict[str, Any]:
    cleaned = pd.to_numeric(series, errors="coerce")
    return {
        "mean": None if cleaned.dropna().empty else float(cleaned.mean()),
        "std": None if cleaned.dropna().empty else float(cleaned.std(ddof=0)),
        "min": None if cleaned.dropna().empty else float(cleaned.min()),
        "p25": None if cleaned.dropna().empty else float(cleaned.quantile(0.25)),
        "median": None if cleaned.dropna().empty else float(cleaned.median()),
        "p75": None if cleaned.dropna().empty else float(cleaned.quantile(0.75)),
        "max": None if cleaned.dropna().empty else float(cleaned.max()),
    }


def build_schema_profile(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for column in df.columns:
        series = df[column]
        profile = {
            "name": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "sample_values": [str(value) for value in series.dropna().head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series):
            profile["numeric_profile"] = _numeric_column_profile(series)
        columns.append(profile)
    return {"columns": columns}


def build_panel_profile(df: pd.DataFrame, *, target: str = DEFAULT_TARGET) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fingerprint": dataframe_fingerprint(df),
    }

    if ENTITY_COLUMN in df.columns:
        profile["entity_count"] = int(df[ENTITY_COLUMN].nunique(dropna=True))

    if TIME_COLUMN in df.columns:
        parsed = parse_panel_months(df[TIME_COLUMN], strict=False)
        if parsed.notna().any():
            profile["time_min"] = parsed.min().strftime("%Y-%m-%d")
            profile["time_max"] = parsed.max().strftime("%Y-%m-%d")
            profile["time_periods"] = int(parsed.dt.to_period("M").nunique())

    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        profile["target_profile"] = _numeric_column_profile(df[target])

    return profile


def build_feature_manifest(bundle: DatasetBundle) -> dict[str, Any]:
    modeling_df = bundle.modeling_df
    raw_columns = set(bundle.raw_df.columns)
    grouped = available_feature_groups(modeling_df.columns)
    feature_records = []
    for feature in bundle.feature_names:
        series = modeling_df[feature]
        feature_records.append(
            {
                "name": feature,
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "engineered": feature not in raw_columns,
                "groups": [
                    group_name
                    for group_name, group_features in grouped.items()
                    if feature in group_features
                ],
            }
        )
    return {
        "target": bundle.target,
        "feature_count": len(bundle.feature_names),
        "feature_groups": grouped,
        "features": feature_records,
    }


def build_ingestion_summary(bundle: DatasetBundle) -> str:
    config = bundle.feature_build_config
    processed = bundle.modeling_df
    missing_counts = (
        processed[bundle.feature_names + [bundle.target]]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )
    top_missing = missing_counts.head(10)
    lines = [
        "# Data Package Summary",
        "",
        f"- Generated at: {_utc_now()}",
        f"- Source path: `{bundle.source_path or 'in-memory dataframe'}`",
        f"- Raw shape: `{bundle.raw_df.shape[0]} x {bundle.raw_df.shape[1]}`",
        f"- Modeling shape: `{processed.shape[0]} x {processed.shape[1]}`",
        f"- Target: `{bundle.target}`",
        f"- Feature count: `{len(bundle.feature_names)}`",
        f"- Feature engineering enabled: `{config.engineer_features}`",
        f"- Raw fingerprint: `{dataframe_fingerprint(bundle.raw_df)}`",
        f"- Modeling fingerprint: `{dataframe_fingerprint(processed)}`",
        "",
        "## Feature Build Config",
        "",
        "```json",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Missing Columns",
        "",
    ]
    for column, value in top_missing.items():
        lines.append(f"- `{column}`: {int(value)} missing values")
    return "\n".join(lines) + "\n"


def write_dataset_package(
    bundle: DatasetBundle,
    output_dir: str | Path = DEFAULT_DATA_ARTIFACT_DIR,
    *,
    dataset_name: str = "base_final",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    metadata_dir = output_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{dataset_name}_snapshot.csv"
    processed_path = processed_dir / "modeling_dataset.csv"
    feature_path = metadata_dir / "feature_list.txt"
    schema_path = metadata_dir / "schema.json"
    profile_path = metadata_dir / "profile.json"
    feature_manifest_path = metadata_dir / "feature_manifest.json"
    summary_path = metadata_dir / "ingestion_summary.md"
    manifest_path = metadata_dir / "manifest.json"

    bundle.raw_df.to_csv(raw_path, index=False)
    bundle.modeling_df.to_csv(processed_path, index=False)
    feature_path.write_text("\n".join(bundle.feature_names), encoding="utf-8")
    schema_payload = {
        "raw": build_schema_profile(bundle.raw_df),
        "processed": build_schema_profile(bundle.modeling_df),
    }
    schema_path.write_text(
        json.dumps(schema_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    profile_payload = {
        "raw": build_panel_profile(bundle.raw_df, target=bundle.target),
        "processed": build_panel_profile(bundle.modeling_df, target=bundle.target),
    }
    profile_path.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_manifest_path.write_text(
        json.dumps(build_feature_manifest(bundle), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(build_ingestion_summary(bundle), encoding="utf-8")

    manifest = {
        "data_contract_version": "1.0",
        "dataset_name": dataset_name,
        "created_at": _utc_now(),
        "source_path": bundle.source_path,
        "target": bundle.target,
        "feature_names": bundle.feature_names,
        "feature_count": len(bundle.feature_names),
        "feature_build_config": asdict(bundle.feature_build_config),
        "raw_rows": int(bundle.raw_df.shape[0]),
        "raw_columns": int(bundle.raw_df.shape[1]),
        "processed_rows": int(bundle.modeling_df.shape[0]),
        "processed_columns": int(bundle.modeling_df.shape[1]),
        "raw_fingerprint": dataframe_fingerprint(bundle.raw_df),
        "processed_fingerprint": dataframe_fingerprint(bundle.modeling_df),
        "artifacts": {
            "raw_snapshot": str(raw_path.relative_to(output_dir)),
            "modeling_dataset": str(processed_path.relative_to(output_dir)),
            "feature_list": str(feature_path.relative_to(output_dir)),
            "schema": str(schema_path.relative_to(output_dir)),
            "profile": str(profile_path.relative_to(output_dir)),
            "feature_manifest": str(feature_manifest_path.relative_to(output_dir)),
            "ingestion_summary": str(summary_path.relative_to(output_dir)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "package_dir": output_dir,
        "manifest": manifest_path,
        "raw_snapshot": raw_path,
        "modeling_dataset": processed_path,
        "feature_list": feature_path,
        "schema": schema_path,
        "profile": profile_path,
        "feature_manifest": feature_manifest_path,
        "ingestion_summary": summary_path,
    }


def load_dataset_package_manifest(package_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(package_dir) / "metadata" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = Path(package_dir) / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_packaged_modeling_dataset(package_dir: str | Path) -> pd.DataFrame:
    manifest = load_dataset_package_manifest(package_dir)
    package_dir = Path(package_dir)
    dataset_path = package_dir / manifest["artifacts"]["modeling_dataset"]
    return pd.read_csv(dataset_path)


def load_packaged_feature_names(package_dir: str | Path) -> list[str]:
    manifest = load_dataset_package_manifest(package_dir)
    feature_names = manifest.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        return [str(value) for value in feature_names]
    feature_path = Path(package_dir) / manifest["artifacts"]["feature_list"]
    return load_feature_list(feature_path) or []

