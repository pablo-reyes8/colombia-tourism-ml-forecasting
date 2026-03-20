"""Dataset drift analysis utilities for monitoring and retraining workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _elevate_severity(current: str, candidate: str) -> str:
    order = {"none": 0, "warning": 1, "critical": 2}
    return candidate if order[candidate] > order[current] else current


@dataclass
class DriftFeatureResult:
    feature: str
    reference_count: int
    current_count: int
    reference_missing_rate: float
    current_missing_rate: float
    psi: float | None
    ks_pvalue: float | None
    mean_shift_std: float | None
    std_ratio: float | None
    severity: str
    drift_detected: bool
    reasons: list[str]

    def as_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = json.dumps(self.reasons, ensure_ascii=False)
        return payload


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_bins(reference: pd.Series, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) <= 2:
        minimum = float(reference.min())
        maximum = float(reference.max())
        if minimum == maximum:
            maximum = minimum + 1e-6
        edges = np.linspace(minimum, maximum, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float | None:
    reference = _coerce_numeric(reference).dropna()
    current = _coerce_numeric(current).dropna()
    if reference.empty or current.empty:
        return None

    edges = _build_bins(reference, bins)
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_share = np.clip(current_counts / current_counts.sum(), epsilon, None)
    psi = np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    return float(psi)


def mean_shift_in_std_units(reference: pd.Series, current: pd.Series) -> float | None:
    reference = _coerce_numeric(reference).dropna()
    current = _coerce_numeric(current).dropna()
    if reference.empty or current.empty:
        return None
    reference_std = float(reference.std(ddof=0))
    if reference_std == 0:
        return None
    return float((current.mean() - reference.mean()) / reference_std)


def std_ratio(reference: pd.Series, current: pd.Series) -> float | None:
    reference = _coerce_numeric(reference).dropna()
    current = _coerce_numeric(current).dropna()
    if reference.empty or current.empty:
        return None
    reference_std = float(reference.std(ddof=0))
    current_std = float(current.std(ddof=0))
    if reference_std == 0:
        return None
    return float(current_std / reference_std)


def analyze_numeric_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature: str,
    psi_warning: float = 0.1,
    psi_critical: float = 0.2,
    ks_alpha: float = 0.05,
    missing_rate_delta_threshold: float = 0.05,
    mean_shift_threshold: float = 0.5,
    bins: int = 10,
) -> DriftFeatureResult:
    reference_numeric = _coerce_numeric(reference)
    current_numeric = _coerce_numeric(current)
    psi = population_stability_index(reference_numeric, current_numeric, bins=bins)
    mean_shift = mean_shift_in_std_units(reference_numeric, current_numeric)
    std_ratio_value = std_ratio(reference_numeric, current_numeric)

    reference_clean = reference_numeric.dropna()
    current_clean = current_numeric.dropna()
    ks_pvalue = None
    if not reference_clean.empty and not current_clean.empty:
        ks_pvalue = float(ks_2samp(reference_clean, current_clean, method="asymp").pvalue)

    reference_missing_rate = float(reference_numeric.isna().mean())
    current_missing_rate = float(current_numeric.isna().mean())
    missing_delta = abs(current_missing_rate - reference_missing_rate)

    reasons: list[str] = []
    severity = "none"
    if psi is not None and psi >= psi_critical:
        severity = "critical"
        reasons.append(f"psi>={psi_critical}")
    elif psi is not None and psi >= psi_warning:
        severity = "warning"
        reasons.append(f"psi>={psi_warning}")

    if ks_pvalue is not None and ks_pvalue < ks_alpha:
        severity = _elevate_severity(severity, "warning")
        reasons.append(f"ks_pvalue<{ks_alpha}")

    if mean_shift is not None and abs(mean_shift) >= mean_shift_threshold:
        severity = _elevate_severity(
            severity,
            "critical" if abs(mean_shift) >= mean_shift_threshold * 2 else "warning",
        )
        reasons.append(f"|mean_shift_std|>={mean_shift_threshold}")

    if missing_delta >= missing_rate_delta_threshold:
        severity = _elevate_severity(severity, "warning")
        reasons.append(f"missing_delta>={missing_rate_delta_threshold}")

    return DriftFeatureResult(
        feature=feature,
        reference_count=int(reference_clean.shape[0]),
        current_count=int(current_clean.shape[0]),
        reference_missing_rate=reference_missing_rate,
        current_missing_rate=current_missing_rate,
        psi=psi,
        ks_pvalue=ks_pvalue,
        mean_shift_std=mean_shift,
        std_ratio=std_ratio_value,
        severity=severity,
        drift_detected=severity in {"warning", "critical"},
        reasons=reasons,
    )


def analyze_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    feature_names: list[str],
    psi_warning: float = 0.1,
    psi_critical: float = 0.2,
    ks_alpha: float = 0.05,
    missing_rate_delta_threshold: float = 0.05,
    mean_shift_threshold: float = 0.5,
    bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results: list[DriftFeatureResult] = []
    for feature in feature_names:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        results.append(
            analyze_numeric_feature_drift(
                reference_df[feature],
                current_df[feature],
                feature=feature,
                psi_warning=psi_warning,
                psi_critical=psi_critical,
                ks_alpha=ks_alpha,
                missing_rate_delta_threshold=missing_rate_delta_threshold,
                mean_shift_threshold=mean_shift_threshold,
                bins=bins,
            )
        )

    frame = pd.DataFrame([result.as_row() for result in results])
    if frame.empty:
        summary = {
            "generated_at": _utc_now(),
            "feature_count": 0,
            "drifted_feature_count": 0,
            "critical_feature_count": 0,
            "mean_psi": None,
            "max_psi": None,
            "dataset_drift_detected": False,
        }
        return frame, summary

    drifted_feature_count = int(frame["drift_detected"].sum())
    critical_feature_count = int((frame["severity"] == "critical").sum())
    psi_values = frame["psi"].dropna()
    summary = {
        "generated_at": _utc_now(),
        "feature_count": int(len(frame)),
        "drifted_feature_count": drifted_feature_count,
        "critical_feature_count": critical_feature_count,
        "drift_share": float(drifted_feature_count / len(frame)),
        "mean_psi": None if psi_values.empty else float(psi_values.mean()),
        "max_psi": None if psi_values.empty else float(psi_values.max()),
        "dataset_drift_detected": bool(drifted_feature_count > 0),
    }
    severity_order = {"critical": 2, "warning": 1, "none": 0}
    frame["_severity_rank"] = frame["severity"].map(severity_order).fillna(0)
    frame = frame.sort_values(["_severity_rank", "psi"], ascending=[False, False]).drop(
        columns=["_severity_rank"]
    )
    return frame, summary


def should_retrain(
    drift_summary: dict[str, Any],
    *,
    min_drifted_features: int = 3,
    critical_feature_count: int = 1,
    mean_psi_threshold: float = 0.15,
) -> tuple[bool, str]:
    critical_count = int(drift_summary.get("critical_feature_count") or 0)
    drifted_count = int(drift_summary.get("drifted_feature_count") or 0)
    mean_psi = drift_summary.get("mean_psi")

    if critical_count >= critical_feature_count:
        return True, f"critical_feature_count>={critical_feature_count}"
    if drifted_count >= min_drifted_features:
        return True, f"drifted_feature_count>={min_drifted_features}"
    if mean_psi is not None and float(mean_psi) >= mean_psi_threshold:
        return True, f"mean_psi>={mean_psi_threshold}"
    return False, "thresholds_not_met"


def write_drift_report(
    drift_frame: pd.DataFrame,
    drift_summary: dict[str, Any],
    output_dir: str | Path,
    *,
    retrain_decision: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "drift_feature_report.csv"
    summary_path = output_dir / "drift_summary.json"
    markdown_path = output_dir / "drift_report.md"
    decision_path = output_dir / "retrain_decision.json"

    drift_frame.to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(drift_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Drift Report",
        "",
        f"- Generated at: {drift_summary.get('generated_at')}",
        f"- Features analyzed: {drift_summary.get('feature_count')}",
        f"- Drifted features: {drift_summary.get('drifted_feature_count')}",
        f"- Critical features: {drift_summary.get('critical_feature_count')}",
        f"- Mean PSI: {drift_summary.get('mean_psi')}",
        f"- Max PSI: {drift_summary.get('max_psi')}",
        f"- Dataset drift detected: {drift_summary.get('dataset_drift_detected')}",
        "",
        "## Top Drifted Features",
        "",
    ]
    if drift_frame.empty:
        lines.append("- No numeric features were available for drift analysis.")
    else:
        for _, row in drift_frame.head(10).iterrows():
            lines.append(
                f"- `{row['feature']}` | severity={row['severity']} | psi={row['psi']} | "
                f"ks_pvalue={row['ks_pvalue']} | reasons={row['reasons']}"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision_payload = retrain_decision or {}
    decision_path.write_text(
        json.dumps(decision_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "feature_report": csv_path,
        "summary": summary_path,
        "markdown": markdown_path,
        "decision": decision_path,
    }
