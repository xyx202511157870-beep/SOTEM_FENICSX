"""Polarization-effect artifact writer."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from atem3d.sotem_metrics import compare_signed_response
from atem3d.validation_3comp import _plot_comparison, _plot_errors


def write_polarization_effect_artifacts(
    noip_dir: str | Path,
    ip_dir: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.10,
) -> dict:
    """Write IP-minus-noIP response, error, and plot artifacts."""

    noip = Path(noip_dir)
    ip = Path(ip_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pred_times, noip_pred, components = _read_response_csv(noip / "predictions.csv")
    ip_pred_times, ip_pred, ip_components = _read_response_csv(ip / "predictions.csv")
    ref_times, noip_ref, ref_components = _read_response_csv(noip / "reference_empymod_or_1d.csv")
    ip_ref_times, ip_ref, ip_ref_components = _read_response_csv(ip / "reference_empymod_or_1d.csv")
    _require_same_axis(pred_times, ip_pred_times, "prediction")
    _require_same_axis(pred_times, ref_times, "no-IP reference")
    _require_same_axis(pred_times, ip_ref_times, "IP reference")
    if components != ip_components or components != ref_components or components != ip_ref_components:
        raise ValueError("no-IP/IP prediction/reference component columns must match")

    effect_pred = ip_pred - noip_pred
    effect_ref = ip_ref - noip_ref
    comparison = compare_signed_response(
        pred_times,
        effect_pred,
        effect_ref,
        components,
        threshold=threshold,
    )
    rows = comparison["rows"]
    summary = {
        **comparison["summary"],
        "artifact_type": "polarization_effect",
        "definition": "ip_minus_noip",
        "component_names": components,
        "time_min": float(np.min(pred_times)),
        "time_max": float(np.max(pred_times)),
        "threshold": float(threshold),
        "floor_by_component": comparison["floor_by_component"],
        "max_robust_error_by_component": comparison["max_robust_error_by_component"],
        "zero_crossings": _json_safe_zero_crossings(comparison["zero_crossings"]),
    }
    _write_response_csv(output / "polarization_effect_predictions.csv", pred_times, effect_pred, components)
    _write_response_csv(output / "polarization_effect_reference.csv", pred_times, effect_ref, components)
    _write_error_csv(output / "polarization_effect_errors.csv", rows)
    (output / "polarization_effect_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _plot_comparison(
        output / "polarization_effect_comparison.png",
        pred_times,
        effect_pred,
        effect_ref,
        components,
    )
    _plot_errors(
        output / "polarization_effect_error_curves.png",
        rows,
        components,
        threshold=threshold,
    )
    return summary


def _json_safe_zero_crossings(zero_crossings: dict) -> dict:
    """Serialize Python ``inf`` count mismatches as ``"count_mismatch"``."""

    safe = {}
    for component, item in zero_crossings.items():
        safe[component] = {
            "prediction": list(item["prediction"]),
            "reference": list(item["reference"]),
            "count_match": bool(item["count_match"]),
            "max_relative_time_error": (
                float(item["max_relative_time_error"])
                if item["count_match"]
                else "count_mismatch"
            ),
        }
    return safe


def _read_response_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_obs" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a time_obs column")
        components = [name for name in reader.fieldnames if name != "time_obs"]
        times = []
        rows = []
        for row in reader:
            times.append(float(row["time_obs"]))
            rows.append([float(row[name]) for name in components])
    if not times:
        raise ValueError(f"{path} contains no response rows")
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float), components


def _require_same_axis(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=0.0):
        raise ValueError(f"{label} time axis does not match no-IP predictions")


def _write_response_csv(path: Path, times: np.ndarray, values: np.ndarray, components: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_obs", *components])
        for time, row in zip(times, values):
            writer.writerow([float(time), *[float(value) for value in row]])


def _write_error_csv(path: Path, rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows.dtype.names)
        for row in rows:
            writer.writerow([row[name] for name in rows.dtype.names])
