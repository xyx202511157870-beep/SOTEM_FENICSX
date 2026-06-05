"""Validation report utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import json
import numpy as np

from .metrics import component_diagnostics, component_group_summary, summarize_errors


@dataclass(frozen=True)
class ValidationCase:
    """A numerical result and reference response to compare."""

    result_path: str | Path
    reference: np.ndarray
    component_names: list[str]
    positive_times_only: bool = False
    skip_positive_times: int = 0
    component_indices: list[int] | None = None
    include_samples: bool = False
    metadata: dict | None = None
    numerical: np.ndarray | None = None


def write_validation_report(case: ValidationCase, report_path: str | Path) -> dict:
    """Write a JSON validation report and return it."""

    with h5py.File(case.result_path, "r") as h5:
        times = h5["times"][:]
        numerical = h5["data"][:] if case.numerical is None else np.asarray(case.numerical, dtype=float)
    reference = np.asarray(case.reference, dtype=float)
    if case.component_indices is not None:
        numerical = numerical[:, case.component_indices]
    if case.positive_times_only:
        mask = times > 0.0
        if case.skip_positive_times:
            selected = np.flatnonzero(mask)
            mask[selected[: case.skip_positive_times]] = False
        times = times[mask]
        numerical = numerical[mask]
    summary = summarize_errors(numerical, reference, case.component_names)
    diagnostics = component_diagnostics(numerical, reference, case.component_names)
    relative_linf_max = max(values["relative_linf"] for values in summary.values())
    report = {
        "result_path": str(case.result_path),
        "n_times": int(times.size),
        "n_components": int(numerical.shape[1]),
        "relative_linf_max": float(relative_linf_max),
        "components": summary,
        "component_groups": component_group_summary(summary),
        "diagnostics": diagnostics,
        "metadata": case.metadata or {},
    }
    if case.include_samples:
        report["samples"] = _component_samples(times, numerical, reference, case.component_names)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _component_samples(
    times: np.ndarray,
    numerical: np.ndarray,
    reference: np.ndarray,
    component_names: list[str],
) -> dict[str, list[dict[str, float]]]:
    samples: dict[str, list[dict[str, float]]] = {}
    for column, name in enumerate(component_names):
        rows = []
        for time, num, ref in zip(times, numerical[:, column], reference[:, column]):
            rows.append(
                {
                    "time": float(time),
                    "numerical": float(num),
                    "reference": float(ref),
                    "difference": float(num - ref),
                    "ratio_numerical_over_reference": _safe_ratio(float(num), float(ref)),
                }
            )
        samples[str(name)] = rows
    return samples


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)
