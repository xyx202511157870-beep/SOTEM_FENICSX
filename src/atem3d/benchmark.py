"""Boundary convergence benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import json
import numpy as np

from .metrics import component_pass_report, summarize_errors


@dataclass(frozen=True)
class BoundaryBenchmarkCase:
    """One boundary-convergence case."""

    name: str
    config: dict[str, Any]


def run_boundary_benchmark(
    cases: list[BoundaryBenchmarkCase],
    runner: Callable[[dict[str, Any]], Any],
    component_names: list[str],
    tolerance: float,
    output_path: str | Path | None = None,
    reference_name: str | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
    absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    """Run cases and compare receiver data to a reference case."""

    if len(cases) < 2:
        raise ValueError("at least two cases are required")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if absolute_tolerance is not None and absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be nonnegative")
    if time_min is not None and time_max is not None and time_min > time_max:
        raise ValueError("time_min must be <= time_max")

    results = {
        case.name: _windowed_data(runner(case.config), time_min=time_min, time_max=time_max)
        for case in cases
    }
    reference = reference_name or cases[-1].name
    if reference not in results:
        raise ValueError(f"unknown reference case: {reference}")
    reference_data = results[reference]

    report_cases = []
    for case in cases:
        summary = summarize_errors(results[case.name], reference_data, component_names)
        components = component_pass_report(
            summary,
            relative_tolerance=float(tolerance),
            absolute_tolerance=absolute_tolerance,
        )
        worst_relative = max(values["relative_linf"] for values in summary.values())
        worst_absolute = max(values["absolute_linf"] for values in summary.values())
        report_cases.append(
            {
                "name": case.name,
                "n_times": int(results[case.name].shape[0]),
                "relative_linf_max": float(worst_relative),
                "absolute_linf_max": float(worst_absolute),
                "passed": all(values["passed"] for values in components.values()),
                "components": components,
            }
        )

    report = {
        "reference": reference,
        "tolerance": float(tolerance),
        "absolute_tolerance": (
            None if absolute_tolerance is None else float(absolute_tolerance)
        ),
        "time_window": {
            "min": None if time_min is None else float(time_min),
            "max": None if time_max is None else float(time_max),
        },
        "cases": report_cases,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _windowed_data(result: Any, time_min: float | None, time_max: float | None) -> np.ndarray:
    data = np.asarray(result.data, dtype=float)
    if time_min is None and time_max is None:
        return data
    if not hasattr(result, "times"):
        raise ValueError("time-windowed benchmarks require runner results with times")
    times = np.asarray(result.times, dtype=float)
    if times.shape[0] != data.shape[0]:
        raise ValueError("result.times length must match result.data rows")
    mask = np.ones(times.shape, dtype=bool)
    if time_min is not None:
        mask &= times >= float(time_min)
    if time_max is not None:
        mask &= times <= float(time_max)
    if not np.any(mask):
        raise ValueError("time window selected no samples")
    return data[mask]


def _json_ready(summary: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {metric: float(value) for metric, value in values.items()}
        for name, values in summary.items()
    }
