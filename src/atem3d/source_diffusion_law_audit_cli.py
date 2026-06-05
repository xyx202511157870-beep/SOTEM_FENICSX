"""Cross-report audit for source-diffusion kernel normalization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceDiffusionTrace:
    """Coefficient trace extracted from one source-neighborhood audit."""

    report: Path
    source_diffusion_time: float
    times: np.ndarray
    coefficients: np.ndarray
    kernel_fits: tuple[dict[str, Any], ...]
    best_kernel_fit: dict[str, Any]
    direction_metrics: dict[str, float]


_DIRECTION_METRIC_KEYS = (
    "all_window_residual_relative_l2",
    "all_window_residual_projection_fraction",
    "per_time_residual_relative_l2",
    "per_time_residual_projection_fraction",
)
_EFFECTIVE_DECAY_AMPLITUDE_FLOOR_FRACTION = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether source-neighborhood coefficient traces share a "
            "source-diffusion law A exp(-(t-t0)/(m*mu*sigma*L^2))."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--candidate", default="active_source")
    parser.add_argument("--multipliers", nargs="+", type=float)
    parser.add_argument(
        "--basis-kind",
        choices=("continuous", "be_decay"),
        default="continuous",
        help=(
            "Kernel basis used for global fits. 'continuous' uses exp(-(t-t0)/tau); "
            "'be_decay' uses the first-gate-normalized backward-Euler Debye decay "
            "on the sampled uniform time grid."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    traces = [
        _load_trace(path, candidate=str(args.candidate)) for path in args.reports
    ]
    multipliers = _multipliers(args.multipliers, traces)
    global_fits = [
        _global_law_fit(multiplier, traces, basis_kind=str(args.basis_kind))
        for multiplier in multipliers
    ]
    best_global = min(
        global_fits,
        key=lambda item: item["combined_coefficient_relative_l2"],
    )
    per_report = [
        _per_report_summary(
            trace,
            multipliers=multipliers,
            basis_kind=str(args.basis_kind),
        )
        for trace in traces
    ]
    report = {
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "basis_kind": str(args.basis_kind),
        "report_count": len(traces),
        "reports": [str(trace.report) for trace in traces],
        "per_report": per_report,
        "normalization_summary": _normalization_summary(per_report),
        "direction_constraint_summary": _direction_constraint_summary(per_report),
        "effective_decay_summary": _effective_decay_global_summary(per_report),
        "diagnostic_source_history_suggestion": (
            _diagnostic_source_history_suggestion(
                traces,
                best_global,
                basis_kind=str(args.basis_kind),
                candidate=str(args.candidate),
            )
        ),
        "global_law_fits": global_fits,
        "best_global_law": best_global,
        "limitations": (
            "Uses per_time_coefficients from source-neighborhood audits. "
            "Receiver-data L2 cannot be recomputed unless the original "
            "reference and recomputed sample arrays are available."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "best global source-diffusion law: "
        f"m={best_global['multiplier']:.6g}, "
        f"A/tau0={best_global['normalized_amplitude']:.6g}, "
        f"coefficient_l2={best_global['combined_coefficient_relative_l2']:.6g}"
    )
    print(f"wrote {args.output}")
    return 0


def _load_trace(path: Path, *, candidate: str) -> SourceDiffusionTrace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit JSON root must be an object")
    source_diffusion_time = float(payload["source_diffusion_time_s"])
    if source_diffusion_time <= 0.0:
        raise ValueError("source_diffusion_time_s must be positive")
    candidates = payload.get("candidate_static_fits")
    if not isinstance(candidates, dict) or candidate not in candidates:
        raise ValueError(f"candidate {candidate!r} is not present in {path}")
    fit = candidates[candidate]
    coefficients = np.asarray(fit["per_time_coefficients"], dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("per_time_coefficients must be a nonempty vector")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("per_time_coefficients must be finite")
    time_count = int(payload.get("time_count", coefficients.size))
    if time_count != coefficients.size:
        raise ValueError("time_count must match per_time_coefficients length")
    times = np.linspace(
        float(payload["time_min"]),
        float(payload["time_max"]),
        coefficients.size,
    )
    kernel_fits = tuple(dict(item) for item in fit["coefficient_kernel_fits"])
    best_kernel_fit = min(
        kernel_fits,
        key=lambda item: float(item["coefficient_relative_l2"]),
    )
    return SourceDiffusionTrace(
        report=path,
        source_diffusion_time=source_diffusion_time,
        times=times,
        coefficients=coefficients,
        kernel_fits=kernel_fits,
        best_kernel_fit=best_kernel_fit,
        direction_metrics=_extract_direction_metrics(fit),
    )


def _extract_direction_metrics(fit: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in _DIRECTION_METRIC_KEYS:
        if key not in fit:
            continue
        value = float(fit[key])
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
        metrics[key] = value
    return metrics


def _multipliers(
    requested: list[float] | None,
    traces: list[SourceDiffusionTrace],
) -> list[float]:
    if requested is not None:
        values = [float(value) for value in requested]
    else:
        values = sorted(
            {
                float(fit["multiplier"])
                for trace in traces
                for fit in trace.kernel_fits
            }
        )
    if not values:
        raise ValueError("at least one multiplier is required")
    if any(value <= 0.0 for value in values):
        raise ValueError("multipliers must be positive")
    return values


def _per_report_summary(
    trace: SourceDiffusionTrace,
    *,
    multipliers: list[float],
    basis_kind: str,
) -> dict[str, Any]:
    tau0 = trace.source_diffusion_time
    first = float(trace.coefficients[0])
    fits = [
        _single_trace_law_fit(trace, multiplier, basis_kind=basis_kind)
        for multiplier in multipliers
    ]
    best = min(fits, key=lambda item: item["coefficient_relative_l2"])
    return {
        "report": str(trace.report),
        "basis_kind": str(basis_kind),
        "source_diffusion_time_s": tau0,
        "time_min": float(trace.times[0]),
        "time_max": float(trace.times[-1]),
        "time_count": int(trace.times.size),
        "best_multiplier": float(best["multiplier"]),
        "best_tau_s": float(best["tau"]),
        "best_amplitude": float(best["amplitude"]),
        "normalized_best_amplitude": float(best["normalized_amplitude"]),
        "best_coefficient_relative_l2": float(best["coefficient_relative_l2"]),
        "fit_candidates": fits,
        "first_coefficient": first,
        "normalized_first_coefficient": first / tau0,
        "direction_metrics": dict(trace.direction_metrics),
        "effective_decay": _effective_decay_summary(trace),
    }


def _normalization_summary(per_report: list[dict[str, Any]]) -> dict[str, Any]:
    best_amplitudes = np.asarray(
        [item["normalized_best_amplitude"] for item in per_report],
        dtype=float,
    )
    first_coefficients = np.asarray(
        [item["normalized_first_coefficient"] for item in per_report],
        dtype=float,
    )
    return {
        "normalized_best_amplitude_mean": float(np.mean(best_amplitudes)),
        "normalized_best_amplitude_std": float(np.std(best_amplitudes)),
        "normalized_best_amplitude_cv_abs": _coefficient_of_variation(best_amplitudes),
        "normalized_first_coefficient_mean": float(np.mean(first_coefficients)),
        "normalized_first_coefficient_std": float(np.std(first_coefficients)),
        "normalized_first_coefficient_cv_abs": _coefficient_of_variation(
            first_coefficients
        ),
        "best_multipliers": [
            float(item["best_multiplier"]) for item in per_report
        ],
    }


def _direction_constraint_summary(per_report: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        item["direction_metrics"]
        for item in per_report
        if item.get("direction_metrics")
    ]
    summary: dict[str, Any] = {
        "available_report_count": len(available),
        "missing_report_count": len(per_report) - len(available),
    }
    if not available:
        return summary

    for key in _DIRECTION_METRIC_KEYS:
        values = np.asarray(
            [metrics[key] for metrics in available if key in metrics],
            dtype=float,
        )
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def _effective_decay_summary(
    trace: SourceDiffusionTrace,
    *,
    amplitude_floor_fraction: float = _EFFECTIVE_DECAY_AMPLITUDE_FLOOR_FRACTION,
) -> dict[str, Any]:
    coefficients = np.asarray(trace.coefficients, dtype=float)
    times = np.asarray(trace.times, dtype=float)
    if coefficients.shape != times.shape:
        raise ValueError("effective decay requires coefficients and times to match")
    pair_count = max(0, coefficients.size - 1)
    summary: dict[str, Any] = {
        "amplitude_floor_fraction": float(amplitude_floor_fraction),
        "candidate_pair_count": int(pair_count),
        "valid_pair_count": 0,
        "valid_pair_fraction": 0.0,
    }
    if pair_count == 0:
        return summary
    if amplitude_floor_fraction < 0.0:
        raise ValueError("amplitude_floor_fraction must be nonnegative")

    diffs = np.diff(times)
    if np.any(diffs <= 0.0):
        raise ValueError("effective decay requires increasing sample times")
    peak = float(np.max(np.abs(coefficients)))
    floor = float(amplitude_floor_fraction) * peak
    be_multipliers = []
    continuous_multipliers = []
    weights = []
    for old, new, dt in zip(coefficients[:-1], coefficients[1:], diffs):
        old = float(old)
        new = float(new)
        if abs(old) < floor or old * new <= 0.0:
            continue
        ratio = new / old
        if not (0.0 < ratio < 1.0):
            continue
        dt = float(dt)
        be_tau = ratio * dt / (1.0 - ratio)
        continuous_tau = -dt / np.log(ratio)
        be_multipliers.append(be_tau / trace.source_diffusion_time)
        continuous_multipliers.append(continuous_tau / trace.source_diffusion_time)
        weights.append(old * old)

    if not be_multipliers:
        return summary
    be = np.asarray(be_multipliers, dtype=float)
    continuous = np.asarray(continuous_multipliers, dtype=float)
    weight = np.asarray(weights, dtype=float)
    summary.update(
        {
            "valid_pair_count": int(be.size),
            "valid_pair_fraction": float(be.size / pair_count),
            "be_multiplier_weighted_mean": _weighted_mean(be, weight),
            "be_multiplier_median": float(np.median(be)),
            "be_multiplier_min": float(np.min(be)),
            "be_multiplier_max": float(np.max(be)),
            "continuous_multiplier_weighted_mean": _weighted_mean(
                continuous,
                weight,
            ),
            "continuous_multiplier_median": float(np.median(continuous)),
            "continuous_multiplier_min": float(np.min(continuous)),
            "continuous_multiplier_max": float(np.max(continuous)),
        }
    )
    return summary


def _effective_decay_global_summary(per_report: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        item["effective_decay"]
        for item in per_report
        if item.get("effective_decay", {}).get("valid_pair_count", 0) > 0
    ]
    summary: dict[str, Any] = {
        "available_report_count": len(available),
        "missing_report_count": len(per_report) - len(available),
        "amplitude_floor_fraction": _EFFECTIVE_DECAY_AMPLITUDE_FLOOR_FRACTION,
    }
    if not available:
        return summary
    for key in (
        "valid_pair_fraction",
        "be_multiplier_weighted_mean",
        "be_multiplier_median",
        "continuous_multiplier_weighted_mean",
        "continuous_multiplier_median",
    ):
        values = np.asarray([item[key] for item in available], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def _diagnostic_source_history_suggestion(
    traces: list[SourceDiffusionTrace],
    best_global: dict[str, Any],
    *,
    basis_kind: str,
    candidate: str,
) -> dict[str, Any]:
    amplitude_times = np.asarray([trace.times[0] for trace in traces], dtype=float)
    amplitude_time_consistent = bool(
        amplitude_times.size > 0
        and np.allclose(amplitude_times, amplitude_times[0], rtol=0.0, atol=1.0e-15)
    )
    suggestion: dict[str, Any] = {
        "diagnostic_only": True,
        "candidate": str(candidate),
        "kind": "source_diffusion_kernel_source_moments",
        "normalized_amplitude": float(best_global["normalized_amplitude"]),
        "tau_multiplier": float(best_global["multiplier"]),
        "basis_kind": str(basis_kind),
        "source_moment_degrees": [0],
        "receiver_matrix": "auto",
        "amplitude_time_consistent": amplitude_time_consistent,
        "amplitude_times": [float(value) for value in amplitude_times],
        "limitations": (
            "Generated from source-neighborhood residual diagnostics; use as a "
            "diagnostic replay block only until the FV/H/J source-channel law "
            "is derived."
        ),
    }
    suggestion["amplitude_time"] = (
        float(amplitude_times[0]) if amplitude_time_consistent else None
    )
    return suggestion


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator == 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / denominator)


def _global_law_fit(
    multiplier: float,
    traces: list[SourceDiffusionTrace],
    *,
    basis_kind: str,
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    for trace in traces:
        basis = _normalized_basis(trace, multiplier, basis_kind=basis_kind)
        numerator += float(np.vdot(basis, trace.coefficients).real)
        denominator += float(np.vdot(basis, basis).real)
    normalized_amplitude = numerator / denominator if denominator > 0.0 else 0.0

    per_report = []
    residual_norm2 = 0.0
    coefficient_norm2 = 0.0
    for trace in traces:
        basis = _normalized_basis(trace, multiplier, basis_kind=basis_kind)
        predicted = normalized_amplitude * basis
        residual = predicted - trace.coefficients
        residual_norm2 += float(np.vdot(residual, residual).real)
        coefficient_norm2 += float(np.vdot(trace.coefficients, trace.coefficients).real)
        per_report.append(
            {
                "report": str(trace.report),
                "coefficient_relative_l2": _relative_l2(
                    predicted,
                    trace.coefficients,
                ),
            }
        )

    combined = (
        float(np.sqrt(residual_norm2 / coefficient_norm2))
        if coefficient_norm2 > 0.0
        else float(np.sqrt(residual_norm2))
    )
    return {
        "multiplier": float(multiplier),
        "normalized_amplitude": float(normalized_amplitude),
        "combined_coefficient_relative_l2": combined,
        "per_report": per_report,
    }


def _single_trace_law_fit(
    trace: SourceDiffusionTrace,
    multiplier: float,
    *,
    basis_kind: str,
) -> dict[str, Any]:
    basis = _normalized_basis(trace, multiplier, basis_kind=basis_kind)
    denominator = float(np.vdot(basis, basis).real)
    normalized_amplitude = (
        float(np.vdot(basis, trace.coefficients).real) / denominator
        if denominator > 0.0
        else 0.0
    )
    predicted = normalized_amplitude * basis
    return {
        "multiplier": float(multiplier),
        "tau": float(multiplier) * trace.source_diffusion_time,
        "amplitude": float(normalized_amplitude * trace.source_diffusion_time),
        "normalized_amplitude": float(normalized_amplitude),
        "coefficient_relative_l2": _relative_l2(predicted, trace.coefficients),
    }


def _normalized_basis(
    trace: SourceDiffusionTrace,
    multiplier: float,
    *,
    basis_kind: str,
) -> np.ndarray:
    tau = float(multiplier) * trace.source_diffusion_time
    kind = str(basis_kind).strip().lower()
    if kind == "continuous":
        return _continuous_normalized_basis(trace, tau)
    if kind == "be_decay":
        return _be_decay_normalized_basis(trace, tau)
    raise ValueError(f"unknown basis_kind {basis_kind!r}")


def _continuous_normalized_basis(
    trace: SourceDiffusionTrace,
    tau: float,
) -> np.ndarray:
    return trace.source_diffusion_time * np.exp(
        -(trace.times - trace.times[0]) / tau
    )


def _be_decay_normalized_basis(
    trace: SourceDiffusionTrace,
    tau: float,
) -> np.ndarray:
    times = np.asarray(trace.times, dtype=float)
    if times.size == 1:
        return np.asarray([trace.source_diffusion_time], dtype=float)
    diffs = np.diff(times)
    if np.any(diffs <= 0.0):
        raise ValueError("BE source-diffusion basis requires increasing times")
    dt = float(diffs[0])
    if not np.allclose(diffs, dt, rtol=1.0e-9, atol=1.0e-15):
        raise ValueError("BE source-diffusion basis requires a uniform time grid")
    offsets = times - times[0]
    steps = np.rint(offsets / dt).astype(int)
    if not np.allclose(steps * dt, offsets, rtol=1.0e-9, atol=1.0e-15):
        raise ValueError("BE source-diffusion basis times must align to the time step")
    alpha = float(tau) / (float(tau) + dt)
    return trace.source_diffusion_time * alpha**steps


def _relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(numerical - reference))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("inf")
    return float(np.std(values) / abs(mean))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
