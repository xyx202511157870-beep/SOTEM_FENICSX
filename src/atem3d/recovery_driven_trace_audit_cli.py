"""Audit driven-response source-moment traces against a clean target trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recovery_spectrum driven-response source-moment projection "
            "traces with a source_history_matrix spatial_time_series target."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-matrix-report", type=Path)
    target_group.add_argument("--target-source-neighborhood-report", type=Path)
    parser.add_argument(
        "--target-candidate",
        default="active_source",
        help=(
            "candidate_static_fits key used with "
            "--target-source-neighborhood-report"
        ),
    )
    parser.add_argument("--compact-normalized-amplitude", nargs="+", type=float)
    parser.add_argument("--combine-report-weights", nargs="+", type=float)
    parser.add_argument("--fit-compact-report-weights", action="store_true")
    parser.add_argument("--time-atol", type=float, default=1.0e-12)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    (
        target_kind,
        target_report_path,
        target_times,
        target_matrix,
        normalization,
        factor,
    ) = _load_target(args)
    compact = (
        None
        if args.compact_normalized_amplitude is None
        else np.asarray(args.compact_normalized_amplitude, dtype=float)
    )
    combine_weights = (
        None
        if args.combine_report_weights is None
        else np.asarray(args.combine_report_weights, dtype=float)
    )
    if combine_weights is not None and combine_weights.shape != (len(args.reports),):
        raise ValueError(
            "combine-report-weights count must match the number of driven reports"
        )
    if args.fit_compact_report_weights and compact is None:
        raise ValueError(
            "fit-compact-report-weights requires compact-normalized-amplitude"
        )

    items = [
        _audit_report(
            report_path,
            target_times=target_times,
            target_matrix=target_matrix,
            normalization_factor=factor,
            normalization_kind=str(normalization["kind"]),
            compact_normalized_amplitude=compact,
            time_atol=float(args.time_atol),
        )
        for report_path in args.reports
    ]
    output = {
        "diagnostic_only": True,
        "target_kind": target_kind,
        "target_report": str(target_report_path),
        "normalization": normalization,
        "items": items,
    }
    if args.target_matrix_report is not None:
        output["target_matrix_report"] = str(args.target_matrix_report)
    if args.target_source_neighborhood_report is not None:
        output["target_source_neighborhood_report"] = str(
            args.target_source_neighborhood_report
        )
        output["target_candidate"] = str(args.target_candidate)
    if combine_weights is not None:
        output["combined"] = _audit_combined_reports(
            args.reports,
            report_weights=combine_weights,
            target_times=target_times,
            target_matrix=target_matrix,
            normalization_factor=factor,
            normalization_kind=str(normalization["kind"]),
            compact_normalized_amplitude=compact,
            time_atol=float(args.time_atol),
        )
    if args.fit_compact_report_weights:
        output["compact_report_weight_fit"] = _fit_compact_report_weights(
            args.reports,
            target_times=target_times,
            target_matrix=target_matrix,
            normalization_factor=factor,
            normalization_kind=str(normalization["kind"]),
            compact_normalized_amplitude=compact,
            time_atol=float(args.time_atol),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for item in items:
        compact_text = ""
        if "compact_relative_l2" in item:
            compact_text = f" compact_l2={item['compact_relative_l2']:.6g}"
        print(
            f"{item['report']}: fit_l2={item['fit_relative_l2']:.6g}"
            f"{compact_text}"
        )
    if "combined" in output:
        combined = output["combined"]
        compact_text = ""
        if "compact_relative_l2" in combined:
            compact_text = f" compact_l2={combined['compact_relative_l2']:.6g}"
        print(f"combined: fit_l2={combined['fit_relative_l2']:.6g}{compact_text}")
    if "compact_report_weight_fit" in output:
        report_fit = output["compact_report_weight_fit"]
        print(f"compact report-weight fit: l2={report_fit['fit_relative_l2']:.6g}")
    print(f"wrote {args.output}")
    return 0


def _audit_report(
    report_path: Path,
    *,
    target_times: np.ndarray,
    target_matrix: np.ndarray,
    normalization_factor: float,
    normalization_kind: str,
    compact_normalized_amplitude: np.ndarray | None,
    time_atol: float,
) -> dict[str, Any]:
    driven, selected = _selected_driven_coefficients(
        report_path,
        target_times=target_times,
        time_atol=time_atol,
    )
    item = _audit_selected_coefficients(
        selected,
        target_times=target_times,
        target_matrix=target_matrix,
        normalization_factor=normalization_factor,
        normalization_kind=normalization_kind,
        compact_normalized_amplitude=compact_normalized_amplitude,
    )
    item.update(
        {
            "report": str(report_path),
            "driver_kind": str(driven.get("driver_kind")),
            "source_projection": str(driven.get("source_projection")),
            "initial_state_kind": str(driven.get("initial_state_kind")),
            "forcing_kind": str(driven.get("forcing_kind")),
        }
    )
    if "driver_fast_tau" in driven:
        item["driver_fast_tau"] = float(driven["driver_fast_tau"])
    selected_driver_values = _selected_driver_values(
        driven,
        target_times=target_times,
        time_atol=time_atol,
    )
    if selected_driver_values is not None:
        item.update(_driver_following_metrics(selected, selected_driver_values))
    return item


def _audit_combined_reports(
    report_paths: list[Path],
    *,
    report_weights: np.ndarray,
    target_times: np.ndarray,
    target_matrix: np.ndarray,
    normalization_factor: float,
    normalization_kind: str,
    compact_normalized_amplitude: np.ndarray | None,
    time_atol: float,
) -> dict[str, Any]:
    selected_reports = [
        _selected_driven_coefficients(
            report_path,
            target_times=target_times,
            time_atol=time_atol,
        )
        for report_path in report_paths
    ]
    selected_shape = selected_reports[0][1].shape
    for report_path, (_, selected) in zip(report_paths, selected_reports):
        if selected.shape != selected_shape:
            raise ValueError(
                "combined driven reports must have matching selected "
                f"coefficient shapes; {report_path} has {selected.shape}, "
                f"expected {selected_shape}"
            )

    combined_selected = np.zeros(selected_shape, dtype=float)
    for weight, (_, selected) in zip(report_weights, selected_reports):
        combined_selected += float(weight) * selected

    item = _audit_selected_coefficients(
        combined_selected,
        target_times=target_times,
        target_matrix=target_matrix,
        normalization_factor=normalization_factor,
        normalization_kind=normalization_kind,
        compact_normalized_amplitude=compact_normalized_amplitude,
    )
    driven_reports = [driven for driven, _ in selected_reports]
    item.update(
        {
            "reports": [str(path) for path in report_paths],
            "report_weights": _float_list(report_weights),
            "driver_kinds": [str(driven.get("driver_kind")) for driven in driven_reports],
            "source_projections": [
                str(driven.get("source_projection")) for driven in driven_reports
            ],
            "initial_state_kinds": [
                str(driven.get("initial_state_kind")) for driven in driven_reports
            ],
            "forcing_kinds": [
                str(driven.get("forcing_kind")) for driven in driven_reports
            ],
        }
    )
    fast_taus = [
        float(driven["driver_fast_tau"])
        for driven in driven_reports
        if "driver_fast_tau" in driven
    ]
    if fast_taus:
        item["driver_fast_taus"] = fast_taus
    return item


def _fit_compact_report_weights(
    report_paths: list[Path],
    *,
    target_times: np.ndarray,
    target_matrix: np.ndarray,
    normalization_factor: float,
    normalization_kind: str,
    compact_normalized_amplitude: np.ndarray | None,
    time_atol: float,
) -> dict[str, Any]:
    if compact_normalized_amplitude is None:
        raise ValueError("compact_normalized_amplitude is required")
    selected_reports = [
        _selected_driven_coefficients(
            report_path,
            target_times=target_times,
            time_atol=time_atol,
        )
        for report_path in report_paths
    ]
    selected_shape = selected_reports[0][1].shape
    for report_path, (_, selected) in zip(report_paths, selected_reports):
        if selected.shape != selected_shape:
            raise ValueError(
                "report-weight fitting requires matching selected "
                f"coefficient shapes; {report_path} has {selected.shape}, "
                f"expected {selected_shape}"
            )
    if compact_normalized_amplitude.shape != (selected_shape[1],):
        raise ValueError(
            "compact normalized amplitude count must match the driven "
            "source-drive count"
        )

    compact_amplitude = compact_normalized_amplitude * normalization_factor
    columns = []
    for _, selected in selected_reports:
        compact = _collapse_selected(selected, compact_amplitude)
        if compact.shape != target_matrix.shape:
            raise ValueError(
                "collapsed compact report trace shape must match target "
                f"matrix shape; got {compact.shape}, expected {target_matrix.shape}"
            )
        columns.append(compact.reshape(-1))
    design = np.column_stack(columns)
    rhs = np.asarray(target_matrix, dtype=float).reshape(-1)
    report_weights, _, rank, singular_values = np.linalg.lstsq(
        design,
        rhs,
        rcond=None,
    )
    fitted_flat = design @ report_weights
    fitted = fitted_flat.reshape(target_matrix.shape)
    return {
        "reports": [str(path) for path in report_paths],
        "compact_amplitude_over_mu_delta_l2": _float_list(
            compact_normalized_amplitude
        ),
        "fit_report_weights": _float_list(report_weights),
        "fit_relative_l2": _relative_l2(fitted_flat, rhs),
        "fit_rank": int(rank),
        "fit_condition_number": _condition_number(singular_values),
        "first_selected_fit_over_normalization": _float_list(
            fitted[0] / normalization_factor
        ),
        "first_selected_target_over_normalization": _float_list(
            target_matrix[0] / normalization_factor
        ),
        "last_selected_fit_over_normalization": _float_list(
            fitted[-1] / normalization_factor
        ),
        "last_selected_target_over_normalization": _float_list(
            target_matrix[-1] / normalization_factor
        ),
    }
    _add_mu_delta_l2_aliases(result, normalization_kind)
    return result


def _selected_driven_coefficients(
    report_path: Path,
    *,
    target_times: np.ndarray,
    time_atol: float,
) -> tuple[dict[str, Any], np.ndarray]:
    payload = _load_json(report_path)
    driven = _driven_response(payload)
    times = np.asarray(driven["times"], dtype=float)
    coefficients = np.asarray(
        driven["source_moment_projection"]["coefficients"],
        dtype=float,
    )
    if coefficients.ndim != 3:
        raise ValueError(
            "driven source_moment_projection.coefficients must have shape "
            "(n_times, n_source_drives, n_source_moments)"
        )
    if coefficients.shape[0] != times.size:
        raise ValueError("driven coefficient time axis does not match times")
    indices = _time_indices(times, target_times, atol=time_atol)
    return driven, coefficients[indices]


def _selected_driver_values(
    driven: dict[str, Any],
    *,
    target_times: np.ndarray,
    time_atol: float,
) -> np.ndarray | None:
    if "driver_values" not in driven:
        return None
    times = np.asarray(driven["times"], dtype=float)
    driver_values = np.asarray(driven["driver_values"], dtype=float)
    if driver_values.ndim != 1:
        raise ValueError("driven driver_values must be a 1D vector")
    if driver_values.size != times.size:
        raise ValueError("driven driver_values time axis does not match times")
    indices = _time_indices(times, target_times, atol=time_atol)
    return driver_values[indices]


def _driver_following_metrics(
    selected: np.ndarray,
    selected_driver_values: np.ndarray,
) -> dict[str, Any]:
    driver = np.asarray(selected_driver_values, dtype=float).reshape(-1)
    if driver.size != selected.shape[0]:
        raise ValueError("selected driver value count must match selected times")
    result: dict[str, Any] = {
        "selected_driver_values": _float_list(driver),
    }
    if driver.size == 0 or driver[0] == 0.0:
        result["driver_follow_column_count"] = 0
        return result
    driver_normalized = driver / driver[0]
    errors = []
    for drive_index in range(selected.shape[1]):
        for moment_index in range(selected.shape[2]):
            trace = np.asarray(selected[:, drive_index, moment_index], dtype=float)
            if trace.size == 0 or trace[0] == 0.0:
                continue
            trace_normalized = trace / trace[0]
            errors.append(_relative_l2(trace_normalized, driver_normalized))
    result["driver_follow_column_count"] = len(errors)
    if errors:
        values = np.asarray(errors, dtype=float)
        result["driver_follow_relative_l2"] = _float_list(values)
        result["driver_follow_relative_l2_mean"] = float(np.mean(values))
        result["driver_follow_relative_l2_max"] = float(np.max(values))
    return result


def _audit_selected_coefficients(
    selected: np.ndarray,
    *,
    target_times: np.ndarray,
    target_matrix: np.ndarray,
    normalization_factor: float,
    normalization_kind: str,
    compact_normalized_amplitude: np.ndarray | None,
) -> dict[str, Any]:
    design = _shared_drive_design(selected)
    rhs = np.asarray(target_matrix, dtype=float).reshape(-1)
    fit_amplitude, _, rank, singular_values = np.linalg.lstsq(
        design,
        rhs,
        rcond=None,
    )
    fitted = _collapse_selected(selected, fit_amplitude)
    fit_normalized = fit_amplitude / normalization_factor
    first_fit_normalized = fitted[0] / normalization_factor
    first_target_normalized = target_matrix[0] / normalization_factor
    last_fit_normalized = fitted[-1] / normalization_factor
    last_target_normalized = target_matrix[-1] / normalization_factor
    item: dict[str, Any] = {
        "normalization_factor": float(normalization_factor),
        "fit_amplitude": _float_list(fit_amplitude),
        "fit_amplitude_over_normalization": _float_list(fit_normalized),
        "fit_relative_l2": _relative_l2(fitted.reshape(-1), rhs),
        "fit_rank": int(rank),
        "fit_condition_number": _condition_number(singular_values),
        "selected_times": _float_list(target_times),
        "first_selected_time": float(target_times[0]),
        "first_selected_fit_over_normalization": _float_list(first_fit_normalized),
        "first_selected_target_over_normalization": _float_list(
            first_target_normalized
        ),
        "last_selected_fit_over_normalization": _float_list(last_fit_normalized),
        "last_selected_target_over_normalization": _float_list(
            last_target_normalized
        ),
    }
    _add_mu_delta_l2_aliases(item, normalization_kind)
    if compact_normalized_amplitude is not None:
        if compact_normalized_amplitude.shape != (selected.shape[1],):
            raise ValueError(
                "compact normalized amplitude count must match the driven "
                "source-drive count"
            )
        compact_amplitude = compact_normalized_amplitude * normalization_factor
        compact = _collapse_selected(selected, compact_amplitude)
        compact_flat = compact.reshape(-1)
        scalar = _optimal_scalar(compact_flat, rhs)
        first_gate_scalar = _optimal_scalar(
            compact[0].reshape(-1),
            target_matrix[0].reshape(-1),
        )
        optimal_compact = scalar * compact
        first_gate_compact = first_gate_scalar * compact
        item.update(
            {
                "compact_amplitude_over_normalization": _float_list(
                    compact_normalized_amplitude
                ),
                "compact_relative_l2": _relative_l2(compact_flat, rhs),
                "compact_optimal_scalar": scalar,
                "compact_optimal_relative_l2": _relative_l2(
                    optimal_compact.reshape(-1),
                    rhs,
                ),
                "compact_time_relative_l2": _time_relative_l2(
                    compact,
                    target_matrix,
                ),
                "compact_time_error_over_target_norm": (
                    _time_error_over_target_norm(compact, target_matrix)
                ),
                "compact_time_error_fraction": _time_error_fraction(
                    compact,
                    target_matrix,
                ),
                "compact_optimal_time_relative_l2": _time_relative_l2(
                    optimal_compact,
                    target_matrix,
                ),
                "compact_optimal_time_error_over_target_norm": (
                    _time_error_over_target_norm(optimal_compact, target_matrix)
                ),
                "compact_optimal_time_error_fraction": _time_error_fraction(
                    optimal_compact,
                    target_matrix,
                ),
                "first_selected_compact_over_normalization": _float_list(
                    compact[0] / normalization_factor
                ),
                "compact_first_gate_scalar": first_gate_scalar,
                "compact_first_gate_amplitude_over_normalization": _float_list(
                    compact_normalized_amplitude * first_gate_scalar
                ),
                "compact_first_gate_relative_l2": _relative_l2(
                    first_gate_compact.reshape(-1),
                    rhs,
                ),
                "compact_first_gate_time_relative_l2": _time_relative_l2(
                    first_gate_compact,
                    target_matrix,
                ),
                "compact_first_gate_time_error_over_target_norm": (
                    _time_error_over_target_norm(first_gate_compact, target_matrix)
                ),
                "compact_first_gate_time_error_fraction": _time_error_fraction(
                    first_gate_compact,
                    target_matrix,
                ),
                "first_selected_compact_first_gate_scaled_over_normalization": (
                    _float_list(first_gate_compact[0] / normalization_factor)
                ),
                "last_selected_compact_first_gate_scaled_over_normalization": (
                    _float_list(first_gate_compact[-1] / normalization_factor)
                ),
            }
        )
        _add_mu_delta_l2_aliases(item, normalization_kind)
    return item


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _driven_response(payload: dict[str, Any]) -> dict[str, Any]:
    driven = payload.get("driven_response")
    if not isinstance(driven, dict):
        raise ValueError("report must contain a driven_response object")
    return driven


def _load_target(args) -> tuple[str, Path, np.ndarray, np.ndarray, dict[str, Any], float]:
    if args.target_matrix_report is not None:
        path = args.target_matrix_report
        payload = _load_json(path)
        target_times, target_matrix = _target_trace(payload)
        normalization = _normalization(payload)
        factor = _normalization_factor(normalization)
        normalization = {
            **normalization,
            "kind": "mu_delta_sigma_l2",
            "factor": factor,
        }
        return (
            "source_history_matrix",
            path,
            target_times,
            target_matrix,
            normalization,
            factor,
        )

    path = args.target_source_neighborhood_report
    payload = _load_json(path)
    target_times, target_matrix = _source_neighborhood_target_trace(
        payload,
        candidate=str(args.target_candidate),
    )
    normalization = _source_neighborhood_normalization(
        payload,
        candidate=str(args.target_candidate),
    )
    return (
        "source_neighborhood",
        path,
        target_times,
        target_matrix,
        normalization,
        float(normalization["factor"]),
    )


def _target_trace(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spatial = payload.get("spatial_time_series")
    if not isinstance(spatial, dict):
        raise ValueError("target report must contain spatial_time_series")
    samples = spatial.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("spatial_time_series.samples must be a nonempty list")
    times = np.asarray([sample["time"] for sample in samples], dtype=float)
    matrix = np.asarray([sample["coefficients"] for sample in samples], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("target coefficients must form a 2D matrix")
    return times, matrix


def _source_neighborhood_target_trace(
    payload: dict[str, Any],
    *,
    candidate: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = payload.get("candidate_static_fits")
    if not isinstance(candidates, dict) or candidate not in candidates:
        raise ValueError(f"candidate {candidate!r} is not present in target report")
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
    return times, coefficients.reshape(-1, 1)


def _normalization(payload: dict[str, Any]) -> dict[str, float]:
    value = payload.get("normalization")
    if not isinstance(value, dict):
        raise ValueError("target report must contain normalization")
    return {
        "mu": float(value["mu"]),
        "delta_sigma": float(value["delta_sigma"]),
        "source_length": float(value["source_length"]),
    }


def _source_neighborhood_normalization(
    payload: dict[str, Any],
    *,
    candidate: str,
) -> dict[str, Any]:
    factor = float(payload["source_diffusion_time_s"])
    if factor <= 0.0:
        raise ValueError("source_diffusion_time_s must be positive")
    return {
        "kind": "source_diffusion_time",
        "candidate": str(candidate),
        "source_diffusion_time_s": factor,
        "factor": factor,
    }


def _normalization_factor(normalization: dict[str, float]) -> float:
    factor = (
        float(normalization["mu"])
        * float(normalization["delta_sigma"])
        * float(normalization["source_length"]) ** 2
    )
    if factor == 0.0:
        raise ValueError("normalization factor must be nonzero")
    return factor


def _add_mu_delta_l2_aliases(item: dict[str, Any], normalization_kind: str) -> None:
    if normalization_kind != "mu_delta_sigma_l2":
        return
    aliases = {
        "fit_amplitude_over_mu_delta_l2": "fit_amplitude_over_normalization",
        "compact_amplitude_over_mu_delta_l2": "compact_amplitude_over_normalization",
        "first_selected_fit_over_mu_delta_l2": "first_selected_fit_over_normalization",
        "first_selected_target_over_mu_delta_l2": (
            "first_selected_target_over_normalization"
        ),
        "last_selected_fit_over_mu_delta_l2": "last_selected_fit_over_normalization",
        "last_selected_target_over_mu_delta_l2": (
            "last_selected_target_over_normalization"
        ),
        "first_selected_compact_over_mu_delta_l2": (
            "first_selected_compact_over_normalization"
        ),
        "compact_first_gate_amplitude_over_mu_delta_l2": (
            "compact_first_gate_amplitude_over_normalization"
        ),
        "first_selected_compact_first_gate_scaled_over_mu_delta_l2": (
            "first_selected_compact_first_gate_scaled_over_normalization"
        ),
        "last_selected_compact_first_gate_scaled_over_mu_delta_l2": (
            "last_selected_compact_first_gate_scaled_over_normalization"
        ),
    }
    for alias, canonical in aliases.items():
        if canonical in item:
            item[alias] = item[canonical]


def _time_indices(times: np.ndarray, target_times: np.ndarray, *, atol: float) -> np.ndarray:
    if atol < 0.0:
        raise ValueError("time_atol must be nonnegative")
    indices = []
    for time in target_times:
        index = int(np.argmin(np.abs(times - time)))
        if abs(float(times[index]) - float(time)) > atol:
            raise ValueError(f"target time {time:g} is not present in driven times")
        indices.append(index)
    return np.asarray(indices, dtype=int)


def _shared_drive_design(selected: np.ndarray) -> np.ndarray:
    n_times, n_drives, n_moments = selected.shape
    design = np.zeros((n_times * n_moments, n_drives), dtype=float)
    for time_index in range(n_times):
        for moment_index in range(n_moments):
            design[time_index * n_moments + moment_index] = selected[
                time_index,
                :,
                moment_index,
            ]
    return design


def _collapse_selected(selected: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    return np.einsum("tdm,d->tm", selected, np.asarray(amplitude, dtype=float))


def _relative_l2(values: np.ndarray, target: np.ndarray) -> float:
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        return float(np.linalg.norm(values - target))
    return float(np.linalg.norm(values - target) / target_norm)


def _time_relative_l2(values: np.ndarray, target: np.ndarray) -> list[float]:
    if values.shape != target.shape:
        raise ValueError(
            f"time relative L2 shapes must match; got {values.shape} and {target.shape}"
        )
    return [
        _relative_l2(values[time_index].reshape(-1), target[time_index].reshape(-1))
        for time_index in range(values.shape[0])
    ]


def _time_error_over_target_norm(values: np.ndarray, target: np.ndarray) -> list[float]:
    if values.shape != target.shape:
        raise ValueError(
            "time error contribution shapes must match; "
            f"got {values.shape} and {target.shape}"
        )
    target_norm = float(np.linalg.norm(target.reshape(-1)))
    errors = values - target
    if target_norm == 0.0:
        return [
            float(np.linalg.norm(errors[time_index].reshape(-1)))
            for time_index in range(values.shape[0])
        ]
    return [
        float(np.linalg.norm(errors[time_index].reshape(-1)) / target_norm)
        for time_index in range(values.shape[0])
    ]


def _time_error_fraction(values: np.ndarray, target: np.ndarray) -> list[float]:
    if values.shape != target.shape:
        raise ValueError(
            f"time error fraction shapes must match; got {values.shape} and {target.shape}"
        )
    errors = values - target
    energy = np.asarray(
        [
            float(np.vdot(errors[time_index], errors[time_index]))
            for time_index in range(values.shape[0])
        ],
        dtype=float,
    )
    total = float(np.sum(energy))
    if total == 0.0:
        return [0.0 for _ in range(values.shape[0])]
    return _float_list(energy / total)


def _optimal_scalar(values: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.vdot(values, values))
    if denominator == 0.0:
        return 0.0
    return float(np.vdot(values, target) / denominator)


def _condition_number(singular_values: np.ndarray) -> float | None:
    values = np.asarray(singular_values, dtype=float)
    if values.size == 0 or values[-1] == 0.0:
        return None
    return float(values[0] / values[-1])


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


if __name__ == "__main__":
    raise SystemExit(main())
