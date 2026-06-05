"""Fit source-moment coefficient traces to rise-decay history kernels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import _build_time_steps
from .metrics import relative_l2
from .source_history_operator import _least_squares_design_diagnostics
from .source_primary import (
    _time_node_indices,
    discrete_driven_relaxation_basis,
    discrete_relaxation_difference_basis,
    normalized_source_primary_scale,
)
from .source_primary_cli import _load_config_from_result


@dataclass(frozen=True)
class TraceKernelFit:
    """Least-squares fit of coefficient traces to an explicit history design."""

    basis_labels: list[str]
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    per_trace_relative_l2: np.ndarray
    rank: int
    singular_values: np.ndarray
    design_shape: tuple[int, int]
    column_norms: np.ndarray
    condition_number: float
    column_normalized_condition_number: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit spatial source-history coefficient traces from a matrix-scan "
            "JSON report to zero-initial BE rise-decay kernels."
        )
    )
    parser.add_argument("matrix_report", type=Path)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="HDF5 result containing config_yaml time_steps for the BE grid",
    )
    parser.add_argument(
        "--slow-tau",
        type=float,
        default=None,
        help="Slow relaxation tau; defaults to the matrix report top-level tau",
    )
    parser.add_argument(
        "--fast-taus",
        nargs="+",
        type=float,
        default=[],
        help="Candidate fast taus for g_slow - g_fast",
    )
    parser.add_argument(
        "--fast-taus-from-recovery-sweep",
        nargs="+",
        type=Path,
        default=[],
        help="Recovery-spectrum sweep JSON files whose diffusion_time_estimate values are candidate fast taus",
    )
    parser.add_argument(
        "--basis-kind",
        choices=("relaxation_difference", "driven_relaxation"),
        default="relaxation_difference",
        help=(
            "History kernel basis: diagnostic g_slow-g_fast difference or "
            "the BE driven recovery state used by the runtime hook."
        ),
    )
    parser.add_argument(
        "--time-atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for matching report sample times to time nodes",
    )
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--prescribed-coefficients", nargs="+", type=float)
    parser.add_argument("--prescribed-normalized-coefficients", nargs="+", type=float)
    parser.add_argument("-o", "--output", type=Path, required=True)
    argv, prescribed_args = _extract_prescribed_coefficient_args(argv)
    args = parser.parse_args(argv)
    for name, values in prescribed_args.items():
        setattr(args, name, values)

    payload = json.loads(args.matrix_report.read_text(encoding="utf-8"))
    sample_times, coefficient_matrix, trace_names = _coefficient_trace_data(payload)
    time_mask = _time_window_mask(
        sample_times,
        time_min=args.time_min,
        time_max=args.time_max,
    )
    sample_times = sample_times[time_mask]
    coefficient_matrix = coefficient_matrix[time_mask]
    slow_tau = _slow_tau(args, payload)
    fast_tau_sources = _fast_tau_sources(args)
    fast_taus = _fast_taus(
        [
            *fast_tau_sources["explicit"],
            *fast_tau_sources["from_recovery_sweeps"],
        ],
        slow_tau=slow_tau,
    )
    time_steps = _time_steps_from_result(args.result)
    normalization = _normalization(payload)

    fits = {}
    for fast_tau in fast_taus:
        design, labels = _rise_decay_design(
            time_steps,
            sample_times,
            slow_tau=slow_tau,
            fast_taus=[fast_tau],
            basis_kind=args.basis_kind,
            time_atol=float(args.time_atol),
        )
        fit = _fit_trace_kernel_design(design, coefficient_matrix, labels)
        fits[f"fast_tau_{fast_tau:g}"] = _fit_report(
            fit,
            trace_names=trace_names,
            normalization=normalization,
        )

    best_key = min(fits, key=lambda key: fits[key]["relative_l2"])
    best_fast_tau = float(fits[best_key]["fast_taus"][0])
    multi_basis_fit = None
    if len(fast_taus) > 1:
        design, labels = _rise_decay_design(
            time_steps,
            sample_times,
            slow_tau=slow_tau,
            fast_taus=fast_taus,
            basis_kind=args.basis_kind,
            time_atol=float(args.time_atol),
        )
        multi_basis_fit = _fit_report(
            _fit_trace_kernel_design(design, coefficient_matrix, labels),
            trace_names=trace_names,
            normalization=normalization,
        )

    report = {
        "diagnostic_only": True,
        "basis_kind": _basis_kind_report_name(args.basis_kind),
        "source_report": str(args.matrix_report),
        "result": str(args.result),
        "slow_tau": float(slow_tau),
        "fast_taus": [float(value) for value in fast_taus],
        "fast_tau_sources": {
            "explicit": [float(value) for value in fast_tau_sources["explicit"]],
            "recovery_sweeps": [
                str(path) for path in args.fast_taus_from_recovery_sweep
            ],
        },
        "best_fast_tau": best_fast_tau,
        "best_key": best_key,
        "time_window": {
            "min": None if args.time_min is None else float(args.time_min),
            "max": None if args.time_max is None else float(args.time_max),
            "selected_count": int(sample_times.size),
        },
        "sample_times": [float(value) for value in sample_times],
        "trace_names": trace_names,
        "normalization": normalization,
        "fits": fits,
    }
    if multi_basis_fit is not None:
        report["multi_basis_fit"] = multi_basis_fit
    prescribed = _prescribed_coefficients(
        args,
        n_basis=fast_taus.size,
        n_traces=coefficient_matrix.shape[1],
        normalization=normalization,
    )
    if prescribed is not None:
        design, labels = _rise_decay_design(
            time_steps,
            sample_times,
            slow_tau=slow_tau,
            fast_taus=fast_taus,
            basis_kind=args.basis_kind,
            time_atol=float(args.time_atol),
        )
        report["prescribed_evaluation"] = _evaluate_trace_kernel_design(
            design,
            coefficient_matrix,
            prescribed,
            labels,
            trace_names=trace_names,
            normalization=normalization,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    best = fits[best_key]
    print(
        f"best_fast_tau={best_fast_tau:g} "
        f"relative_l2={best['relative_l2']:.6e}"
    )
    print(f"wrote {args.output}")
    return 0


def _extract_prescribed_coefficient_args(
    argv: list[str] | None,
) -> tuple[list[str], dict[str, list[float]]]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    option_to_dest = {
        "--prescribed-coefficients": "prescribed_coefficients",
        "--prescribed-normalized-coefficients": "prescribed_normalized_coefficients",
    }
    extracted: dict[str, list[float]] = {}
    cleaned: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        matched_option = None
        attached_value = None
        for option in option_to_dest:
            if token == option:
                matched_option = option
                break
            prefix = f"{option}="
            if token.startswith(prefix):
                matched_option = option
                attached_value = token[len(prefix) :]
                break
        if matched_option is None:
            cleaned.append(token)
            index += 1
            continue

        values: list[float] = []
        if attached_value is not None:
            if attached_value == "":
                raise ValueError(f"{matched_option} requires at least one value")
            values.append(float(attached_value))
        index += 1
        while index < len(tokens) and _looks_like_float_token(tokens[index]):
            values.append(float(tokens[index]))
            index += 1
        if not values:
            raise ValueError(f"{matched_option} requires at least one value")
        extracted[option_to_dest[matched_option]] = values
    return cleaned, extracted


def _looks_like_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _coefficient_trace_data(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spatial = payload.get("spatial_time_series")
    if not isinstance(spatial, dict):
        raise ValueError("matrix report does not contain spatial_time_series")
    samples = spatial.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("spatial_time_series.samples must be a non-empty list")
    sample_times = np.asarray([row["time"] for row in samples], dtype=float)
    if sample_times.ndim != 1 or np.any(sample_times < 0.0):
        raise ValueError("sample times must be a nonnegative 1D sequence")

    matrix = None
    matrix_payload = spatial.get("coefficient_matrix")
    if isinstance(matrix_payload, dict) and "values" in matrix_payload:
        matrix = np.asarray(matrix_payload["values"], dtype=float)
    if matrix is None:
        matrix = np.asarray([row["coefficients"] for row in samples], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("coefficient matrix must have shape (n_times, n_traces)")
    if matrix.shape[0] != sample_times.size:
        raise ValueError("coefficient matrix row count must match sample times")

    names = spatial.get("coefficient_names")
    if names is None:
        names = [f"trace:{index}" for index in range(matrix.shape[1])]
    trace_names = [str(name) for name in names]
    if len(trace_names) != matrix.shape[1]:
        raise ValueError("coefficient_names must match coefficient matrix columns")
    return sample_times, matrix, trace_names


def _time_window_mask(times: np.ndarray, *, time_min, time_max) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if time_min is not None and float(time_min) < 0.0:
        raise ValueError("--time-min must be nonnegative")
    if time_max is not None and float(time_max) < 0.0:
        raise ValueError("--time-max must be nonnegative")
    if (
        time_min is not None
        and time_max is not None
        and float(time_min) > float(time_max)
    ):
        raise ValueError("--time-min must be less than or equal to --time-max")
    mask = np.ones(times.shape, dtype=bool)
    atol = 1.0e-12
    if time_min is not None:
        mask &= times >= float(time_min) - atol
    if time_max is not None:
        mask &= times <= float(time_max) + atol
    if not np.any(mask):
        raise ValueError("time window selects no samples")
    return mask


def _slow_tau(args: argparse.Namespace, payload: dict[str, Any]) -> float:
    value = args.slow_tau
    if value is None:
        value = payload.get("tau")
    if value is None:
        raise ValueError("--slow-tau is required when the report has no top-level tau")
    value = float(value)
    if value <= 0.0:
        raise ValueError("slow_tau must be positive")
    return value


def _fast_taus(values, *, slow_tau: float) -> np.ndarray:
    fast_taus = np.asarray(_unique_float_values(values), dtype=float)
    if fast_taus.ndim != 1 or fast_taus.size == 0:
        raise ValueError("--fast-taus must contain at least one value")
    if np.any(fast_taus <= 0.0):
        raise ValueError("fast taus must be positive")
    if np.any(fast_taus >= float(slow_tau)):
        raise ValueError("fast taus must be smaller than slow_tau")
    return fast_taus


def _unique_float_values(values) -> list[float]:
    unique: list[float] = []
    for value in values:
        number = float(value)
        if not any(np.isclose(number, existing, rtol=1.0e-12, atol=1.0e-15) for existing in unique):
            unique.append(number)
    return unique


def _fast_tau_sources(args: argparse.Namespace) -> dict[str, list[float]]:
    return {
        "explicit": [float(value) for value in args.fast_taus],
        "from_recovery_sweeps": [
            value
            for path in args.fast_taus_from_recovery_sweep
            for value in _fast_taus_from_recovery_sweep(path)
        ],
    }


def _fast_taus_from_recovery_sweep(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sweep = payload.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("recovery sweep report must contain a sweep object")
    cases = sweep.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("recovery sweep report must contain sweep.cases")
    values = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or "diffusion_time_estimate" not in case:
            raise ValueError(
                f"recovery sweep case {index} has no diffusion_time_estimate"
            )
        value = float(case["diffusion_time_estimate"])
        if value <= 0.0:
            raise ValueError("diffusion_time_estimate values must be positive")
        values.append(value)
    return values


def _time_steps_from_result(path: Path) -> np.ndarray:
    config = _load_config_from_result(path)
    if "time_steps" not in config:
        raise ValueError("result config does not contain time_steps")
    return np.asarray(_build_time_steps(config["time_steps"]), dtype=float)


def _normalization(payload: dict[str, Any]) -> dict[str, float | None]:
    raw = payload.get("normalization", {})
    if not isinstance(raw, dict):
        raw = {}
    result: dict[str, float | None] = {}
    for key in ("delta_sigma", "source_length", "mu"):
        value = raw.get(key)
        result[key] = None if value is None else float(value)
    return result


def _rise_decay_design(
    time_steps,
    sample_times: np.ndarray,
    *,
    slow_tau: float,
    fast_taus,
    basis_kind: str,
    time_atol: float,
) -> tuple[np.ndarray, list[str]]:
    columns = []
    labels = []
    kind = str(basis_kind).strip().lower()
    for fast_tau in fast_taus:
        if kind == "relaxation_difference":
            basis = discrete_relaxation_difference_basis(
                time_steps,
                slow_tau=slow_tau,
                fast_tau=float(fast_tau),
            )
            label = f"BE relaxation difference slow-fast fast_tau={float(fast_tau):g}"
        elif kind == "driven_relaxation":
            basis = discrete_driven_relaxation_basis(
                time_steps,
                driver_tau=slow_tau,
                response_tau=float(fast_tau),
            )
            label = f"BE driven relaxation response response_tau={float(fast_tau):g}"
        else:
            raise ValueError(f"unsupported basis kind: {basis_kind}")
        indices = _time_node_indices(basis.times, sample_times, atol=time_atol)
        columns.append(basis.values[indices])
        labels.append(label)
    return np.column_stack(columns), labels


def _basis_kind_report_name(value: str) -> str:
    kind = str(value).strip().lower()
    if kind == "relaxation_difference":
        return "discrete_relaxation_difference"
    if kind == "driven_relaxation":
        return "discrete_driven_relaxation"
    raise ValueError(f"unsupported basis kind: {value}")


def _fit_trace_kernel_design(
    design,
    coefficient_matrix,
    basis_labels: list[str],
) -> TraceKernelFit:
    design = np.asarray(design, dtype=float)
    coefficient_matrix = np.asarray(coefficient_matrix, dtype=float)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("design must have shape (n_times, n_basis)")
    if coefficient_matrix.ndim != 2 or coefficient_matrix.shape[0] != design.shape[0]:
        raise ValueError("coefficient_matrix must have shape (n_times, n_traces)")
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        coefficient_matrix,
        rcond=None,
    )
    fitted = design @ coefficients
    residual = fitted - coefficient_matrix
    diagnostics = _least_squares_design_diagnostics(design, singular_values, int(rank))
    per_trace = np.asarray(
        [
            relative_l2(fitted[:, index], coefficient_matrix[:, index])
            for index in range(coefficient_matrix.shape[1])
        ],
        dtype=float,
    )
    return TraceKernelFit(
        basis_labels=list(basis_labels),
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, coefficient_matrix),
        per_trace_relative_l2=per_trace,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def _evaluate_trace_kernel_design(
    design,
    coefficient_matrix,
    coefficients,
    basis_labels: list[str],
    *,
    trace_names: list[str],
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    design = np.asarray(design, dtype=float)
    coefficient_matrix = np.asarray(coefficient_matrix, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (design.shape[1], coefficient_matrix.shape[1]):
        raise ValueError("prescribed coefficient matrix has wrong shape")
    fitted = design @ coefficients
    residual = fitted - coefficient_matrix
    per_trace = [
        relative_l2(fitted[:, index], coefficient_matrix[:, index])
        for index in range(coefficient_matrix.shape[1])
    ]
    report = {
        "basis_labels": list(basis_labels),
        "fast_taus": [_fast_tau_from_label(label) for label in basis_labels],
        "coefficients": coefficients.reshape(-1).astype(float).tolist(),
        "coefficient_table": {
            "row_labels": list(basis_labels),
            "column_labels": list(trace_names),
            "values": coefficients.astype(float).tolist(),
        },
        "coefficient_sum_table": _coefficient_sum_table(coefficients, trace_names),
        "relative_l2": float(relative_l2(fitted, coefficient_matrix)),
        "per_trace_relative_l2": [float(value) for value in per_trace],
        "fitted": fitted.astype(float).tolist(),
        "residual": residual.astype(float).tolist(),
    }
    normalized = _normalized_coefficient_table(
        coefficients,
        basis_labels,
        trace_names,
        normalization,
    )
    if normalized is not None:
        report["coefficients_over_mu_delta_l2"] = (
            np.asarray(normalized["values"], dtype=float).reshape(-1).tolist()
        )
        report["coefficient_table_over_mu_delta_l2"] = normalized
        report["coefficient_sum_table_over_mu_delta_l2"] = _coefficient_sum_table(
            np.asarray(normalized["values"], dtype=float),
            trace_names,
        )
    return report


def _fit_report(
    fit: TraceKernelFit,
    *,
    trace_names: list[str],
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    report = {
        "basis_labels": list(fit.basis_labels),
        "fast_taus": [_fast_tau_from_label(label) for label in fit.basis_labels],
        "coefficients": fit.coefficients.reshape(-1).astype(float).tolist(),
        "coefficient_table": {
            "row_labels": list(fit.basis_labels),
            "column_labels": list(trace_names),
            "values": fit.coefficients.astype(float).tolist(),
        },
        "coefficient_sum_table": _coefficient_sum_table(
            fit.coefficients,
            trace_names,
        ),
        "relative_l2": float(fit.relative_l2),
        "per_trace_relative_l2": [float(value) for value in fit.per_trace_relative_l2],
        "fitted": fit.fitted.astype(float).tolist(),
        "residual": fit.residual.astype(float).tolist(),
        "rank": int(fit.rank),
        "singular_values": [float(value) for value in fit.singular_values],
        "design_matrix": {
            "shape": [int(value) for value in fit.design_shape],
            "column_norms": [float(value) for value in fit.column_norms],
            "condition_number": float(fit.condition_number),
            "column_normalized_condition_number": float(
                fit.column_normalized_condition_number
            ),
        },
    }
    normalized = _normalized_coefficient_table(
        fit.coefficients,
        fit.basis_labels,
        trace_names,
        normalization,
    )
    if normalized is not None:
        report["coefficients_over_mu_delta_l2"] = (
            np.asarray(normalized["values"], dtype=float).reshape(-1).tolist()
        )
        report["coefficient_table_over_mu_delta_l2"] = normalized
        report["coefficient_sum_table_over_mu_delta_l2"] = _coefficient_sum_table(
            np.asarray(normalized["values"], dtype=float),
            trace_names,
        )
    return report


def _prescribed_coefficients(
    args: argparse.Namespace,
    *,
    n_basis: int,
    n_traces: int,
    normalization: dict[str, float | None],
) -> np.ndarray | None:
    has_physical = args.prescribed_coefficients is not None
    has_normalized = args.prescribed_normalized_coefficients is not None
    if has_physical and has_normalized:
        raise ValueError(
            "use only one of --prescribed-coefficients or "
            "--prescribed-normalized-coefficients"
        )
    if not has_physical and not has_normalized:
        return None

    values = np.asarray(
        args.prescribed_coefficients
        if has_physical
        else args.prescribed_normalized_coefficients,
        dtype=float,
    )
    expected = int(n_basis) * int(n_traces)
    if values.size != expected:
        raise ValueError(
            "prescribed coefficient count must equal "
            "len(fast_taus) * number_of_traces"
        )

    coefficients = values.reshape(int(n_basis), int(n_traces))
    if has_normalized:
        coefficients = coefficients * _normalization_factor(normalization)
    return coefficients


def _normalization_factor(normalization: dict[str, float | None]) -> float:
    delta_sigma = normalization.get("delta_sigma")
    source_length = normalization.get("source_length")
    mu = normalization.get("mu")
    if delta_sigma is None or source_length is None or mu is None:
        raise ValueError(
            "prescribed normalized coefficients require delta_sigma, "
            "source_length, and mu in report normalization"
        )
    return float(mu) * float(delta_sigma) * float(source_length) ** 2


def _coefficient_sum_table(coefficients: np.ndarray, column_labels: list[str]) -> dict[str, Any]:
    values = np.sum(np.asarray(coefficients, dtype=float), axis=0, keepdims=True)
    return {
        "row_labels": ["sum_over_history_basis"],
        "column_labels": list(column_labels),
        "values": values.tolist(),
    }


def _normalized_coefficient_table(
    coefficients: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    normalization: dict[str, float | None],
) -> dict[str, Any] | None:
    delta_sigma = normalization.get("delta_sigma")
    source_length = normalization.get("source_length")
    mu = normalization.get("mu")
    if delta_sigma is None or source_length is None or mu is None:
        return None
    values = np.asarray(
        [
            [
                normalized_source_primary_scale(
                    value,
                    delta_sigma=float(delta_sigma),
                    source_length=float(source_length),
                    mu=float(mu),
                )
                for value in row
            ]
            for row in np.asarray(coefficients, dtype=float)
        ],
        dtype=float,
    )
    return {
        "row_labels": list(row_labels),
        "column_labels": list(column_labels),
        "values": values.tolist(),
    }


def _fast_tau_from_label(label: str) -> float:
    for marker in ("fast_tau=", "response_tau="):
        if marker in label:
            return float(label.split(marker, 1)[1])
    return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
