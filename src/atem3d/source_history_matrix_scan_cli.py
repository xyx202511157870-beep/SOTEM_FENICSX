"""Batch source-history matrix-basis scans over orders and time windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import build_simulation
from .source_history_matrix_cli import (
    _coefficient_table,
    _field_location,
    _fit_report,
    _local_edge_basis_report,
    _local_spatial_edge_basis,
    _normalized_coefficient_table,
    _parse_source_moment_degrees,
    _per_column_fit_report,
    _prescribed_coefficients_from_args,
    _receiver_columns,
    _receiver_matrix,
    _source_moment_basis_report,
    _source_moment_spatial_basis,
    _source_normalization,
    _source_vector,
    _static_response_for_spatial_vectors,
    _static_response_matrix_report,
    _static_response_matrix_report_from_static_response,
    _spatial_time_series_report,
    _time_window_mask,
    _wire_source_vector,
)
from .local_coupling import local_edge_basis
from .source_history_operator import (
    evaluate_source_history_coefficients_for_components,
    fit_static_spatial_coefficients_from_static_response,
    fit_static_spatial_coefficients_for_components,
    fit_source_history_coefficients_for_components,
    source_history_receiver_basis,
    source_history_receiver_basis_from_spatial_vectors,
    source_history_receiver_basis_from_static_response,
    source_history_receiver_basis_from_vectors,
)
from .source_primary import _time_node_indices
from .source_primary_cli import (
    _load_config_from_result,
    _load_sampled_report,
    _report_result_path,
    _single_config_tau,
    _target_data,
    _validate_sample_alignment,
)


@dataclass(frozen=True)
class TimeWindowSpec:
    """Named positive-time sample window for matrix-basis scans."""

    label: str
    time_min: float | None
    time_max: float | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan source-history matrix fits over several BE basis orders and "
            "sample time windows."
        )
    )
    parser.add_argument("ip_report", type=Path)
    parser.add_argument("noip_report", type=Path)
    parser.add_argument(
        "--target",
        choices=[
            "reference_delta",
            "numerical_delta",
            "ip_residual",
            "noip_residual",
            "delta_residual",
        ],
        default="reference_delta",
    )
    parser.add_argument("--component-prefix", default="Hz")
    parser.add_argument(
        "--receiver-matrix",
        choices=[
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        ],
        default="edge_current",
    )
    parser.add_argument(
        "--field-location",
        choices=["auto", "edge", "face"],
        default="auto",
        help="Source-history vector location. 'auto' uses face for formulation: hj.",
    )
    parser.add_argument(
        "--source-vector",
        choices=[
            "wire",
            "dc_conduction_current",
            "dc_total_current",
            "dc_polarization_current",
        ],
        default="wire",
    )
    parser.add_argument(
        "--source-vectors",
        default=None,
        help=(
            "Comma-separated source-vector choices.  When provided, the scan "
            "uses the first max_order + 1 names for each order."
        ),
    )
    parser.add_argument(
        "--spatial-basis",
        choices=["source_vector", "source_moments", "local_edges"],
        default="source_vector",
    )
    parser.add_argument("--source-cell-radius", type=int, default=0)
    parser.add_argument("--receiver-cell-radius", type=int, default=0)
    parser.add_argument("--source-edge-atol", type=float, default=0.0)
    parser.add_argument(
        "--local-basis-scope",
        choices=["support_edges", "source_edges", "source_cell_edges", "source_moments"],
        default="support_edges",
    )
    parser.add_argument("--source-moment-degree", type=int, default=2)
    parser.add_argument(
        "--source-moment-degrees",
        default=None,
        help="Comma-separated source moment degrees, e.g. 0,2. Overrides --source-moment-degree.",
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--orders",
        required=True,
        help="Comma-separated nonnegative BE max_order values, e.g. 1,2,3.",
    )
    parser.add_argument(
        "--windows",
        default="all",
        help=(
            "Comma-separated time windows: 'all' or min:max, with either bound "
            "optional."
        ),
    )
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument(
        "--prescribed-coefficients",
        default=None,
        help="Comma-separated coefficients to evaluate without fitting for every scanned case.",
    )
    parser.add_argument(
        "--prescribed-normalized-coefficients",
        default=None,
        help=(
            "Comma-separated coefficients in units of "
            "mu*delta_sigma*source_length**2."
        ),
    )
    parser.add_argument(
        "--per-column",
        action="store_true",
        help="Also fit independent coefficients for each selected receiver/component column.",
    )
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    ip = _load_sampled_report(args.ip_report, args.component_prefix)
    noip = _load_sampled_report(args.noip_report, args.component_prefix)
    _validate_sample_alignment(ip, noip)
    target = _target_data(ip, noip, args.target)

    result_path = args.result or _report_result_path(args.ip_report, ip.payload)
    config = _load_config_from_result(result_path)
    sim = build_simulation(config)
    field_location = _field_location(args.field_location, config)
    tau = float(args.tau) if args.tau is not None else _single_config_tau(config)
    original_receiver_indices, component_indices, locations = _receiver_columns(
        config,
        sim,
        ip.names,
    )
    local_receiver_indices = np.arange(len(component_indices), dtype=int)
    matrix_free_static_response = (
        args.spatial_basis in {"local_edges", "source_moments"}
        and args.receiver_matrix in {"edge_basis", "face_basis"}
    )
    receiver_matrix = None
    if not matrix_free_static_response:
        receiver_matrix = _receiver_matrix(
            sim,
            locations,
            kind=args.receiver_matrix,
            subdivisions=int(args.subdivisions),
            field_location=field_location,
        )
    normalization = _source_normalization(sim)
    fit_normalization = normalization

    orders = _parse_orders(args.orders)
    windows = _parse_windows(args.windows)
    prescribed_coefficients = _prescribed_coefficients_from_args(
        args,
        normalization=normalization,
    )
    local_basis = None
    edge_basis = None
    local_basis_report = None
    source_moment_basis = None
    source_moment_basis_report = None
    static_response_matrix = None
    spatial_time_series = None
    local_static_response = None
    source_moment_static_response = None
    if args.spatial_basis == "local_edges":
        if field_location != "edge":
            raise ValueError("--spatial-basis local_edges currently requires edge field location")
        if args.source_vectors is not None:
            raise ValueError("--source-vectors is only supported with --spatial-basis source_vector")
        local_basis = local_edge_basis(
            sim.mesh,
            _wire_source_vector(sim),
            locations,
            source_cell_radius=int(args.source_cell_radius),
            receiver_cell_radius=int(args.receiver_cell_radius),
            source_edge_atol=float(args.source_edge_atol),
        )
        edge_basis = _local_spatial_edge_basis(
            sim,
            local_basis,
            scope=args.local_basis_scope,
            source_vector=_wire_source_vector(sim),
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
            source_edge_atol=float(args.source_edge_atol),
        )
        source_vector_names = [args.local_basis_scope]
        local_basis_report = _local_edge_basis_report(
            local_basis,
            source_cell_radius=int(args.source_cell_radius),
            receiver_cell_radius=int(args.receiver_cell_radius),
            source_edge_atol=float(args.source_edge_atol),
            basis_scope=args.local_basis_scope,
            edge_basis=edge_basis,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
        )
        if receiver_matrix is None:
            local_static_response = _static_response_for_spatial_vectors(
                sim,
                locations,
                kind=args.receiver_matrix,
                subdivisions=int(args.subdivisions),
                field_location=field_location,
                spatial_vectors=edge_basis.basis_vectors,
            )
        if local_static_response is None:
            if receiver_matrix is None:
                receiver_matrix = _receiver_matrix(
                    sim,
                    locations,
                    kind=args.receiver_matrix,
                    subdivisions=int(args.subdivisions),
                    field_location=field_location,
                )
            static_response_matrix = _static_response_matrix_report(
                receiver_matrix,
                edge_basis.basis_vectors,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
            spatial_fit = fit_static_spatial_coefficients_for_components(
                receiver_matrix,
                edge_basis.basis_vectors,
                target,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
        else:
            static_response_matrix = _static_response_matrix_report_from_static_response(
                local_static_response,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
            spatial_fit = fit_static_spatial_coefficients_from_static_response(
                local_static_response,
                target,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
        fit_normalization = {"delta_sigma": None, "source_length": None, "mu": None}
        spatial_time_series = _spatial_time_series_report(
            spatial_fit,
            times=ip.times,
            spatial_labels=edge_basis.basis_labels,
            time_steps=sim.time_steps,
            tau=tau,
            history_orders=orders,
            normalization=fit_normalization,
            prescribed_coefficients=prescribed_coefficients,
        )
    elif args.spatial_basis == "source_moments":
        if args.source_vectors is not None:
            raise ValueError("--source-vectors is only supported with --spatial-basis source_vector")
        source_moment_basis = _source_moment_spatial_basis(
            sim,
            field_location=field_location,
            source_vector_kind=args.source_vector,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
            source_edge_atol=float(args.source_edge_atol),
        )
        source_moment_basis_report = _source_moment_basis_report(
            source_moment_basis,
            field_location=field_location,
            source_vector=args.source_vector,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
            source_edge_atol=float(args.source_edge_atol),
        )
        if receiver_matrix is None:
            source_moment_static_response = _static_response_for_spatial_vectors(
                sim,
                locations,
                kind=args.receiver_matrix,
                subdivisions=int(args.subdivisions),
                field_location=field_location,
                spatial_vectors=source_moment_basis.basis_vectors,
            )
        if source_moment_static_response is None:
            if receiver_matrix is None:
                receiver_matrix = _receiver_matrix(
                    sim,
                    locations,
                    kind=args.receiver_matrix,
                    subdivisions=int(args.subdivisions),
                    field_location=field_location,
                )
            static_response_matrix = _static_response_matrix_report(
                receiver_matrix,
                source_moment_basis.basis_vectors,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
            spatial_fit = fit_static_spatial_coefficients_for_components(
                receiver_matrix,
                source_moment_basis.basis_vectors,
                target,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
        else:
            static_response_matrix = _static_response_matrix_report_from_static_response(
                source_moment_static_response,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
            spatial_fit = fit_static_spatial_coefficients_from_static_response(
                source_moment_static_response,
                target,
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
        spatial_time_series = _spatial_time_series_report(
            spatial_fit,
            times=ip.times,
            spatial_labels=source_moment_basis.basis_labels,
            time_steps=sim.time_steps,
            tau=tau,
            history_orders=orders,
            normalization=fit_normalization,
            prescribed_coefficients=prescribed_coefficients,
        )
        source_vector_names = ["source_moments"]
    else:
        source_vector_names = _source_vector_names_for_scan(args, max(orders))

    cases = []
    for max_order in orders:
        if local_basis is not None:
            active_names = [args.local_basis_scope]
            if local_static_response is None:
                basis = source_history_receiver_basis_from_spatial_vectors(
                    sim.time_steps,
                    tau=tau,
                    spatial_vectors=edge_basis.basis_vectors,
                    receiver_matrix=receiver_matrix,
                    max_order=max_order,
                    spatial_labels=edge_basis.basis_labels,
                )
            else:
                basis = source_history_receiver_basis_from_static_response(
                    sim.time_steps,
                    tau=tau,
                    static_response=local_static_response,
                    max_order=max_order,
                    spatial_labels=edge_basis.basis_labels,
                )
        elif source_moment_basis is not None:
            active_names = ["source_moments"]
            if source_moment_static_response is None:
                basis = source_history_receiver_basis_from_spatial_vectors(
                    sim.time_steps,
                    tau=tau,
                    spatial_vectors=source_moment_basis.basis_vectors,
                    receiver_matrix=receiver_matrix,
                    max_order=max_order,
                    spatial_labels=source_moment_basis.basis_labels,
                )
            else:
                basis = source_history_receiver_basis_from_static_response(
                    sim.time_steps,
                    tau=tau,
                    static_response=source_moment_static_response,
                    max_order=max_order,
                    spatial_labels=source_moment_basis.basis_labels,
                )
        else:
            active_names = source_vector_names[: max_order + 1]
            source_vectors = np.vstack(
                [_source_vector(sim, name, field_location) for name in active_names]
            )
            if field_location == "edge" and args.receiver_matrix == "edge_basis":
                source_vectors = source_vectors / sim.unit_edge_mass_diagonal[None, :]
            if args.source_vectors is None:
                basis = source_history_receiver_basis(
                    sim.time_steps,
                    tau=tau,
                    source_vector=source_vectors[0],
                    receiver_matrix=receiver_matrix,
                    max_order=max_order,
                )
            else:
                basis = source_history_receiver_basis_from_vectors(
                    sim.time_steps,
                    tau=tau,
                    source_vectors=source_vectors,
                    receiver_matrix=receiver_matrix,
                )
        for window in windows:
            time_mask = _time_window_mask(
                ip.times,
                time_min=window.time_min,
                time_max=window.time_max,
            )
            fit_times = ip.times[time_mask]
            indices = _time_node_indices(basis.times, fit_times, atol=1.0e-12)
            fit = fit_source_history_coefficients_for_components(
                basis.responses[indices],
                target[time_mask],
                receiver_indices=local_receiver_indices,
                component_indices=component_indices,
            )
            per_column_fit = None
            prescribed_report = None
            if prescribed_coefficients is not None:
                prescribed = evaluate_source_history_coefficients_for_components(
                    basis.responses[indices],
                    target[time_mask],
                    coefficients=prescribed_coefficients,
                    receiver_indices=local_receiver_indices,
                    component_indices=component_indices,
                )
                prescribed_report = _fit_report(
                    prescribed,
                    normalization=fit_normalization,
                )
                prescribed_report["coefficient_table"] = _coefficient_table(
                    prescribed.coefficients,
                    basis.basis_labels,
                )
                normalized_table = _normalized_coefficient_table(
                    prescribed.coefficients,
                    basis.basis_labels,
                    fit_normalization,
                )
                if normalized_table is not None:
                    prescribed_report["coefficient_table_over_mu_delta_l2"] = (
                        normalized_table
                    )
            if args.per_column:
                per_column_fit = _per_column_fit_report(
                    basis.responses[indices],
                    target[time_mask],
                    component_names=ip.names,
                    original_receiver_indices=original_receiver_indices,
                    local_receiver_indices=local_receiver_indices,
                    component_indices=component_indices,
                    receiver_locations=locations,
                    normalization=fit_normalization,
                )
            cases.append(
                _case_report(
                    fit,
                    normalization=fit_normalization,
                    max_order=max_order,
                    source_vector_names=active_names,
                    window=window,
                    selected_count=fit_times.size,
                    basis_labels=basis.basis_labels,
                    per_column_fit=per_column_fit,
                    prescribed_report=prescribed_report,
                )
            )

    report = {
        "diagnostic_only": True,
        "target": args.target,
        "component_names": ip.names,
        "receiver_indices": [int(value) for value in original_receiver_indices],
        "local_receiver_indices": [int(value) for value in local_receiver_indices],
        "receiver_matrix": args.receiver_matrix,
        "field_location": field_location,
        "spatial_basis": args.spatial_basis,
        "source_vector": args.source_vector,
        "source_vectors": source_vector_names,
        "tau": tau,
        "orders": orders,
        "windows": [
            {
                "label": window.label,
                "min": window.time_min,
                "max": window.time_max,
            }
            for window in windows
        ],
        "normalization": normalization,
        "cases": cases,
    }
    if local_basis_report is not None:
        report["local_edge_basis"] = local_basis_report
    if source_moment_basis_report is not None:
        report["source_moment_basis"] = source_moment_basis_report
    if static_response_matrix is not None:
        report["static_response_matrix"] = static_response_matrix
    if spatial_time_series is not None:
        report["spatial_time_series"] = spatial_time_series
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"cases={len(cases)} wrote {args.output}")
    return 0


def _case_report(
    fit,
    *,
    normalization: dict[str, float | None],
    max_order: int,
    source_vector_names: list[str],
    window: TimeWindowSpec,
    selected_count: int,
    basis_labels: list[str],
    per_column_fit: list[dict[str, Any]] | None = None,
    prescribed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "label": f"order_{max_order}_{window.label}",
        "max_order": int(max_order),
        "source_vectors": list(source_vector_names),
        "time_window": {
            "label": window.label,
            "min": window.time_min,
            "max": window.time_max,
            "selected_count": int(selected_count),
        },
        "basis_labels": list(basis_labels),
        "fit": _fit_report(fit, normalization=normalization),
        "coefficient_table": _coefficient_table(fit.coefficients, basis_labels),
    }
    if prescribed_report is not None:
        report["prescribed"] = prescribed_report
    if per_column_fit is not None:
        report["per_column_fit"] = per_column_fit
    return report


def _parse_orders(value: str) -> list[int]:
    orders: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            raise ValueError("--orders contains an empty entry")
        order = int(text)
        if order < 0:
            raise ValueError("--orders must contain nonnegative integers")
        orders.append(order)
    if not orders:
        raise ValueError("--orders must contain at least one value")
    return orders


def _parse_windows(value: str) -> list[TimeWindowSpec]:
    windows: list[TimeWindowSpec] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            raise ValueError("--windows contains an empty entry")
        if text.lower() == "all":
            windows.append(TimeWindowSpec(label="all", time_min=None, time_max=None))
            continue
        if ":" not in text:
            raise ValueError("--windows entries must be 'all' or min:max")
        start_text, end_text = [piece.strip() for piece in text.split(":", 1)]
        if not start_text and not end_text:
            raise ValueError("--windows entries cannot have two empty bounds")
        time_min = None if not start_text else float(start_text)
        time_max = None if not end_text else float(end_text)
        label = f"{start_text or 'start'}_{end_text or 'end'}"
        windows.append(TimeWindowSpec(label=label, time_min=time_min, time_max=time_max))
    if not windows:
        raise ValueError("--windows must contain at least one value")
    return windows


def _source_vector_names_for_scan(args, max_order: int) -> list[str]:
    allowed = {
        "wire",
        "dc_conduction_current",
        "dc_total_current",
        "dc_polarization_current",
    }
    if args.source_vectors is None:
        return [str(args.source_vector)] * (int(max_order) + 1)
    names = [part.strip() for part in str(args.source_vectors).split(",")]
    if len(names) < int(max_order) + 1 or any(not name for name in names):
        raise ValueError("--source-vectors must provide at least max(orders) + 1 names")
    unsupported = [name for name in names if name not in allowed]
    if unsupported:
        raise ValueError(f"unsupported source vector(s): {', '.join(unsupported)}")
    return names[: int(max_order) + 1]


if __name__ == "__main__":
    raise SystemExit(main())
