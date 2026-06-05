"""Fit source-history matrix-basis coefficients from sampled reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import build_simulation
from .hj import hj_dc_initial_current_density, hj_dc_initial_electric_field
from .local_coupling import (
    EdgeBasis,
    FaceBasis,
    LocalEdgeBasis,
    canonical_edge_basis,
    edge_indices_for_cells,
    local_edge_basis,
    source_edge_moment_basis,
    source_face_moment_basis,
)
from .magnetic_recovery import (
    biot_savart_h_from_edge_basis_currents,
    biot_savart_h_from_face_basis_currents,
    cell_current_biot_matrix,
    edge_basis_biot_matrix,
    edge_current_biot_matrix,
    face_basis_biot_matrix,
    face_current_biot_matrix,
)
from .source_history_operator import (
    evaluate_spatial_coefficient_traces_with_history_basis,
    evaluate_source_history_coefficients_for_components,
    fit_static_spatial_coefficients_from_static_response,
    fit_spatial_coefficient_traces_to_history_basis,
    fit_static_spatial_coefficients_for_components,
    fit_source_history_coefficients_for_components,
    project_vector_to_spatial_basis,
    source_history_receiver_basis,
    source_history_receiver_basis_from_spatial_vectors,
    source_history_receiver_basis_from_static_response,
    source_history_receiver_basis_from_vectors,
)
from .source_primary import _time_node_indices, normalized_source_primary_scale
from .source_primary_cli import (
    _load_config_from_result,
    _load_sampled_report,
    _receiver_name_map,
    _report_result_path,
    _single_config_tau,
    _target_data,
    _validate_sample_alignment,
)
from .source_history_runtime import charge_conserving_face_current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit source-history coefficients using explicit FV/MMR receiver "
            "matrices and sampled validation reports."
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
        help=(
            "Edge-current moment vector used as the static source-history "
            "spatial basis."
        ),
    )
    parser.add_argument(
        "--source-vectors",
        default=None,
        help=(
            "Comma-separated source-vector choices, one per BE basis order. "
            "Overrides --source-vector when provided."
        ),
    )
    parser.add_argument(
        "--spatial-basis",
        choices=["source_vector", "source_moments", "local_edges"],
        default="source_vector",
        help=(
            "Use the legacy source-vector basis, low-order source moments, "
            "or a canonical local edge basis over source/receiver neighborhoods."
        ),
    )
    parser.add_argument("--source-cell-radius", type=int, default=0)
    parser.add_argument("--receiver-cell-radius", type=int, default=0)
    parser.add_argument("--source-edge-atol", type=float, default=0.0)
    parser.add_argument(
        "--local-basis-scope",
        choices=["support_edges", "source_edges", "source_cell_edges", "source_moments"],
        default="support_edges",
        help=(
            "Which edge set is converted to local spatial basis vectors when "
            "--spatial-basis local_edges is used."
        ),
    )
    parser.add_argument("--source-moment-degree", type=int, default=2)
    parser.add_argument(
        "--source-moment-degrees",
        default=None,
        help="Comma-separated source moment degrees, e.g. 0,2. Overrides --source-moment-degree.",
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument(
        "--prescribed-coefficients",
        default=None,
        help="Comma-separated coefficients to evaluate without fitting.",
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
        "--prescribed-coefficients-file",
        type=Path,
        default=None,
        help=(
            "JSON file containing prescribed coefficients, e.g. an "
            "atem3d-ampere-source-projection report."
        ),
    )
    parser.add_argument(
        "--prescribed-coefficients-key",
        default="discrete_basis_fit.coefficients",
        help="Dotted JSON key used with --prescribed-coefficients-file.",
    )
    parser.add_argument(
        "--prescribed-spatial-trace-index",
        type=int,
        default=None,
        help=(
            "Treat loaded coefficients as a single history trace and place "
            "them in this spatial-basis column, filling other columns with zero."
        ),
    )
    parser.add_argument(
        "--prescribed-candidate",
        choices=[
            "initial_polarization",
            "charge_conserving_initial_polarization",
        ],
        default=None,
        help="Generate prescribed coefficients from a built-in non-fitted FV/MMR candidate.",
    )
    parser.add_argument(
        "--prescribed-candidate-projection",
        choices=["receiver_l2", "dof_l2"],
        default="receiver_l2",
        help="Projection used by --prescribed-candidate.",
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
    time_mask = _time_window_mask(
        ip.times,
        time_min=args.time_min,
        time_max=args.time_max,
    )
    fit_times = ip.times[time_mask]
    target = target[time_mask]

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
    local_basis = None
    source_moment_basis = None
    static_response_matrix = None
    spatial_time_series = None
    basis_spatial_count = None
    prescribed_coefficients_for_basis = None
    prescribed_candidate_report = None
    local_static_response = None
    source_moment_static_response = None
    normalization = _source_normalization(sim)
    prescribed_coefficients = _prescribed_coefficients_from_args(
        args,
        normalization=normalization,
    )
    fit_normalization = normalization
    if args.spatial_basis == "local_edges":
        if field_location != "edge":
            raise ValueError("--spatial-basis local_edges currently requires edge field location")
        if args.source_vectors is not None:
            raise ValueError("--source-vectors is only supported with --spatial-basis source_vector")
        support_vector = _wire_source_vector(sim)
        local_basis = local_edge_basis(
            sim.mesh,
            support_vector,
            locations,
            source_cell_radius=int(args.source_cell_radius),
            receiver_cell_radius=int(args.receiver_cell_radius),
            source_edge_atol=float(args.source_edge_atol),
        )
        edge_basis = _local_spatial_edge_basis(
            sim,
            local_basis,
            scope=args.local_basis_scope,
            source_vector=support_vector,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
            source_edge_atol=float(args.source_edge_atol),
        )
        basis_spatial_count = len(edge_basis.basis_labels)
        source_vector_names = [args.local_basis_scope]
        fit_normalization = {"delta_sigma": None, "source_length": None, "mu": None}
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
            basis = source_history_receiver_basis_from_spatial_vectors(
                sim.time_steps,
                tau=tau,
                spatial_vectors=edge_basis.basis_vectors,
                receiver_matrix=receiver_matrix,
                max_order=int(args.max_order),
                spatial_labels=edge_basis.basis_labels,
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
            basis = source_history_receiver_basis_from_static_response(
                sim.time_steps,
                tau=tau,
                static_response=local_static_response,
                max_order=int(args.max_order),
                spatial_labels=edge_basis.basis_labels,
            )
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
        if args.prescribed_candidate is not None and receiver_matrix is None:
            receiver_matrix = _receiver_matrix(
                sim,
                locations,
                kind=args.receiver_matrix,
                subdivisions=int(args.subdivisions),
                field_location=field_location,
            )
        (
            prescribed_coefficients_for_basis,
            prescribed_candidate_report,
        ) = _prescribed_coefficients_for_spatial_basis(
            args,
            prescribed_coefficients,
            sim=sim,
            receiver_matrix=receiver_matrix,
            spatial_vectors=edge_basis.basis_vectors,
            field_location=field_location,
            max_order=int(args.max_order),
        )
        spatial_time_series = _spatial_time_series_report(
            spatial_fit,
            times=fit_times,
            spatial_labels=edge_basis.basis_labels,
            time_steps=sim.time_steps,
            tau=tau,
            history_orders=[int(args.max_order)],
            normalization=fit_normalization,
            prescribed_coefficients=prescribed_coefficients_for_basis,
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
        basis_spatial_count = len(source_moment_basis.basis_labels)
        source_vector_names = ["source_moments"]
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
            basis = source_history_receiver_basis_from_spatial_vectors(
                sim.time_steps,
                tau=tau,
                spatial_vectors=source_moment_basis.basis_vectors,
                receiver_matrix=receiver_matrix,
                max_order=int(args.max_order),
                spatial_labels=source_moment_basis.basis_labels,
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
            basis = source_history_receiver_basis_from_static_response(
                sim.time_steps,
                tau=tau,
                static_response=source_moment_static_response,
                max_order=int(args.max_order),
                spatial_labels=source_moment_basis.basis_labels,
            )
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
        if args.prescribed_candidate is not None and receiver_matrix is None:
            receiver_matrix = _receiver_matrix(
                sim,
                locations,
                kind=args.receiver_matrix,
                subdivisions=int(args.subdivisions),
                field_location=field_location,
            )
        (
            prescribed_coefficients_for_basis,
            prescribed_candidate_report,
        ) = _prescribed_coefficients_for_spatial_basis(
            args,
            prescribed_coefficients,
            sim=sim,
            receiver_matrix=receiver_matrix,
            spatial_vectors=source_moment_basis.basis_vectors,
            field_location=field_location,
            max_order=int(args.max_order),
        )
        spatial_time_series = _spatial_time_series_report(
            spatial_fit,
            times=fit_times,
            spatial_labels=source_moment_basis.basis_labels,
            time_steps=sim.time_steps,
            tau=tau,
            history_orders=[int(args.max_order)],
            normalization=fit_normalization,
            prescribed_coefficients=prescribed_coefficients_for_basis,
        )
    else:
        if args.prescribed_candidate is not None:
            raise ValueError(
                "--prescribed-candidate requires --spatial-basis source_moments or local_edges"
            )
        source_vector_names = _source_vector_names(args)
        basis_spatial_count = len(source_vector_names)
        source_vectors = np.vstack(
            [_source_vector(sim, name, field_location) for name in source_vector_names]
        )
        if field_location == "edge" and args.receiver_matrix == "edge_basis":
            source_vectors = source_vectors / sim.unit_edge_mass_diagonal[None, :]
        if args.source_vectors is None:
            basis = source_history_receiver_basis(
                sim.time_steps,
                tau=tau,
                source_vector=source_vectors[0],
                receiver_matrix=receiver_matrix,
                max_order=int(args.max_order),
            )
        else:
            basis = source_history_receiver_basis_from_vectors(
                sim.time_steps,
                tau=tau,
                source_vectors=source_vectors,
                receiver_matrix=receiver_matrix,
            )
        prescribed_coefficients_for_basis = _expand_prescribed_coefficients_for_spatial_trace(
            prescribed_coefficients,
            n_spatial=basis_spatial_count,
            trace_index=args.prescribed_spatial_trace_index,
        )
    indices = _time_node_indices(basis.times, fit_times, atol=1.0e-12)
    fit = fit_source_history_coefficients_for_components(
        basis.responses[indices],
        target,
        receiver_indices=local_receiver_indices,
        component_indices=component_indices,
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
        "time_window": {
            "min": None if args.time_min is None else float(args.time_min),
            "max": None if args.time_max is None else float(args.time_max),
            "selected_count": int(fit_times.size),
        },
        "tau": tau,
        "max_order": int(args.max_order),
        "basis_labels": basis.basis_labels,
        "normalization": normalization,
        "fit": _fit_report(fit, normalization=fit_normalization),
        "coefficient_table": _coefficient_table(fit.coefficients, basis.basis_labels),
    }
    if prescribed_candidate_report is not None:
        report["prescribed_candidate"] = prescribed_candidate_report
    if prescribed_coefficients_for_basis is not None:
        prescribed = evaluate_source_history_coefficients_for_components(
            basis.responses[indices],
            target,
            coefficients=prescribed_coefficients_for_basis,
            receiver_indices=local_receiver_indices,
            component_indices=component_indices,
        )
        prescribed_report = _fit_report(prescribed, normalization=fit_normalization)
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
            prescribed_report["coefficient_table_over_mu_delta_l2"] = normalized_table
        report["prescribed"] = prescribed_report
    if local_basis is not None:
        report["local_edge_basis"] = _local_edge_basis_report(
            local_basis,
            source_cell_radius=int(args.source_cell_radius),
            receiver_cell_radius=int(args.receiver_cell_radius),
            source_edge_atol=float(args.source_edge_atol),
            basis_scope=args.local_basis_scope,
            edge_basis=edge_basis,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
        )
    if source_moment_basis is not None:
        report["source_moment_basis"] = _source_moment_basis_report(
            source_moment_basis,
            field_location=field_location,
            source_vector=args.source_vector,
            source_moment_degree=int(args.source_moment_degree),
            source_moment_degrees=_parse_source_moment_degrees(args),
            source_edge_atol=float(args.source_edge_atol),
        )
    if static_response_matrix is not None:
        report["static_response_matrix"] = static_response_matrix
    if spatial_time_series is not None:
        report["spatial_time_series"] = spatial_time_series
    if args.per_column:
        report["per_column_fit"] = _per_column_fit_report(
            basis.responses[indices],
            target,
            component_names=ip.names,
            original_receiver_indices=original_receiver_indices,
            local_receiver_indices=local_receiver_indices,
            component_indices=component_indices,
            receiver_locations=locations,
            normalization=fit_normalization,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "coefficients="
        + _format_coefficients(fit.coefficients)
        + f" relative_l2={fit.relative_l2:.6e}"
    )
    print(f"wrote {args.output}")
    return 0


def _receiver_columns(config: dict[str, Any], sim, names: list[str]):
    receiver_by_name = _receiver_name_map(config, sim.receivers)
    receiver_indices = []
    component_indices = []
    locations = []
    for name in names:
        receiver = receiver_by_name[name]
        receiver_index = next(
            index for index, candidate in enumerate(sim.receivers) if candidate is receiver
        )
        receiver_indices.append(receiver_index)
        component_indices.append({"x": 0, "y": 1, "z": 2}[receiver.component[-1].lower()])
        locations.append(receiver.location)
    return (
        np.asarray(receiver_indices, dtype=int),
        np.asarray(component_indices, dtype=int),
        np.asarray(locations, dtype=float),
    )


def _receiver_matrix(
    sim,
    locations: np.ndarray,
    *,
    kind: str,
    subdivisions: int,
    field_location: str = "edge",
):
    field_location = _normalize_field_location(field_location)
    if kind == "current_biot":
        if field_location == "face":
            return face_current_biot_matrix(
                sim.mesh,
                locations,
                subdivisions=subdivisions,
            )
        return cell_current_biot_matrix(
            sim.mesh,
            locations,
            subdivisions=subdivisions,
        )
    if kind == "edge_current":
        _require_field_location(field_location, "edge", kind)
        return edge_current_biot_matrix(sim.mesh, locations)
    if kind == "edge_basis":
        _require_field_location(field_location, "edge", kind)
        return edge_basis_biot_matrix(
            sim.mesh,
            locations,
            subdivisions=subdivisions,
        )
    if kind == "face_current":
        _require_field_location(field_location, "face", kind)
        return face_current_biot_matrix(
            sim.mesh,
            locations,
            subdivisions=subdivisions,
        )
    if kind == "face_basis":
        _require_field_location(field_location, "face", kind)
        return face_basis_biot_matrix(
            sim.mesh,
            locations,
            subdivisions=subdivisions,
        )
    raise ValueError(f"unsupported receiver matrix: {kind}")


def _static_response_for_spatial_vectors(
    sim,
    locations: np.ndarray,
    *,
    kind: str,
    subdivisions: int,
    field_location: str,
    spatial_vectors: np.ndarray,
) -> np.ndarray | None:
    """Return ``B_R v_s`` directly when full receiver matrices are expensive."""

    field_location = _normalize_field_location(field_location)
    kind = str(kind).strip().lower()
    if kind == "edge_basis":
        _require_field_location(field_location, "edge", kind)
        return np.asarray(
            [
                biot_savart_h_from_edge_basis_currents(
                    sim.mesh,
                    vector,
                    locations,
                    subdivisions=subdivisions,
                )
                for vector in np.asarray(spatial_vectors, dtype=float)
            ],
            dtype=float,
        )
    if kind == "face_basis":
        _require_field_location(field_location, "face", kind)
        return np.asarray(
            [
                biot_savart_h_from_face_basis_currents(
                    sim.mesh,
                    vector,
                    locations,
                    subdivisions=subdivisions,
                )
                for vector in np.asarray(spatial_vectors, dtype=float)
            ],
            dtype=float,
        )
    return None


def _field_location(value: str, config: dict[str, Any]) -> str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        formulation = str(config.get("formulation", "eb")).strip().lower()
        return "face" if formulation == "hj" else "edge"
    return _normalize_field_location(normalized)


def _normalize_field_location(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"edge", "face"}:
        raise ValueError("field location must be 'edge' or 'face'")
    return normalized


def _require_field_location(actual: str, expected: str, receiver_matrix: str) -> None:
    if actual != expected:
        raise ValueError(
            f"receiver_matrix='{receiver_matrix}' requires field_location='{expected}'"
        )


def _time_window_mask(times: np.ndarray, *, time_min, time_max) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    atol = 1.0e-12
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
    if time_min is not None:
        mask &= times >= float(time_min) - atol
    if time_max is not None:
        mask &= times <= float(time_max) + atol
    if not np.any(mask):
        raise ValueError("time window selects no samples")
    return mask


def _source_vector_names(args) -> list[str]:
    max_order = int(args.max_order)
    if max_order < 0:
        raise ValueError("--max-order must be nonnegative")
    if args.source_vectors is None:
        return [str(args.source_vector)] * (max_order + 1)
    names = [part.strip() for part in str(args.source_vectors).split(",")]
    if len(names) != max_order + 1 or any(not name for name in names):
        raise ValueError("--source-vectors must provide max_order + 1 names")
    allowed = {
        "wire",
        "dc_conduction_current",
        "dc_total_current",
        "dc_polarization_current",
    }
    unsupported = [name for name in names if name not in allowed]
    if unsupported:
        raise ValueError(f"unsupported source vector(s): {', '.join(unsupported)}")
    return names


def _parse_coefficients(value: str) -> np.ndarray:
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("coefficients must be a comma-separated list of numbers")
    return np.asarray([float(part) for part in parts], dtype=float)


def _prescribed_coefficients_from_args(
    args,
    *,
    normalization: dict[str, float | None] | None = None,
) -> np.ndarray | None:
    explicit_count = sum(
        value is not None
        for value in (
            getattr(args, "prescribed_coefficients", None),
            getattr(args, "prescribed_normalized_coefficients", None),
            getattr(args, "prescribed_coefficients_file", None),
            getattr(args, "prescribed_candidate", None),
        )
    )
    if explicit_count > 1:
        raise ValueError(
            "choose only one of --prescribed-coefficients, "
            "--prescribed-normalized-coefficients, "
            "--prescribed-coefficients-file, or --prescribed-candidate"
        )
    if getattr(args, "prescribed_candidate", None) is not None:
        if getattr(args, "prescribed_spatial_trace_index", None) is not None:
            raise ValueError(
                "--prescribed-spatial-trace-index is only used with explicit coefficient series"
            )
        return None
    if getattr(args, "prescribed_coefficients_file", None) is not None:
        payload = json.loads(
            Path(args.prescribed_coefficients_file).read_text(encoding="utf-8")
        )
        value = _json_dotted_value(payload, str(args.prescribed_coefficients_key))
        coefficients = np.asarray(value, dtype=float)
    elif getattr(args, "prescribed_coefficients", None) is not None:
        coefficients = _parse_coefficients(args.prescribed_coefficients)
    elif getattr(args, "prescribed_normalized_coefficients", None) is not None:
        coefficients = _parse_coefficients(args.prescribed_normalized_coefficients)
        coefficients = coefficients * _source_normalization_factor(normalization)
    else:
        if getattr(args, "prescribed_spatial_trace_index", None) is not None:
            raise ValueError(
                "--prescribed-spatial-trace-index requires prescribed coefficients"
            )
        return None
    if coefficients.size == 0:
        raise ValueError("prescribed coefficients must not be empty")
    return np.asarray(coefficients, dtype=float)


def _json_dotted_value(payload, key: str):
    value = payload
    for part in str(key).split("."):
        if not part:
            raise ValueError("prescribed coefficient key contains an empty segment")
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"prescribed coefficient key not found: {key}")
        value = value[part]
    return value


def _expand_prescribed_coefficients_for_spatial_trace(
    coefficients: np.ndarray | None,
    *,
    n_spatial: int,
    trace_index: int | None,
) -> np.ndarray | None:
    if coefficients is None:
        return None
    values = np.asarray(coefficients, dtype=float)
    if trace_index is None:
        return values.reshape(-1)

    n_spatial = int(n_spatial)
    trace_index = int(trace_index)
    if n_spatial <= 0:
        raise ValueError("n_spatial must be positive")
    if trace_index < 0 or trace_index >= n_spatial:
        raise ValueError("--prescribed-spatial-trace-index is out of range")
    if values.ndim == 2:
        if values.shape[1] == n_spatial:
            return values.reshape(-1)
        if values.shape[1] != 1:
            raise ValueError(
                "2D prescribed coefficients must have one column or n_spatial columns"
            )
        series = values[:, 0]
    else:
        series = values.reshape(-1)
    table = np.zeros((series.size, n_spatial), dtype=float)
    table[:, trace_index] = series
    return table.reshape(-1)


def _prescribed_coefficients_for_spatial_basis(
    args,
    explicit_coefficients: np.ndarray | None,
    *,
    sim,
    receiver_matrix: np.ndarray,
    spatial_vectors: np.ndarray,
    field_location: str,
    max_order: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if args.prescribed_candidate is None:
        return (
            _expand_prescribed_coefficients_for_spatial_trace(
                explicit_coefficients,
                n_spatial=np.asarray(spatial_vectors).shape[0],
                trace_index=args.prescribed_spatial_trace_index,
            ),
            None,
        )
    if args.prescribed_candidate == "initial_polarization":
        coefficients = _initial_polarization_candidate_coefficients(
            sim,
            receiver_matrix,
            spatial_vectors,
            field_location=field_location,
            max_order=max_order,
            projection=args.prescribed_candidate_projection,
        )
        return coefficients, {
            "kind": "initial_polarization",
            "requires_ip": True,
            "projection": str(args.prescribed_candidate_projection),
            "field_location": _normalize_field_location(field_location),
            "max_order": int(max_order),
            "coefficient_count": int(coefficients.size),
        }
    if args.prescribed_candidate == "charge_conserving_initial_polarization":
        coefficients = _charge_conserving_initial_polarization_candidate_coefficients(
            sim,
            receiver_matrix,
            spatial_vectors,
            field_location=field_location,
            max_order=max_order,
            projection=args.prescribed_candidate_projection,
        )
        return coefficients, {
            "kind": "charge_conserving_initial_polarization",
            "requires_ip": True,
            "projection": str(args.prescribed_candidate_projection),
            "field_location": _normalize_field_location(field_location),
            "max_order": int(max_order),
            "coefficient_count": int(coefficients.size),
        }
    raise ValueError(f"unsupported prescribed candidate: {args.prescribed_candidate}")


def _initial_polarization_candidate_coefficients(
    sim,
    receiver_matrix: np.ndarray,
    spatial_vectors: np.ndarray,
    *,
    field_location: str,
    max_order: int,
    projection: str,
) -> np.ndarray:
    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    target_vector = -_dc_polarization_current_vector(sim, field_location)
    leading_coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        spatial_vectors,
        target_vector,
        projection=projection,
    )
    table = np.zeros((max_order + 1, spatial_vectors.shape[0]), dtype=float)
    table[0] = leading_coefficients
    return table.reshape(-1)


def _charge_conserving_initial_polarization_candidate_coefficients(
    sim,
    receiver_matrix: np.ndarray,
    spatial_vectors: np.ndarray,
    *,
    field_location: str,
    max_order: int,
    projection: str,
) -> np.ndarray:
    field_location = _normalize_field_location(field_location)
    if field_location != "face":
        raise ValueError(
            "charge_conserving_initial_polarization candidate requires face field location"
        )
    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    target_vector = charge_conserving_face_current(
        sim.mesh,
        sim.ip_model.sigma_infinity,
        -_dc_polarization_current_vector(sim, field_location),
    )
    leading_coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        spatial_vectors,
        target_vector,
        projection=projection,
    )
    table = np.zeros((max_order + 1, spatial_vectors.shape[0]), dtype=float)
    table[0] = leading_coefficients
    return table.reshape(-1)


def _local_edge_basis_report(
    basis: LocalEdgeBasis,
    *,
    source_cell_radius: int,
    receiver_cell_radius: int,
    source_edge_atol: float,
    basis_scope: str = "support_edges",
    edge_basis: EdgeBasis | None = None,
    source_moment_degree: int | None = None,
    source_moment_degrees: list[int] | None = None,
) -> dict[str, Any]:
    basis_edge_indices = (
        basis.support_edge_indices if edge_basis is None else edge_basis.edge_indices
    )
    report = {
        "source_cell_radius": int(source_cell_radius),
        "receiver_cell_radius": int(receiver_cell_radius),
        "source_edge_atol": float(source_edge_atol),
        "basis_scope": str(basis_scope),
        "source_edge_count": int(basis.source_edge_indices.size),
        "source_cell_count": int(basis.source_cell_indices.size),
        "support_cell_count": int(basis.support_cell_indices.size),
        "support_edge_count": int(basis.support_edge_indices.size),
        "basis_edge_count": int(basis_edge_indices.size),
        "basis_vector_count": (
            int(edge_basis.basis_vectors.shape[0])
            if edge_basis is not None
            else int(basis_edge_indices.size)
        ),
        "source_edge_indices": [int(value) for value in basis.source_edge_indices],
        "source_cell_indices": [int(value) for value in basis.source_cell_indices],
        "receiver_cell_indices": [
            [int(value) for value in indices] for indices in basis.receiver_cell_indices
        ],
        "support_cell_indices": [int(value) for value in basis.support_cell_indices],
        "support_edge_indices": [int(value) for value in basis.support_edge_indices],
        "basis_edge_indices": [int(value) for value in basis_edge_indices],
    }
    if source_moment_degree is not None and str(basis_scope) == "source_moments":
        report["source_moment_degree"] = int(source_moment_degree)
    if source_moment_degrees is not None and str(basis_scope) == "source_moments":
        report["source_moment_degrees"] = [int(value) for value in source_moment_degrees]
    return report


def _local_spatial_edge_basis(
    sim,
    basis: LocalEdgeBasis,
    *,
    scope: str,
    source_vector: np.ndarray | None = None,
    source_moment_degree: int = 2,
    source_moment_degrees: list[int] | None = None,
    source_edge_atol: float = 0.0,
) -> EdgeBasis:
    scope = str(scope).strip().lower()
    mesh = sim.mesh
    if scope == "support_edges":
        return canonical_edge_basis(mesh, basis.support_edge_indices)
    if scope == "source_edges":
        return canonical_edge_basis(
            mesh,
            basis.source_edge_indices,
            label_prefix="source_edge",
        )
    if scope == "source_cell_edges":
        return canonical_edge_basis(
            mesh,
            edge_indices_for_cells(mesh, basis.source_cell_indices),
            label_prefix="source_cell_edge",
        )
    if scope == "source_moments":
        if len(sim.sources) != 1:
            raise ValueError("source_moments local basis requires exactly one source")
        if source_vector is None:
            source_vector = _wire_source_vector(sim)
        source = sim.sources[0]
        return source_edge_moment_basis(
            mesh,
            source_vector,
            start=source.locations[0],
            end=source.locations[-1],
            max_degree=int(source_moment_degree),
            degrees=source_moment_degrees,
            source_edge_atol=float(source_edge_atol),
        )
    raise ValueError(f"unsupported local basis scope: {scope}")


def _source_moment_spatial_basis(
    sim,
    *,
    field_location: str,
    source_vector_kind: str = "wire",
    source_moment_degree: int = 2,
    source_moment_degrees: list[int] | None = None,
    source_edge_atol: float = 0.0,
) -> EdgeBasis | FaceBasis:
    field_location = _normalize_field_location(field_location)
    if len(sim.sources) != 1:
        raise ValueError("source_moments spatial basis requires exactly one source")
    source = sim.sources[0]
    source_vector = _source_vector(sim, source_vector_kind, field_location)
    if field_location == "edge":
        return source_edge_moment_basis(
            sim.mesh,
            source_vector,
            start=source.locations[0],
            end=source.locations[-1],
            max_degree=int(source_moment_degree),
            degrees=source_moment_degrees,
            source_edge_atol=float(source_edge_atol),
        )
    return source_face_moment_basis(
        sim.mesh,
        source_vector,
        start=source.locations[0],
        end=source.locations[-1],
        max_degree=int(source_moment_degree),
        degrees=source_moment_degrees,
        source_face_atol=float(source_edge_atol),
    )


def _source_moment_basis_report(
    basis: EdgeBasis | FaceBasis,
    *,
    field_location: str,
    source_vector: str,
    source_moment_degree: int,
    source_moment_degrees: list[int] | None,
    source_edge_atol: float,
) -> dict[str, Any]:
    if isinstance(basis, FaceBasis):
        source_indices = basis.face_indices
    else:
        source_indices = basis.edge_indices
    report = {
        "field_location": _normalize_field_location(field_location),
        "source_vector": str(source_vector),
        "source_edge_atol": float(source_edge_atol),
        "source_moment_degree": int(source_moment_degree),
        "source_index_count": int(source_indices.size),
        "basis_vector_count": int(basis.basis_vectors.shape[0]),
        "source_indices": [int(value) for value in source_indices],
        "basis_labels": list(basis.basis_labels),
    }
    if source_moment_degrees is not None:
        report["source_moment_degrees"] = [int(value) for value in source_moment_degrees]
    return report


def _parse_source_moment_degrees(args) -> list[int] | None:
    value = getattr(args, "source_moment_degrees", None)
    if value is None:
        return None
    degrees: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            raise ValueError("--source-moment-degrees contains an empty entry")
        degree = int(text)
        if degree < 0:
            raise ValueError("--source-moment-degrees must be nonnegative")
        degrees.append(degree)
    if not degrees:
        raise ValueError("--source-moment-degrees must contain at least one degree")
    return degrees


def _source_vector(sim, kind: str, field_location: str = "edge") -> np.ndarray:
    kind = str(kind).strip().lower()
    field_location = _normalize_field_location(field_location)
    if kind == "wire":
        vector = _wire_source_vector(sim, field_location)
    elif kind == "dc_conduction_current":
        vector = _dc_conduction_current_vector(sim, field_location)
    elif kind == "dc_total_current":
        vector = _dc_conduction_current_vector(sim, field_location) + _wire_source_vector(
            sim,
            field_location,
        )
    elif kind == "dc_polarization_current":
        vector = _dc_polarization_current_vector(sim, field_location)
    else:
        raise ValueError(f"unsupported source vector: {kind}")
    if np.linalg.norm(vector) == 0.0:
        raise ValueError(f"{kind} source vector has zero norm")
    return np.asarray(vector, dtype=float)


def _wire_source_vector(sim, field_location: str = "edge") -> np.ndarray:
    field_location = _normalize_field_location(field_location)
    n_dofs = sim.mesh.n_faces if field_location == "face" else sim.mesh.n_edges
    vector = np.zeros(n_dofs, dtype=float)
    for source in sim.sources:
        if field_location == "face":
            vector -= source.initial_face_vector(sim.mesh)
        else:
            vector += source.initial_edge_vector(sim.mesh)
    return np.asarray(vector, dtype=float)


def _dc_conduction_current_vector(sim, field_location: str = "edge") -> np.ndarray:
    field_location = _normalize_field_location(field_location)
    if field_location == "face":
        return hj_dc_initial_current_density(sim.mesh, sim.ip_model, sim.sources)
    e0 = sim.initial_electric_field()
    sigma0 = sim.mesh.get_edge_inner_product(sim.ip_model.low_frequency_sigma()).tocsr()
    return np.asarray(sigma0 @ e0, dtype=float)


def _dc_polarization_current_vector(sim, field_location: str = "edge") -> np.ndarray:
    field_location = _normalize_field_location(field_location)
    if not sim.ip_model.terms:
        n_dofs = sim.mesh.n_faces if field_location == "face" else sim.mesh.n_edges
        return np.zeros(n_dofs, dtype=float)
    if field_location == "face":
        e0 = hj_dc_initial_electric_field(sim.mesh, sim.ip_model, sim.sources)
        unit_face = sim.mesh.get_face_inner_product(np.ones(sim.mesh.n_cells)).tocsr()
        unit_diagonal = np.asarray(unit_face.diagonal(), dtype=float)
        vector = np.zeros(sim.mesh.n_faces, dtype=float)
        for term in sim.ip_model.terms:
            delta_face = sim.mesh.get_face_inner_product(term.delta_sigma).tocsr()
            vector += (np.asarray(delta_face.diagonal(), dtype=float) / unit_diagonal) * e0
        return np.asarray(vector, dtype=float)
    e0 = sim.initial_electric_field()
    vector = np.zeros(sim.mesh.n_edges, dtype=float)
    for term in sim.ip_model.terms:
        vector += sim.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ e0
    return np.asarray(vector, dtype=float)


def _source_normalization(sim) -> dict[str, float | None]:
    source_length = None
    delta_sigma = None
    if sim.sources:
        source = sim.sources[0]
        source_length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
        if sim.ip_model.terms:
            cell_index = _source_midpoint_cell_index(sim, source)
            delta_sigma = float(
                sum(term.delta_sigma[cell_index] for term in sim.ip_model.terms)
            )
    return {
        "delta_sigma": delta_sigma,
        "source_length": source_length,
        "mu": float(sim.mu),
    }


def _source_normalization_factor(normalization: dict[str, float | None] | None) -> float:
    if normalization is None:
        raise ValueError(
            "--prescribed-normalized-coefficients requires source normalization"
        )
    missing = [
        key
        for key in ("mu", "delta_sigma", "source_length")
        if normalization.get(key) is None
    ]
    if missing:
        raise ValueError(
            "--prescribed-normalized-coefficients requires "
            + ", ".join(missing)
            + " in source normalization"
        )
    mu = float(normalization["mu"])
    delta_sigma = float(normalization["delta_sigma"])
    source_length = float(normalization["source_length"])
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if delta_sigma <= 0.0:
        raise ValueError("delta_sigma must be positive")
    if source_length <= 0.0:
        raise ValueError("source_length must be positive")
    return mu * delta_sigma * source_length**2


def _source_midpoint_cell_index(sim, source) -> int:
    if hasattr(sim, "_source_midpoint_cell_index"):
        return int(sim._source_midpoint_cell_index(source))
    midpoint = 0.5 * (source.locations[0] + source.locations[-1])
    centers = np.asarray(sim.mesh.cell_centers, dtype=float)
    return int(np.argmin(np.sum((centers - midpoint[None, :]) ** 2, axis=1)))


def _fit_report(fit, *, normalization: dict[str, float | None]) -> dict[str, Any]:
    report = {
        "coefficients": [float(value) for value in fit.coefficients],
        "relative_l2": float(fit.relative_l2),
        "rank": int(fit.rank),
        "singular_values": [float(value) for value in fit.singular_values],
        "design_matrix": {
            "shape": [int(value) for value in fit.design_shape],
            "column_norms": [float(value) for value in fit.column_norms],
            "condition_number": _finite_json_number(fit.condition_number),
            "column_normalized_condition_number": _finite_json_number(
                fit.column_normalized_condition_number
            ),
            "rank_deficient": not np.isfinite(fit.condition_number),
            "column_normalized_rank_deficient": not np.isfinite(
                fit.column_normalized_condition_number
            ),
        },
    }
    if (
        normalization.get("delta_sigma") is not None
        and normalization.get("source_length") is not None
        and normalization.get("mu") is not None
    ):
        report["coefficients_over_mu_delta_l2"] = [
            normalized_source_primary_scale(
                value,
                delta_sigma=float(normalization["delta_sigma"]),
                source_length=float(normalization["source_length"]),
                mu=float(normalization["mu"]),
            )
            for value in fit.coefficients
        ]
    return report


def _normalized_coefficient_table(
    coefficients,
    basis_labels: list[str],
    normalization: dict[str, float | None],
) -> dict[str, Any] | None:
    if (
        normalization.get("delta_sigma") is None
        or normalization.get("source_length") is None
        or normalization.get("mu") is None
    ):
        return None
    normalized = [
        normalized_source_primary_scale(
            value,
            delta_sigma=float(normalization["delta_sigma"]),
            source_length=float(normalization["source_length"]),
            mu=float(normalization["mu"]),
        )
        for value in np.asarray(coefficients, dtype=float).reshape(-1)
    ]
    return _coefficient_table(normalized, basis_labels)


def _static_response_matrix_report(
    receiver_matrix: np.ndarray,
    spatial_vectors: np.ndarray,
    *,
    receiver_indices: np.ndarray,
    component_indices: np.ndarray,
) -> dict[str, Any]:
    """Return rank diagnostics for ``B_R v_s`` before time-history expansion."""

    receiver_matrix = np.asarray(receiver_matrix, dtype=float)
    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if receiver_matrix.ndim != 3 or receiver_matrix.shape[1] != 3:
        raise ValueError("receiver_matrix must have shape (n_locations, 3, n_edges)")
    if spatial_vectors.ndim != 2:
        raise ValueError("spatial_vectors must have shape (n_spatial, n_edges)")
    if spatial_vectors.shape[1] != receiver_matrix.shape[2]:
        raise ValueError("spatial_vectors edge dimension must match receiver_matrix")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= receiver_matrix.shape[0]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")

    static_response = np.einsum("lce,se->slc", receiver_matrix, spatial_vectors)
    design = static_response[:, receiver_indices, component_indices].T
    singular_values = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    column_norms = np.linalg.norm(design, axis=0)
    normalized = design.copy()
    nonzero_columns = column_norms > 0.0
    normalized[:, nonzero_columns] /= column_norms[nonzero_columns]
    normalized_singular_values = np.linalg.svd(
        normalized,
        full_matrices=False,
        compute_uv=False,
    )
    normalized_rank = int(np.linalg.matrix_rank(normalized))
    n_spatial = int(spatial_vectors.shape[0])
    return {
        "shape": [int(design.shape[0]), int(design.shape[1])],
        "rank": rank,
        "rank_deficient": rank < n_spatial,
        "singular_values": [float(value) for value in singular_values],
        "column_norms": [float(value) for value in column_norms],
        "condition_number": _finite_json_number(
            _matrix_condition_number(singular_values, rank=rank, n_columns=n_spatial)
        ),
        "column_normalized_condition_number": _finite_json_number(
            _matrix_condition_number(
                normalized_singular_values,
                rank=normalized_rank,
                n_columns=n_spatial,
            )
        ),
        "column_normalized_rank_deficient": normalized_rank < n_spatial,
    }


def _static_response_matrix_report_from_static_response(
    static_response: np.ndarray,
    *,
    receiver_indices: np.ndarray,
    component_indices: np.ndarray,
) -> dict[str, Any]:
    """Return rank diagnostics for a precomputed ``B_R v_s`` matrix."""

    static_response = np.asarray(static_response, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if static_response.ndim != 3 or static_response.shape[2] != 3:
        raise ValueError("static_response must have shape (n_spatial, n_locations, 3)")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= static_response.shape[1]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")

    design = static_response[:, receiver_indices, component_indices].T
    singular_values = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    column_norms = np.linalg.norm(design, axis=0)
    normalized = design.copy()
    nonzero_columns = column_norms > 0.0
    normalized[:, nonzero_columns] /= column_norms[nonzero_columns]
    normalized_singular_values = np.linalg.svd(
        normalized,
        full_matrices=False,
        compute_uv=False,
    )
    normalized_rank = int(np.linalg.matrix_rank(normalized))
    n_spatial = int(static_response.shape[0])
    return {
        "shape": [int(design.shape[0]), int(design.shape[1])],
        "rank": rank,
        "rank_deficient": rank < n_spatial,
        "singular_values": [float(value) for value in singular_values],
        "column_norms": [float(value) for value in column_norms],
        "condition_number": _finite_json_number(
            _matrix_condition_number(singular_values, rank=rank, n_columns=n_spatial)
        ),
        "column_normalized_condition_number": _finite_json_number(
            _matrix_condition_number(
                normalized_singular_values,
                rank=normalized_rank,
                n_columns=n_spatial,
            )
        ),
        "column_normalized_rank_deficient": normalized_rank < n_spatial,
    }


def _spatial_time_series_report(
    fit,
    *,
    times: np.ndarray,
    spatial_labels: list[str],
    time_steps=None,
    tau: float | None = None,
    history_orders: list[int] | None = None,
    normalization: dict[str, float | None] | None = None,
    prescribed_coefficients=None,
) -> dict[str, Any]:
    coefficients = np.asarray(fit.coefficients, dtype=float)
    times = np.asarray(times, dtype=float)
    if coefficients.shape[0] != times.size:
        raise ValueError("coefficient row count must match times")
    if coefficients.shape[1] != len(spatial_labels):
        raise ValueError("spatial_labels must match coefficient columns")
    report = {
        "coefficient_names": list(spatial_labels),
        "time_count": int(times.size),
        "coefficient_matrix": {
            "shape": [int(value) for value in coefficients.shape],
            "values": [
                [float(value) for value in row]
                for row in coefficients
            ],
        },
        "samples": [
            {
                "time": float(time),
                "coefficients": [float(value) for value in row],
            }
            for time, row in zip(times, coefficients)
        ],
        "projection_fit": {
            "relative_l2": float(fit.relative_l2),
            "rank": int(fit.rank),
            "singular_values": [float(value) for value in fit.singular_values],
            "design_matrix": {
                "shape": [int(value) for value in fit.design_shape],
                "column_norms": [float(value) for value in fit.column_norms],
                "condition_number": _finite_json_number(fit.condition_number),
                "column_normalized_condition_number": _finite_json_number(
                    fit.column_normalized_condition_number
                ),
                "rank_deficient": not np.isfinite(fit.condition_number),
                "column_normalized_rank_deficient": not np.isfinite(
                    fit.column_normalized_condition_number
                ),
            },
        },
    }
    if time_steps is not None and tau is not None and history_orders:
        report["history_basis_fits"] = [
            _spatial_trace_history_fit_report(
                fit_spatial_coefficient_traces_to_history_basis(
                    time_steps,
                    times,
                    coefficients,
                    tau=float(tau),
                    max_order=int(max_order),
                ),
                spatial_labels=spatial_labels,
                max_order=int(max_order),
                normalization=normalization or {},
            )
            for max_order in sorted({int(order) for order in history_orders})
        ]
    if prescribed_coefficients is not None:
        if time_steps is None or tau is None:
            raise ValueError(
                "prescribed trace-history evaluation requires time_steps and tau"
            )
        prescribed = evaluate_spatial_coefficient_traces_with_history_basis(
            time_steps,
            times,
            coefficients,
            tau=float(tau),
            coefficients=prescribed_coefficients,
        )
        report["prescribed_history_basis"] = _spatial_trace_history_fit_report(
            prescribed,
            spatial_labels=spatial_labels,
            max_order=int(prescribed.coefficients.shape[0] - 1),
            normalization=normalization or {},
        )
    return report


def _spatial_trace_history_fit_report(
    fit,
    *,
    spatial_labels: list[str],
    max_order: int,
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    basis_labels = [
        f"{history_label} * {spatial_label}"
        for history_label in fit.basis_labels
        for spatial_label in spatial_labels
    ]
    flat_coefficients = np.asarray(fit.coefficients, dtype=float).reshape(-1)
    report = {
        "max_order": int(max_order),
        "tau": float(fit.tau),
        "basis_labels": list(fit.basis_labels),
        "relative_l2": float(fit.relative_l2),
        "per_trace_relative_l2": [float(value) for value in fit.per_trace_relative_l2],
        "rank": int(fit.rank),
        "singular_values": [float(value) for value in fit.singular_values],
        "coefficient_table": _coefficient_table(flat_coefficients, basis_labels),
        "design_matrix": {
            "shape": [int(value) for value in fit.design_shape],
            "column_norms": [float(value) for value in fit.column_norms],
            "condition_number": _finite_json_number(fit.condition_number),
            "column_normalized_condition_number": _finite_json_number(
                fit.column_normalized_condition_number
            ),
            "rank_deficient": not np.isfinite(fit.condition_number),
            "column_normalized_rank_deficient": not np.isfinite(
                fit.column_normalized_condition_number
            ),
        },
    }
    if (
        normalization.get("delta_sigma") is not None
        and normalization.get("source_length") is not None
        and normalization.get("mu") is not None
    ):
        normalized = [
            normalized_source_primary_scale(
                value,
                delta_sigma=float(normalization["delta_sigma"]),
                source_length=float(normalization["source_length"]),
                mu=float(normalization["mu"]),
            )
            for value in flat_coefficients
        ]
        report["coefficient_table_over_mu_delta_l2"] = _coefficient_table(
            normalized,
            basis_labels,
        )
    return report


def _matrix_condition_number(
    singular_values: np.ndarray,
    *,
    rank: int,
    n_columns: int,
) -> float:
    singular_values = np.asarray(singular_values, dtype=float)
    if n_columns <= 0 or singular_values.size == 0:
        return float("nan")
    if int(rank) < int(n_columns):
        return float("inf")
    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def _per_column_fit_report(
    basis_responses: np.ndarray,
    target: np.ndarray,
    *,
    component_names: list[str],
    original_receiver_indices: np.ndarray,
    local_receiver_indices: np.ndarray,
    component_indices: np.ndarray,
    receiver_locations: np.ndarray,
    normalization: dict[str, float | None],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for column_index, name in enumerate(component_names):
        fit = fit_source_history_coefficients_for_components(
            basis_responses,
            target[:, [column_index]],
            receiver_indices=np.asarray([local_receiver_indices[column_index]], dtype=int),
            component_indices=np.asarray([component_indices[column_index]], dtype=int),
        )
        items.append(
            {
                "component_name": name,
                "receiver_index": int(original_receiver_indices[column_index]),
                "local_receiver_index": int(local_receiver_indices[column_index]),
                "component_index": int(component_indices[column_index]),
                "receiver_location": [
                    float(value) for value in receiver_locations[column_index]
                ],
                "fit": _fit_report(fit, normalization=normalization),
            }
        )
    return items


def _finite_json_number(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _format_coefficients(values, *, max_values: int = 12) -> str:
    values = np.asarray(values, dtype=float).reshape(-1)
    max_values = int(max_values)
    if max_values <= 0:
        raise ValueError("max_values must be positive")
    if values.size <= max_values:
        return ",".join(f"{value:.12g}" for value in values)
    head_count = max_values // 2
    tail_count = max_values - head_count
    head = ",".join(f"{value:.12g}" for value in values[:head_count])
    tail = ",".join(f"{value:.12g}" for value in values[-tail_count:])
    return f"{head},...,{tail} (n={values.size})"


def _coefficient_table(coefficients, basis_labels: list[str]) -> dict[str, Any]:
    values = np.asarray(coefficients, dtype=float).reshape(-1)
    labels = list(basis_labels)
    if values.size != len(labels):
        raise ValueError("basis_labels must have one entry per coefficient")

    split_labels = [label.split(" * ", 1) for label in labels]
    if all(len(parts) == 2 for parts in split_labels):
        row_labels = _unique_in_order(parts[0] for parts in split_labels)
        column_labels = _unique_in_order(parts[1] for parts in split_labels)
        row_index = {label: index for index, label in enumerate(row_labels)}
        column_index = {label: index for index, label in enumerate(column_labels)}
        table = np.full((len(row_labels), len(column_labels)), np.nan, dtype=float)
        for coefficient, parts in zip(values, split_labels):
            table[row_index[parts[0]], column_index[parts[1]]] = coefficient
        return {
            "row_labels": row_labels,
            "column_labels": column_labels,
            "values": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in table
            ],
        }

    return {
        "row_labels": labels,
        "column_labels": ["coefficient"],
        "values": [[float(value)] for value in values],
    }


def _unique_in_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
