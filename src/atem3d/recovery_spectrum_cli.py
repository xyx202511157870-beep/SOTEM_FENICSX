"""CLI for local H/J magnetic-diffusion recovery spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from .config import build_simulation, load_config
from .local_coupling import local_cell_support, source_face_moment_basis
from .magnetic_recovery import face_current_biot_matrix
from .recovery_spectrum import (
    local_magnetic_diffusion_positive_spectrum,
    magnetic_diffusion_driven_response,
    magnetic_diffusion_matrices,
    magnetic_diffusion_mmr_initial_state,
    magnetic_diffusion_modal_coupling,
    project_modal_response_to_source_moments,
    tensor_mesh_cell_submesh,
)
from .source_primary import (
    discrete_debye_history_basis,
    discrete_relaxation_difference_basis,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a local H/J magnetic-diffusion recovery spectrum."
    )
    parser.add_argument("config", type=Path, help="ATEM3D YAML configuration")
    parser.add_argument(
        "--support",
        choices=("cell_indices", "source_receiver"),
        default=None,
        help="How to choose the local cell support",
    )
    parser.add_argument("--cell-indices", nargs="+", type=int)
    parser.add_argument(
        "--field-location",
        choices=("auto", "edge", "face"),
        default="auto",
        help="Dof location used to find active source support",
    )
    parser.add_argument("--source-cell-radius", type=int, default=0)
    parser.add_argument("--receiver-cell-radius", type=int, default=0)
    parser.add_argument("--source-atol", type=float, default=0.0)
    parser.add_argument("--sweep-source-cell-radii", nargs="+", type=int)
    parser.add_argument("--sweep-receiver-cell-radii", nargs="+", type=int)
    parser.add_argument("--sweep-padding", nargs="+", type=int)
    parser.add_argument("--include-modal-coupling", action="store_true")
    parser.add_argument("--include-driven-response", action="store_true")
    parser.add_argument(
        "--driver-tau",
        type=float,
        default=None,
        help="Debye driver tau for --include-driven-response; defaults to the single IP tau",
    )
    parser.add_argument(
        "--driver-kind",
        choices=("debye_decay", "debye_build_up", "relaxation_difference"),
        default="debye_decay",
        help="Time history used to drive --include-driven-response",
    )
    parser.add_argument(
        "--driver-fast-tau",
        type=float,
        default=None,
        help=(
            "Fast tau for --driver-kind relaxation_difference; defaults to "
            "the support mu*sigma*L^2 estimate"
        ),
    )
    parser.add_argument(
        "--driven-source-projection",
        choices=("raw", "charge_conserving"),
        default="raw",
        help="Source current used to force the local driven response",
    )
    parser.add_argument(
        "--driven-initial-state",
        choices=("zero", "charge_conserving_mmr", "global_charge_conserving_mmr"),
        default="zero",
        help=(
            "Initial local H state used by --include-driven-response. "
            "'charge_conserving_mmr' builds a non-fitted MMR state from the "
            "charge-conserving source-moment face currents. "
            "'global_charge_conserving_mmr' builds the same state on the full "
            "mesh and restricts it to the local support."
        ),
    )
    parser.add_argument(
        "--driven-forcing",
        choices=("source_edge_rhs", "global_mmr_steady"),
        default="source_edge_rhs",
        help=(
            "Edge RHS used by --include-driven-response. "
            "'source_edge_rhs' uses C.T M_rho s. "
            "'global_mmr_steady' uses K_local h_global, where h_global is the "
            "full-mesh charge-conserving MMR source-moment state restricted to "
            "the local support."
        ),
    )
    parser.add_argument("--source-moment-degrees", nargs="+", type=int, default=[0, 2])
    parser.add_argument(
        "--modal-receiver-mode",
        choices=("stored_h", "face_current_biot"),
        default="face_current_biot",
    )
    parser.add_argument("--modal-subdivisions", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--max-modes", type=int, default=12)
    parser.add_argument("--dense-dof-limit", type=int, default=2000)
    parser.add_argument(
        "--skip-spectrum",
        action="store_true",
        help="Skip dense eigen-spectrum computation and only build support/driven reports",
    )
    parser.add_argument(
        "--conductivity-model",
        choices=("sigma0", "sigma_infinity"),
        default="sigma0",
        help="Conductivity used in C.T M_rho C for the local recovery operator",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.padding < 0:
        parser.error("--padding must be nonnegative")
    if args.source_cell_radius < 0:
        parser.error("--source-cell-radius must be nonnegative")
    if args.receiver_cell_radius < 0:
        parser.error("--receiver-cell-radius must be nonnegative")
    if args.source_atol < 0.0:
        parser.error("--source-atol must be nonnegative")
    if args.max_modes <= 0:
        parser.error("--max-modes must be positive")
    if args.dense_dof_limit <= 0:
        parser.error("--dense-dof-limit must be positive")
    if args.modal_subdivisions <= 0:
        parser.error("--modal-subdivisions must be positive")
    if args.skip_spectrum and args.include_modal_coupling:
        parser.error("--skip-spectrum cannot be combined with --include-modal-coupling")
    if args.driver_tau is not None and args.driver_tau <= 0.0:
        parser.error("--driver-tau must be positive")
    if args.driver_fast_tau is not None and args.driver_fast_tau <= 0.0:
        parser.error("--driver-fast-tau must be positive")
    for name in ("sweep_source_cell_radii", "sweep_receiver_cell_radii", "sweep_padding"):
        values = getattr(args, name)
        if values is not None and any(value < 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} values must be nonnegative")

    config = load_config(args.config)
    simulation = build_simulation(config)
    support_kind = args.support or ("cell_indices" if args.cell_indices else "source_receiver")
    conductivity = _conductivity_model(simulation.ip_model, args.conductivity_model)
    if _has_sweep(args):
        report = _sweep_report(
            simulation,
            config,
            conductivity,
            support_kind=support_kind,
            args=args,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.output}")
        print(f"sweep cases: {report['sweep']['case_count']}")
        return 0

    cell_indices, support_selection = _support_cell_indices(
        simulation,
        config,
        kind=support_kind,
        cell_indices=args.cell_indices,
        field_location=args.field_location,
        source_cell_radius=args.source_cell_radius,
        receiver_cell_radius=args.receiver_cell_radius,
        source_atol=args.source_atol,
    )
    if args.skip_spectrum:
        support = tensor_mesh_cell_submesh(
            simulation.mesh,
            cell_indices,
            padding=args.padding,
        )
        local = None
        report = _support_only_report(
            requested_cell_indices=cell_indices,
            support_selection=support_selection,
            padding=args.padding,
            conductivity_model=args.conductivity_model,
            support=support,
        )
    else:
        local = local_magnetic_diffusion_positive_spectrum(
            simulation.mesh,
            conductivity,
            cell_indices,
            padding=args.padding,
            max_modes=args.max_modes,
            dense_dof_limit=args.dense_dof_limit,
        )
        support = local.support
        report = _report(
            requested_cell_indices=cell_indices,
            support_selection=support_selection,
            padding=args.padding,
            conductivity_model=args.conductivity_model,
            local=local,
        )
    if args.include_modal_coupling:
        if local is None:
            raise ValueError("modal coupling requires a computed local spectrum")
        report["modal_coupling"] = _modal_coupling_report(
            simulation,
            local,
            conductivity,
            source_moment_degrees=args.source_moment_degrees,
            max_modes=args.max_modes,
            receiver_mode=args.modal_receiver_mode,
            subdivisions=args.modal_subdivisions,
            include_driven_response=args.include_driven_response,
            time_steps=simulation.time_steps,
            driver_tau=_driver_tau(simulation.ip_model, args.driver_tau)
            if args.include_driven_response
            else None,
            driven_source_projection=args.driven_source_projection,
            driven_initial_state=args.driven_initial_state,
            driven_forcing=args.driven_forcing,
            driver_kind=args.driver_kind,
            driver_fast_tau=args.driver_fast_tau,
        )
    elif args.include_driven_response:
        report["driven_response"] = _driven_response_for_support_report(
            simulation,
            support,
            conductivity,
            source_moment_degrees=args.source_moment_degrees,
            receiver_mode=args.modal_receiver_mode,
            subdivisions=args.modal_subdivisions,
            time_steps=simulation.time_steps,
            driver_tau=_driver_tau(simulation.ip_model, args.driver_tau),
            driver_kind=args.driver_kind,
            driver_fast_tau=args.driver_fast_tau,
            source_projection=args.driven_source_projection,
            initial_state=args.driven_initial_state,
            forcing=args.driven_forcing,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {args.output}")
    print(
        "local support: "
        f"{report['support']['n_cells']} cells; "
        f"{report['support']['n_edges']} edge dofs"
    )
    if "spectrum" in report and report["spectrum"]["time_constants"]:
        print(
            "time constants: "
            f"{report['spectrum']['time_constants'][0]:.6e} s"
            f" .. {report['spectrum']['time_constants'][-1]:.6e} s"
        )
    return 0


def _support_cell_indices(
    simulation,
    config: dict,
    *,
    kind: str,
    cell_indices: list[int] | None,
    field_location: str,
    source_cell_radius: int,
    receiver_cell_radius: int,
    source_atol: float,
) -> tuple[list[int], dict]:
    if kind == "cell_indices":
        if not cell_indices:
            raise ValueError("--cell-indices is required when --support=cell_indices")
        return [int(index) for index in cell_indices], {
            "kind": "cell_indices",
        }
    if kind != "source_receiver":
        raise ValueError(f"unsupported support kind: {kind}")
    if len(simulation.sources) != 1:
        raise ValueError("source_receiver support requires exactly one source")

    resolved_location = _field_location(config, field_location)
    source = simulation.sources[0]
    if resolved_location == "face":
        source_vector = source.initial_face_vector(simulation.mesh)
    else:
        source_vector = source.initial_edge_vector(simulation.mesh)
    receiver_locations = _receiver_locations(simulation.receivers)
    support = local_cell_support(
        simulation.mesh,
        source_vector,
        receiver_locations,
        field_location=resolved_location,
        source_cell_radius=source_cell_radius,
        receiver_cell_radius=receiver_cell_radius,
        source_atol=source_atol,
    )
    if support.support_cell_indices.size == 0:
        raise ValueError("source_receiver support selected no cells")
    return [int(index) for index in support.support_cell_indices], {
        "kind": "source_receiver",
        "field_location": resolved_location,
        "source_cell_radius": int(source_cell_radius),
        "receiver_cell_radius": int(receiver_cell_radius),
        "source_atol": float(source_atol),
        "source_dof_indices": [int(index) for index in support.source_dof_indices],
        "source_cell_indices": [int(index) for index in support.source_cell_indices],
        "receiver_cell_indices": [
            [int(index) for index in cells] for cells in support.receiver_cell_indices
        ],
    }


def _field_location(config: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    return "face" if str(config.get("formulation", "eb")).lower() == "hj" else "edge"


def _has_sweep(args) -> bool:
    return any(
        value is not None
        for value in (
            args.sweep_source_cell_radii,
            args.sweep_receiver_cell_radii,
            args.sweep_padding,
        )
    )


def _sweep_report(
    simulation,
    config: dict,
    conductivity: np.ndarray,
    *,
    support_kind: str,
    args,
) -> dict:
    if support_kind != "source_receiver":
        raise ValueError("sweep mode currently requires --support source_receiver")
    cases = []
    source_radii = args.sweep_source_cell_radii or [args.source_cell_radius]
    receiver_radii = args.sweep_receiver_cell_radii or [args.receiver_cell_radius]
    paddings = args.sweep_padding or [args.padding]
    for source_radius in source_radii:
        for receiver_radius in receiver_radii:
            for padding in paddings:
                cells, selection = _support_cell_indices(
                    simulation,
                    config,
                    kind="source_receiver",
                    cell_indices=None,
                    field_location=args.field_location,
                    source_cell_radius=source_radius,
                    receiver_cell_radius=receiver_radius,
                    source_atol=args.source_atol,
                )
                support = tensor_mesh_cell_submesh(
                    simulation.mesh,
                    cells,
                    padding=padding,
                )
                local_sigma = conductivity[support.global_cell_indices]
                case = {
                    "source_cell_radius": int(source_radius),
                    "receiver_cell_radius": int(receiver_radius),
                    "padding": int(padding),
                    "support_selection": selection,
                    "support": _support_report(support),
                    "diffusion_time_estimate": _diffusion_time_estimate(
                        support,
                        local_sigma,
                    ),
                }
                if support.mesh.n_edges > args.dense_dof_limit:
                    case["spectrum_skipped"] = {
                        "reason": "dense_dof_limit",
                        "dense_dof_limit": int(args.dense_dof_limit),
                    }
                else:
                    local = local_magnetic_diffusion_positive_spectrum(
                        simulation.mesh,
                        conductivity,
                        cells,
                        padding=padding,
                        max_modes=args.max_modes,
                        dense_dof_limit=args.dense_dof_limit,
                    )
                    case["spectrum"] = _spectrum_report(local.spectrum)
                cases.append(case)
    return {
        "conductivity_model": args.conductivity_model,
        "sweep": {
            "case_count": len(cases),
            "cases": cases,
        },
    }


def _receiver_locations(receivers) -> np.ndarray:
    if not receivers:
        return np.empty((0, 3), dtype=float)
    return np.asarray([receiver.location for receiver in receivers], dtype=float)


def _conductivity_model(ip_model, name: str) -> np.ndarray:
    sigma = np.asarray(ip_model.sigma_infinity, dtype=float).copy()
    if name == "sigma_infinity":
        return sigma
    for term in ip_model.terms:
        sigma -= np.asarray(term.delta_sigma, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("sigma0 conductivity must be positive for recovery spectrum")
    return sigma


def _modal_coupling_report(
    simulation,
    local,
    global_conductivity: np.ndarray,
    *,
    source_moment_degrees: list[int],
    max_modes: int,
    receiver_mode: str,
    subdivisions: int,
    include_driven_response: bool,
    time_steps,
    driver_tau: float | None,
    driven_source_projection: str,
    driven_initial_state: str,
    driven_forcing: str,
    driver_kind: str,
    driver_fast_tau: float | None,
) -> dict:
    if len(simulation.sources) != 1:
        raise ValueError("modal coupling requires exactly one source")
    source = simulation.sources[0]
    local_source = -source.initial_face_vector(local.support.mesh)
    source_basis = source_face_moment_basis(
        local.support.mesh,
        local_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=source_moment_degrees,
    )
    locations = _receiver_locations(simulation.receivers)
    coupling = magnetic_diffusion_modal_coupling(
        local.support.mesh,
        global_conductivity[local.support.global_cell_indices],
        source_basis.basis_vectors,
        locations,
        max_modes=max_modes,
        receiver_mode=receiver_mode,
        subdivisions=subdivisions,
    )
    report = {
        "receiver_mode": coupling.receiver_mode,
        "source_moment_degrees": [int(degree) for degree in source_moment_degrees],
        "source_basis_labels": list(source_basis.basis_labels),
        "modal_forcing": coupling.modal_forcing.tolist(),
        "modal_receiver_response": coupling.modal_receiver_response.tolist(),
        "source_receiver_response": coupling.source_receiver_response.tolist(),
    }
    if coupling.receiver_mode == "face_current_biot":
        static_response = _face_source_static_response(
            local.support.mesh,
            source_basis.basis_vectors,
            locations,
            subdivisions=subdivisions,
        )
        projection = project_modal_response_to_source_moments(
            static_response,
            coupling.source_receiver_response,
        )
        report["source_moment_projection"] = _source_moment_projection_report(
            projection,
            source_basis_labels=source_basis.basis_labels,
        )
    if include_driven_response:
        if driver_tau is None:
            raise ValueError("driver_tau is required for driven response")
        resolved_fast_tau = _driver_fast_tau(
            local.support,
            global_conductivity[local.support.global_cell_indices],
            driver_fast_tau,
            driver_kind=driver_kind,
        )
        driver = _driver_history(
            time_steps,
            driver_tau=driver_tau,
            driver_kind=driver_kind,
            driver_fast_tau=resolved_fast_tau,
        )
        initial_state = _driven_initial_state(
            local.support,
            global_conductivity[local.support.global_cell_indices],
            source_basis.basis_vectors,
            kind=driven_initial_state,
            simulation=simulation,
            global_conductivity=global_conductivity,
            source_moment_degrees=source_moment_degrees,
        )
        forcing_vectors = _driven_forcing_vectors(
            local.support,
            global_conductivity[local.support.global_cell_indices],
            kind=driven_forcing,
            simulation=simulation,
            global_conductivity=global_conductivity,
            source_moment_degrees=source_moment_degrees,
        )
        driven = magnetic_diffusion_driven_response(
            local.support.mesh,
            global_conductivity[local.support.global_cell_indices],
            source_basis.basis_vectors,
            locations,
            time_steps=time_steps,
            driver_values=driver,
            receiver_mode=receiver_mode,
            subdivisions=subdivisions,
            source_projection=driven_source_projection,
            initial_state=initial_state,
            initial_state_kind=driven_initial_state,
            forcing_vectors=forcing_vectors,
            forcing_kind=driven_forcing,
        )
        report["driven_response"] = _driven_response_report(
            driven,
            driver_tau=driver_tau,
            driver_kind=driver_kind,
            driver_fast_tau=resolved_fast_tau,
            source_basis_labels=source_basis.basis_labels,
        )
    return report


def _driven_response_for_support_report(
    simulation,
    support,
    global_conductivity: np.ndarray,
    *,
    source_moment_degrees: list[int],
    receiver_mode: str,
    subdivisions: int,
    time_steps,
    driver_tau: float,
    driver_kind: str,
    driver_fast_tau: float | None,
    source_projection: str,
    initial_state: str,
    forcing: str,
) -> dict:
    if len(simulation.sources) != 1:
        raise ValueError("driven response requires exactly one source")
    source = simulation.sources[0]
    local_source = -source.initial_face_vector(support.mesh)
    source_basis = source_face_moment_basis(
        support.mesh,
        local_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=source_moment_degrees,
    )
    resolved_fast_tau = _driver_fast_tau(
        support,
        global_conductivity[support.global_cell_indices],
        driver_fast_tau,
        driver_kind=driver_kind,
    )
    driver = _driver_history(
        time_steps,
        driver_tau=driver_tau,
        driver_kind=driver_kind,
        driver_fast_tau=resolved_fast_tau,
    )
    initial_state_values = _driven_initial_state(
        support,
        global_conductivity[support.global_cell_indices],
        source_basis.basis_vectors,
        kind=initial_state,
        simulation=simulation,
        global_conductivity=global_conductivity,
        source_moment_degrees=source_moment_degrees,
    )
    forcing_vectors = _driven_forcing_vectors(
        support,
        global_conductivity[support.global_cell_indices],
        kind=forcing,
        simulation=simulation,
        global_conductivity=global_conductivity,
        source_moment_degrees=source_moment_degrees,
    )
    driven = magnetic_diffusion_driven_response(
        support.mesh,
        global_conductivity[support.global_cell_indices],
        source_basis.basis_vectors,
        _receiver_locations(simulation.receivers),
        time_steps=time_steps,
        driver_values=driver,
        receiver_mode=receiver_mode,
        subdivisions=subdivisions,
        source_projection=source_projection,
        initial_state=initial_state_values,
        initial_state_kind=initial_state,
        forcing_vectors=forcing_vectors,
        forcing_kind=forcing,
    )
    return _driven_response_report(
        driven,
        driver_tau=driver_tau,
        driver_kind=driver_kind,
        driver_fast_tau=resolved_fast_tau,
        source_basis_labels=source_basis.basis_labels,
    )


def _face_source_static_response(
    mesh,
    source_vectors: np.ndarray,
    receiver_locations: np.ndarray,
    *,
    subdivisions: int,
) -> np.ndarray:
    receiver_matrix = face_current_biot_matrix(
        mesh,
        receiver_locations,
        subdivisions=subdivisions,
    )
    return np.einsum("lcf,sf->slc", receiver_matrix, source_vectors)


def _driven_initial_state(
    support,
    conductivity: np.ndarray,
    source_vectors: np.ndarray,
    *,
    kind: str,
    simulation=None,
    global_conductivity: np.ndarray | None = None,
    source_moment_degrees: list[int] | None = None,
):
    kind = str(kind).strip().lower()
    if kind == "zero":
        return None
    if kind == "charge_conserving_mmr":
        return magnetic_diffusion_mmr_initial_state(
            support.mesh,
            source_vectors,
            conductivity=conductivity,
            source_projection="charge_conserving",
        )
    if kind == "global_charge_conserving_mmr":
        if simulation is None or global_conductivity is None or source_moment_degrees is None:
            raise ValueError(
                "global_charge_conserving_mmr initial state requires simulation, "
                "global_conductivity, and source_moment_degrees"
            )
        if len(simulation.sources) != 1:
            raise ValueError("global_charge_conserving_mmr requires exactly one source")
        source = simulation.sources[0]
        global_source = -source.initial_face_vector(simulation.mesh)
        global_basis = source_face_moment_basis(
            simulation.mesh,
            global_source,
            start=source.locations[0],
            end=source.locations[-1],
            degrees=source_moment_degrees,
        )
        global_state = magnetic_diffusion_mmr_initial_state(
            simulation.mesh,
            global_basis.basis_vectors,
            conductivity=global_conductivity,
            source_projection="charge_conserving",
        )
        restricted = np.asarray(global_state[support.global_edge_indices], dtype=float)
        if restricted.shape != (support.mesh.n_edges, np.asarray(source_vectors).shape[0]):
            raise ValueError(
                "restricted global MMR state does not match the local source-vector count"
            )
        return restricted
    raise ValueError(f"unsupported driven initial state: {kind}")


def _driven_forcing_vectors(
    support,
    conductivity: np.ndarray,
    *,
    kind: str,
    simulation=None,
    global_conductivity: np.ndarray | None = None,
    source_moment_degrees: list[int] | None = None,
):
    kind = str(kind).strip().lower()
    if kind == "source_edge_rhs":
        return None
    if kind == "global_mmr_steady":
        global_state = _driven_initial_state(
            support,
            conductivity,
            np.empty((len(source_moment_degrees or []), support.mesh.n_faces)),
            kind="global_charge_conserving_mmr",
            simulation=simulation,
            global_conductivity=global_conductivity,
            source_moment_degrees=source_moment_degrees,
        )
        stiffness, _ = magnetic_diffusion_matrices(support.mesh, conductivity)
        return np.asarray(stiffness @ global_state, dtype=float)
    raise ValueError(f"unsupported driven forcing: {kind}")


def _source_moment_projection_report(
    projection,
    *,
    source_basis_labels: list[str],
) -> dict:
    return {
        "coefficient_axes": [
            "mode_index",
            "source_drive",
            "source_moment",
        ],
        "source_drive_labels": list(source_basis_labels),
        "source_moment_labels": list(source_basis_labels),
        "coefficients": projection.coefficients.tolist(),
        "relative_l2": projection.relative_l2.tolist(),
        "aggregate_relative_l2": float(projection.aggregate_relative_l2),
        "rank": int(projection.rank),
        "singular_values": [float(value) for value in projection.singular_values],
        "design_matrix": {
            "shape": [int(value) for value in projection.design_shape],
            "column_norms": [
                float(value) for value in projection.column_norms
            ],
            "condition_number": float(projection.condition_number),
            "column_normalized_condition_number": float(
                projection.column_normalized_condition_number
            ),
        },
    }


def _driven_response_report(
    driven,
    *,
    driver_tau: float,
    driver_kind: str,
    driver_fast_tau: float | None,
    source_basis_labels: list[str],
) -> dict:
    report = {
        "driver_tau": float(driver_tau),
        "driver_kind": str(driver_kind),
        "source_projection": driven.source_projection,
        "initial_state_kind": driven.initial_state_kind,
        "forcing_kind": driven.forcing_kind,
        "times": [float(value) for value in driven.times],
        "driver_values": [float(value) for value in driven.driver_values],
        "receiver_response": driven.receiver_response.tolist(),
        "source_moment_projection": _source_moment_projection_report(
            driven.source_moment_projection,
            source_basis_labels=source_basis_labels,
        ),
    }
    if driver_fast_tau is not None:
        report["driver_fast_tau"] = float(driver_fast_tau)
    return report


def _driver_tau(ip_model, value: float | None) -> float:
    if value is not None:
        return float(value)
    taus = sorted({float(term.tau) for term in getattr(ip_model, "terms", [])})
    if len(taus) != 1:
        raise ValueError(
            "--driver-tau is required when the model does not have exactly one IP tau"
        )
    return float(taus[0])


def _driver_history(
    time_steps,
    *,
    driver_tau: float,
    driver_kind: str,
    driver_fast_tau: float | None,
) -> np.ndarray:
    if driver_kind == "debye_decay":
        basis = discrete_debye_history_basis(
            time_steps,
            tau=driver_tau,
            max_order=0,
        )
        return np.asarray(basis.values[:, 0], dtype=float)
    if driver_kind == "debye_build_up":
        basis = discrete_debye_history_basis(
            time_steps,
            tau=driver_tau,
            max_order=0,
        )
        values = np.asarray(basis.values[:, 0], dtype=float)
        return 1.0 - values
    if driver_kind == "relaxation_difference":
        if driver_fast_tau is None:
            raise ValueError("relaxation_difference requires driver_fast_tau")
        basis = discrete_relaxation_difference_basis(
            time_steps,
            slow_tau=driver_tau,
            fast_tau=driver_fast_tau,
        )
        return np.asarray(basis.values, dtype=float)
    raise ValueError(
        "driver_kind must be 'debye_decay', 'debye_build_up', "
        "or 'relaxation_difference'"
    )


def _driver_fast_tau(
    support,
    conductivity: np.ndarray,
    value: float | None,
    *,
    driver_kind: str,
) -> float | None:
    if driver_kind != "relaxation_difference":
        return None
    if value is not None:
        return float(value)
    return _diffusion_time_estimate(support, conductivity)


def _report(
    *,
    requested_cell_indices: list[int],
    support_selection: dict,
    padding: int,
    conductivity_model: str,
    local,
) -> dict:
    support = local.support
    spectrum = local.spectrum
    return {
        "conductivity_model": conductivity_model,
        "requested_cell_indices": [int(index) for index in requested_cell_indices],
        "support_selection": support_selection,
        "padding": int(padding),
        "support": _support_report(support),
        "spectrum": _spectrum_report(spectrum),
    }


def _support_only_report(
    *,
    requested_cell_indices: list[int],
    support_selection: dict,
    padding: int,
    conductivity_model: str,
    support,
) -> dict:
    return {
        "conductivity_model": conductivity_model,
        "requested_cell_indices": [int(index) for index in requested_cell_indices],
        "support_selection": support_selection,
        "padding": int(padding),
        "support": _support_report(support),
        "spectrum_skipped": {"reason": "requested"},
    }


def _support_report(support) -> dict:
    return {
        "global_cell_indices": [
            int(index) for index in support.global_cell_indices
        ],
        "global_face_indices": [
            int(index) for index in support.global_face_indices
        ],
        "global_edge_indices": [
            int(index) for index in support.global_edge_indices
        ],
        "ijk_min": [int(index) for index in support.ijk_min],
        "ijk_max": [int(index) for index in support.ijk_max],
        "n_cells": int(support.mesh.n_cells),
        "n_faces": int(support.mesh.n_faces),
        "n_edges": int(support.mesh.n_edges),
        "origin": [float(value) for value in support.mesh.origin],
        "h": [
            [float(value) for value in np.asarray(widths, dtype=float)]
            for widths in support.mesh.h
        ],
        "extents": [
            float(np.sum(np.asarray(widths, dtype=float)))
            for widths in support.mesh.h
        ],
    }


def _spectrum_report(spectrum) -> dict:
    return {
        "eigenvalues": [float(value) for value in spectrum.eigenvalues],
        "time_constants": [
            float(value) for value in spectrum.time_constants
        ],
        "eigenvalue_filter": {
            "eigenvalue_floor": float(spectrum.eigenvalue_floor),
            "discarded_count": int(spectrum.discarded_count),
            "raw_eigenvalue_min": float(spectrum.raw_eigenvalue_min),
            "raw_eigenvalue_max": float(spectrum.raw_eigenvalue_max),
            "last_discarded_eigenvalue": (
                None
                if spectrum.last_discarded_eigenvalue is None
                else float(spectrum.last_discarded_eigenvalue)
            ),
            "first_kept_eigenvalue": (
                None
                if spectrum.first_kept_eigenvalue is None
                else float(spectrum.first_kept_eigenvalue)
            ),
        },
    }


def _diffusion_time_estimate(support, conductivity: np.ndarray) -> float:
    extents = [
        float(np.sum(np.asarray(widths, dtype=float))) for widths in support.mesh.h
    ]
    return float(mu_0 * float(np.mean(conductivity)) * max(extents) ** 2)


if __name__ == "__main__":
    raise SystemExit(main())
