"""Canonical Lei/Song benchmark adapter for the ATEM3D EB-Debye solver.

The adapter deliberately contains no reference-response code.  It only maps the
immutable ``BenchmarkCase`` description onto the existing SimPEG/Discretize-
compatible ATEM3D configuration and solver path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import build_simulation
from .fit import pelton_resistivity_to_conductivity
from .sotem_benchmark import BenchmarkCase


_COMPONENTS = ("Ex", "Ey", "Hz", "dBzdt")
_SPATIAL_LEVELS = {
    "S0": {"source_cell_m": 40.0, "receiver_cell_m": 20.0},
    "S1": {"source_cell_m": 20.0, "receiver_cell_m": 10.0},
    "S2": {"source_cell_m": 10.0, "receiver_cell_m": 5.0},
}
_BOUNDARY_LEVELS = {"B0": 25_000.0, "B1": 50_000.0, "B2": 100_000.0}
_PADDING_GROWTH = 1.4
_FIT_FREQUENCIES = np.logspace(-3.0, 4.0, 81)
_FIT_TERMS = 16
_MATERIAL_RELATIVE_L2_LIMIT = 0.01
_DC_ABSOLUTE_TOLERANCE = 1.0e-14
_COORDINATE_TRANSFORM = {
    "public_coordinates": "z_down",
    "internal_coordinates": "z_up",
    "position_mapping": "(x, y, z_up) = (x, y, -z_down)",
    "output_component_signs": {
        "Ex": 1.0,
        "Ey": 1.0,
        "Hz": 1.0,
        "dBzdt": 1.0,
    },
    "Hz_vector_type": "axial",
    "vertical_axial_mapping": "Hz_up = Hz_down; dBzdt_up = dBzdt_down",
}


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def build_internal_time_steps(
    outputs: Any,
    *,
    substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every output interval into equal internal time steps."""

    output_array = np.asarray(outputs, dtype=float)
    if (
        output_array.ndim != 1
        or output_array.size == 0
        or not np.all(np.isfinite(output_array))
        or np.any(output_array <= 0.0)
        or np.any(np.diff(output_array) <= 0.0)
    ):
        raise ValueError(
            "outputs must be a non-empty finite positive strictly increasing 1-D array"
        )
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps <= 0:
        raise ValueError("substeps must be a positive integer excluding bool")

    interval_widths = np.diff(np.r_[0.0, output_array])
    steps = np.concatenate(
        [np.full(int(substeps), interval / int(substeps)) for interval in interval_widths]
    )
    indices = np.arange(int(substeps) - 1, steps.size, int(substeps), dtype=np.int64)
    return (
        _readonly_array(steps, dtype=np.float64),
        _readonly_array(indices, dtype=np.int64),
    )


def _validate_case(case: BenchmarkCase) -> None:
    if not isinstance(case, BenchmarkCase):
        raise TypeError("case must be a BenchmarkCase")
    if case.coordinates != "z_down":
        raise ValueError("benchmark coordinates must be z_down")
    if tuple(case.components) != _COMPONENTS:
        raise ValueError("benchmark components must be [Ex, Ey, Hz, dBzdt] in that order")


def _validate_level(value: str, levels: dict[str, Any], name: str) -> str:
    if value not in levels:
        raise ValueError(f"{name} must be one of {sorted(levels)}")
    return value


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_anchors(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    by_location: dict[float, float] = {}
    for location, target_width in values:
        location = float(location)
        target_width = float(target_width)
        by_location[location] = min(target_width, by_location.get(location, target_width))
    return sorted(by_location.items())


def _padding_widths(length: float, local_width: float) -> np.ndarray:
    widths = []
    width = float(local_width)
    while sum(widths) < length:
        widths.append(width)
        width *= _PADDING_GROWTH
    result = np.asarray(widths, dtype=float)
    return result * (float(length) / float(np.sum(result)))


def _bridge_widths(
    length: float,
    left_width: float,
    right_width: float,
) -> np.ndarray:
    count = 1
    while True:
        indices = np.arange(count, dtype=float)
        raw = np.minimum(
            float(left_width) * _PADDING_GROWTH**indices,
            float(right_width) * _PADDING_GROWTH**indices[::-1],
        )
        if float(np.sum(raw)) >= float(length):
            return raw * (float(length) / float(np.sum(raw)))
        count += 1


def _graded_axis(
    anchors: list[tuple[float, float]],
    *,
    lower: float,
    upper: float,
) -> tuple[list[float], float]:
    anchors = _unique_anchors(anchors)
    if not anchors or not lower < anchors[0][0] < anchors[-1][0] < upper:
        raise ValueError("mesh anchors must lie strictly inside ordered bounds")

    widths: list[float] = []
    left_padding = _padding_widths(anchors[0][0] - lower, anchors[0][1])
    widths.extend(left_padding[::-1].tolist())
    for (left, left_width), (right, right_width) in zip(anchors[:-1], anchors[1:]):
        widths.extend(_bridge_widths(right - left, left_width, right_width).tolist())
    right_padding = _padding_widths(upper - anchors[-1][0], anchors[-1][1])
    widths.extend(right_padding.tolist())
    # Reconstruct the origin from the first anchor and the exact generated
    # prefix.  This makes canonical anchor nodes (notably z=0) numerically
    # stable instead of inheriting a long-padding summation residual.
    left_prefix = float(np.cumsum(left_padding[::-1])[-1])
    origin = float(anchors[0][0] - left_prefix)
    nodes = origin + np.r_[0.0, np.cumsum(np.asarray(widths, dtype=float))]
    if not np.all(np.diff(nodes) > 0.0):
        raise RuntimeError("generated mesh axis contains non-positive widths")
    if not np.isclose(nodes[-1], upper, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("generated mesh axis does not terminate at requested bound")
    return [float(value) for value in widths], origin


def _snap_axis_nodes(
    widths: list[float],
    origin: float,
    anchors: tuple[float, ...],
) -> list[float]:
    """Apply roundoff-sized width corrections so selected nodes are bit-exact."""

    result = np.asarray(widths, dtype=float).copy()
    for anchor in sorted(anchors):
        for _ in range(8):
            # TensorMesh uses cumulative summation over ``[origin, *widths]``.
            # Mirroring that exact operation is required for bit-exact nodes.
            nodes = np.cumsum(np.r_[float(origin), result])
            index = int(np.argmin(np.abs(nodes - float(anchor))))
            if nodes[index] == float(anchor):
                break
            if index == 0:
                raise RuntimeError("cannot snap the mesh origin through a cell width")
            residual = float(anchor) - float(nodes[index])
            updated = float(result[index - 1] + residual)
            if updated == result[index - 1]:
                updated = float(
                    np.nextafter(
                        result[index - 1],
                        np.inf if residual > 0.0 else -np.inf,
                    )
                )
            if updated <= 0.0:
                raise RuntimeError("anchor correction produced a non-positive width")
            result[index - 1] = updated
        else:
            raise RuntimeError(f"could not make mesh anchor {anchor} bit-exact")
    return [float(value) for value in result]


def _z_up_point(point_down: tuple[float, float, float]) -> list[float]:
    return [float(point_down[0]), float(point_down[1]), -float(point_down[2])]


def _build_mesh_dict(
    case: BenchmarkCase,
    spatial_level: str,
    boundary_level: str,
) -> dict[str, Any]:
    spatial = _SPATIAL_LEVELS[spatial_level]
    source_h = spatial["source_cell_m"]
    receiver_h = spatial["receiver_cell_m"]
    extent = _BOUNDARY_LEVELS[boundary_level]

    x_anchors = [
        (case.source_start_down[0], source_h),
        (case.source_end_down[0], source_h),
        (case.receiver_down[0], receiver_h),
    ]
    y_anchors = [
        (case.source_start_down[1], source_h),
        (case.source_end_down[1], source_h),
        (case.receiver_down[1], receiver_h),
    ]
    # Two horizontal anchor locations are required by _graded_axis.  Add a
    # physically neutral nearby refinement node when all y coordinates agree.
    if len({value for value, _ in y_anchors}) == 1:
        y_anchors.append((y_anchors[0][0] + source_h, source_h))

    z_anchors = [(0.0, source_h), (300.0, source_h)]
    hx, x0 = _graded_axis(x_anchors, lower=-extent, upper=extent)
    hy, y0 = _graded_axis(y_anchors, lower=-extent, upper=extent)
    hz_down, z0_down = _graded_axis(z_anchors, lower=-extent, upper=extent)
    hz_down = _snap_axis_nodes(hz_down, z0_down, (0.0, 300.0))
    z_down_upper = z0_down + float(np.sum(hz_down))
    hz = list(reversed(hz_down))
    z0 = -z_down_upper
    hz = _snap_axis_nodes(hz, z0, (-300.0, 0.0))

    counts = {"x": len(hx), "y": len(hy), "z": len(hz)}
    internal_bounds = {
        "x": [x0, x0 + float(np.sum(hx))],
        "y": [y0, y0 + float(np.sum(hy))],
        "z": [z0, z0 + float(np.sum(hz))],
    }
    public_bounds = {
        "x": copy.deepcopy(internal_bounds["x"]),
        "y": copy.deepcopy(internal_bounds["y"]),
        "z": [-internal_bounds["z"][1], -internal_bounds["z"][0]],
    }
    hash_input = {"hx": hx, "hy": hy, "hz": hz, "origin": [x0, y0, z0]}
    mesh_hash = _hash_payload(hash_input)
    n_cells = counts["x"] * counts["y"] * counts["z"]
    nx, ny, nz = counts["x"], counts["y"], counts["z"]
    n_edges = (
        nx * (ny + 1) * (nz + 1)
        + (nx + 1) * ny * (nz + 1)
        + (nx + 1) * (ny + 1) * nz
    )
    metadata = {
        "spatial_level": spatial_level,
        "boundary_level": boundary_level,
        "source_cell_m": source_h,
        "receiver_cell_m": receiver_h,
        "nominal_far_extent_m": extent,
        "padding_growth_max": _PADDING_GROWTH,
        "axis_cell_counts": counts,
        "bounds_m": internal_bounds,
        "public_bounds_z_down_m": public_bounds,
        "public_coordinate_system": "z_down",
        "internal_coordinate_system": "z_up",
        "n_cells": n_cells,
        "n_edges": n_edges,
        "mesh_hash": mesh_hash,
    }
    return {
        **hash_input,
        "mesh_hash": mesh_hash,
        "metadata": metadata,
    }


def _song_material_fit(polarization: Any) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rho0 = float(polarization["rho0_ohm_m"])
    chargeability = float(polarization["m"])
    tau = float(polarization["tau_s"])
    exponent = float(polarization["c"])
    sigma_dc = 1.0 / rho0
    sigma_infinity = 1.0 / (rho0 * (1.0 - chargeability))
    delta_sum_required = sigma_infinity - sigma_dc

    if chargeability == 0.0:
        metadata = {
            "rho0_ohm_m": rho0,
            "m": chargeability,
            "tau_s": tau,
            "c": exponent,
            "fit_frequency_min_hz": 1.0e-3,
            "fit_frequency_max_hz": 1.0e4,
            "fit_frequency_count": 81,
            "fit_term_count": 0,
            "relative_l2": 0.0,
            "relative_l2_limit": _MATERIAL_RELATIVE_L2_LIMIT,
            "sigma_dc": sigma_dc,
            "sigma_infinity": sigma_infinity,
            "delta_sum": 0.0,
            "dc_residual": 0.0,
            "dc_absolute_tolerance": _DC_ABSOLUTE_TOLERANCE,
            "material_gate_pass": True,
            "debye_terms": [],
        }
        return [], metadata

    frequencies = _FIT_FREQUENCIES.copy()
    tau_grid = np.logspace(
        np.log10(1.0 / (2.0 * np.pi * frequencies.max()) / 10.0),
        np.log10(1.0 / (2.0 * np.pi * frequencies.min()) * 10.0),
        _FIT_TERMS,
    )
    basis = 1.0 / (
        1.0 + 1j * 2.0 * np.pi * frequencies[:, None] * tau_grid[None, :]
    )
    target = pelton_resistivity_to_conductivity(
        frequencies,
        rho0,
        chargeability,
        tau,
        exponent,
    )
    rhs = sigma_infinity - target
    design = np.vstack([basis.real, basis.imag])
    data = np.r_[rhs.real, rhs.imag]

    def objective(delta: np.ndarray) -> float:
        residual = design @ delta - data
        return 0.5 * float(residual @ residual)

    def gradient(delta: np.ndarray) -> np.ndarray:
        return design.T @ (design @ delta - data)

    positive_floor = delta_sum_required * 1.0e-12
    result = minimize(
        objective,
        np.full(_FIT_TERMS, delta_sum_required / _FIT_TERMS),
        jac=gradient,
        method="SLSQP",
        bounds=[(positive_floor, None)] * _FIT_TERMS,
        constraints={
            "type": "eq",
            "fun": lambda delta: float(np.sum(delta) - delta_sum_required),
            "jac": lambda delta: np.ones_like(delta),
        },
        options={"ftol": 1.0e-18, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"DC-constrained Song Debye fit failed: {result.message}")

    delta = np.asarray(result.x, dtype=float)
    fitted = sigma_infinity - basis @ delta
    relative_l2 = float(
        np.linalg.norm(np.r_[fitted.real - target.real, fitted.imag - target.imag])
        / np.linalg.norm(np.r_[target.real, target.imag])
    )
    delta_sum = float(np.sum(delta))
    dc_residual = float((sigma_infinity - delta_sum) - sigma_dc)
    material_gate_pass = bool(
        relative_l2 <= _MATERIAL_RELATIVE_L2_LIMIT
        and abs(dc_residual) <= _DC_ABSOLUTE_TOLERANCE
        and np.all(delta > 0.0)
        and delta.size == _FIT_TERMS
    )
    metadata = {
        "rho0_ohm_m": rho0,
        "m": chargeability,
        "tau_s": tau,
        "c": exponent,
        "fit_frequency_min_hz": 1.0e-3,
        "fit_frequency_max_hz": 1.0e4,
        "fit_frequency_count": 81,
        "fit_term_count": int(delta.size),
        "relative_l2": relative_l2,
        "relative_l2_limit": _MATERIAL_RELATIVE_L2_LIMIT,
        "sigma_dc": sigma_dc,
        "sigma_infinity": sigma_infinity,
        "delta_sum": delta_sum,
        "dc_residual": dc_residual,
        "dc_absolute_tolerance": _DC_ABSOLUTE_TOLERANCE,
        "minimum_delta_sigma": float(np.min(delta)),
        "material_gate_pass": material_gate_pass,
    }
    if not material_gate_pass:
        raise ValueError(
            "Song material fit failed adapter gate: "
            f"relative_l2={relative_l2}, dc_residual={dc_residual}"
        )
    terms = [
        {"tau": float(term_tau), "delta_sigma": float(term_delta)}
        for term_tau, term_delta in zip(tau_grid, delta)
    ]
    metadata["debye_terms"] = copy.deepcopy(terms)
    return terms, metadata


def _earth_layers(
    case: BenchmarkCase,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    air = {
        "top": float("inf"),
        "bottom": 0.0,
        "sigma_infinity": 1.0 / float(case.rho_air_ohm_m),
        "debye_terms": [],
        "material": "air",
    }
    material_fit = None

    if "rho_ohm_m" in case.earth:
        earth_sigma = 1.0 / float(case.earth["rho_ohm_m"])
        return [
            air,
            {
                "top": 0.0,
                "bottom": float("-inf"),
                "sigma_infinity": earth_sigma,
                "debye_terms": [],
                "material": "earth_noip",
            },
        ], None

    earth_layers = list(case.earth["layers"])
    layers: list[dict[str, Any]] = [air]
    for index, earth in enumerate(earth_layers):
        top_down = float(earth["top_m"])
        bottom_down = (
            float("inf")
            if earth["bottom_m"] is None
            else float(earth["bottom_m"])
        )
        top = -top_down
        bottom = -bottom_down
        sigma = 1.0 / float(earth["rho_ohm_m"])
        layer = {
            "top": top,
            "bottom": bottom,
            "sigma_infinity": sigma,
            "debye_terms": [],
            "material": "earth_noip",
        }
        if variant == "ip" and index == 0:
            polarization = case.polarization
            if polarization is None:
                raise ValueError("IP variant requires a polarizable benchmark case")
            if (
                float(polarization["top_m"]) != top_down
                or float(polarization["bottom_m"]) != bottom_down
            ):
                raise ValueError("polarization interval must match the first Song earth layer")
            terms, material_fit = _song_material_fit(polarization)
            layer["sigma_infinity"] = material_fit["sigma_infinity"]
            layer["debye_terms"] = terms
            layer["material"] = "song_pelton_debye" if terms else "earth_noip"
        layers.append(layer)
    return layers, material_fit


def build_benchmark_config(
    case: BenchmarkCase,
    *,
    variant: str,
    spatial_level: str,
    boundary_level: str,
    substeps: int,
) -> dict[str, Any]:
    """Build a deterministic plain config for the canonical EB-Debye solver."""

    _validate_case(case)
    if variant not in {"noip", "ip"}:
        raise ValueError("variant must be 'noip' or 'ip'")
    if variant == "ip" and case.polarization is None:
        raise ValueError("IP variant requires a polarizable benchmark case")
    spatial_level = _validate_level(spatial_level, _SPATIAL_LEVELS, "spatial_level")
    boundary_level = _validate_level(boundary_level, _BOUNDARY_LEVELS, "boundary_level")
    steps, output_indices = build_internal_time_steps(
        case.observation_times,
        substeps=substeps,
    )
    mesh = _build_mesh_dict(case, spatial_level, boundary_level)
    layers, material_fit = _earth_layers(case, variant)
    time_hash = _hash_payload(
        {"time_steps": steps.tolist(), "output_indices": output_indices.tolist()}
    )
    return {
        "coordinate_system": "z_up",
        "formulation": "eb",
        "initial_magnetic_field": "ampere",
        "solver": {
            "type": "petsc_ams",
            "tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "maxiter": 2000,
            "preconditioner": "hypre_ams",
            "ksp_type": "gmres",
            "residual_replacement_steps": 2,
        },
        "initialization_solver": {
            "type": "petsc_hypre",
            "tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "maxiter": 2000,
            "residual_replacement_steps": 2,
            "dc_ksp_type": "cg",
            "dc_preconditioner": "hypre_boomeramg",
            "magnetic_ksp_type": "gmres",
            "magnetic_preconditioner": "hypre_ams",
        },
        "mesh": mesh,
        "mesh_hash": mesh["mesh_hash"],
        "time_hash": time_hash,
        "model": {
            "coordinate_system": "z_up",
            "layers": layers,
            "require_layer_boundary_alignment": True,
            "layer_boundary_tolerance": 1.0e-10,
        },
        "source": {
            "start": _z_up_point(case.source_start_down),
            "end": _z_up_point(case.source_end_down),
            "current": float(case.current_a),
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "receivers": [
            {
                "location": _z_up_point(case.receiver_down),
                "component": component,
                "type": "point",
            }
            for component in _COMPONENTS
        ],
        "time_steps": steps.tolist(),
        "boundary": {"kind": "none", "thickness_cells": 0},
        "magnetic_receiver_mode": "stored_b",
        "adapter_metadata": {
            "case_id": case.case_id,
            "variant": variant,
            "components": list(_COMPONENTS),
            "output_indices": output_indices.tolist(),
            "observation_times": np.asarray(case.observation_times, dtype=float).tolist(),
            "mesh_hash": mesh["mesh_hash"],
            "time_hash": time_hash,
            "material_fit": material_fit,
            "coordinate_transform": copy.deepcopy(_COORDINATE_TRANSFORM),
            "initial_magnetic_field": "ampere",
            "transient_solver": "petsc_gmres_hypre_ams",
            "initialization_solver": {
                "dc_electric": "petsc_cg_hypre_boomeramg",
                "ampere_magnetic": "petsc_gmres_hypre_ams",
            },
            "resource_note": (
                "Canonical PETSc/HYPRE initialization uses independent external residual "
                "and physical-balance gates after the prior sparse-direct OOM; the "
                "full-scale S0 case has not yet rerun with this initialization path."
            ),
        },
    }


def paired_model_dicts(
    case: BenchmarkCase,
    *,
    spatial_level: str,
    boundary_level: str,
    substeps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return independent no-IP/IP configs for a polarizable Song case."""

    _validate_case(case)
    if case.polarization is None:
        raise ValueError("paired_model_dicts requires a polarizable benchmark case")
    noip = build_benchmark_config(
        case,
        variant="noip",
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=substeps,
    )
    ip = build_benchmark_config(
        case,
        variant="ip",
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=substeps,
    )
    # The separate builds are already independent.  Deep copies make that API
    # guarantee explicit even if construction is internally memoized later.
    return copy.deepcopy(noip), copy.deepcopy(ip)


def run_simpeg_benchmark(
    case: BenchmarkCase,
    *,
    variant: str,
    spatial_level: str = "S0",
    boundary_level: str = "B0",
    substeps: int = 1,
) -> dict[str, Any]:
    """Run the real ATEM3D SimPEG/Discretize-compatible EB-Debye path."""

    config = build_benchmark_config(
        case,
        variant=variant,
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=substeps,
    )
    metadata = config["adapter_metadata"]
    material_fit = metadata["material_fit"]
    if material_fit is not None and not material_fit["material_gate_pass"]:
        raise ValueError("material fit gate must pass before simulation")

    simulation = build_simulation(config)
    result = simulation.run_data_only()
    initialization_solver_diagnostics = _validated_initialization_solver_diagnostics(
        simulation.initialization_solver_diagnostics,
        config,
    )
    linear_solver_diagnostics = _validated_linear_solver_diagnostics(
        simulation.linear_solver_diagnostics,
        config,
    )
    output_indices = np.asarray(metadata["output_indices"], dtype=np.int64)
    result_times = np.asarray(result.times, dtype=float)
    result_data = np.asarray(result.data, dtype=float)
    expected_rows = len(config["time_steps"]) + 1
    if result_times.shape != (expected_rows,) or result_data.shape != (expected_rows, 4):
        raise RuntimeError("solver returned an unexpected time/data shape")
    selected_times = result_times[1:][output_indices]
    expected_times = np.asarray(case.observation_times, dtype=float)
    if not np.allclose(selected_times, expected_times, rtol=2.0e-15, atol=0.0):
        raise RuntimeError("solver output times do not match benchmark observation times")
    selected_data = np.array(result_data[1:][output_indices], dtype=float, copy=True)
    component_signs = np.asarray(
        [
            metadata["coordinate_transform"]["output_component_signs"][component]
            for component in _COMPONENTS
        ],
        dtype=float,
    )
    selected_data *= component_signs[None, :]
    if not np.all(np.isfinite(selected_data)):
        raise RuntimeError("solver returned non-finite benchmark data")

    mesh_metadata = config["mesh"]["metadata"]
    mesh_stats = {
        "n_cells": int(simulation.mesh.n_cells),
        "n_edges": int(simulation.mesh.n_edges),
        "axis_cell_counts": copy.deepcopy(mesh_metadata["axis_cell_counts"]),
        "bounds_m": copy.deepcopy(mesh_metadata["public_bounds_z_down_m"]),
        "internal_bounds_z_up_m": copy.deepcopy(mesh_metadata["bounds_m"]),
        "spatial_level": spatial_level,
        "boundary_level": boundary_level,
        "mesh_hash": metadata["mesh_hash"],
    }
    return {
        "times": _readonly_array(expected_times, dtype=np.float64),
        "data": selected_data,
        "components": list(_COMPONENTS),
        "coordinate_system": "z_down",
        "coordinate_transform": copy.deepcopy(metadata["coordinate_transform"]),
        "mesh_stats": mesh_stats,
        "mesh_hash": metadata["mesh_hash"],
        "time_hash": metadata["time_hash"],
        "solver_id": "atem3d_simpeg_discretize_debye",
        "initialization_solver_diagnostics": initialization_solver_diagnostics,
        "linear_solver_diagnostics": linear_solver_diagnostics,
        "variant": variant,
        "material_fit": copy.deepcopy(material_fit),
    }


def _validated_initialization_solver_diagnostics(
    diagnostics: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = (
        (
            "dc_electric",
            "petsc_ksp_hypre_boomeramg",
            config["initialization_solver"]["dc_ksp_type"],
            "hypre_boomeramg",
            "discrete_current_divergence",
        ),
        (
            "ampere_magnetic",
            "petsc_ksp_hypre_ams",
            config["initialization_solver"]["magnetic_ksp_type"],
            "hypre_ams",
            "static_ampere",
        ),
    )
    if not isinstance(diagnostics, list) or len(diagnostics) != len(expected):
        raise RuntimeError(
            "initialization solver diagnostics must contain DC and Ampere records"
        )
    solver_config = config["initialization_solver"]
    tolerance = float(solver_config["tolerance"])
    configured_internal = solver_config.get("internal_tolerance")
    internal_tolerance = (
        min(1.0e-11, tolerance * 1.0e-3)
        if configured_internal is None
        else float(configured_internal)
    )
    maximum_replacements = int(solver_config["residual_replacement_steps"])

    def finite_number(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(float(value))
        )

    validated: list[dict[str, Any]] = []
    for record, (phase, solver, ksp_type, pc_type, balance_name) in zip(
        diagnostics,
        expected,
    ):
        if not isinstance(record, dict):
            raise RuntimeError("initialization solver diagnostics records must be mappings")
        residual = record.get("external_true_relative_residual")
        balance = record.get("balance_relative_residual")
        reason = record.get("backend_reason")
        iterations = record.get("backend_iterations")
        replacements = record.get("residual_replacement_steps")
        solve_mode = record.get("solve_mode")
        external_tolerance = record.get("external_tolerance")
        balance_tolerance = record.get("balance_tolerance")
        reported_internal_tolerance = record.get("internal_tolerance")
        invalid = (
            record.get("phase") != phase
            or record.get("solver") != solver
            or record.get("ksp_type") != ksp_type
            or record.get("pc_type") != pc_type
            or record.get("balance_name") != balance_name
            or solve_mode not in {"exact_zero_rhs", "petsc_ksp"}
            or isinstance(reason, bool)
            or not isinstance(reason, (int, np.integer))
            or isinstance(iterations, bool)
            or not isinstance(iterations, (int, np.integer))
            or int(iterations) < 0
            or isinstance(replacements, bool)
            or not isinstance(replacements, (int, np.integer))
            or int(replacements) < 0
            or int(replacements) > maximum_replacements
            or not finite_number(residual)
            or float(residual) > tolerance
            or not finite_number(balance)
            or float(balance) > tolerance
            or not finite_number(external_tolerance)
            or float(external_tolerance) != tolerance
            or not finite_number(balance_tolerance)
            or float(balance_tolerance) != tolerance
            or not finite_number(reported_internal_tolerance)
            or float(reported_internal_tolerance) != internal_tolerance
        )
        if not invalid and solve_mode == "exact_zero_rhs":
            invalid = (
                int(reason) != 0
                or record.get("backend_reported_converged") is not False
                or int(iterations) != 0
                or float(residual) != 0.0
                or int(replacements) != 0
                or float(balance) != 0.0
            )
        elif not invalid:
            invalid = (
                int(reason) <= 0
                or record.get("backend_reported_converged") is not True
            )
        if invalid:
            raise RuntimeError(
                "initialization solver diagnostics failed the external residual or "
                "physical balance gate"
            )
        validated.append(copy.deepcopy(record))
    return validated


def _validated_linear_solver_diagnostics(
    diagnostics: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(diagnostics, list) or len(diagnostics) != len(config["time_steps"]):
        raise RuntimeError("linear solver diagnostics must contain one record per time step")
    tolerance = float(config["solver"]["tolerance"])
    validated: list[dict[str, Any]] = []
    for index, (record, dt) in enumerate(zip(diagnostics, config["time_steps"])):
        if not isinstance(record, dict):
            raise RuntimeError("linear solver diagnostics records must be mappings")
        residual = record.get("external_true_relative_residual")
        if (
            record.get("step_index") != index
            or not bool(record.get("backend_reported_converged", False))
            or isinstance(residual, bool)
            or not isinstance(residual, (int, float, np.integer, np.floating))
            or not np.isfinite(float(residual))
            or float(residual) > tolerance
            or not np.isclose(float(record.get("dt_s", np.nan)), float(dt), rtol=0.0, atol=0.0)
        ):
            raise RuntimeError("linear solver diagnostics failed the external residual gate")
        validated.append(copy.deepcopy(record))
    return validated
