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
    hz, z0 = _graded_axis(z_anchors, lower=-extent, upper=extent)

    counts = {"x": len(hx), "y": len(hy), "z": len(hz)}
    bounds = {
        "x": [x0, x0 + float(np.sum(hx))],
        "y": [y0, y0 + float(np.sum(hy))],
        "z": [z0, z0 + float(np.sum(hz))],
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
        "bounds_m": bounds,
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
        "top": float("-inf"),
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
                "bottom": float("inf"),
                "sigma_infinity": earth_sigma,
                "debye_terms": [],
                "material": "earth_noip",
            },
        ], None

    earth_layers = list(case.earth["layers"])
    layers: list[dict[str, Any]] = [air]
    for index, earth in enumerate(earth_layers):
        top = float(earth["top_m"])
        bottom = float("inf") if earth["bottom_m"] is None else float(earth["bottom_m"])
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
                float(polarization["top_m"]) != top
                or float(polarization["bottom_m"]) != bottom
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
        "coordinate_system": "depth_down",
        "formulation": "eb",
        "initial_magnetic_field": "ampere",
        "solver": {
            "type": "cg",
            "tolerance": 1.0e-8,
            "maxiter": 2000,
            "preconditioner": "jacobi",
        },
        "mesh": mesh,
        "mesh_hash": mesh["mesh_hash"],
        "time_hash": time_hash,
        "model": {
            "coordinate_system": "depth_down",
            "layers": layers,
            "require_layer_boundary_alignment": True,
            "layer_boundary_tolerance": 1.0e-10,
        },
        "source": {
            "start": list(case.source_start_down),
            "end": list(case.source_end_down),
            "current": float(case.current_a),
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "receivers": [
            {"location": list(case.receiver_down), "component": component, "type": "point"}
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
            "initial_magnetic_field": "ampere",
            "transient_solver": "cg_jacobi",
            "initialization_solver": "scipy_sparse_direct",
            "resource_note": (
                "initial DC electric and Ampere-consistent magnetic fields use "
                "sparse direct solves in the existing core solver"
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
    if not np.all(np.isfinite(selected_data)):
        raise RuntimeError("solver returned non-finite benchmark data")

    mesh_metadata = config["mesh"]["metadata"]
    mesh_stats = {
        "n_cells": int(simulation.mesh.n_cells),
        "n_edges": int(simulation.mesh.n_edges),
        "axis_cell_counts": copy.deepcopy(mesh_metadata["axis_cell_counts"]),
        "bounds_m": copy.deepcopy(mesh_metadata["bounds_m"]),
        "spatial_level": spatial_level,
        "boundary_level": boundary_level,
        "mesh_hash": metadata["mesh_hash"],
    }
    return {
        "times": _readonly_array(expected_times, dtype=np.float64),
        "data": selected_data,
        "components": list(_COMPONENTS),
        "mesh_stats": mesh_stats,
        "mesh_hash": metadata["mesh_hash"],
        "time_hash": metadata["time_hash"],
        "solver_id": "atem3d_simpeg_discretize_debye",
        "variant": variant,
        "material_fit": copy.deepcopy(material_fit),
    }
