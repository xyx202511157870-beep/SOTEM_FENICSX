#!/usr/bin/env python3
"""3D FETD grounded-wire SOTEM verification against empymod.

This script is intentionally self-contained.  The default run implements the
stage-1 non-polarizable air-earth benchmark requested for DOLFINx/PETSc.  The
Cole-Cole branch is opt-in and first fits the complex conductivity to Debye
terms before any time-domain memory variables are used.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass, field, replace
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


CORE_MODULES = ("dolfinx", "ufl", "basix", "mpi4py", "petsc4py")
PIP_MODULES = {
    "gmsh": "gmsh",
    "empymod": "empymod",
    "meshio": "meshio",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}
POSTPROCESS_MODULES = {
    "empymod": "empymod",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}

PHYS_AIR = 100
PHYS_EARTH = 101
PHYS_OUTER = 201
PHYS_SURFACE = 202
PHYS_SOURCE_LINE = 301
SPONGE_ALL_SIDES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@dataclass
class PipelineConfig:
    """Runtime configuration for the verification model."""

    workdir: Path = field(default_factory=lambda: Path.cwd())
    msh_name: str = "verification_mesh.msh"
    force_mesh: bool = False

    x_extent: float = 25_000.0
    y_extent: float = 25_000.0
    air_height: float = 10_000.0
    earth_depth: float = 25_000.0

    source_start: tuple[float, float, float] = (-500.0, 200.0, -0.1)
    source_end: tuple[float, float, float] = (500.0, 200.0, -0.1)
    source_current: float = 10.0
    ramp_off_time: float = 1.0e-5
    wire_radius: float = 2.5
    source_mesh_size: float = 5.0
    source_refinement_radius: float = 100.0
    source_quadrature_points: int = 0

    receiver: tuple[float, float, float] = (0.0, -300.0, -0.1)
    receiver_type: str = "point"  # point, volume_average, disk_average
    receiver_average_radius: float = 2.0
    receiver_diagnostic_types: tuple[str, ...] | str = ()
    receiver_mesh_size: float = 10.0
    receiver_anchor_mesh_size: float = 0.0
    receiver_refinement_radius: float = 60.0
    outer_boundary_mode: str = "pec"  # pec, natural, robin
    outer_boundary_robin_scale: float = 1.0
    diffusion_refinement_factor: float = 0.0
    diffusion_refinement_mesh_size: float = 80.0
    sponge_strength: float = 0.0
    sponge_thickness: float = 0.0
    sponge_power: float = 2.0
    sponge_apply_to_initial: bool = False
    sponge_sides: tuple[str, ...] = SPONGE_ALL_SIDES
    expected_source_length: float = 1000.0
    expected_parallel_offset: float = 500.0
    geometry_tolerance: float = 1.0e-8

    rho_air: float = 1.0e8
    rho_earth: float = 100.0
    layer_depths: tuple[float, ...] = ()
    layer_resistivities: tuple[float, ...] = ()
    mu_r_air: float = 1.0
    mu_r_earth: float = 1.0
    nedelec_order: int = 1

    t_min: float = 1.0e-6
    t_max: float = 1.0
    time_growth: float = 1.05
    observation_times: tuple[float, ...] = ()
    time_method: str = "theta"  # theta, bdf2
    time_theta: float = 1.0
    time_origin: str = "after_ramp"  # after_ramp, ramp_start
    ramp_solver_t_min: float = 1.0e-6
    min_steps_during_turnoff: int = 10
    min_steps_before_first_observation: int = 1

    source_mode: str = "auto"  # auto, line, manual_line, regularized
    source_projection_mode: str = "charge_conserving"  # charge_conserving, raw
    source_rhs_sign: float = -1.0
    source_term_mode: str = "impressed_current"  # impressed_current, primary_dc
    formulation: str = "e"  # e, h
    initial_dc_mode: str = "fem"  # fem, analytic_halfspace
    magnetic_receiver_mode: str = "curl"  # curl, biot_current, biot_ohmic, faraday_integrated
    magnetic_dbdt_mode: str = "curl"  # curl, biot_rate
    receiver_evaluation_mode: str = "median"  # first_cell, mean, median, nearest_center, shallowest
    divergence_cleaning: str = "none"  # none, conductivity
    divergence_cleaning_strength: float = 1.0
    divergence_cleaning_t_obs_min: float = 0.0
    divergence_control_weight: float = 0.0
    divergence_control_t_obs_min: float = 0.0
    divergence_control_scale: str = "absolute"  # absolute, mass, stiffness, lhs
    polarization: str = "none"  # none, cole-cole
    cole_rho0: float = 100.0
    cole_m: float = 0.2
    cole_tau: float = 0.1
    cole_c: float = 0.6
    cole_layer_top: float = 0.0
    cole_layer_bottom: float = float("inf")
    cole_n_terms: int = 10
    cole_f_min: float = 1.0e-2
    cole_f_max: float = 1.0e5
    cole_n_freq: int = 96
    empymod_srcpts: int = 5
    empymod_ht: str = "dlf"
    empymod_ft: str = "dlf"
    reference_audit_srcpts: int = 0

    ksp_type: str = "gmres"
    rtol: float = 1.0e-8
    atol: float = 1.0e-12
    max_it: int = 1000

    error_tolerance: float = 0.05
    error_min_time: float = 0.0
    weak_component_reference_fraction: float = 0.1
    checkpoint_forward: bool = False
    resume_forward: bool = False
    stop_after_outputs: int = 0
    source_only: bool = False
    memory_limit_gb: float = 32.0
    memory_safety_fraction: float = 0.95

    def mesh_path(self) -> Path:
        return self.workdir / self.msh_name

    def output_png(self) -> Path:
        return self.workdir / "verification_result.png"

    def output_npz(self) -> Path:
        return self.workdir / "verification_data.npz"

    def forward_partial_npz(self) -> Path:
        return self.workdir / "forward_partial.npz"

    def forward_checkpoint_npz(self) -> Path:
        return self.workdir / "forward_checkpoint.npz"

    def receiver_diagnostics_csv(self) -> Path:
        return self.workdir / "receiver_diagnostics.csv"

    def receiver_diagnostics_png(self) -> Path:
        return self.workdir / "receiver_diagnostics.png"

    def output_report(self) -> Path:
        return self.workdir / "verification_report.txt"


def validate_geometry_consistency(config: PipelineConfig) -> dict[str, float]:
    """Validate the intended 1000 m wire and 500 m parallel-line survey geometry."""

    import numpy as np

    start = np.asarray(config.source_start, dtype=float)
    end = np.asarray(config.source_end, dtype=float)
    receiver = np.asarray(config.receiver, dtype=float)
    axis = end - start
    source_length = float(np.linalg.norm(axis))
    if source_length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")

    horizontal_axis = axis[:2]
    horizontal_length = float(np.linalg.norm(horizontal_axis))
    if horizontal_length <= 0.0:
        raise ValueError("parallel survey offset requires a horizontal source projection")
    rel = receiver[:2] - start[:2]
    inline = float(np.dot(rel, horizontal_axis) / horizontal_length)
    cross = float((horizontal_axis[0] * rel[1] - horizontal_axis[1] * rel[0]) / horizontal_length)
    parallel_offset = abs(cross)

    tol = float(config.geometry_tolerance)
    if abs(source_length - float(config.expected_source_length)) > tol:
        raise ValueError(
            "source length is inconsistent with the configured model: "
            f"got {source_length:.12g} m, expected {config.expected_source_length:.12g} m"
        )
    if abs(parallel_offset - float(config.expected_parallel_offset)) > tol:
        raise ValueError(
            "parallel survey offset is inconsistent with the configured model: "
            f"got {parallel_offset:.12g} m from receiver {config.receiver}, "
            f"expected {config.expected_parallel_offset:.12g} m"
        )
    return {
        "source_length": source_length,
        "parallel_offset": parallel_offset,
        "inline_distance_from_source_start": inline,
        "signed_crossline_offset": cross,
    }


def _normalise_layer_model(config: PipelineConfig) -> tuple[list[float], list[float]]:
    """Return earth-interface depths and earth-layer resistivities."""

    layer_depths = [float(value) for value in tuple(config.layer_depths)]
    layer_resistivities = [float(value) for value in tuple(config.layer_resistivities)]
    if not layer_depths and not layer_resistivities:
        return [], [float(config.rho_earth)]
    if len(layer_resistivities) != len(layer_depths) + 1:
        raise ValueError(
            "layer_resistivities must contain exactly one more value than layer_depths "
            f"(got {len(layer_resistivities)} resistivities for {len(layer_depths)} layer depths)"
        )
    previous = 0.0
    for depth in layer_depths:
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"layer_depths must be finite positive depths below z=0; got {depth:.12g}")
        if depth <= previous:
            raise ValueError("layer_depths must be strictly increasing")
        previous = depth
    for resistivity in layer_resistivities:
        if not math.isfinite(resistivity) or resistivity <= 0.0:
            raise ValueError(
                f"layer_resistivities must be finite positive resistivities; got {resistivity:.12g}"
            )
    return layer_depths, layer_resistivities


def _earth_resistivity_at_depth(depth: float, config: PipelineConfig) -> float:
    """Return the earth-layer resistivity for a positive depth below z=0."""

    depth = float(depth)
    if not math.isfinite(depth) or depth < 0.0:
        raise ValueError(f"layer lookup depth must be finite and nonnegative; got {depth:.12g}")
    layer_depths, layer_resistivities = _normalise_layer_model(config)
    index = bisect.bisect_right(layer_depths, depth)
    return float(layer_resistivities[index])


def _empymod_depth_res(config: PipelineConfig) -> tuple[list[float], list[float]]:
    """Return empymod depth/res arrays for the configured air + earth layers."""

    layer_depths, layer_resistivities = _normalise_layer_model(config)
    return [0.0, *layer_depths], [float(config.rho_air), *layer_resistivities]


def _uniform_halfspace_resistivity(config: PipelineConfig) -> float:
    """Return the earth resistivity for analytic uniform-halfspace formulas."""

    _layer_depths, layer_resistivities = _normalise_layer_model(config)
    first = float(layer_resistivities[0])
    if any(abs(float(value) - first) > max(1.0e-10, 1.0e-10 * abs(first)) for value in layer_resistivities[1:]):
        raise ValueError("analytic_halfspace initial_dc_mode requires a uniform earth resistivity")
    return first


def _max_earth_diffusion_length(config: PipelineConfig, time: float | None = None) -> float:
    """Return sqrt(2*rho*t/mu) using the most diffusive earth resistivity."""

    _layer_depths, layer_resistivities = _normalise_layer_model(config)
    rho = max(float(value) for value in layer_resistivities)
    t = float(config.t_max if time is None else time)
    mu = 4.0 * math.pi * 1.0e-7 * float(config.mu_r_earth)
    if rho <= 0.0 or t <= 0.0 or mu <= 0.0:
        raise ValueError("rho, time, and mu must be positive for diffusion-length audit")
    return math.sqrt(2.0 * rho * t / mu)


def _diffusion_refinement_box(config: PipelineConfig) -> dict[str, float]:
    """Return the late-diffusion mesh box implied by config."""

    base_radius = 1000.0
    base_depth = 500.0
    base_top = 200.0
    factor = float(config.diffusion_refinement_factor)
    if factor > 0.0:
        length = _max_earth_diffusion_length(config)
        radius = max(base_radius, factor * length)
        depth = max(base_depth, factor * length)
    else:
        radius = base_radius
        depth = base_depth
    return {
        "radius": float(radius),
        "depth": float(depth),
        "top": float(base_top),
        "mesh_size": float(config.diffusion_refinement_mesh_size),
    }


def _diffusion_refinement_audit(config: PipelineConfig) -> dict[str, Any]:
    """Audit late diffusion coverage for both refinement and finite domain."""

    diffusion_length = _max_earth_diffusion_length(config)
    recommended_factor = 2.0
    recommended_radius = recommended_factor * diffusion_length
    box = _diffusion_refinement_box(config)
    underresolved = bool(box["radius"] < recommended_radius or box["depth"] < recommended_radius)
    domain_horizontal_radius = min(float(config.x_extent), float(config.y_extent))
    domain_depth = float(config.earth_depth)
    domain_underresolved = bool(domain_horizontal_radius < recommended_radius or domain_depth < recommended_radius)
    return {
        "diffusion_length": float(diffusion_length),
        "recommended_factor": float(recommended_factor),
        "recommended_radius": float(recommended_radius),
        "box_radius": float(box["radius"]),
        "box_depth": float(box["depth"]),
        "box_top": float(box["top"]),
        "mesh_size": float(box["mesh_size"]),
        "underresolved": underresolved,
        "domain_horizontal_radius": float(domain_horizontal_radius),
        "domain_depth": float(domain_depth),
        "domain_underresolved": domain_underresolved,
    }


def _format_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def _normalise_sponge_sides(sides: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    """Return validated sponge side names."""

    if sides is None:
        return SPONGE_ALL_SIDES
    if isinstance(sides, str):
        raw_sides = tuple(part.strip().lower() for part in sides.split(","))
    else:
        raw_sides = tuple(str(part).strip().lower() for part in sides)
    out = tuple(side for side in raw_sides if side)
    unknown = sorted(set(out) - set(SPONGE_ALL_SIDES))
    if unknown:
        raise ValueError(f"unknown sponge side(s): {', '.join(unknown)}")
    return out


def _sponge_sigma_addition_for_centers(centers, config: PipelineConfig):
    """Return direct-time sponge conductivity increments for local cell centers."""

    import numpy as np

    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (n_cells, 3)")
    strength = float(config.sponge_strength)
    thickness = float(config.sponge_thickness)
    power = float(config.sponge_power)
    sides = _normalise_sponge_sides(config.sponge_sides)
    addition = np.zeros(centers.shape[0], dtype=float)
    if strength == 0.0 or thickness == 0.0 or not sides:
        return addition

    bounds = {
        "x_min": centers[:, 0] - (-float(config.x_extent)),
        "x_max": float(config.x_extent) - centers[:, 0],
        "y_min": centers[:, 1] - (-float(config.y_extent)),
        "y_max": float(config.y_extent) - centers[:, 1],
        "z_min": centers[:, 2] - (-float(config.earth_depth)),
        "z_max": float(config.air_height) - centers[:, 2],
    }
    for side in sides:
        distance = bounds[side]
        active = distance < thickness
        if np.any(active):
            weight = np.zeros_like(addition)
            weight[active] = ((thickness - distance[active]) / thickness) ** power
            addition = np.maximum(addition, strength * weight)
    return addition


def _sponge_diagnostics(config: PipelineConfig) -> dict[str, Any]:
    sides = _normalise_sponge_sides(config.sponge_sides)
    return {
        "strength": float(config.sponge_strength),
        "thickness": float(config.sponge_thickness),
        "power": float(config.sponge_power),
        "apply_to_initial": bool(config.sponge_apply_to_initial),
        "sides": sides,
        "enabled": float(config.sponge_strength) > 0.0 and float(config.sponge_thickness) > 0.0 and bool(sides),
    }


def validate_model_consistency(config: PipelineConfig, reference_mode: str | None = None) -> dict[str, Any]:
    """Validate FEM and empymod inputs before constructing the mesh or reference."""

    import numpy as np

    diagnostics: dict[str, Any] = {}

    def require_positive(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; got {value:.12g}")
        return value

    def require_finite_point(name: str, point: tuple[float, float, float]) -> np.ndarray:
        arr = np.asarray(point, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite 3D coordinate; got {point}")
        return arr

    start = require_finite_point("source_start", config.source_start)
    end = require_finite_point("source_end", config.source_end)
    receiver = require_finite_point("receiver", config.receiver)

    for name, value in [
        ("x_extent", config.x_extent),
        ("y_extent", config.y_extent),
        ("air_height", config.air_height),
        ("earth_depth", config.earth_depth),
        ("rho_air", config.rho_air),
        ("rho_earth", config.rho_earth),
        ("mu_r_air", config.mu_r_air),
        ("mu_r_earth", config.mu_r_earth),
        ("wire_radius", config.wire_radius),
        ("source_mesh_size", config.source_mesh_size),
        ("source_refinement_radius", config.source_refinement_radius),
        ("receiver_mesh_size", config.receiver_mesh_size),
        ("receiver_refinement_radius", config.receiver_refinement_radius),
        ("diffusion_refinement_mesh_size", config.diffusion_refinement_mesh_size),
        ("sponge_power", config.sponge_power),
        ("geometry_tolerance", config.geometry_tolerance),
        ("t_min", config.t_min),
        ("t_max", config.t_max),
        ("ramp_solver_t_min", config.ramp_solver_t_min),
        ("memory_limit_gb", config.memory_limit_gb),
    ]:
        require_positive(name, value)
    ramp_off_time = float(config.ramp_off_time)
    if not math.isfinite(ramp_off_time) or ramp_off_time < 0.0:
        raise ValueError("ramp_off_time must be finite and nonnegative")
    diagnostics["ramp_off_time"] = ramp_off_time
    receiver_anchor_mesh_size = float(config.receiver_anchor_mesh_size)
    if not math.isfinite(receiver_anchor_mesh_size) or receiver_anchor_mesh_size < 0.0:
        raise ValueError(
            "receiver_anchor_mesh_size must be finite and nonnegative; "
            f"got {receiver_anchor_mesh_size:.12g}"
        )
    memory_safety_fraction = float(config.memory_safety_fraction)
    if not math.isfinite(memory_safety_fraction) or memory_safety_fraction <= 0.0 or memory_safety_fraction > 1.0:
        raise ValueError(
            f"memory_safety_fraction must be finite and in (0, 1]; got {memory_safety_fraction:.12g}"
        )
    for name, value in [
        ("sponge_strength", config.sponge_strength),
        ("sponge_thickness", config.sponge_thickness),
    ]:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative; got {value:.12g}")
    sponge = _sponge_diagnostics(config)
    if sponge["strength"] > 0.0 and sponge["thickness"] <= 0.0:
        raise ValueError("sponge_thickness must be positive when sponge_strength is positive")
    if sponge["strength"] > 0.0 and not sponge["sides"]:
        raise ValueError("sponge_sides must contain at least one side when sponge_strength is positive")
    if not math.isfinite(float(config.source_current)) or float(config.source_current) == 0.0:
        raise ValueError(f"source_current must be finite and nonzero; got {float(config.source_current):.12g}")
    if float(config.source_rhs_sign) not in {-1.0, 1.0}:
        raise ValueError(f"source_rhs_sign must be -1 or 1; got {float(config.source_rhs_sign):.12g}")
    source_quadrature_points = int(config.source_quadrature_points)
    if source_quadrature_points < 0:
        raise ValueError(f"source_quadrature_points must be nonnegative; got {source_quadrature_points}")
    stop_after_outputs = int(config.stop_after_outputs)
    if stop_after_outputs < 0:
        raise ValueError(f"stop_after_outputs must be nonnegative; got {stop_after_outputs}")
    nedelec_order = int(config.nedelec_order)
    if nedelec_order not in {1, 2}:
        raise ValueError(f"nedelec_order must be 1 or 2; got {nedelec_order}")
    source_term_mode = str(config.source_term_mode).strip().lower()
    if source_term_mode not in {"impressed_current", "primary_dc"}:
        raise ValueError("source_term_mode must be 'impressed_current' or 'primary_dc'")
    source_projection_mode = str(config.source_projection_mode).strip().lower()
    if source_projection_mode not in {"charge_conserving", "raw"}:
        raise ValueError("source_projection_mode must be 'charge_conserving' or 'raw'")
    magnetic_receiver_mode = str(config.magnetic_receiver_mode).strip().lower()
    if magnetic_receiver_mode not in {"curl", "biot_current", "biot_ohmic", "faraday_integrated"}:
        raise ValueError(
            "magnetic_receiver_mode must be 'curl', 'biot_current', 'biot_ohmic', or 'faraday_integrated'"
        )
    magnetic_dbdt_mode = str(config.magnetic_dbdt_mode).strip().lower()
    if magnetic_dbdt_mode not in {"curl", "biot_rate"}:
        raise ValueError("magnetic_dbdt_mode must be 'curl' or 'biot_rate'")
    if magnetic_dbdt_mode == "biot_rate" and magnetic_receiver_mode not in {"biot_current", "biot_ohmic"}:
        raise ValueError("magnetic_dbdt_mode='biot_rate' requires a Biot magnetic_receiver_mode")
    outer_boundary_mode = str(config.outer_boundary_mode).strip().lower()
    if outer_boundary_mode not in {"pec", "natural", "robin"}:
        raise ValueError("outer_boundary_mode must be 'pec', 'natural', or 'robin'")
    if not math.isfinite(float(config.outer_boundary_robin_scale)) or float(config.outer_boundary_robin_scale) < 0.0:
        raise ValueError(
            "outer_boundary_robin_scale must be finite and nonnegative; "
            f"got {float(config.outer_boundary_robin_scale):.12g}"
        )
    receiver_evaluation_mode = str(config.receiver_evaluation_mode).strip().lower()
    if receiver_evaluation_mode not in {"first_cell", "mean", "median", "nearest_center", "shallowest"}:
        raise ValueError(
            "receiver_evaluation_mode must be 'first_cell', 'mean', 'median', 'nearest_center', or 'shallowest'"
        )
    receiver_type = str(config.receiver_type).strip().lower()
    if receiver_type not in {"point", "volume_average", "disk_average"}:
        raise ValueError("receiver_type must be 'point', 'volume_average', or 'disk_average'")
    receiver_average_radius = float(config.receiver_average_radius)
    if receiver_type != "point" and (not math.isfinite(receiver_average_radius) or receiver_average_radius <= 0.0):
        raise ValueError("receiver_average_radius must be positive for average receiver types")
    receiver_diagnostic_types = _parse_receiver_diagnostic_types(config)
    if any(item != "point" for item in receiver_diagnostic_types) and (
        not math.isfinite(receiver_average_radius) or receiver_average_radius <= 0.0
    ):
        raise ValueError("receiver_average_radius must be positive when receiver diagnostics include average types")
    divergence_cleaning = str(config.divergence_cleaning).strip().lower()
    if divergence_cleaning not in {"none", "conductivity"}:
        raise ValueError("divergence_cleaning must be 'none' or 'conductivity'")
    divergence_cleaning_strength = float(config.divergence_cleaning_strength)
    if (
        not math.isfinite(divergence_cleaning_strength)
        or divergence_cleaning_strength < 0.0
        or divergence_cleaning_strength > 1.0
    ):
        raise ValueError(
            "divergence_cleaning_strength must be finite and in [0, 1]; "
            f"got {divergence_cleaning_strength:.12g}"
        )
    divergence_cleaning_t_obs_min = float(config.divergence_cleaning_t_obs_min)
    if not math.isfinite(divergence_cleaning_t_obs_min) or divergence_cleaning_t_obs_min < 0.0:
        raise ValueError(
            "divergence_cleaning_t_obs_min must be finite and nonnegative; "
            f"got {divergence_cleaning_t_obs_min:.12g}"
        )
    divergence_control_weight = float(config.divergence_control_weight)
    if not math.isfinite(divergence_control_weight) or divergence_control_weight < 0.0:
        raise ValueError(
            "divergence_control_weight must be finite and nonnegative; "
            f"got {divergence_control_weight:.12g}"
        )
    divergence_control_t_obs_min = float(config.divergence_control_t_obs_min)
    if not math.isfinite(divergence_control_t_obs_min) or divergence_control_t_obs_min < 0.0:
        raise ValueError(
            "divergence_control_t_obs_min must be finite and nonnegative; "
            f"got {divergence_control_t_obs_min:.12g}"
        )
    divergence_control_scale = str(config.divergence_control_scale).strip().lower()
    if divergence_control_scale not in {"absolute", "mass", "stiffness", "lhs"}:
        raise ValueError("divergence_control_scale must be 'absolute', 'mass', 'stiffness', or 'lhs'")
    initial_dc_mode = str(config.initial_dc_mode).strip().lower()
    if initial_dc_mode not in {"fem", "analytic_halfspace"}:
        raise ValueError("initial_dc_mode must be 'analytic_halfspace' or 'fem'")
    time_origin = str(config.time_origin).strip().lower()
    if time_origin not in {"after_ramp", "ramp_start"}:
        raise ValueError("time_origin must be 'after_ramp' or 'ramp_start'")
    reference_ramp_window = "after_ramp" if time_origin == "after_ramp" else "ramp_start"
    if float(config.t_max) <= float(config.t_min):
        raise ValueError(f"t_max must be greater than t_min; got t_min={config.t_min:.12g}, t_max={config.t_max:.12g}")
    if not math.isfinite(float(config.time_growth)) or float(config.time_growth) <= 1.0:
        raise ValueError(f"time_growth must be finite and greater than 1; got {float(config.time_growth):.12g}")
    if not math.isfinite(float(config.diffusion_refinement_factor)) or float(config.diffusion_refinement_factor) < 0.0:
        raise ValueError(
            "diffusion_refinement_factor must be finite and nonnegative; "
            f"got {float(config.diffusion_refinement_factor):.12g}"
        )
    time_theta = float(config.time_theta)
    if not math.isfinite(time_theta) or not (0.5 <= time_theta <= 1.0):
        raise ValueError(f"time_theta must be finite and in [0.5, 1]; got {time_theta:.12g}")
    time_method = str(config.time_method).strip().lower()
    if time_method not in {"theta", "bdf2"}:
        raise ValueError("time_method must be 'theta' or 'bdf2'")
    min_steps_during_turnoff = int(config.min_steps_during_turnoff)
    if min_steps_during_turnoff <= 0:
        raise ValueError(f"min_steps_during_turnoff must be positive; got {min_steps_during_turnoff}")
    min_steps_before_first_observation = int(config.min_steps_before_first_observation)
    if min_steps_before_first_observation <= 0:
        raise ValueError(
            "min_steps_before_first_observation must be positive; "
            f"got {min_steps_before_first_observation}"
        )
    if not math.isfinite(float(config.error_min_time)) or float(config.error_min_time) < 0.0:
        raise ValueError(f"error_min_time must be finite and nonnegative; got {float(config.error_min_time):.12g}")
    weak_fraction = float(config.weak_component_reference_fraction)
    if not math.isfinite(weak_fraction) or not (0.0 < weak_fraction < 1.0):
        raise ValueError(
            "weak_component_reference_fraction must be finite and in (0, 1); "
            f"got {weak_fraction:.12g}"
        )
    empymod_srcpts = int(config.empymod_srcpts)
    if empymod_srcpts <= 0:
        raise ValueError(f"empymod_srcpts must be positive; got {empymod_srcpts}")
    empymod_ht = str(config.empymod_ht).strip().lower()
    if empymod_ht not in {"dlf", "qwe", "quad"}:
        raise ValueError("empymod_ht must be 'dlf', 'qwe', or 'quad'")
    empymod_ft = str(config.empymod_ft).strip().lower()
    if empymod_ft not in {"dlf", "sin", "cos", "qwe", "fftlog", "fft"}:
        raise ValueError("empymod_ft must be 'dlf', 'sin', 'cos', 'qwe', 'fftlog', or 'fft'")
    reference_audit_srcpts = int(config.reference_audit_srcpts)
    if reference_audit_srcpts < 0:
        raise ValueError(f"reference_audit_srcpts must be nonnegative; got {reference_audit_srcpts}")

    tol = float(config.geometry_tolerance)
    source_depth_start = float(-start[2])
    source_depth_end = float(-end[2])
    receiver_depth = float(-receiver[2])
    for name, depth in [
        ("source_start", source_depth_start),
        ("source_end", source_depth_end),
        ("receiver", receiver_depth),
    ]:
        if depth <= tol:
            raise ValueError(f"{name} must be below the z=0 earth surface for the air-earth empymod comparison")
        if depth >= float(config.earth_depth) - tol:
            raise ValueError(f"{name} depth {depth:.12g} m exceeds the finite FEM earth depth {config.earth_depth:.12g} m")
    if abs(source_depth_start - source_depth_end) > tol:
        raise ValueError(
            "source_start and source_end must have matching depths for the horizontal finite-wire empymod comparison"
        )
    if abs(float(start[0])) >= float(config.x_extent) - tol or abs(float(end[0])) >= float(config.x_extent) - tol:
        raise ValueError("source x coordinates must lie inside the finite FEM x_extent")
    if abs(float(start[1])) >= float(config.y_extent) - tol or abs(float(end[1])) >= float(config.y_extent) - tol:
        raise ValueError("source y coordinates must lie inside the finite FEM y_extent")
    if abs(float(receiver[0])) >= float(config.x_extent) - tol or abs(float(receiver[1])) >= float(config.y_extent) - tol:
        raise ValueError("receiver coordinates must lie inside the finite FEM horizontal extents")

    layer_depths, layer_resistivities = _normalise_layer_model(config)
    if layer_depths and layer_depths[-1] >= float(config.earth_depth) - tol:
        raise ValueError(
            "layer_depths must lie inside the finite FEM earth depth: "
            f"deepest layer interface={layer_depths[-1]:.12g} m, earth_depth={float(config.earth_depth):.12g} m"
        )
    if initial_dc_mode == "analytic_halfspace":
        _uniform_halfspace_resistivity(config)

    polarization = str(config.polarization).strip().lower()
    if polarization not in {"none", "cole-cole"}:
        raise ValueError("polarization must be 'none' or 'cole-cole'")
    if polarization == "cole-cole" and (abs(time_theta - 1.0) > 1.0e-12 or time_method != "theta"):
        raise ValueError("Cole-Cole Debye memory terms currently require time_method='theta' and time_theta=1.0")
    if polarization == "cole-cole" and divergence_cleaning != "none":
        raise ValueError("divergence_cleaning='conductivity' is currently implemented for non-polarizable runs only")
    if time_method == "bdf2" and bool(config.resume_forward):
        raise ValueError("time_method='bdf2' does not support resume_forward checkpoints yet")
    if polarization == "cole-cole":
        require_positive("cole_rho0", config.cole_rho0)
        require_positive("cole_tau", config.cole_tau)
        require_positive("cole_c", config.cole_c)
        require_positive("cole_n_terms", config.cole_n_terms)
        require_positive("cole_f_min", config.cole_f_min)
        require_positive("cole_f_max", config.cole_f_max)
        require_positive("cole_n_freq", config.cole_n_freq)
        cole_layer_top = float(config.cole_layer_top)
        cole_layer_bottom = float(config.cole_layer_bottom)
        if not math.isfinite(cole_layer_top) or cole_layer_top < 0.0:
            raise ValueError(f"cole_layer_top must be finite and nonnegative; got {cole_layer_top:.12g}")
        if math.isinf(cole_layer_bottom):
            cole_layer_bottom = float(config.earth_depth)
        if (
            not math.isfinite(cole_layer_bottom)
            or cole_layer_bottom <= cole_layer_top
            or cole_layer_bottom > float(config.earth_depth) + tol
        ):
            raise ValueError(
                "cole_layer_bottom must be finite, greater than cole_layer_top, and inside earth_depth; "
                f"got top={cole_layer_top:.12g}, bottom={float(config.cole_layer_bottom):.12g}, "
                f"earth_depth={float(config.earth_depth):.12g}"
            )
        sampled_depths = [0.5 * (cole_layer_top + cole_layer_bottom)]
        sampled_depths.extend(depth for depth in layer_depths if cole_layer_top < depth < cole_layer_bottom)
        polarizable_rhos = {
            round(float(_earth_resistivity_at_depth(depth, config)), 12)
            for depth in sampled_depths
        }
        if len(polarizable_rhos) != 1:
            raise ValueError(
                "cole_layer depth window spans multiple resistivity layers; split the run or use a uniform "
                f"polarizable interval. resistivities={sorted(polarizable_rhos)}"
            )
        cole_layer_rho = next(iter(polarizable_rhos))
        if abs(float(config.cole_rho0) - cole_layer_rho) > tol:
            raise ValueError(
                "cole_rho0 must match the resistivity of the polarizable layer so the FEM material and "
                f"reference share the same earth model; got cole_rho0={float(config.cole_rho0):.12g}, "
                f"layer_rho={cole_layer_rho:.12g}"
            )
        if not math.isfinite(float(config.cole_m)) or not (0.0 <= float(config.cole_m) < 1.0):
            raise ValueError(f"cole_m must be finite and in [0, 1); got {float(config.cole_m):.12g}")
        if float(config.cole_f_max) <= float(config.cole_f_min):
            raise ValueError("cole_f_max must be greater than cole_f_min")

    ref_mode = reference_mode
    if ref_mode is None:
        ref_mode = "cole-cole" if polarization == "cole-cole" else "noip"
    if ref_mode not in {"noip", "cole-cole"}:
        raise ValueError("reference_mode must be 'noip' or 'cole-cole'")
    if ref_mode == "cole-cole" and polarization != "cole-cole":
        raise ValueError("reference_mode='cole-cole' requires polarization='cole-cole'")
    if ref_mode == "noip" and polarization == "cole-cole":
        raise ValueError("polarization='cole-cole' requires reference_mode='cole-cole'")

    diagnostics.update(validate_geometry_consistency(config))
    diagnostics.update(
        {
            "reference_mode": ref_mode,
            "source_depth_start": source_depth_start,
            "source_depth_end": source_depth_end,
            "receiver_depth": receiver_depth,
            "rho_air": float(config.rho_air),
            "rho_earth": float(layer_resistivities[0]),
            "sigma_air": 1.0 / float(config.rho_air),
            "sigma_earth": 1.0 / float(layer_resistivities[0]),
            "layer_depths": layer_depths,
            "layer_resistivities": layer_resistivities,
            "cole_layer_top": float(config.cole_layer_top),
            "cole_layer_bottom": float(config.cole_layer_bottom)
            if math.isfinite(float(config.cole_layer_bottom))
            else float(config.earth_depth),
            "source_current": float(config.source_current),
            "source_rhs_sign": float(config.source_rhs_sign),
            "source_quadrature_points": source_quadrature_points,
            "source_projection_mode": source_projection_mode,
            "source_only": bool(config.source_only),
            "nedelec_order": nedelec_order,
            "source_term_mode": source_term_mode,
            "initial_dc_mode": initial_dc_mode,
            "magnetic_receiver_mode": magnetic_receiver_mode,
            "magnetic_dbdt_mode": magnetic_dbdt_mode,
            "outer_boundary_mode": outer_boundary_mode,
            "outer_boundary_robin_scale": float(config.outer_boundary_robin_scale),
            "receiver_evaluation_mode": receiver_evaluation_mode,
            "receiver_type": receiver_type,
            "receiver_average_radius": receiver_average_radius,
            "receiver_diagnostic_types": receiver_diagnostic_types,
            "divergence_cleaning": divergence_cleaning,
            "divergence_cleaning_strength": divergence_cleaning_strength,
            "divergence_cleaning_t_obs_min": divergence_cleaning_t_obs_min,
            "divergence_control_weight": divergence_control_weight,
            "divergence_control_t_obs_min": divergence_control_t_obs_min,
            "divergence_control_scale": divergence_control_scale,
            "diffusion_refinement": _diffusion_refinement_audit(config),
            "sponge": sponge,
            "time_method": time_method,
            "time_origin": time_origin,
            "time_theta": time_theta,
            "reference_ramp_window": reference_ramp_window,
            "ramp_off_time": ramp_off_time,
            "ramp_solver_t_min": float(config.ramp_solver_t_min),
            "min_steps_during_turnoff": min_steps_during_turnoff,
            "min_steps_before_first_observation": min_steps_before_first_observation,
            "empymod_srcpts": empymod_srcpts,
            "empymod_ht": empymod_ht,
            "empymod_ft": empymod_ft,
            "reference_audit_srcpts": reference_audit_srcpts,
            "error_min_time": float(config.error_min_time),
        }
    )
    return diagnostics


def validate_formulation(config: PipelineConfig) -> str:
    """Return the normalized field formulation name."""

    formulation = str(config.formulation).strip().lower()
    if formulation not in {"e", "h"}:
        raise ValueError("formulation must be 'e' or 'h'")
    return formulation


@dataclass
class DebyeTerm:
    delta_sigma: float
    tau: float


@dataclass
class DebyeFit:
    sigma_infinity: float
    terms: list[DebyeTerm]
    frequencies: Any
    target_sigma: Any
    fitted_sigma: Any
    relative_l2: float


def log(message: str, *, comm: Any | None = None) -> None:
    """Rank-aware logging."""

    if comm is None:
        print(message, flush=True)
        return
    if comm.rank == 0:
        print(message, flush=True)


def _import_module(name: str):
    return importlib.import_module(name)


def _pip_install(package: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", package]
    print(f"[environment] Installing missing pip package: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def check_environment(install_missing: bool = True, *, require_core: bool = True) -> dict[str, str]:
    """Check interpreter, Python modules, and PETSc/HYPRE support."""

    print(f"[environment] Python executable: {sys.executable}", flush=True)
    expected_prefix = "/home/paidaxin/miniconda3/envs/fenicsx"
    if not sys.executable.startswith(expected_prefix):
        print(
            "[environment] WARNING: this is not the requested fenicsx interpreter "
            f"({expected_prefix}). Continue only if this is intentional.",
            flush=True,
        )

    versions: dict[str, str] = {"python": sys.executable}
    if require_core:
        missing_core: list[str] = []
        for name in CORE_MODULES:
            try:
                mod = _import_module(name)
                versions[name] = str(getattr(mod, "__version__", "unknown"))
            except Exception as exc:
                missing_core.append(f"{name}: {exc}")
        if missing_core:
            message = (
                "Missing FEniCSx/PETSc core components. Do not pip install dolfinx, "
                "petsc4py, mpi4py, ufl, or basix in this script.\n"
                + "\n".join(missing_core)
            )
            raise SystemExit(message)

    runtime_modules = PIP_MODULES if require_core else POSTPROCESS_MODULES
    for module_name, pip_name in runtime_modules.items():
        try:
            mod = _import_module(module_name)
            versions[module_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            if not install_missing:
                raise SystemExit(f"Missing optional/runtime package {module_name}: {exc}")
            _pip_install(pip_name)
            mod = _import_module(module_name)
            versions[module_name] = str(getattr(mod, "__version__", "unknown"))

    if require_core:
        from petsc4py import PETSc

        if not PETSc.Sys().hasExternalPackage("hypre"):
            raise SystemExit(
                "PETSc was not built with HYPRE. The requested AMS preconditioner "
                "cannot be used, so this script must exit."
            )
        versions["PETSc_hypre"] = "True"
        try:
            import dolfinx

            dx_ver = tuple(int(part) for part in dolfinx.__version__.split(".")[:2])
            if dx_ver < (0, 10):
                print(
                    f"[environment] DOLFINx {dolfinx.__version__} detected. "
                    "Recommended version is >= 0.10.0; ridge line source tags are "
                    "not assumed available, so regularized volume source fallback "
                    "will be used unless explicitly overridden.",
                    flush=True,
                )
        except Exception:
            pass

    print("[environment] Module versions:", flush=True)
    for key in sorted(versions):
        print(f"  {key}: {versions[key]}", flush=True)
    return versions


def generate_time_array(config: PipelineConfig):
    """Generate logarithmically increasing time samples."""

    import numpy as np

    if len(config.observation_times) > 0:
        values = np.asarray(config.observation_times, dtype=float)
        if (
            values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError("observation_times must be finite, positive, and strictly increasing")
        return values

    if config.t_min <= 0.0 or config.t_max <= config.t_min:
        raise ValueError("time bounds must satisfy 0 < t_min < t_max")
    if config.time_growth <= 1.0:
        raise ValueError("time_growth must be > 1")
    values = [float(config.t_min)]
    while values[-1] < config.t_max:
        nxt = values[-1] * config.time_growth
        if nxt >= config.t_max:
            values.append(float(config.t_max))
            break
        values.append(float(nxt))
    return np.asarray(values, dtype=float)


def _forward_observation_schedule(times, config: PipelineConfig):
    """Map reported observation times to internal ramp-start solve times."""

    import numpy as np

    observation_times = np.asarray(times, dtype=float)
    if observation_times.ndim != 1 or observation_times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(observation_times)) or np.any(observation_times <= 0.0):
        raise ValueError("times must be finite and positive")
    if np.any(np.diff(observation_times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    time_origin = str(config.time_origin).strip().lower()
    if time_origin == "ramp_start":
        step_times = observation_times.copy()
        output_internal_times = observation_times.copy()
    elif time_origin == "after_ramp":
        ramp_time = float(config.ramp_off_time)
        if ramp_time == 0.0:
            return {
                "step_times": observation_times.copy(),
                "output_internal_times": observation_times.copy(),
                "return_times": observation_times.copy(),
                "output_step_indices": list(range(observation_times.size)),
            }
        min_steps = max(1, int(config.min_steps_during_turnoff))
        ramp_start = min(float(config.ramp_solver_t_min), ramp_time / min_steps)
        ramp_steps = np.linspace(ramp_start, ramp_time, min_steps)
        output_internal_times = ramp_time + observation_times
        first_obs_steps = int(config.min_steps_before_first_observation)
        first_obs_internal_steps = np.linspace(ramp_time, output_internal_times[0], first_obs_steps + 1)[1:]
        step_times = np.unique(np.r_[ramp_steps, first_obs_internal_steps, output_internal_times])
    else:
        raise ValueError("time_origin must be 'after_ramp' or 'ramp_start'")

    output_step_indices: list[int] = []
    for output_time in output_internal_times:
        matches = np.flatnonzero(np.isclose(step_times, output_time, rtol=1.0e-12, atol=1.0e-30))
        if matches.size == 0:
            raise RuntimeError(f"internal output time {output_time:.12e} is missing from the solve schedule")
        output_step_indices.append(int(matches[0]))
    return {
        "step_times": step_times,
        "output_internal_times": output_internal_times,
        "return_times": observation_times,
        "output_step_indices": output_step_indices,
    }


def _add_box_field(gmsh, xmin, xmax, ymin, ymax, zmin, zmax, vin, vout, thickness):
    tag = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(tag, "VIn", vin)
    gmsh.model.mesh.field.setNumber(tag, "VOut", vout)
    gmsh.model.mesh.field.setNumber(tag, "XMin", xmin)
    gmsh.model.mesh.field.setNumber(tag, "XMax", xmax)
    gmsh.model.mesh.field.setNumber(tag, "YMin", ymin)
    gmsh.model.mesh.field.setNumber(tag, "YMax", ymax)
    gmsh.model.mesh.field.setNumber(tag, "ZMin", zmin)
    gmsh.model.mesh.field.setNumber(tag, "ZMax", zmax)
    gmsh.model.mesh.field.setNumber(tag, "Thickness", thickness)
    return tag


def _mesh_size_statistics(msh_path: Path) -> dict[str, float]:
    import meshio
    import numpy as np

    mesh = meshio.read(msh_path)
    points = mesh.points
    lengths: list[float] = []
    n_cells = 0
    for block in mesh.cells:
        if block.type not in {"tetra", "triangle", "line"}:
            continue
        cells = block.data
        n_cells += int(cells.shape[0])
        for cell in cells:
            coords = points[cell]
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    lengths.append(float(np.linalg.norm(coords[i] - coords[j])))
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        stats = {
            "edge_min": float(arr.min()),
            "edge_p50": float(np.median(arr)),
            "edge_p95": float(np.percentile(arr, 95)),
            "edge_max": float(arr.max()),
        }
    else:
        stats = {"edge_min": math.nan, "edge_p50": math.nan, "edge_p95": math.nan, "edge_max": math.nan}
    stats["nodes"] = int(points.shape[0])
    stats["cells_blocks"] = int(n_cells)
    return stats


def _mesh_memory_preflight(config: PipelineConfig, mesh_stats: dict[str, Any]) -> dict[str, float | bool | int]:
    cells = int(mesh_stats.get("cells_blocks", 0))
    nodes = int(mesh_stats.get("nodes", 0))
    if cells <= 0 or nodes <= 0:
        raise ValueError(f"mesh statistics must include positive cells_blocks and nodes; got {mesh_stats}")

    order_factor = 1.0 if int(config.nedelec_order) == 1 else 4.0
    polarization_factor = 1.0
    if str(config.polarization).strip().lower() == "cole-cole":
        polarization_factor += 0.03 * max(0, int(config.cole_n_terms))
    base_gb = cells * 2.85e-5 + nodes * 1.5e-6
    estimated_gb = base_gb * order_factor * polarization_factor
    limit_gb = float(config.memory_limit_gb)
    usable_limit_gb = limit_gb * float(config.memory_safety_fraction)
    audit = {
        "cells": cells,
        "nodes": nodes,
        "nedelec_order": int(config.nedelec_order),
        "limit_gb": limit_gb,
        "usable_limit_gb": usable_limit_gb,
        "estimated_gb": estimated_gb,
        "ok": bool(estimated_gb <= usable_limit_gb),
    }
    if not audit["ok"]:
        raise MemoryError(
            "estimated solver memory exceeds configured workstation budget: "
            f"estimated solver memory={estimated_gb:.2f} GB, usable limit={usable_limit_gb:.2f} GB "
            f"({limit_gb:.2f} GB * safety fraction {float(config.memory_safety_fraction):.2f}), "
            f"cells={cells}, nodes={nodes}. Reduce refinement/extent or use a smaller validation chunk."
        )
    return audit


def _mesh_memory_preflight_for_path(config: PipelineConfig, msh_path: Path) -> dict[str, float | bool | int] | None:
    try:
        stats = _mesh_size_statistics(msh_path)
        print(f"[mesh] size statistics={json.dumps(stats, indent=2)}", flush=True)
        memory_audit = _mesh_memory_preflight(config, stats)
        print(f"[mesh] memory preflight={json.dumps(memory_audit, indent=2)}", flush=True)
        return memory_audit
    except Exception as exc:
        if isinstance(exc, MemoryError):
            raise
        print(f"[mesh] mesh size statistics unavailable: {exc}", flush=True)
        return None


def _receiver_refinement_cloud_points(config: PipelineConfig) -> list[tuple[float, float, float]]:
    """Return extra embedded points that keep receiver cells locally small."""

    x, y, z = (float(v) for v in config.receiver)
    h = _receiver_anchor_mesh_size(config)
    radius = max(h, float(config.receiver_refinement_radius))
    n_layers = max(1, min(4, int(math.ceil(radius / h))))
    raw_points: list[tuple[float, float, float]] = []

    for dx, dy in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h), (h, h), (h, -h), (-h, h), (-h, -h)):
        raw_points.append((x + dx, y + dy, z))

    for layer in range(1, n_layers + 1):
        zz = z - layer * h
        raw_points.extend(
            [
                (x, y, zz),
                (x + h, y, zz),
                (x - h, y, zz),
                (x, y + h, zz),
                (x, y - h, zz),
            ]
        )

    points: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    receiver = (x, y, z)
    for point in raw_points:
        key = tuple(round(v, 12) for v in point)
        if key == receiver or key in seen:
            continue
        seen.add(key)
        points.append(key)
    return points


def _receiver_surface_refinement_points(config: PipelineConfig) -> list[tuple[float, float, float]]:
    """Return interface points above the receiver to prevent long near-surface cells."""

    import numpy as np

    x, y, _z = (float(v) for v in config.receiver)
    h = _receiver_anchor_mesh_size(config)
    radius = max(h, float(config.receiver_refinement_radius))
    n_layers = max(1, min(4, int(math.ceil(radius / h))))
    raw_points: list[tuple[float, float, float]] = []
    for ix in range(-n_layers, n_layers + 1):
        for iy in range(-n_layers, n_layers + 1):
            offset = np.asarray([float(ix) * h, float(iy) * h], dtype=float)
            if float(np.linalg.norm(offset)) <= radius:
                raw_points.append((x + float(offset[0]), y + float(offset[1]), 0.0))

    points: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in raw_points:
        key = tuple(round(v, 12) for v in point)
        if key in seen:
            continue
        seen.add(key)
        points.append(key)
    return points


def _receiver_anchor_mesh_size(config: PipelineConfig) -> float:
    """Return the local receiver anchoring length scale used for embedded points."""

    anchor = float(config.receiver_anchor_mesh_size)
    if anchor > 0.0:
        return max(anchor, 1.0e-9)
    return max(float(config.receiver_mesh_size), 1.0e-9)


def _source_refinement_cloud_points(config: PipelineConfig) -> list[tuple[float, float, float]]:
    """Return extra embedded points that keep source-line cells locally small."""

    import numpy as np

    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length
    horizontal = np.asarray([-tangent[1], tangent[0], 0.0], dtype=float)
    hnorm = float(np.linalg.norm(horizontal))
    if hnorm <= 1.0e-12:
        horizontal = np.asarray([0.0, 1.0, 0.0], dtype=float)
    else:
        horizontal /= hnorm

    h = max(float(config.source_mesh_size), 1.0e-9)
    radius = max(h, float(config.source_refinement_radius))
    n_layers = max(1, min(4, int(math.ceil(radius / h))))
    n_segments = max(1, int(math.ceil(length / h)))
    raw_points: list[tuple[float, float, float]] = []

    for i in range(n_segments + 1):
        base = p0 + (float(i) / float(n_segments)) * axis
        for sign in (-1.0, 1.0):
            p = base + sign * h * horizontal
            raw_points.append((float(p[0]), float(p[1]), float(p[2])))
        for layer in range(1, n_layers + 1):
            down = base.copy()
            down[2] -= layer * h
            raw_points.append((float(down[0]), float(down[1]), float(down[2])))
            for sign in (-1.0, 1.0):
                p = down + sign * h * horizontal
                raw_points.append((float(p[0]), float(p[1]), float(p[2])))

    for base in (p0 - h * tangent, p1 + h * tangent):
        raw_points.append((float(base[0]), float(base[1]), float(base[2])))
        for sign in (-1.0, 1.0):
            p = base + sign * h * horizontal
            raw_points.append((float(p[0]), float(p[1]), float(p[2])))
        for layer in range(1, n_layers + 1):
            down = base.copy()
            down[2] -= layer * h
            raw_points.append((float(down[0]), float(down[1]), float(down[2])))
            for sign in (-1.0, 1.0):
                p = down + sign * h * horizontal
                raw_points.append((float(p[0]), float(p[1]), float(p[2])))

    endpoints = {tuple(round(float(v), 12) for v in p0), tuple(round(float(v), 12) for v in p1)}
    points: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in raw_points:
        key = tuple(round(v, 12) for v in point)
        if key in endpoints or key in seen:
            continue
        seen.add(key)
        points.append(key)
    return points


def generate_verification_mesh(config: PipelineConfig) -> Path:
    """Generate the two-volume air-earth Gmsh model."""

    msh_path = config.mesh_path()
    if msh_path.exists() and not config.force_mesh:
        print(f"[mesh] Reusing existing mesh: {msh_path}", flush=True)
        _mesh_memory_preflight_for_path(config, msh_path)
        return msh_path

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.add("sotem_air_earth")

        occ = gmsh.model.occ
        x0, x1 = -config.x_extent, config.x_extent
        y0, y1 = -config.y_extent, config.y_extent
        air = occ.addBox(x0, y0, 0.0, 2.0 * config.x_extent, 2.0 * config.y_extent, config.air_height)
        layer_depths, _layer_resistivities = _normalise_layer_model(config)
        earth_depth_breaks = [0.0, *layer_depths, float(config.earth_depth)]
        earth_boxes: list[int] = []
        for top_depth, bottom_depth in zip(earth_depth_breaks[:-1], earth_depth_breaks[1:]):
            earth_boxes.append(
                occ.addBox(
                    x0,
                    y0,
                    -bottom_depth,
                    2.0 * config.x_extent,
                    2.0 * config.y_extent,
                    bottom_depth - top_depth,
                )
            )
        occ.fragment([(3, air)], [(3, earth) for earth in earth_boxes])

        p0 = occ.addPoint(*config.source_start, 5.0)
        p1 = occ.addPoint(*config.source_end, 5.0)
        source_line = occ.addLine(p0, p1)
        source_cloud = [occ.addPoint(*point, config.source_mesh_size) for point in _source_refinement_cloud_points(config)]
        receiver_anchor_mesh_size = _receiver_anchor_mesh_size(config)
        rp = occ.addPoint(*config.receiver, receiver_anchor_mesh_size)
        receiver_cloud = [
            occ.addPoint(*point, receiver_anchor_mesh_size) for point in _receiver_refinement_cloud_points(config)
        ]
        receiver_surface_cloud = [
            occ.addPoint(*point, receiver_anchor_mesh_size) for point in _receiver_surface_refinement_points(config)
        ]
        occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        air_vols: list[int] = []
        earth_vols: list[int] = []
        for _, tag in volumes:
            _, _, zc = occ.getCenterOfMass(3, tag)
            if zc > 0.0:
                air_vols.append(tag)
            else:
                earth_vols.append(tag)
        if not air_vols or not earth_vols:
            raise RuntimeError("failed to identify air and earth volumes")
        gmsh.model.addPhysicalGroup(3, air_vols, PHYS_AIR)
        gmsh.model.setPhysicalName(3, PHYS_AIR, "air")
        gmsh.model.addPhysicalGroup(3, earth_vols, PHYS_EARTH)
        gmsh.model.setPhysicalName(3, PHYS_EARTH, "earth")

        surfaces = gmsh.model.getEntities(2)
        outer: list[int] = []
        interface: list[int] = []
        tol = 1.0e-6
        for _, tag in surfaces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            if abs(zmin) < tol and abs(zmax) < tol:
                interface.append(tag)
            elif (
                abs(xmin - x0) < tol
                and abs(xmax - x0) < tol
                or abs(xmin - x1) < tol
                and abs(xmax - x1) < tol
                or abs(ymin - y0) < tol
                and abs(ymax - y0) < tol
                or abs(ymin - y1) < tol
                and abs(ymax - y1) < tol
                or abs(zmin + config.earth_depth) < tol
                and abs(zmax + config.earth_depth) < tol
                or abs(zmin - config.air_height) < tol
                and abs(zmax - config.air_height) < tol
            ):
                outer.append(tag)
        if not outer:
            raise RuntimeError("failed to identify outer boundary surfaces")
        if not interface:
            raise RuntimeError("failed to identify z=0 air-earth interface surfaces")
        gmsh.model.addPhysicalGroup(2, outer, PHYS_OUTER)
        gmsh.model.setPhysicalName(2, PHYS_OUTER, "outer_boundary")
        gmsh.model.addPhysicalGroup(2, interface, PHYS_SURFACE)
        gmsh.model.setPhysicalName(2, PHYS_SURFACE, "air_earth_interface")
        gmsh.model.addPhysicalGroup(1, [source_line], PHYS_SOURCE_LINE)
        gmsh.model.setPhysicalName(1, PHYS_SOURCE_LINE, "source_wire")

        top_layer_bottom = layer_depths[0] if layer_depths else float(config.earth_depth)
        top_earth_vols = [
            earth_vol
            for earth_vol in earth_vols
            if -top_layer_bottom <= occ.getCenterOfMass(3, earth_vol)[2] <= 0.0
        ]
        if not top_earth_vols:
            raise RuntimeError("failed to identify top earth layer volumes for source and receiver embedding")
        for earth_vol in top_earth_vols:
            gmsh.model.mesh.embed(1, [source_line], 3, earth_vol)
            gmsh.model.mesh.embed(0, [*source_cloud, rp, *receiver_cloud], 3, earth_vol)
        for interface_surf in interface:
            gmsh.model.mesh.embed(0, receiver_surface_cloud, 2, interface_surf)

        f_line = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_line, "CurvesList", [source_line])
        gmsh.model.mesh.field.setNumber(f_line, "Sampling", 100)
        f_source = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_source, "InField", f_line)
        gmsh.model.mesh.field.setNumber(f_source, "SizeMin", config.source_mesh_size)
        gmsh.model.mesh.field.setNumber(f_source, "SizeMax", 2500.0)
        gmsh.model.mesh.field.setNumber(f_source, "DistMin", config.source_refinement_radius)
        gmsh.model.mesh.field.setNumber(f_source, "DistMax", 2500.0)

        f_point = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_point, "PointsList", [rp, *receiver_surface_cloud])
        f_receiver = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_receiver, "InField", f_point)
        gmsh.model.mesh.field.setNumber(f_receiver, "SizeMin", receiver_anchor_mesh_size)
        gmsh.model.mesh.field.setNumber(f_receiver, "SizeMax", 2500.0)
        gmsh.model.mesh.field.setNumber(f_receiver, "DistMin", config.receiver_refinement_radius)
        gmsh.model.mesh.field.setNumber(f_receiver, "DistMax", 3000.0)

        f_receiver_ball = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "XCenter", config.receiver[0])
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "YCenter", config.receiver[1])
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "ZCenter", config.receiver[2])
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "Radius", config.receiver_refinement_radius)
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "VIn", receiver_anchor_mesh_size)
        gmsh.model.mesh.field.setNumber(f_receiver_ball, "VOut", 2500.0)

        diffusion_box = _diffusion_refinement_box(config)
        f_box = _add_box_field(
            gmsh,
            -diffusion_box["radius"],
            diffusion_box["radius"],
            -diffusion_box["radius"],
            diffusion_box["radius"],
            -diffusion_box["depth"],
            diffusion_box["top"],
            diffusion_box["mesh_size"],
            2500.0,
            diffusion_box["radius"],
        )
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_source, f_receiver, f_receiver_ball, f_box])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(5.0, config.source_mesh_size, receiver_anchor_mesh_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", 3000.0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        gmsh.model.mesh.generate(3)
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(msh_path))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
        n_elems = sum(len(tags) for tags in elem_tags)
        physical = {
            "air": PHYS_AIR,
            "earth": PHYS_EARTH,
            "outer_boundary": PHYS_OUTER,
            "air_earth_interface": PHYS_SURFACE,
            "source_wire": PHYS_SOURCE_LINE,
        }
        print(f"[mesh] Wrote {msh_path}", flush=True)
        print(f"[mesh] nodes={len(node_tags)}, elements={n_elems}, element_types={list(elem_types)}", flush=True)
        print(f"[mesh] physical tags={physical}", flush=True)
    finally:
        gmsh.finalize()

    _mesh_memory_preflight_for_path(config, msh_path)
    return msh_path


def load_mesh(config: PipelineConfig):
    """Load the Gmsh mesh and physical tags into DOLFINx."""

    from mpi4py import MPI
    from dolfinx.io import gmshio

    msh_path = config.mesh_path()
    msh, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, 0)
    log(
        f"[load_mesh] local cells={msh.topology.index_map(msh.topology.dim).size_local}, "
        f"global cells={msh.topology.index_map(msh.topology.dim).size_global}",
        comm=msh.comm,
    )
    return msh, cell_tags, facet_tags


def build_function_spaces(msh, config: PipelineConfig | None = None):
    """Build Nedelec, DG0 scalar/vector, and H1 spaces."""

    from dolfinx import fem

    nedelec_order = int(config.nedelec_order) if config is not None else 1
    V = fem.functionspace(msh, ("N1curl", nedelec_order))
    Q = fem.functionspace(msh, ("DG", 0))
    W = fem.functionspace(msh, ("DG", 0, (3,)))
    S = fem.functionspace(msh, ("Lagrange", 1))
    log(
        f"[spaces] V=N1curl({nedelec_order}) dofs global={V.dofmap.index_map.size_global}, "
        f"DG0 cells global={Q.dofmap.index_map.size_global}",
        comm=msh.comm,
    )
    return {"V": V, "Q": Q, "W": W, "S": S}


def _assign_dg0_by_cell(function, cells, value: float) -> None:
    dm = function.function_space.dofmap
    arr = function.x.array
    for cell in cells:
        dofs = dm.cell_dofs(int(cell))
        arr[dofs] = value


def _assign_dg0_by_cell_values(function, cells, values) -> None:
    dm = function.function_space.dofmap
    arr = function.x.array
    for cell, value in zip(cells, values):
        dofs = dm.cell_dofs(int(cell))
        arr[dofs] = float(value)


def _make_dolfinx_materials_from_cell_material_map(
    msh,
    spaces,
    material_map,
    *,
    dt: float,
    mu_inv_value: float = 1.0,
):
    """Convert a cell-marker material map into DOLFINx DG0 material fields."""

    import numpy as np
    from dolfinx import fem

    from atem3d.materials.prony import DebyeTerm, PronyConductivity

    Q = spaces["Q"]
    n_local = msh.topology.index_map(msh.topology.dim).size_local
    markers = np.asarray(material_map.markers, dtype=int)
    if markers.shape != (n_local,):
        raise ValueError("material_map markers must match the number of local cells")
    dt_value = float(dt)
    if dt_value <= 0.0 or not math.isfinite(dt_value):
        raise ValueError("dt must be finite and positive")
    cells = np.arange(n_local, dtype=np.int32)

    sigma_initial_values = np.asarray(material_map.sigma0(), dtype=float)
    sigma_infinity_values = np.asarray(material_map.sigma_inf(), dtype=float)
    term_taus = []
    for cell in range(n_local):
        for term in material_map.material_for_cell(cell).terms:
            tau = float(term.tau)
            if tau not in term_taus:
                term_taus.append(tau)
    term_taus.sort()
    delta_values_by_tau = {tau: np.zeros(n_local, dtype=float) for tau in term_taus}
    for cell in range(n_local):
        for term in material_map.material_for_cell(cell).terms:
            delta_values_by_tau[float(term.tau)][cell] += float(term.delta_sigma)

    sigma_eff_values = sigma_infinity_values.copy()
    representative_terms = []
    delta_functions = []
    for index, tau in enumerate(term_taus):
        delta_values = delta_values_by_tau[tau]
        beta = dt_value / (tau + dt_value)
        sigma_eff_values -= beta * delta_values
        representative_terms.append(DebyeTerm(delta_sigma=float(np.max(delta_values)), tau=tau))
        delta_fn = fem.Function(Q, name=f"delta_sigma_term_{index}")
        _assign_dg0_by_cell_values(delta_fn, cells, delta_values)
        delta_fn.x.scatter_forward()
        delta_functions.append(delta_fn)

    sigma = fem.Function(Q, name="sigma")
    sigma_initial = fem.Function(Q, name="sigma_initial")
    sigma_infinity = fem.Function(Q, name="sigma_infinity")
    mu_inv = fem.Function(Q, name="mu_inv")
    _assign_dg0_by_cell_values(sigma, cells, sigma_eff_values)
    _assign_dg0_by_cell_values(sigma_initial, cells, sigma_initial_values)
    _assign_dg0_by_cell_values(sigma_infinity, cells, sigma_infinity_values)
    _assign_dg0_by_cell_values(mu_inv, cells, np.full(n_local, float(mu_inv_value), dtype=float))
    for function in (sigma, sigma_initial, sigma_infinity, mu_inv):
        function.x.scatter_forward()

    if representative_terms:
        representative_material = PronyConductivity(
            sigma_inf=float(np.max(sigma_infinity_values)),
            terms=representative_terms,
        )
        polarizable_cells = np.flatnonzero(
            np.sum(np.vstack([delta_values_by_tau[tau] for tau in term_taus]), axis=0) > 0.0
        ).astype(np.int32)
        debye = {
            "terms": tuple(representative_terms),
            "delta_functions": delta_functions,
            "polarizable_cells": polarizable_cells,
            "sigma_infinity": float(representative_material.sigma_inf),
        }
    else:
        representative_material = PronyConductivity.no_ip(float(np.max(sigma_infinity_values)))
        debye = None

    background_marker = int(np.min(markers))
    marker_counts = {str(int(marker)): int(np.count_nonzero(markers == marker)) for marker in np.unique(markers)}
    return {
        "materials": {
            "sigma": sigma,
            "sigma_initial": sigma_initial,
            "sigma_infinity": sigma_infinity,
            "mu_inv": mu_inv,
        },
        "debye": debye,
        "representative_material": representative_material,
        "diagnostics": {
            "cell_count": int(n_local),
            "marker_counts": marker_counts,
            "leakage_cell_count": int(np.count_nonzero(markers != background_marker)),
            "sigma0_min": float(np.min(sigma_initial_values)),
            "sigma0_max": float(np.max(sigma_initial_values)),
            "sigma_inf_min": float(np.min(sigma_infinity_values)),
            "sigma_inf_max": float(np.max(sigma_infinity_values)),
        },
    }


def _write_small_gmsh_terrain_leakage_mesh(mesh_path, *, mesh_size: float = 0.5):
    """Write a tiny Gmsh terrain volume for leakage-channel forward smokes."""

    from pathlib import Path

    import gmsh

    path = Path(mesh_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_size = float(mesh_size)
    if mesh_size <= 0.0 or not math.isfinite(mesh_size):
        raise ValueError("mesh_size must be finite and positive")

    top = {
        "p5": 0.02,
        "p6": 0.14,
        "p7": 0.06,
        "p8": -0.04,
    }
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("small_terrain_leakage")
        geo = gmsh.model.geo

        p1 = geo.addPoint(0.0, 0.0, -1.0, mesh_size)
        p2 = geo.addPoint(1.0, 0.0, -1.0, mesh_size)
        p3 = geo.addPoint(1.0, 1.0, -1.0, mesh_size)
        p4 = geo.addPoint(0.0, 1.0, -1.0, mesh_size)
        p5 = geo.addPoint(0.0, 0.0, top["p5"], mesh_size)
        p6 = geo.addPoint(1.0, 0.0, top["p6"], mesh_size)
        p7 = geo.addPoint(1.0, 1.0, top["p7"], mesh_size)
        p8 = geo.addPoint(0.0, 1.0, top["p8"], mesh_size)

        b1 = geo.addLine(p1, p2)
        b2 = geo.addLine(p2, p3)
        b3 = geo.addLine(p3, p4)
        b4 = geo.addLine(p4, p1)
        t1 = geo.addLine(p5, p6)
        t2 = geo.addLine(p6, p7)
        t3 = geo.addLine(p7, p8)
        t4 = geo.addLine(p8, p5)
        v1 = geo.addLine(p1, p5)
        v2 = geo.addLine(p2, p6)
        v3 = geo.addLine(p3, p7)
        v4 = geo.addLine(p4, p8)

        bottom = geo.addPlaneSurface([geo.addCurveLoop([b1, b2, b3, b4])])
        top_surface = geo.addPlaneSurface([geo.addCurveLoop([t1, t2, t3, t4])])
        side_y0 = geo.addPlaneSurface([geo.addCurveLoop([b1, v2, -t1, -v1])])
        side_x1 = geo.addPlaneSurface([geo.addCurveLoop([b2, v3, -t2, -v2])])
        side_y1 = geo.addPlaneSurface([geo.addCurveLoop([b3, v4, -t3, -v3])])
        side_x0 = geo.addPlaneSurface([geo.addCurveLoop([b4, v1, -t4, -v4])])
        surface_loop = geo.addSurfaceLoop([bottom, top_surface, side_y0, side_x1, side_y1, side_x0])
        volume = geo.addVolume([surface_loop])
        geo.synchronize()

        gmsh.model.addPhysicalGroup(3, [volume], PHYS_EARTH)
        gmsh.model.setPhysicalName(3, PHYS_EARTH, "earth")
        outer_surfaces = [bottom, top_surface, side_y0, side_x1, side_y1, side_x0]
        gmsh.model.addPhysicalGroup(2, outer_surfaces, PHYS_OUTER)
        gmsh.model.setPhysicalName(2, PHYS_OUTER, "outer_boundary")
        gmsh.model.addPhysicalGroup(2, [top_surface], PHYS_SURFACE)
        gmsh.model.setPhysicalName(2, PHYS_SURFACE, "terrain_surface")
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        element_count = sum(len(tags) for tags in elem_tags)
    finally:
        gmsh.finalize()

    elevations = list(top.values())
    return {
        "mesh_path": str(path),
        "node_count": int(len(node_tags)),
        "cell_count": int(element_count),
        "terrain_elevation_min": float(min(elevations)),
        "terrain_elevation_max": float(max(elevations)),
    }


def _copy_dg0_function(function, name: str):
    from dolfinx import fem

    copied = fem.Function(function.function_space, name=name)
    copied.x.array[:] = function.x.array
    copied.x.scatter_forward()
    return copied


def apply_transient_sponge(msh, materials: dict[str, Any], config: PipelineConfig) -> None:
    """Add a direct-time-domain sponge conductivity to transient material fields."""

    import numpy as np
    from dolfinx import fem

    Q = materials["sigma"].function_space
    sponge_sigma = fem.Function(Q, name="sponge_sigma")
    centers = _cell_centers(msh)
    addition = _sponge_sigma_addition_for_centers(centers, config)
    n_local = centers.shape[0]
    sponge_sigma.x.array[:n_local] = addition
    sponge_sigma.x.scatter_forward()
    materials["sponge_sigma"] = sponge_sigma

    enabled = bool(np.any(addition > 0.0))
    materials["sponge_enabled"] = enabled
    if not enabled:
        log("[sponge] disabled: transient material fields remain physical.", comm=msh.comm)
        return

    for key in ("sigma", "sigma_infinity"):
        materials[key].x.array[:n_local] += addition
        materials[key].x.scatter_forward()
    if bool(config.sponge_apply_to_initial):
        materials["sigma_initial"].x.array[:n_local] += addition
        materials["sigma_initial"].x.scatter_forward()

    log(
        "[sponge] enabled: "
        f"strength={config.sponge_strength:.6e} S/m, thickness={config.sponge_thickness:.6g} m, "
        f"power={config.sponge_power:.6g}, apply_to_initial={bool(config.sponge_apply_to_initial)}, "
        f"sides={','.join(_normalise_sponge_sides(config.sponge_sides))}, "
        f"local_changed_cells={int(np.count_nonzero(addition > 0.0))}, "
        f"max_added_sigma={float(np.max(addition)):.6e} S/m",
        comm=msh.comm,
    )


def assign_materials(msh, cell_tags, spaces: dict[str, Any], config: PipelineConfig):
    """Assign cell-wise sigma and mu^{-1} with DG0 functions."""

    import numpy as np
    from scipy.constants import mu_0
    from dolfinx import fem

    Q = spaces["Q"]
    sigma = fem.Function(Q, name="sigma")
    sigma_inf = fem.Function(Q, name="sigma_infinity")
    rho = fem.Function(Q, name="rho")
    mu = fem.Function(Q, name="mu")
    mu_inv = fem.Function(Q, name="mu_inv")

    air_cells = cell_tags.find(PHYS_AIR)
    earth_cells = cell_tags.find(PHYS_EARTH)
    if len(air_cells) == 0 or len(earth_cells) == 0:
        raise RuntimeError("cell tags must include air=100 and earth=101")

    layer_depths, layer_resistivities = _normalise_layer_model(config)
    sigma_air = 1.0 / config.rho_air
    _assign_dg0_by_cell(sigma, air_cells, sigma_air)
    _assign_dg0_by_cell(sigma_inf, air_cells, sigma_air)
    _assign_dg0_by_cell(rho, air_cells, config.rho_air)
    if layer_depths:
        centers = _cell_centers(msh)
        earth_depths = -centers[np.asarray(earth_cells, dtype=int), 2]
        earth_rho = np.asarray([_earth_resistivity_at_depth(depth, config) for depth in earth_depths], dtype=float)
        earth_sigma = 1.0 / earth_rho
        _assign_dg0_by_cell_values(sigma, earth_cells, earth_sigma)
        _assign_dg0_by_cell_values(sigma_inf, earth_cells, earth_sigma)
        _assign_dg0_by_cell_values(rho, earth_cells, earth_rho)
    else:
        sigma_earth = 1.0 / layer_resistivities[0]
        _assign_dg0_by_cell(sigma, earth_cells, sigma_earth)
        _assign_dg0_by_cell(sigma_inf, earth_cells, sigma_earth)
        _assign_dg0_by_cell(rho, earth_cells, layer_resistivities[0])
    _assign_dg0_by_cell(mu, air_cells, mu_0 * config.mu_r_air)
    _assign_dg0_by_cell(mu, earth_cells, mu_0 * config.mu_r_earth)
    _assign_dg0_by_cell(mu_inv, air_cells, 1.0 / (mu_0 * config.mu_r_air))
    _assign_dg0_by_cell(mu_inv, earth_cells, 1.0 / (mu_0 * config.mu_r_earth))
    sigma.x.scatter_forward()
    sigma_inf.x.scatter_forward()
    rho.x.scatter_forward()
    mu.x.scatter_forward()
    mu_inv.x.scatter_forward()
    sigma_physical = _copy_dg0_function(sigma, "sigma_physical")
    sigma_initial = _copy_dg0_function(sigma, "sigma_initial")
    sigma_infinity_physical = _copy_dg0_function(sigma_inf, "sigma_infinity_physical")

    log(
        "[materials] "
        f"sigma_air={sigma_air:.6e} S/m, earth_layers={_format_float_list(layer_resistivities)} ohm m, "
        f"air_cells(local)={len(air_cells)}, earth_cells(local)={len(earth_cells)}",
        comm=msh.comm,
    )
    return {
        "sigma": sigma,
        "sigma_infinity": sigma_inf,
        "sigma_physical": sigma_physical,
        "sigma_initial": sigma_initial,
        "sigma_infinity_physical": sigma_infinity_physical,
        "rho": rho,
        "mu": mu,
        "mu_inv": mu_inv,
    }


def _dolfinx_supports_ridge_integral() -> bool:
    try:
        import dolfinx
        import dolfinx.io

        version = tuple(int(part) for part in dolfinx.__version__.split(".")[:2])
        return version >= (0, 10) and hasattr(dolfinx.io, "gmsh")
    except Exception:
        return False


def _cell_centers(msh):
    import numpy as np

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geometry = msh.geometry.x
    n_local = msh.topology.index_map(tdim).size_local
    centers = np.zeros((n_local, 3), dtype=float)
    for cell in range(n_local):
        vertices = c_to_v.links(cell)
        centers[cell] = geometry[vertices].mean(axis=0)
    return centers


def _cell_centers_radii_volumes(msh):
    import numpy as np

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geometry = msh.geometry.x
    n_local = msh.topology.index_map(tdim).size_local
    centers = np.zeros((n_local, 3), dtype=float)
    radii = np.zeros(n_local, dtype=float)
    volumes = np.zeros(n_local, dtype=float)
    for cell in range(n_local):
        vertices = c_to_v.links(cell)
        coords = geometry[vertices]
        center = coords.mean(axis=0)
        centers[cell] = center
        radii[cell] = float(np.max(np.linalg.norm(coords - center, axis=1)))
        if coords.shape[0] >= 4:
            mat = np.vstack((coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0])).T
            volumes[cell] = abs(float(np.linalg.det(mat))) / 6.0
    return centers, radii, volumes


def _cell_geometry(msh, cell: int):
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    vertices = msh.topology.connectivity(tdim, 0).links(cell)
    return msh.geometry.x[vertices]


def _nedelec_interpolation_points(msh, spaces):
    """Return physical Nedelec interpolation points for local cells.

    The returned table is intended for primary-provider sampling before
    interpolating tabulated primary fields back into the same Nedelec space.
    It currently assumes affine tetrahedral geometry, which is the mesh type
    used by the production Gmsh pipeline and the WSL smokes.
    """

    import numpy as np

    V = spaces["V"]
    reference_points = np.asarray(V.element.interpolation_points(), dtype=float)
    if reference_points.ndim != 2 or reference_points.shape[1] != 3:
        raise ValueError("Nedelec interpolation points must have shape (n_points_per_cell, 3)")
    tdim = msh.topology.dim
    n_local = msh.topology.index_map(tdim).size_local
    points = []
    cells = []
    for cell in range(n_local):
        coords = _cell_geometry(msh, cell)
        if coords.shape[0] != 4:
            raise ValueError("Nedelec interpolation point export currently requires tetrahedral cells")
        origin = coords[0]
        jacobian = np.column_stack((coords[1] - origin, coords[2] - origin, coords[3] - origin))
        physical = origin + reference_points @ jacobian.T
        points.append(physical)
        cells.extend([cell] * reference_points.shape[0])
    if points:
        point_array = np.vstack(points)
    else:
        point_array = np.empty((0, 3), dtype=float)
    return {
        "points": point_array,
        "cells": np.asarray(cells, dtype=np.int32),
        "points_per_cell": int(reference_points.shape[0]),
    }


def _manual_line_source_quadrature_count(length: float, config: PipelineConfig) -> int:
    """Return Gauss points for the discontinuous cellwise line-source integrand."""

    requested = int(config.source_quadrature_points)
    if requested > 0:
        return max(2, requested)
    length = float(length)
    if length <= 0.0:
        raise ValueError("length must be positive")
    mesh_size = max(float(config.source_mesh_size), 1.0e-12)
    target_spacing = min(2.0, max(mesh_size / 200.0, length / 20001.0))
    return max(51, int(math.ceil(length / target_spacing)) + 1)


def _source_line_segments_from_meshio_blocks(points, cell_blocks, physical_blocks, config: PipelineConfig):
    """Return source-wire line segments from meshio-style line blocks."""

    import numpy as np

    pts = np.asarray(points, dtype=float)
    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length
    selected: list[np.ndarray] = []
    for block, physical in zip(cell_blocks, physical_blocks):
        block_type, data = block
        if str(block_type) != "line":
            continue
        lines = np.asarray(data, dtype=np.int64)
        tags = np.asarray(physical, dtype=np.int64)
        if lines.size == 0:
            continue
        mask = tags == int(PHYS_SOURCE_LINE)
        if not np.any(mask):
            continue
        for edge in lines[mask]:
            segment = pts[np.asarray(edge, dtype=np.int64)]
            if segment.shape != (2, 3):
                continue
            selected.append(segment)
    if not selected:
        return None

    segments = np.asarray(selected, dtype=float)
    starts = np.dot(segments[:, 0, :] - p0[None, :], tangent) / length
    ends = np.dot(segments[:, 1, :] - p0[None, :], tangent) / length
    flip = ends < starts
    if np.any(flip):
        segments[flip] = segments[flip, ::-1, :]
        starts, ends = np.minimum(starts, ends), np.maximum(starts, ends)
    order = np.argsort(starts)
    segments = segments[order]
    lengths = np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1)
    return {
        "segments": segments,
        "segment_count": int(segments.shape[0]),
        "total_length": float(np.sum(lengths)),
        "min_segment_length": float(np.min(lengths)),
        "max_segment_length": float(np.max(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
    }


def _source_line_segments_from_msh(mesh_path: Path, config: PipelineConfig):
    """Read source-wire line segments from the Gmsh .msh file when available."""

    try:
        import meshio
    except Exception:
        return None
    path = Path(mesh_path)
    if not path.is_file():
        return None
    try:
        mesh = meshio.read(path)
    except Exception:
        return None
    physical = mesh.cell_data.get("gmsh:physical") if hasattr(mesh, "cell_data") else None
    if not physical:
        return None
    blocks = [(block.type, block.data) for block in mesh.cells]
    return _source_line_segments_from_meshio_blocks(mesh.points, blocks, physical, config)


def _manual_line_source_integration_points(config: PipelineConfig, *, mesh_segments=None):
    """Build line-integration points and weights for the manual Nedelec source."""

    import numpy as np

    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length

    if mesh_segments is not None:
        segments = np.asarray(mesh_segments["segments"], dtype=float)
        global_npts = _manual_line_source_quadrature_count(length, config)
        target_spacing = length / max(global_npts - 1, 1)
        min_rule_points = max(2, int(config.nedelec_order) + 1)
        all_points: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        all_svals: list[np.ndarray] = []
        segment_rule_points: list[int] = []
        for segment in segments:
            a = np.asarray(segment[0], dtype=float)
            b = np.asarray(segment[1], dtype=float)
            seg_axis = b - a
            seg_length = float(np.linalg.norm(seg_axis))
            if seg_length <= 0.0:
                continue
            per_segment = max(min_rule_points, int(math.ceil(seg_length / max(target_spacing, 1.0e-12))) + 1)
            qx, qw = np.polynomial.legendre.leggauss(per_segment)
            local_s = 0.5 * (qx + 1.0)
            seg_points = a[None, :] + local_s[:, None] * seg_axis[None, :]
            seg_weights = 0.5 * seg_length * qw
            seg_svals = np.dot(seg_points - p0[None, :], tangent) / length
            all_points.append(seg_points)
            all_weights.append(seg_weights)
            all_svals.append(seg_svals)
            segment_rule_points.append(int(per_segment))
        if all_points:
            points = np.vstack(all_points)
            weights = np.concatenate(all_weights)
            svals = np.concatenate(all_svals)
            order = np.argsort(svals)
            diagnostics = {
                "integration_mode": "mesh_segments",
                "segment_count": int(mesh_segments["segment_count"]),
                "segment_total_length": float(mesh_segments["total_length"]),
                "segment_min_length": float(mesh_segments["min_segment_length"]),
                "segment_max_length": float(mesh_segments["max_segment_length"]),
                "segment_mean_length": float(mesh_segments["mean_segment_length"]),
                "quadrature_points_per_segment_min": int(min(segment_rule_points)),
                "quadrature_points_per_segment_max": int(max(segment_rule_points)),
                "quadrature_points_per_segment_mean": float(sum(segment_rule_points) / len(segment_rule_points)),
                "quadrature_points": int(points.shape[0]),
            }
            diagnostics["line_orientation"] = _manual_line_orientation_diagnostics(
                config,
                weights=np.asarray(weights, dtype=float),
                svals=np.asarray(svals, dtype=float),
            )
            return points[order], weights[order], svals[order], diagnostics

    npts = _manual_line_source_quadrature_count(length, config)
    qx, qw = np.polynomial.legendre.leggauss(npts)
    svals = 0.5 * (qx + 1.0)
    weights = 0.5 * length * qw
    points = p0[None, :] + svals[:, None] * axis[None, :]
    diagnostics = {
        "integration_mode": "global_gauss",
        "segment_count": 0,
        "segment_total_length": 0.0,
        "segment_min_length": 0.0,
        "segment_max_length": 0.0,
        "segment_mean_length": 0.0,
        "quadrature_points_per_segment": 0,
        "quadrature_points": int(npts),
    }
    diagnostics["line_orientation"] = _manual_line_orientation_diagnostics(
        config,
        weights=weights,
        svals=svals,
    )
    return points, weights, svals, diagnostics


def _manual_line_orientation_diagnostics(config: PipelineConfig, *, weights, svals):
    """Return source-line quadrature orientation diagnostics for manual_line."""

    import numpy as np

    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    displacement = p1 - p0
    source_length = float(np.linalg.norm(displacement))
    if source_length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = displacement / source_length
    weights = np.asarray(weights, dtype=float).reshape(-1)
    svals = np.asarray(svals, dtype=float).reshape(-1)
    if weights.size != svals.size:
        raise ValueError("weights and svals must have the same size")
    if weights.size == 0:
        raise ValueError("source line orientation diagnostics require at least one quadrature point")
    weight_sum = float(np.sum(weights))
    integrated = tangent * weight_sum
    signed_parallel = float(np.dot(integrated, tangent))
    transverse = integrated - signed_parallel * tangent
    integrated_norm = float(np.linalg.norm(integrated))
    orientation_cosine = (
        0.0
        if integrated_norm == 0.0
        else float(np.dot(integrated, displacement) / (integrated_norm * source_length))
    )
    return {
        "source_start": _numeric_list(p0),
        "source_end": _numeric_list(p1),
        "source_length_m": source_length,
        "expected_displacement_m": _numeric_list(displacement),
        "integrated_displacement_m": _numeric_list(integrated),
        "quadrature_weight_sum_m": weight_sum,
        "signed_parallel_projection_m": signed_parallel,
        "transverse_residual_m": float(np.linalg.norm(transverse)),
        "orientation_cosine": orientation_cosine,
        "relative_parallel_length_error": float(abs(signed_parallel - source_length) / source_length),
        "s_parameter_min": float(np.min(svals)),
        "s_parameter_max": float(np.max(svals)),
        "s_parameter_monotonic": bool(np.all(np.diff(svals) >= -1.0e-14)),
        "reversed_orientation": bool(orientation_cosine <= -0.99),
    }


def _numeric_list(values) -> list[float]:
    import numpy as np

    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _build_manual_line_source(msh, spaces: dict[str, Any], config: PipelineConfig):
    """Assemble -dI/dt int_Gamma t.v dl by direct Nedelec basis tabulation."""

    import numpy as np
    from dolfinx import fem
    from dolfinx import geometry
    from petsc4py import PETSc

    V = spaces["V"]
    source_fun = fem.Function(V, name="manual_line_source_work")
    source_vec = source_fun.x.petsc_vec.copy()
    source_vec.set(0.0)
    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length
    mesh_segments = _source_line_segments_from_msh(config.mesh_path(), config)
    points, weights, svals, integration_diagnostics = _manual_line_source_integration_points(
        config,
        mesh_segments=mesh_segments,
    )
    npts = int(points.shape[0])

    tree = geometry.bb_tree(msh, msh.topology.dim, padding=1.0e-8)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    be = V.element.basix_element
    dofmap = V.dofmap
    index_map = dofmap.index_map
    local_cells = msh.topology.index_map(msh.topology.dim).size_local
    msh.topology.create_entity_permutations()
    permutations = msh.topology.get_cell_permutation_info()
    added = 0
    missed = 0
    hit_cell_ids: list[int | None] = []
    cell_l1_contributions: dict[int, float] = {}
    dof_l1_contributions: dict[int, float] = {}
    for ip, point in enumerate(points):
        cells = [int(c) for c in colliding.links(ip) if int(c) < local_cells]
        if not cells:
            missed += 1
            hit_cell_ids.append(None)
            continue
        cell = cells[0]
        hit_cell_ids.append(cell)
        cell_geom = _cell_geometry(msh, cell)
        X = msh.geometry.cmap.pull_back(point.reshape(1, 3), cell_geom)
        tab = be.tabulate(0, X)[0]
        J = np.column_stack((cell_geom[1] - cell_geom[0], cell_geom[2] - cell_geom[0], cell_geom[3] - cell_geom[0]))
        Jb = J.reshape(1, 3, 3)
        detJ = np.asarray([np.linalg.det(J)], dtype=float)
        K = np.linalg.inv(J).reshape(1, 3, 3)
        phi = be.push_forward(tab, Jb, detJ, K)[0]
        local = np.asarray(phi @ tangent, dtype=float) * float(weights[ip])
        if V.element.needs_dof_transformations:
            local_t = local.copy()
            V.element.Tt_inv_apply(local_t, permutations[cell : cell + 1], 1)
            local = local_t
        local_dofs = dofmap.cell_dofs(cell)
        global_dofs = index_map.local_to_global(local_dofs).astype(PETSc.IntType)
        source_vec.setValues(global_dofs, local, addv=PETSc.InsertMode.ADD_VALUES)
        local_l1 = float(np.sum(np.abs(local)))
        cell_l1_contributions[cell] = cell_l1_contributions.get(cell, 0.0) + local_l1
        for dof, value in zip(global_dofs, local):
            key = int(dof)
            dof_l1_contributions[key] = dof_l1_contributions.get(key, 0.0) + abs(float(value))
        added += 1
    source_vec.assemble()
    local_diagnostics = _summarize_manual_line_source_local_diagnostics(
        npts=npts,
        added=added,
        missed=missed,
        hit_cell_ids=hit_cell_ids,
        svals=svals,
        cell_l1_contributions=cell_l1_contributions,
        dof_l1_contributions=dof_l1_contributions,
    )
    local_diagnostics.update(integration_diagnostics)
    log(
        f"[source] mode=manual_line; integration={integration_diagnostics['integration_mode']}; "
        f"quadrature points={npts}; added={added}; missed={missed}; "
        "assembled direct Nedelec line integral int_Gamma t.v dl",
        comm=msh.comm,
    )
    return {"mode": "manual_line", "coeff": None, "vector": source_vec, "local_projection_diagnostics": local_diagnostics}


def build_source(msh, spaces: dict[str, Any], config: PipelineConfig, cell_tags=None):
    """Build the source vector for a unit current derivative."""

    import numpy as np
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    requested = config.source_mode.lower()
    ridge_ok = _dolfinx_supports_ridge_integral()
    if requested in {"auto", "line", "manual_line"}:
        source = _build_manual_line_source(msh, spaces, config)
        source["vector"], source["projection_diagnostics"] = _enforce_source_charge_conservation(
            msh,
            spaces,
            source["vector"],
            config,
            apply_projection=str(config.source_projection_mode).strip().lower() == "charge_conserving",
        )
        return source
    if requested in {"auto", "line"} and ridge_ok:
        raise SystemExit(
            "Ridge line integral mode was selected by feature detection, but this "
            "single-file implementation has not enabled a verified ridge_tags "
            "path for the current API. Re-run with --source-mode regularized."
        )

    mode = "regularized_volume"
    V = spaces["V"]
    W = spaces["W"]
    v = ufl.TestFunction(V)
    source_coeff = fem.Function(W, name="J_unit_regularized")
    nominal_area = math.pi * config.wire_radius**2
    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length
    centers, cell_radii, cell_volumes = _cell_centers_radii_volumes(msh)
    arr = source_coeff.x.array
    dm = W.dofmap
    block_size = int(dm.bs)
    kernel_radius = max(float(config.wire_radius), 1.0e-12)
    weights: list[tuple[int, float]] = []
    nearest: list[tuple[float, int]] = []
    earth_cells = None
    if cell_tags is not None:
        earth_cells = {int(cell) for cell in cell_tags.find(PHYS_EARTH)}
    for cell, center in enumerate(centers):
        if earth_cells is not None and cell not in earth_cells:
            continue
        rel = center - p0
        s = float(np.dot(rel, tangent))
        closest = p0 + min(max(s, 0.0), length) * tangent
        radius = float(np.linalg.norm(center - closest))
        if 0.0 <= s <= length:
            nearest.append((radius, cell))
            weight = math.exp(-((radius / kernel_radius) ** 2))
            if weight > 1.0e-14 and cell_volumes[cell] > 0.0:
                weights.append((cell, weight))
    if not weights:
        nearest.sort()
        for _, cell in nearest[: max(1, min(32, len(nearest)))]:
            weights.append((cell, 1.0))
    kernel_integral = float(sum(weight * cell_volumes[cell] for cell, weight in weights))
    if kernel_integral <= 0.0:
        raise RuntimeError("regularized source kernel has zero discrete volume integral")
    density_scale = length / kernel_integral
    effective_area = kernel_integral / length
    for cell, weight in weights:
        dofs = dm.cell_dofs(cell)
        start = int(dofs[0]) * block_size
        arr[start : start + 3] = density_scale * weight * tangent
    source_coeff.x.scatter_forward()
    if len(weights) == 0:
        log(
            "[source] WARNING: no cell centers fell inside the regularized wire cylinder. "
            "The source vector may be zero; refine/increase --wire-radius.",
            comm=msh.comm,
        )
    Ls = fem.form(ufl.inner(source_coeff, v) * ufl.dx)
    source_vec = fem_petsc.assemble_vector(Ls)
    source_vec.assemble()
    log(
        f"[source] mode={mode}; radius={config.wire_radius} m; "
        f"nominal_area={nominal_area:.6e} m^2; effective_area={effective_area:.6e} m^2; "
        f"density_scale={density_scale:.6e} 1/m^2; weighted local cells={len(weights)}",
        comm=msh.comm,
    )
    source_vec, projection_diagnostics = _enforce_source_charge_conservation(
        msh,
        spaces,
        source_vec,
        config,
        apply_projection=str(config.source_projection_mode).strip().lower() == "charge_conserving",
    )
    return {"mode": mode, "coeff": source_coeff, "vector": source_vec, "projection_diagnostics": projection_diagnostics}


def _build_regularized_current_density(msh, spaces: dict[str, Any], config: PipelineConfig, cell_tags=None):
    """Build a DG0 vector current density with unit current moment."""

    import numpy as np
    from dolfinx import fem

    W = spaces["W"]
    source_coeff = fem.Function(W, name="J_unit_regularized_for_h")
    p0 = np.asarray(config.source_start, dtype=float)
    p1 = np.asarray(config.source_end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    tangent = axis / length
    centers, _cell_radii, cell_volumes = _cell_centers_radii_volumes(msh)
    dm = W.dofmap
    block_size = int(dm.bs)
    kernel_radius = max(float(config.wire_radius), 1.0e-12)
    earth_cells = None
    if cell_tags is not None:
        earth_cells = {int(cell) for cell in cell_tags.find(PHYS_EARTH)}
    weights: list[tuple[int, float]] = []
    nearest: list[tuple[float, int]] = []
    for cell, center in enumerate(centers):
        if earth_cells is not None and cell not in earth_cells:
            continue
        rel = center - p0
        s = float(np.dot(rel, tangent))
        closest = p0 + min(max(s, 0.0), length) * tangent
        radius = float(np.linalg.norm(center - closest))
        if 0.0 <= s <= length:
            nearest.append((radius, cell))
            weight = math.exp(-((radius / kernel_radius) ** 2))
            if weight > 1.0e-14 and cell_volumes[cell] > 0.0:
                weights.append((cell, weight))
    if not weights:
        nearest.sort()
        for _, cell in nearest[: max(1, min(32, len(nearest)))]:
            weights.append((cell, 1.0))
    kernel_integral = float(sum(weight * cell_volumes[cell] for cell, weight in weights))
    if kernel_integral <= 0.0:
        raise RuntimeError("regularized H-form source kernel has zero discrete volume integral")
    density_scale = length / kernel_integral
    arr = source_coeff.x.array
    for cell, weight in weights:
        dofs = dm.cell_dofs(cell)
        start = int(dofs[0]) * block_size
        arr[start : start + 3] = density_scale * weight * tangent
    source_coeff.x.scatter_forward()
    log(
        f"[source:h] regularized current density for H-form; radius={config.wire_radius} m; "
        f"weighted local cells={len(weights)}; density_scale={density_scale:.6e}",
        comm=msh.comm,
    )
    return source_coeff


def _make_zero_tangential_bc(msh, spaces: dict[str, Any], facet_tags, config: PipelineConfig):
    import numpy as np
    from dolfinx import fem

    V = spaces["V"]
    boundary_mode = str(config.outer_boundary_mode).strip().lower()
    if boundary_mode in {"natural", "robin"}:
        log(f"[bc] {boundary_mode} outer boundary: no strong tangential E constraint applied", comm=msh.comm)
        return None, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    facets = facet_tags.find(PHYS_OUTER)
    if len(facets) == 0:
        raise RuntimeError("facet tags must include outer boundary=201")
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    dofs = np.unique(dofs.astype(np.int32))
    zero = fem.Function(V)
    bc = fem.dirichletbc(zero, dofs)
    global_dofs = V.dofmap.index_map.local_to_global(dofs).astype(np.int32)
    log(f"[bc] E x n = 0 on outer boundary facets={len(facets)}, dofs(local)={len(dofs)}", comm=msh.comm)
    return bc, dofs, global_dofs


def _copy_and_combine_matrix(K, M, scale: float, *, k_scale: float = 1.0):
    from petsc4py import PETSc

    A = K.copy()
    if float(k_scale) != 1.0:
        A.scale(float(k_scale))
    A.axpy(scale, M, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    A.assemble()
    return A


def _outer_boundary_robin_admittance(config: PipelineConfig, dt: float) -> float:
    from scipy.constants import mu_0

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive for Robin boundary admittance")
    sigma = 1.0 / float(config.rho_earth)
    return float(config.outer_boundary_robin_scale) * math.sqrt(sigma / (mu_0 * dt))


def _zero_rows_columns(A, global_dofs, diag: float = 1.0) -> None:
    from petsc4py import PETSc

    iset = PETSc.IS().createGeneral(global_dofs, comm=A.comm)
    A.zeroRowsColumns(iset, diag=diag)
    A.assemble()
    iset.destroy()


def _zero_rhs_entries(vec, global_dofs) -> None:
    import numpy as np

    if len(global_dofs) == 0:
        return
    vec.setValues(global_dofs, np.zeros(len(global_dofs), dtype=float))
    vec.assemble()


def _endpoint_load_current(config: PipelineConfig, *, use_unit_current: bool) -> float:
    return 1.0 if bool(use_unit_current) else float(config.source_current)


def _build_endpoint_scalar_load(msh, S, config: PipelineConfig, *, use_unit_current: bool = False):
    """Build scalar endpoint load I*(q_end - q_start) for the wire divergence."""

    from dolfinx import fem

    current = _endpoint_load_current(config, use_unit_current=use_unit_current)
    phi = fem.Function(S)
    b = phi.x.petsc_vec.copy()
    b.set(0.0)
    _add_scalar_point_load(b, msh, S, config.source_start, -current)
    _add_scalar_point_load(b, msh, S, config.source_end, current)
    b.assemble()
    return b


def _enforce_source_charge_conservation(
    msh,
    spaces: dict[str, Any],
    source_vec,
    config: PipelineConfig,
    *,
    apply_projection: bool = True,
):
    """Project the edge source load so G.T * source equals the endpoint load."""

    import numpy as np
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from petsc4py import PETSc

    S = spaces["S"]
    V = spaces["V"]
    G = fem_petsc.discrete_gradient(S, V)
    G.assemble()

    endpoint = _build_endpoint_scalar_load(msh, S, config, use_unit_current=True)
    current_div = endpoint.duplicate()
    current_div.set(0.0)
    G.multTranspose(source_vec, current_div)
    residual = endpoint.copy()
    residual.axpy(-1.0, current_div)
    endpoint_norm = endpoint.norm()
    residual_norm = residual.norm()
    raw_source_l2_norm = float(source_vec.norm())
    raw_source_l1_norm = float(source_vec.norm(PETSc.NormType.NORM_1))
    scalar_balance = _scalar_source_balance_vector_diagnostics(
        endpoint.getArray(readonly=True).copy(),
        current_div.getArray(readonly=True).copy(),
        residual.getArray(readonly=True).copy(),
    )
    projection_mode = "charge_conserving" if bool(apply_projection) else "raw"
    if not apply_projection:
        log(
            "[source] charge-conservation projection disabled for raw diagnostic mode; "
            f"divergence residual={residual_norm:.6e}, endpoint_norm={endpoint_norm:.6e}",
            comm=msh.comm,
        )
        endpoint.destroy()
        current_div.destroy()
        residual.destroy()
        G.destroy()
        return source_vec, {
            "projection_mode": projection_mode,
            "applied": False,
            "before_residual": float(residual_norm),
            "after_residual": float(residual_norm),
            "endpoint_norm": float(endpoint_norm),
            "raw_source_l2_norm": raw_source_l2_norm,
            "projected_source_l2_norm": raw_source_l2_norm,
            "correction_l2_norm": 0.0,
            "correction_l2_over_raw": 0.0,
            "raw_source_l1_norm": raw_source_l1_norm,
            "projected_source_l1_norm": raw_source_l1_norm,
            "correction_l1_norm": 0.0,
            "correction_l1_over_raw": 0.0,
            "divergence_residual_reduction": 0.0,
            "scalar_balance": scalar_balance,
            "ksp_iterations": 0,
            "ksp_reason": 0,
            "ksp_residual": 0.0,
        }
    if residual_norm <= max(1.0e-12, 1.0e-10 * endpoint_norm):
        log(
            f"[source] charge-conservation projection skipped; divergence residual={residual_norm:.6e}",
            comm=msh.comm,
        )
        endpoint.destroy()
        current_div.destroy()
        residual.destroy()
        G.destroy()
        return source_vec, {
            "projection_mode": projection_mode,
            "applied": False,
            "before_residual": float(residual_norm),
            "after_residual": float(residual_norm),
            "endpoint_norm": float(endpoint_norm),
            "raw_source_l2_norm": raw_source_l2_norm,
            "projected_source_l2_norm": raw_source_l2_norm,
            "correction_l2_norm": 0.0,
            "correction_l2_over_raw": 0.0,
            "raw_source_l1_norm": raw_source_l1_norm,
            "projected_source_l1_norm": raw_source_l1_norm,
            "correction_l1_norm": 0.0,
            "correction_l1_over_raw": 0.0,
            "divergence_residual_reduction": 0.0,
            "scalar_balance": scalar_balance,
            "ksp_iterations": 0,
            "ksp_reason": 0,
            "ksp_residual": 0.0,
        }

    A = G.transposeMatMult(G)
    A.assemble()
    gauge_dofs = _scalar_potential_gauge_dofs(S)
    gauge_global = S.dofmap.index_map.local_to_global(gauge_dofs).astype(np.int32)
    _zero_rows_columns(A, gauge_global, diag=1.0)
    _zero_rhs_entries(residual, gauge_global)

    y_fun = fem.Function(S)
    y_fun.x.array[:] = 0.0
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=max(config.max_it, 1000))
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    ksp.setFromOptions()
    ksp.solve(residual, y_fun.x.petsc_vec)
    y_fun.x.scatter_forward()
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(
            f"source charge-conservation projection failed, reason={reason}, residual={ksp.getResidualNorm():.6e}"
        )

    correction = source_vec.duplicate()
    correction.set(0.0)
    G.mult(y_fun.x.petsc_vec, correction)
    correction_l2_norm = float(correction.norm())
    correction_l1_norm = float(correction.norm(PETSc.NormType.NORM_1))
    source_vec.axpy(1.0, correction)
    source_vec.assemble()
    projected_source_l2_norm = float(source_vec.norm())
    projected_source_l1_norm = float(source_vec.norm(PETSc.NormType.NORM_1))

    corrected_div = endpoint.duplicate()
    corrected_div.set(0.0)
    G.multTranspose(source_vec, corrected_div)
    corrected_div.axpy(-1.0, endpoint)
    corrected_norm = corrected_div.norm()
    log(
        "[source] charge-conservation projection applied; "
        f"before={residual_norm:.6e}, after={corrected_norm:.6e}, endpoint_norm={endpoint_norm:.6e}",
        comm=msh.comm,
    )
    diagnostics = {
        "projection_mode": projection_mode,
        "applied": True,
        "before_residual": float(residual_norm),
        "after_residual": float(corrected_norm),
        "endpoint_norm": float(endpoint_norm),
        "raw_source_l2_norm": raw_source_l2_norm,
        "projected_source_l2_norm": projected_source_l2_norm,
        "correction_l2_norm": correction_l2_norm,
        "correction_l2_over_raw": float(correction_l2_norm / raw_source_l2_norm) if raw_source_l2_norm > 0.0 else 0.0,
        "raw_source_l1_norm": raw_source_l1_norm,
        "projected_source_l1_norm": projected_source_l1_norm,
        "correction_l1_norm": correction_l1_norm,
        "correction_l1_over_raw": float(correction_l1_norm / raw_source_l1_norm) if raw_source_l1_norm > 0.0 else 0.0,
        "divergence_residual_reduction": (
            float((residual_norm - corrected_norm) / residual_norm) if residual_norm > 0.0 else 0.0
        ),
        "scalar_balance": scalar_balance,
        "ksp_iterations": int(ksp.getIterationNumber()),
        "ksp_reason": int(reason),
        "ksp_residual": float(ksp.getResidualNorm()),
    }

    corrected_div.destroy()
    correction.destroy()
    ksp.destroy()
    A.destroy()
    endpoint.destroy()
    current_div.destroy()
    residual.destroy()
    G.destroy()
    return source_vec, diagnostics


def _build_conductivity_divergence_cleaner(spaces: dict[str, Any], operators, config: PipelineConfig):
    """Build a sigma-weighted gradient projection for post-ramp E cleaning."""

    import numpy as np
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from petsc4py import PETSc

    S = spaces["S"]
    V = spaces["V"]
    G = fem_petsc.discrete_gradient(S, V)
    G.assemble()
    MG = operators["M"].matMult(G)
    A = G.transposeMatMult(MG)
    A.assemble()
    gauge_dofs = _scalar_potential_gauge_dofs(S)
    gauge_global = S.dofmap.index_map.local_to_global(gauge_dofs).astype(np.int32)
    _zero_rows_columns(A, gauge_global, diag=1.0)

    phi = fem.Function(S, name="divergence_cleaning_phi")
    rhs = phi.x.petsc_vec.duplicate()
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=max(config.max_it, 1000))
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    ksp.setFromOptions()
    log("[div-clean] conductivity divergence cleaning enabled: projecting G^T M_sigma E to zero after source turn-off.")
    return {
        "G": G,
        "MG": MG,
        "A": A,
        "phi": phi,
        "rhs": rhs,
        "ksp": ksp,
        "gauge_global": gauge_global,
    }


def _build_conductivity_divergence_control_matrix(spaces: dict[str, Any], operators, config: PipelineConfig):
    """Build M_sigma G G^T M_sigma for an implicit weak divergence-control diagnostic."""

    from dolfinx.fem import petsc as fem_petsc

    S = spaces["S"]
    V = spaces["V"]
    G = fem_petsc.discrete_gradient(S, V)
    G.assemble()
    MG = operators["M"].matMult(G)
    MG.assemble()
    control = MG.matTransposeMult(MG)
    control.assemble()
    G.destroy()
    MG.destroy()
    log(
        "[div-control] conductivity divergence-control matrix enabled: "
        f"weight={float(config.divergence_control_weight):.6g}, "
        f"t_obs_min={float(config.divergence_control_t_obs_min):.6g} s."
    )
    return control


def _petsc_matrix_norm(mat) -> float:
    """Return a robust PETSc matrix norm for diagnostics and relative scaling."""

    try:
        from petsc4py import PETSc

        norm_type = getattr(PETSc.NormType, "FROBENIUS", None)
        if norm_type is not None:
            return float(mat.norm(norm_type))
    except Exception:
        pass
    return float(mat.norm())


def _apply_conductivity_divergence_cleaning(cleaner, E, operators, config: PipelineConfig) -> dict[str, float | int]:
    """Project E onto the M_sigma-orthogonal complement of gradient fields."""

    import numpy as np

    rhs = cleaner["rhs"]
    rhs.set(0.0)
    mass_e = E.x.petsc_vec.duplicate()
    operators["M"].mult(E.x.petsc_vec, mass_e)
    cleaner["G"].multTranspose(mass_e, rhs)
    rhs.assemble()
    before = float(rhs.norm())
    _zero_rhs_entries(rhs, cleaner["gauge_global"])

    phi = cleaner["phi"]
    phi.x.array[:] = 0.0
    cleaner["ksp"].solve(rhs, phi.x.petsc_vec)
    phi.x.scatter_forward()
    reason = int(cleaner["ksp"].getConvergedReason())
    residual = float(cleaner["ksp"].getResidualNorm())
    its = int(cleaner["ksp"].getIterationNumber())
    if reason < 0:
        mass_e.destroy()
        raise RuntimeError(
            f"conductivity divergence cleaning failed, reason={reason}, residual={residual:.6e}"
        )

    correction = E.x.petsc_vec.duplicate()
    cleaner["G"].mult(phi.x.petsc_vec, correction)
    strength = float(config.divergence_cleaning_strength)
    E.x.petsc_vec.axpy(-strength, correction)
    E.x.petsc_vec.assemble()
    E.x.scatter_forward()

    operators["M"].mult(E.x.petsc_vec, mass_e)
    rhs.set(0.0)
    cleaner["G"].multTranspose(mass_e, rhs)
    rhs.assemble()
    after = float(rhs.norm())
    correction_norm = float(correction.norm())
    applied_correction_norm = float(abs(strength) * correction_norm)
    correction.destroy()
    mass_e.destroy()
    return {
        "before": before,
        "after": after,
        "correction_norm": correction_norm,
        "applied_correction_norm": applied_correction_norm,
        "strength": strength,
        "its": its,
        "reason": reason,
        "residual": residual,
    }


def _validate_h_solver_state(label: str, ksp, solution, *, rhs_norm: float, config: PipelineConfig) -> dict[str, float | int]:
    """Reject failed, non-finite, or numerically useless H-form solver states."""

    import numpy as np

    reason = int(ksp.getConvergedReason())
    residual = float(ksp.getResidualNorm())
    its = int(ksp.getIterationNumber())
    rhs_norm = float(rhs_norm)
    if reason < 0:
        raise RuntimeError(f"{label} KSP failed, reason={reason}, residual={residual:.6e}")
    if not np.isfinite(residual) or not np.isfinite(rhs_norm):
        raise RuntimeError(f"{label} KSP reported non-finite residual/RHS norm: residual={residual}, rhs_norm={rhs_norm}")

    arr = np.asarray(solution, dtype=float)
    nonfinite = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if nonfinite:
        raise RuntimeError(f"{label} solution contains {nonfinite} non-finite values")

    relative_residual = residual / max(rhs_norm, 1.0)
    residual_limit = max(float(config.rtol) * 100.0, 1.0e-8)
    if relative_residual > residual_limit:
        raise RuntimeError(
            f"{label} KSP relative residual is too large: "
            f"relative residual={relative_residual:.6e}, residual={residual:.6e}, "
            f"rhs_norm={rhs_norm:.6e}, limit={residual_limit:.6e}, reason={reason}"
        )
    return {"its": its, "reason": reason, "residual": residual, "relative_residual": float(relative_residual)}


def assemble_operators(msh, spaces: dict[str, Any], materials: dict[str, Any], facet_tags, config: PipelineConfig, debye=None):
    """Assemble curl-curl and conductivity mass matrices."""

    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    V = spaces["V"]
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bc, bc_dofs, bc_global = _make_zero_tangential_bc(msh, spaces, facet_tags, config)

    K_form = fem.form(materials["mu_inv"] * ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
    M_form = fem.form(materials["sigma"] * ufl.inner(u, v) * ufl.dx)
    Minf_form = fem.form(materials["sigma_infinity"] * ufl.inner(u, v) * ufl.dx)
    K = fem_petsc.assemble_matrix(K_form)
    K.assemble()
    M = fem_petsc.assemble_matrix(M_form)
    M.assemble()
    M_inf = fem_petsc.assemble_matrix(Minf_form)
    M_inf.assemble()

    B_robin = None
    if str(config.outer_boundary_mode).strip().lower() == "robin":
        n = ufl.FacetNormal(msh)
        u_t = u - n * ufl.inner(u, n)
        v_t = v - n * ufl.inner(v, n)
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
        B_form = fem.form(ufl.inner(u_t, v_t) * ds(PHYS_OUTER))
        B_robin = fem_petsc.assemble_matrix(B_form)
        B_robin.assemble()

    debye_mats = []
    if debye is not None and debye["terms"]:
        for delta_fun in debye["delta_functions"]:
            form = fem.form(delta_fun * ufl.inner(u, v) * ufl.dx)
            mat = fem_petsc.assemble_matrix(form)
            mat.assemble()
            debye_mats.append(mat)

    log("[operators] Assembled K, M_sigma, and M_sigma_infinity matrices.", comm=msh.comm)
    return {
        "K": K,
        "M": M,
        "M_inf": M_inf,
        "B_robin": B_robin,
        "M_debye": debye_mats,
        "bc": bc,
        "bc_dofs": bc_dofs,
        "bc_global": bc_global,
    }


def configure_ams_solver(A, spaces: dict[str, Any], config: PipelineConfig):
    """Configure a PETSc Krylov solver with HYPRE AMS."""

    import numpy as np
    from petsc4py import PETSc
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    V = spaces["V"]
    S = spaces["S"]
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType(config.ksp_type)
    ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=config.max_it)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")

    try:
        G = fem_petsc.discrete_gradient(S, V)
        G.assemble()
        pc.setHYPREDiscreteGradient(G)

        ex = fem.Function(V)
        ey = fem.Function(V)
        ez = fem.Function(V)
        ex.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
        ey.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]), np.zeros(x.shape[1]))))
        ez.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.ones(x.shape[1]))))
        ex.x.scatter_forward()
        ey.x.scatter_forward()
        ez.x.scatter_forward()
        pc.setHYPRESetEdgeConstantVectors(ex.x.petsc_vec, ey.x.petsc_vec, ez.x.petsc_vec)
        pc.setUp()
    except Exception as exc:
        raise SystemExit(f"Failed to configure HYPRE AMS with discrete gradient and edge constants: {exc}") from exc

    log("[solver] Configured KSP " + config.ksp_type + " with PC hypre/ams.", comm=A.comm)
    return {"ksp": ksp, "G": G, "edge_constants": (ex, ey, ez)}


def configure_lu_solver(A, *, comm=None):
    """Configure a robust serial LU solver for H-form diagnostics."""

    from petsc4py import PETSc

    ksp = PETSc.KSP().create(comm=A.comm if comm is None else comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setFromOptions()
    log("[solver:h] Configured KSP preonly with PC lu.", comm=A.comm)
    return {"ksp": ksp}


def _source_current(t: float, config: PipelineConfig) -> float:
    if t <= 0.0:
        return config.source_current
    if t >= config.ramp_off_time:
        return 0.0
    return config.source_current * (1.0 - t / config.ramp_off_time)


def _should_apply_divergence_cleaning(t_internal: float, config: PipelineConfig) -> bool:
    if _source_current(float(t_internal), config) != 0.0:
        return False
    t_obs = max(0.0, float(t_internal) - float(config.ramp_off_time))
    return t_obs >= float(config.divergence_cleaning_t_obs_min)


def _should_apply_divergence_control(t_internal: float, config: PipelineConfig) -> bool:
    if float(config.divergence_control_weight) <= 0.0:
        return False
    if _source_current(float(t_internal), config) != 0.0:
        return False
    t_obs = max(0.0, float(t_internal) - float(config.ramp_off_time))
    return t_obs >= float(config.divergence_control_t_obs_min)


def _divergence_control_step_stats(
    config: PipelineConfig,
    *,
    dt: float,
    lhs_mass: float,
    lhs_stiffness: float,
    mass_norm: float,
    stiffness_norm: float,
    control_norm: float,
) -> dict[str, float | str]:
    """Return the actual divergence-control coefficient for this implicit step."""

    scale = str(config.divergence_control_scale).strip().lower()
    base_weight = float(config.divergence_control_weight)
    mass_part = abs(float(lhs_mass)) * float(mass_norm)
    stiffness_part = abs(float(lhs_stiffness)) * float(stiffness_norm)
    if scale == "absolute":
        reference_norm = float("nan")
        applied_weight = base_weight
        relative_weight = (
            abs(applied_weight) * float(control_norm) / (mass_part + stiffness_part)
            if mass_part + stiffness_part > 0.0
            else float("nan")
        )
    elif scale == "mass":
        reference_norm = mass_part
        applied_weight = base_weight * reference_norm / float(control_norm) if control_norm > 0.0 else 0.0
        relative_weight = base_weight if reference_norm > 0.0 and control_norm > 0.0 else float("nan")
    elif scale == "stiffness":
        reference_norm = stiffness_part
        applied_weight = base_weight * reference_norm / float(control_norm) if control_norm > 0.0 else 0.0
        relative_weight = base_weight if reference_norm > 0.0 and control_norm > 0.0 else float("nan")
    elif scale == "lhs":
        reference_norm = mass_part + stiffness_part
        applied_weight = base_weight * reference_norm / float(control_norm) if control_norm > 0.0 else 0.0
        relative_weight = base_weight if reference_norm > 0.0 and control_norm > 0.0 else float("nan")
    else:
        raise ValueError("divergence_control_scale must be 'absolute', 'mass', 'stiffness', or 'lhs'")
    return {
        "divergence_control_scale": scale,
        "divergence_control_weight": base_weight,
        "divergence_control_applied_weight": float(applied_weight),
        "divergence_control_reference_norm": float(reference_norm),
        "divergence_control_matrix_norm": float(control_norm),
        "divergence_control_relative_weight": float(relative_weight),
        "divergence_control_dt": float(dt),
    }


def _source_interval_average_didt(t0: float, t1: float, config: PipelineConfig) -> float:
    """Return interval-average dI/dt used by Backward-Euler source loading."""

    t0 = float(t0)
    t1 = float(t1)
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    return (_source_current(t1, config) - _source_current(t0, config)) / (t1 - t0)


def _find_cell_for_point(msh, point):
    cells = _find_cells_for_point(msh, point)
    if len(cells) == 0:
        return None
    return int(cells[0])


def _find_cells_for_point(msh, point):
    import numpy as np
    from dolfinx import geometry

    points = np.asarray(point, dtype=float).reshape(1, 3)
    tree = geometry.bb_tree(msh, msh.topology.dim, padding=1.0e-10)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    links = colliding.links(0)
    return np.asarray(links, dtype=np.int32)


def compute_dbdt(E, spaces: dict[str, Any]):
    """Compute dB/dt = -curl(E) as a DG0 vector field."""

    import ufl
    from dolfinx import fem

    W = spaces["W"]
    dbdt = fem.Function(W, name="dBdt")
    expr = fem.Expression(-ufl.curl(E), W.element.interpolation_points(), comm=E.function_space.mesh.comm)
    dbdt.interpolate(expr)
    dbdt.x.scatter_forward()
    return dbdt


def _initial_field_curl_diagnostics(E_initial, spaces: dict[str, Any]) -> dict[str, float | str]:
    """Measure the residual curl of the DC initial electric field."""

    import numpy as np
    from mpi4py import MPI

    dbdt0 = compute_dbdt(E_initial, spaces)
    values = np.asarray(dbdt0.x.array, dtype=float)
    local_max = float(np.max(np.abs(values))) if values.size else 0.0
    max_abs = float(E_initial.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX))
    return {
        "quantity": "-curl(E_initial)",
        "initial_curl_residual": float(dbdt0.x.petsc_vec.norm()),
        "initial_curl_max_abs": max_abs,
    }


def _receiver_sampling_points(config: PipelineConfig):
    """Return deterministic diagnostic sample points for the configured receiver."""

    import numpy as np

    center = np.asarray(config.receiver, dtype=float).reshape(1, 3)
    receiver_type = str(config.receiver_type).strip().lower()
    if receiver_type == "point":
        return center
    radius = float(config.receiver_average_radius)
    if receiver_type == "disk_average":
        offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
            ],
            dtype=float,
        )
    elif receiver_type == "volume_average":
        offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=float,
        )
    else:
        raise ValueError("receiver_type must be 'point', 'volume_average', or 'disk_average'")
    return center + offsets


def _parse_receiver_diagnostic_types(config: PipelineConfig) -> tuple[str, ...]:
    raw = config.receiver_diagnostic_types
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.split(",")]
    else:
        items = [str(item).strip().lower() for item in raw]
    allowed = {"point", "volume_average", "disk_average"}
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in allowed:
            raise ValueError("receiver_diagnostic_types may contain only point, volume_average, or disk_average")
        if item not in out:
            out.append(item)
    return tuple(out)


def _receiver_config_for_type(config: PipelineConfig, receiver_type: str) -> PipelineConfig:
    return replace(config, receiver_type=receiver_type)


def _collapse_receiver_cell_candidates(values, mode: str, *, centers=None, point=None):
    collapsed, _metadata = _collapse_receiver_cell_candidates_with_metadata(
        values,
        mode,
        centers=centers,
        point=point,
    )
    return collapsed


def _collapse_receiver_cell_candidates_with_metadata(values, mode: str, *, centers=None, point=None):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("receiver candidate values must have shape (n_candidates, n_components)")
    mode = str(mode).strip().lower()
    selected_index = None
    if mode == "first_cell":
        selected_index = 0
        collapsed = arr[0]
    elif mode == "median":
        collapsed = np.median(arr, axis=0)
    elif mode == "mean":
        collapsed = np.mean(arr, axis=0)
    elif mode in {"nearest_center", "shallowest"}:
        if centers is None:
            raise ValueError(f"receiver_evaluation_mode='{mode}' requires candidate cell centers")
        center_arr = np.asarray(centers, dtype=float)
        if center_arr.shape != (arr.shape[0], 3):
            raise ValueError("candidate cell centers must have shape (n_candidates, 3)")
        if mode == "nearest_center":
            if point is None:
                raise ValueError("receiver_evaluation_mode='nearest_center' requires the receiver point")
            point_arr = np.asarray(point, dtype=float).reshape(1, 3)
            idx = int(np.argmin(np.linalg.norm(center_arr - point_arr, axis=1)))
        else:
            idx = int(np.argmax(center_arr[:, 2]))
        selected_index = idx
        collapsed = arr[idx]
    else:
        raise ValueError(
            "receiver_evaluation_mode must be 'first_cell', 'mean', 'median', 'nearest_center', or 'shallowest'"
        )

    metadata: dict[str, Any] = {"selected_index": selected_index, "candidate_count": int(arr.shape[0])}
    if centers is not None:
        center_arr = np.asarray(centers, dtype=float)
        if center_arr.shape != (arr.shape[0], 3):
            raise ValueError("candidate cell centers must have shape (n_candidates, 3)")
        metadata["candidate_center_z_min"] = float(np.min(center_arr[:, 2]))
        metadata["candidate_center_z_max"] = float(np.max(center_arr[:, 2]))
        if point is not None:
            point_arr = np.asarray(point, dtype=float).reshape(1, 3)
            distances = np.linalg.norm(center_arr - point_arr, axis=1)
            metadata["candidate_center_distance_min"] = float(np.min(distances))
            metadata["candidate_center_distance_max"] = float(np.max(distances))
            metadata["candidate_center_distance_mean"] = float(np.mean(distances))
            if selected_index is not None:
                metadata["selected_center_distance"] = float(distances[int(selected_index)])
                metadata["selected_center_z"] = float(center_arr[int(selected_index), 2])
    return collapsed, metadata


def _aggregate_receiver_sample_values(sample_values, mode: str, *, sample_centers=None, sample_points=None):
    aggregated, _metadata = _aggregate_receiver_sample_values_with_metadata(
        sample_values,
        mode,
        sample_centers=sample_centers,
        sample_points=sample_points,
    )
    return aggregated


def _aggregate_receiver_sample_values_with_metadata(sample_values, mode: str, *, sample_centers=None, sample_points=None):
    import numpy as np

    centers = list(sample_centers or [None] * len(sample_values))
    points = list(sample_points or [None] * len(sample_values))
    collapsed = []
    sample_metadata = []
    for i, values in enumerate(sample_values):
        value, metadata = _collapse_receiver_cell_candidates_with_metadata(
            values,
            mode,
            centers=centers[i],
            point=points[i],
        )
        collapsed.append(value)
        sample_metadata.append(metadata)
    if not collapsed:
        raise ValueError("at least one receiver sample value is required")
    return np.mean(np.vstack(collapsed), axis=0), _receiver_candidate_geometry_stats(sample_metadata)


def _receiver_candidate_count_stats(counts) -> dict[str, float | int]:
    import numpy as np

    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return {
            "sample_count": 0,
            "candidate_count_min": 0,
            "candidate_count_max": 0,
            "candidate_count_mean": float("nan"),
        }
    return {
        "sample_count": int(arr.size),
        "candidate_count_min": int(np.min(arr)),
        "candidate_count_max": int(np.max(arr)),
        "candidate_count_mean": float(np.mean(arr)),
    }


def _receiver_candidate_geometry_stats(sample_metadata) -> dict[str, float | int]:
    import numpy as np

    items = [dict(item) for item in sample_metadata or []]
    if not items:
        return {
            "multi_candidate_sample_count": 0,
            "candidate_center_distance_min": float("nan"),
            "candidate_center_distance_max": float("nan"),
            "candidate_center_distance_mean": float("nan"),
            "selected_center_distance_mean": float("nan"),
            "selected_center_distance_max": float("nan"),
            "candidate_center_z_min": float("nan"),
            "candidate_center_z_max": float("nan"),
            "selected_center_z_mean": float("nan"),
        }

    distances = []
    distance_mins = []
    distance_maxs = []
    selected_distances = []
    selected_z = []
    z_mins = []
    z_maxs = []
    multi_count = 0
    for item in items:
        if int(item.get("candidate_count", 0)) > 1:
            multi_count += 1
        if math.isfinite(float(item.get("candidate_center_distance_mean", math.nan))):
            distances.append(float(item["candidate_center_distance_mean"]))
            distance_mins.append(float(item["candidate_center_distance_min"]))
            distance_maxs.append(float(item["candidate_center_distance_max"]))
        if math.isfinite(float(item.get("selected_center_distance", math.nan))):
            selected_distances.append(float(item["selected_center_distance"]))
        if math.isfinite(float(item.get("candidate_center_z_min", math.nan))):
            z_mins.append(float(item["candidate_center_z_min"]))
            z_maxs.append(float(item["candidate_center_z_max"]))
        if math.isfinite(float(item.get("selected_center_z", math.nan))):
            selected_z.append(float(item["selected_center_z"]))

    return {
        "multi_candidate_sample_count": int(multi_count),
        "candidate_center_distance_min": float(np.min(distance_mins)) if distance_mins else float("nan"),
        "candidate_center_distance_max": float(np.max(distance_maxs)) if distance_maxs else float("nan"),
        "candidate_center_distance_mean": float(np.mean(distances)) if distances else float("nan"),
        "selected_center_distance_mean": float(np.mean(selected_distances)) if selected_distances else float("nan"),
        "selected_center_distance_max": float(np.max(selected_distances)) if selected_distances else float("nan"),
        "candidate_center_z_min": float(np.min(z_mins)) if z_mins else float("nan"),
        "candidate_center_z_max": float(np.max(z_maxs)) if z_maxs else float("nan"),
        "selected_center_z_mean": float(np.mean(selected_z)) if selected_z else float("nan"),
    }


def evaluate_receivers(E, dbdt, msh, config: PipelineConfig):
    """Evaluate Ex, Ey and dBz/dt at the configured receiver point."""

    import numpy as np

    if not getattr(evaluate_receivers, "_logged_dbdt_mode", False):
        log(
            f"[receiver] Ex/Ey and dBz/dt are evaluated with receiver_type={config.receiver_type}, "
            f"receiver_evaluation_mode={config.receiver_evaluation_mode}. "
            "dBz/dt comes from a DG0 interpolation of -curl(E); at shared receiver points, "
            "the selected mode combines the colliding cellwise-constant candidates.",
            comm=msh.comm,
        )
        setattr(evaluate_receivers, "_logged_dbdt_mode", True)
    mode = str(config.receiver_evaluation_mode).strip().lower()
    e_samples = []
    dbdt_samples = []
    center_samples = []
    point_samples = []
    candidate_counts = []
    needs_geometry_stats = bool(_parse_receiver_diagnostic_types(config)) or mode in {"nearest_center", "shallowest"}
    cell_centers = _cell_centers(msh) if needs_geometry_stats else None
    for sample_point in _receiver_sampling_points(config):
        cells = _find_cells_for_point(msh, sample_point)
        if len(cells) == 0:
            continue
        candidate_counts.append(int(len(cells)))
        point = np.repeat(np.asarray(sample_point, dtype=float).reshape(1, 3), len(cells), axis=0)
        e_vals = np.asarray(E.eval(point, cells), dtype=float).reshape(len(cells), -1)
        dbdt_vals = np.asarray(dbdt.eval(point, cells), dtype=float).reshape(len(cells), -1)
        e_samples.append(e_vals)
        dbdt_samples.append(dbdt_vals)
        center_samples.append(cell_centers[np.asarray(cells, dtype=int)] if cell_centers is not None else None)
        point_samples.append(np.asarray(sample_point, dtype=float))
    if not e_samples:
        raise RuntimeError(
            f"receiver {config.receiver} was not found in a local cell; run in serial "
            "for point extraction or add MPI point ownership handling."
        )
    e_val = _aggregate_receiver_sample_values(
        e_samples,
        mode,
        sample_centers=center_samples,
        sample_points=point_samples,
    )
    dbdt_val, candidate_geometry_stats = _aggregate_receiver_sample_values_with_metadata(
        dbdt_samples,
        mode,
        sample_centers=center_samples,
        sample_points=point_samples,
    )
    rec = {"Ex": float(e_val[0]), "Ey": float(e_val[1]), "dBzdt": float(dbdt_val[2])}
    rec.update(_receiver_candidate_count_stats(candidate_counts))
    rec.update(candidate_geometry_stats)
    return rec


def _make_secondary_receiver_projector_from_evaluate_receivers(
    electric_getter,
    dbdt_getter,
    *,
    msh,
    config: PipelineConfig,
):
    """Build a primary-secondary receiver projector from existing DOLFINx sampling.

    ``electric_getter`` and ``dbdt_getter`` are small DOLFINx-specific hooks
    that return the secondary electric and magnetic-rate fields for the current
    secondary state.  The returned callable matches
    ``PrimarySecondaryForwardOperator.secondary_receiver_projector``.
    """

    def projector(state, Ep_new, time_value: float, dt: float, components):
        import numpy as np

        E_secondary = electric_getter(state, Ep_new, time_value, dt)
        dbdt_secondary = dbdt_getter(state, Ep_new, time_value, dt)
        rec = evaluate_receivers(E_secondary, dbdt_secondary, msh, config)
        row = []
        for component in components:
            if component not in rec:
                raise ValueError(f"secondary receiver record missing component {component!r}")
            row.append(float(rec[component]))
        return np.asarray([row], dtype=float)

    return projector


def _make_dolfinx_zero_secondary_receiver_projector(msh, spaces, config: PipelineConfig):
    """Return a DOLFINx receiver projector for zero secondary fields."""

    from dolfinx import fem

    zero_E = fem.Function(spaces["V"], name="E_secondary_zero")
    zero_E.x.array[:] = 0.0
    zero_E.x.scatter_forward()
    zero_dbdt = compute_dbdt(zero_E, spaces)

    def electric_getter(_state, _Ep_new, _time_value, _dt):
        return zero_E

    def dbdt_getter(_state, _Ep_new, _time_value, _dt):
        return zero_dbdt

    return _make_secondary_receiver_projector_from_evaluate_receivers(
        electric_getter,
        dbdt_getter,
        msh=msh,
        config=config,
    )


def _evaluate_receiver_diagnostics(E, dbdt, msh, config: PipelineConfig, *, time_obs: float, main_record=None):
    import numpy as np

    rows = []
    for receiver_type in _parse_receiver_diagnostic_types(config):
        diag_config = _receiver_config_for_type(config, receiver_type)
        if main_record is not None and receiver_type == str(config.receiver_type).strip().lower():
            rec = dict(main_record)
        else:
            rec = evaluate_receivers(E, dbdt, msh, diag_config)
        rows.append(
            {
                "time_obs": float(time_obs),
                "receiver_type": receiver_type,
                "radius": 0.0 if receiver_type == "point" else float(config.receiver_average_radius),
                "Ex": float(rec.get("Ex", np.nan)),
                "Ey": float(rec.get("Ey", np.nan)),
                "Hz": float(rec.get("Hz", np.nan)),
                "dBzdt": float(rec.get("dBzdt", np.nan)),
                "dBzdt_curl": float(rec.get("dBzdt_curl", np.nan)),
                "dBzdt_biot_rate": float(rec.get("dBzdt_biot_rate", np.nan)),
                "sample_count": int(rec.get("sample_count", 0)),
                "candidate_count_min": int(rec.get("candidate_count_min", 0)),
                "candidate_count_max": int(rec.get("candidate_count_max", 0)),
                "candidate_count_mean": float(rec.get("candidate_count_mean", np.nan)),
                "multi_candidate_sample_count": int(rec.get("multi_candidate_sample_count", 0)),
                "candidate_center_distance_min": float(rec.get("candidate_center_distance_min", np.nan)),
                "candidate_center_distance_max": float(rec.get("candidate_center_distance_max", np.nan)),
                "candidate_center_distance_mean": float(rec.get("candidate_center_distance_mean", np.nan)),
                "selected_center_distance_mean": float(rec.get("selected_center_distance_mean", np.nan)),
                "selected_center_distance_max": float(rec.get("selected_center_distance_max", np.nan)),
                "candidate_center_z_min": float(rec.get("candidate_center_z_min", np.nan)),
                "candidate_center_z_max": float(rec.get("candidate_center_z_max", np.nan)),
                "selected_center_z_mean": float(rec.get("selected_center_z_mean", np.nan)),
            }
        )
    return rows


def _receiver_diagnostic_summary(receiver_diagnostic_rows, *, threshold: float = 0.05) -> dict[str, Any]:
    rows = list(receiver_diagnostic_rows or [])
    if not rows:
        return {"enabled": False}

    receiver_types: list[str] = []
    by_time: dict[float, dict[str, dict[str, Any]]] = {}
    for row in rows:
        receiver_type = str(row.get("receiver_type", ""))
        if not receiver_type:
            continue
        if receiver_type not in receiver_types:
            receiver_types.append(receiver_type)
        time_obs = float(row.get("time_obs", math.nan))
        if not math.isfinite(time_obs):
            continue
        by_time.setdefault(time_obs, {})[receiver_type] = row

    if not receiver_types:
        return {"enabled": False}
    baseline = "point" if "point" in receiver_types else receiver_types[0]
    components = ["Ex", "Ey", "Hz", "dBzdt"]
    comparisons: dict[str, Any] = {}
    issue_suspected = False
    for receiver_type in receiver_types:
        if receiver_type == baseline:
            continue
        component_summary: dict[str, Any] = {}
        for component in components:
            relatives = []
            absolutes = []
            component_times = []
            for time_obs, grouped in sorted(by_time.items()):
                if baseline not in grouped or receiver_type not in grouped:
                    continue
                base_value = float(grouped[baseline].get(component, math.nan))
                value = float(grouped[receiver_type].get(component, math.nan))
                if not (math.isfinite(base_value) and math.isfinite(value)):
                    continue
                abs_diff = abs(value - base_value)
                rel_diff = abs_diff / max(abs(base_value), 1.0e-300)
                relatives.append(rel_diff)
                absolutes.append(abs_diff)
                component_times.append(time_obs)
            if not relatives:
                continue
            max_index = max(range(len(relatives)), key=lambda i: relatives[i])
            max_relative = float(round(relatives[max_index], 15))
            issue_suspected = issue_suspected or max_relative > float(threshold)
            component_summary[component] = {
                "sample_count": len(relatives),
                "max_relative_difference": max_relative,
                "mean_relative_difference": float(round(sum(relatives) / len(relatives), 15)),
                "max_absolute_difference": float(absolutes[max_index]),
                "time_at_max_relative_difference": float(component_times[max_index]),
            }
        comparisons[receiver_type] = component_summary

    return {
        "enabled": True,
        "threshold": float(threshold),
        "baseline_receiver_type": baseline,
        "receiver_types": receiver_types,
        "time_count": len(by_time),
        "comparisons": comparisons,
        "receiver_sampling_issue_suspected": bool(issue_suspected),
    }


def _receiver_reference_summary(receiver_diagnostic_rows, times, ref_data, components, *, threshold: float = 0.05) -> dict[str, Any]:
    import numpy as np

    rows = list(receiver_diagnostic_rows or [])
    if not rows:
        return {"enabled": False}
    component_names = [str(component) for component in components]
    t = np.asarray(times, dtype=float)
    ref = np.asarray(ref_data, dtype=float)
    if ref.ndim != 2 or ref.shape[0] != t.size or ref.shape[1] != len(component_names):
        return {"enabled": False, "reason": "inconsistent reference/time/component shapes"}
    time_to_index = {float(time): i for i, time in enumerate(t)}
    receiver_types: list[str] = []
    for row in rows:
        receiver_type = str(row.get("receiver_type", ""))
        if receiver_type and receiver_type not in receiver_types:
            receiver_types.append(receiver_type)
    if not receiver_types:
        return {"enabled": False}
    baseline = "point" if "point" in receiver_types else receiver_types[0]

    by_type: dict[str, Any] = {}
    for receiver_type in receiver_types:
        receiver_rows = [row for row in rows if str(row.get("receiver_type", "")) == receiver_type]
        component_errors: dict[str, Any] = {}
        for col, component in enumerate(component_names):
            if component == "Hz" and not any(math.isfinite(float(row.get(component, math.nan))) for row in receiver_rows):
                continue
            floor = _validation_floor(component, ref[:, col])
            max_abs_ref = float(np.max(np.abs(ref[:, col]))) if ref.shape[0] else 0.0
            robust_errors = []
            peak_errors = []
            error_times = []
            for row in receiver_rows:
                time_obs = float(row.get("time_obs", math.nan))
                idx = time_to_index.get(time_obs)
                if idx is None:
                    continue
                pred_value = float(row.get(component, math.nan))
                if not math.isfinite(pred_value):
                    continue
                ref_value = float(ref[idx, col])
                abs_error = abs(pred_value - ref_value)
                robust_errors.append(abs_error / max(abs(ref_value), floor))
                peak_errors.append(abs_error / max(max_abs_ref, floor))
                error_times.append(time_obs)
            if not robust_errors:
                continue
            max_index = max(range(len(robust_errors)), key=lambda i: robust_errors[i])
            component_errors[component] = {
                "sample_count": len(robust_errors),
                "max_relative_error": float(robust_errors[max_index]),
                "mean_relative_error": float(sum(robust_errors) / len(robust_errors)),
                "max_peak_normalized_error": float(max(peak_errors)),
                "time_at_max_relative_error": float(error_times[max_index]),
                "passes_threshold": bool(max(robust_errors) <= float(threshold) and max(peak_errors) <= float(threshold)),
            }
        by_type[receiver_type] = component_errors

    comparisons: dict[str, Any] = {}
    baseline_errors = by_type.get(baseline, {})
    for receiver_type, component_errors in by_type.items():
        if receiver_type == baseline:
            continue
        comparisons[receiver_type] = {}
        for component, metrics in component_errors.items():
            baseline_metric = baseline_errors.get(component)
            if baseline_metric is None:
                continue
            comparisons[receiver_type][component] = {
                "baseline_max_relative_error": float(baseline_metric["max_relative_error"]),
                "candidate_max_relative_error": float(metrics["max_relative_error"]),
                "improves_over_baseline": bool(metrics["max_relative_error"] < baseline_metric["max_relative_error"]),
            }

    return {
        "enabled": True,
        "threshold": float(threshold),
        "baseline_receiver_type": baseline,
        "receiver_types": receiver_types,
        "metrics": by_type,
        "comparisons": comparisons,
    }


def _receiver_reference_error_rows(receiver_diagnostic_rows, times, ref_data, components, *, threshold: float = 0.05) -> list[dict[str, Any]]:
    import numpy as np

    rows = list(receiver_diagnostic_rows or [])
    if not rows:
        return []
    component_names = [str(component) for component in components]
    t = np.asarray(times, dtype=float)
    ref = np.asarray(ref_data, dtype=float)
    if ref.ndim != 2 or ref.shape[0] != t.size or ref.shape[1] != len(component_names):
        return []
    time_to_index = {float(time): i for i, time in enumerate(t)}
    out = []
    for row in rows:
        receiver_type = str(row.get("receiver_type", ""))
        time_obs = float(row.get("time_obs", math.nan))
        idx = time_to_index.get(time_obs)
        if not receiver_type or idx is None:
            continue
        for col, component in enumerate(component_names):
            pred_value = float(row.get(component, math.nan))
            if not math.isfinite(pred_value):
                continue
            ref_value = float(ref[idx, col])
            floor = _validation_floor(component, ref[:, col])
            max_abs_ref = float(np.max(np.abs(ref[:, col]))) if ref.shape[0] else 0.0
            abs_error = abs(pred_value - ref_value)
            ordinary = abs_error / abs(ref_value) if ref_value != 0.0 else float("inf")
            robust = abs_error / max(abs(ref_value), floor)
            peak = abs_error / max(max_abs_ref, floor)
            out.append(
                {
                    "time_obs": time_obs,
                    "receiver_type": receiver_type,
                    "component": component,
                    "pred": pred_value,
                    "ref": ref_value,
                    "abs_error": float(abs_error),
                    "ordinary_relative_error": float(ordinary),
                    "relative_error_with_floor": float(robust),
                    "peak_normalized_error": float(peak),
                    "pass_5pct": bool(robust <= float(threshold) and peak <= float(threshold)),
                }
            )
    return out


def _write_receiver_reference_error_artifacts(workdir: Path, receiver_rows: list[dict[str, Any]]) -> None:
    csv_path = workdir / "receiver_reference_errors.csv"
    png_path = workdir / "receiver_reference_error_curves.png"
    if not receiver_rows:
        for path in (csv_path, png_path):
            if path.exists():
                path.unlink()
        return
    fields = [
        "time_obs",
        "receiver_type",
        "component",
        "pred",
        "ref",
        "abs_error",
        "ordinary_relative_error",
        "relative_error_with_floor",
        "peak_normalized_error",
        "pass_5pct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in receiver_rows:
            writer.writerow(row)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    components = []
    receiver_types = []
    for row in receiver_rows:
        component = str(row["component"])
        receiver_type = str(row["receiver_type"])
        if component not in components:
            components.append(component)
        if receiver_type not in receiver_types:
            receiver_types.append(receiver_type)
    fig, axes = plt.subplots(len(components), 1, figsize=(7.2, 2.6 * len(components) + 1.0), sharex=True)
    axes_arr = np.atleast_1d(axes)
    markers = ["o", "s", "^", "d", "x"]
    for ax, component in zip(axes_arr, components):
        for i, receiver_type in enumerate(receiver_types):
            subset = [
                row for row in receiver_rows
                if str(row["component"]) == component and str(row["receiver_type"]) == receiver_type
            ]
            subset.sort(key=lambda row: float(row["time_obs"]))
            if not subset:
                continue
            ax.semilogx(
                [float(row["time_obs"]) for row in subset],
                [float(row["relative_error_with_floor"]) for row in subset],
                marker=markers[i % len(markers)],
                label=f"{receiver_type} robust",
            )
            ax.semilogx(
                [float(row["time_obs"]) for row in subset],
                [float(row["peak_normalized_error"]) for row in subset],
                linestyle="--",
                marker=markers[i % len(markers)],
                alpha=0.75,
                label=f"{receiver_type} peak",
            )
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1.2, label="5%")
        ax.set_ylabel(component)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    axes_arr[-1].set_xlabel("t_obs (s)")
    fig.suptitle("Receiver diagnostics vs reference")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _forward_components(config: PipelineConfig) -> list[str]:
    components = ["Ex", "Ey"]
    magnetic_mode = str(config.magnetic_receiver_mode).strip().lower()
    formulation = str(config.formulation).strip().lower()
    if formulation == "h" or magnetic_mode in {"biot_current", "biot_ohmic", "faraday_integrated"}:
        components.append("Hz")
    components.append("dBzdt")
    return components


def cole_cole_complex_conductivity(freqs, rho0: float, m: float, tau: float, c: float):
    """Pelton/Cole-Cole resistivity form converted to complex conductivity."""

    import numpy as np

    freqs = np.asarray(freqs, dtype=float)
    s_tau = 1j * 2.0 * np.pi * freqs * tau
    rho = rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + s_tau**c)))
    return 1.0 / rho


def fit_cole_cole_to_debye(config: PipelineConfig) -> DebyeFit:
    """Fit Cole-Cole conductivity to nonnegative Debye/Prony poles."""

    import numpy as np
    from scipy.optimize import lsq_linear

    if config.cole_n_terms <= 0:
        raise ValueError("cole_n_terms must be positive")
    freqs = np.logspace(math.log10(config.cole_f_min), math.log10(config.cole_f_max), config.cole_n_freq)
    target = cole_cole_complex_conductivity(freqs, config.cole_rho0, config.cole_m, config.cole_tau, config.cole_c)
    sigma_inf = 1.0 / (config.cole_rho0 * (1.0 - config.cole_m))
    tau_min = 1.0 / (2.0 * math.pi * freqs.max()) / 10.0
    tau_max = 1.0 / (2.0 * math.pi * freqs.min()) * 10.0
    tau_grid = np.logspace(math.log10(tau_min), math.log10(tau_max), config.cole_n_terms)
    basis = 1.0 / (1.0 + 1j * 2.0 * np.pi * freqs[:, None] * tau_grid[None, :])
    rhs = sigma_inf - target
    sigma_dc = 1.0 / float(config.cole_rho0)
    dc_delta_sum = sigma_inf - sigma_dc
    dc_weight = 1.0e6
    design = np.vstack([basis.real, basis.imag, dc_weight * np.ones((1, config.cole_n_terms))])
    data = np.r_[rhs.real, rhs.imag, dc_weight * dc_delta_sum]
    fit = lsq_linear(design, data, bounds=(0.0, np.inf), lsmr_tol="auto")
    delta = fit.x
    fitted = sigma_inf - basis @ delta
    denom = np.linalg.norm(np.r_[target.real, target.imag])
    rel_l2 = float(np.linalg.norm(np.r_[fitted.real - target.real, fitted.imag - target.imag]) / denom)
    terms = [DebyeTerm(float(d), float(t)) for d, t in zip(delta, tau_grid)]
    print(
        f"[polarization] Cole-Cole fitted to {len(terms)} Debye terms; "
        f"relative L2 error={rel_l2:.6e}",
        flush=True,
    )
    for i, term in enumerate(terms):
        print(f"  term {i}: A={term.delta_sigma:.6e} S/m, tau={term.tau:.6e} s", flush=True)
    return DebyeFit(sigma_inf, terms, freqs, target, fitted, rel_l2)


def _polarizable_earth_cells(msh, cell_tags, config: PipelineConfig):
    import numpy as np

    earth_cells = np.asarray(cell_tags.find(PHYS_EARTH), dtype=np.int32)
    if earth_cells.size == 0:
        return earth_cells
    top = float(config.cole_layer_top)
    bottom = float(config.cole_layer_bottom)
    if math.isinf(bottom):
        return earth_cells
    centers = _cell_centers(msh)
    depths = -centers[earth_cells.astype(int), 2]
    mask = (depths >= top) & (depths <= bottom)
    return earth_cells[mask]


def _build_debye_materials(msh, cell_tags, spaces: dict[str, Any], fit: DebyeFit, config: PipelineConfig):
    from dolfinx import fem

    Q = spaces["Q"]
    delta_functions = []
    polarizable_cells = _polarizable_earth_cells(msh, cell_tags, config)
    if len(polarizable_cells) == 0:
        raise RuntimeError(
            "Cole-Cole polarization is enabled but no local earth cells fall inside "
            f"cole_layer_top={float(config.cole_layer_top):.6g} m and "
            f"cole_layer_bottom={float(config.cole_layer_bottom):.6g} m"
        )
    for index, term in enumerate(fit.terms):
        fn = fem.Function(Q, name=f"delta_sigma_term_{index}")
        fn.x.array[:] = 0.0
        _assign_dg0_by_cell(fn, polarizable_cells, term.delta_sigma)
        fn.x.scatter_forward()
        delta_functions.append(fn)
    return {
        "fit": fit,
        "terms": fit.terms,
        "delta_functions": delta_functions,
        "polarizable_cells": polarizable_cells,
    }


def _debye_backward_euler_coefficients(term: DebyeTerm, dt: float) -> tuple[float, float]:
    """Return alpha, beta for tau*dpsi/dt + psi = E under backward Euler."""

    tau = float(term.tau)
    dt = float(dt)
    if tau <= 0.0 or dt <= 0.0:
        raise ValueError("Debye tau and timestep must be positive")
    denom = tau + dt
    return tau / denom, dt / denom


def _debye_total_field_step_metadata(debye, dt: float) -> dict[str, Any]:
    """Return the task-book E-form total-field Debye BE convention.

    The DOLFINx total-field IP step uses
    ``J = sigma_inf E - sum(delta_sigma_k chi_k)`` and
    ``chi_k_new = alpha_k chi_k_old + beta_k E_new``. Eliminating ``chi`` gives
    ``[K + R + M(sigma_eff)/dt] E_new =
    M[J_old + sum(delta_sigma_k alpha_k chi_old_k)]/dt + source``.
    """

    if debye is None or not debye.get("terms", ()):
        return {
            "enabled": False,
            "time_scheme": "backward_euler",
            "reason": "no_debye_terms",
        }
    dt = float(dt)
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")

    fit = debye.get("fit")
    if fit is not None:
        sigma_inf = float(getattr(fit, "sigma_infinity"))
    elif "sigma_infinity" in debye:
        sigma_inf = float(debye["sigma_infinity"])
    else:
        raise ValueError("Debye metadata requires fit.sigma_infinity or sigma_infinity")
    terms = tuple(debye.get("terms", ()))
    alpha: list[float] = []
    beta: list[float] = []
    delta_sigma: list[float] = []
    sigma_eff = sigma_inf
    for term in terms:
        alpha_i, beta_i = _debye_backward_euler_coefficients(term, dt)
        delta_i = float(term.delta_sigma)
        alpha.append(float(alpha_i))
        beta.append(float(beta_i))
        delta_sigma.append(delta_i)
        sigma_eff -= delta_i * float(beta_i)

    sum_delta = float(sum(delta_sigma))
    return {
        "enabled": True,
        "time_scheme": "backward_euler",
        "current_convention": "J = sigma_inf E - sum(delta_sigma_k chi_k)",
        "memory_update": "chi_k_new = alpha_k * chi_k_old + beta_k * E_new",
        "memory_initial_condition": "chi_k0 = E0",
        "lhs_operator": "K + R + M(sigma_eff)/dt",
        "rhs_history": "M[J_old + sum(delta_sigma_k * alpha_k * chi_old_k)]/dt",
        "sigma_inf": sigma_inf,
        "sum_delta_sigma": sum_delta,
        "sigma0": float(sigma_inf - sum_delta),
        "sigma_eff": float(sigma_eff),
        "delta_sigma": delta_sigma,
        "tau": [float(term.tau) for term in terms],
        "alpha": alpha,
        "beta": beta,
        "delta_sigma_zero_degenerates_to_noip": bool(all(value == 0.0 for value in delta_sigma)),
    }


def _matrix_for_effective_conductivity(operators, debye, dt: float):
    from petsc4py import PETSc

    if debye is None or not debye["terms"]:
        return operators["M"]
    M_eff = operators["M_inf"].copy()
    for term, M_delta in zip(debye["terms"], operators["M_debye"]):
        _alpha, beta = _debye_backward_euler_coefficients(term, dt)
        M_eff.axpy(-beta, M_delta, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    M_eff.assemble()
    return M_eff


def _assemble_history_rhs(operators, debye, memories, E_old, dt: float):
    if debye is None or not debye["terms"]:
        b = E_old.x.petsc_vec.duplicate()
        operators["M"].mult(E_old.x.petsc_vec, b)
        b.scale(1.0 / dt)
        return b

    b = E_old.x.petsc_vec.duplicate()
    operators["M_inf"].mult(E_old.x.petsc_vec, b)
    tmp = E_old.x.petsc_vec.duplicate()
    for term, M_delta, chi in zip(debye["terms"], operators["M_debye"], memories):
        _alpha, beta = _debye_backward_euler_coefficients(term, dt)
        M_delta.mult(chi.x.petsc_vec, tmp)
        b.axpy(-beta, tmp)
    b.scale(1.0 / dt)
    tmp.destroy()
    return b


def _theta_step_coefficients(dt: float, theta: float) -> dict[str, float]:
    dt = float(dt)
    theta = float(theta)
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(theta) or not (0.5 <= theta <= 1.0):
        raise ValueError("theta must be finite and in [0.5, 1]")
    return {
        "mass_lhs": 1.0 / dt,
        "stiffness_lhs": theta,
        "mass_rhs": 1.0 / dt,
        "stiffness_rhs": -(1.0 - theta),
    }


def _bdf2_step_coefficients(dt: float, previous_dt: float) -> dict[str, float]:
    """Return variable-step BDF2 derivative coefficients for M*dE/dt."""

    dt = float(dt)
    previous_dt = float(previous_dt)
    if dt <= 0.0 or previous_dt <= 0.0 or not math.isfinite(dt) or not math.isfinite(previous_dt):
        raise ValueError("BDF2 timesteps must be finite and positive")
    ratio = dt / previous_dt
    return {
        "lhs": (1.0 + 2.0 * ratio) / (dt * (1.0 + ratio)),
        "old": (1.0 + ratio) / dt,
        "older": -(ratio * ratio) / (dt * (1.0 + ratio)),
    }


def _add_scalar_point_load(vec, msh, S, point, value: float) -> None:
    """Add value*q(point) to a scalar H1 load vector."""

    import numpy as np
    from petsc4py import PETSc

    cell = _find_cell_for_point(msh, point)
    if cell is None:
        raise RuntimeError(f"point electrode {point} was not found in a local cell")
    cell_geom = _cell_geometry(msh, cell)
    X = msh.geometry.cmap.pull_back(np.asarray(point, dtype=float).reshape(1, 3), cell_geom)
    basis = S.element.basix_element.tabulate(0, X)[0, 0, :, 0]
    local_dofs = S.dofmap.cell_dofs(cell)
    global_dofs = S.dofmap.index_map.local_to_global(local_dofs).astype(PETSc.IntType)
    vec.setValues(global_dofs, float(value) * basis, addv=PETSc.InsertMode.ADD_VALUES)


def _scalar_potential_gauge_dofs(S):
    """Return local scalar-potential dofs used to fix one global gauge value."""

    import numpy as np

    local_start, local_end = S.dofmap.index_map.local_range
    if local_start <= 0 < local_end:
        return np.asarray([0 - local_start], dtype=np.int32)
    return np.asarray([], dtype=np.int32)


def _analytic_halfspace_dc_electric_field(points, config: PipelineConfig):
    """Return the static grounded-wire E field for a uniform conducting halfspace."""

    import numpy as np

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    start = np.asarray(config.source_start, dtype=float)
    end = np.asarray(config.source_end, dtype=float)
    r_start = pts - start[None, :]
    r_end = pts - end[None, :]
    n_start = np.linalg.norm(r_start, axis=1)
    n_end = np.linalg.norm(r_end, axis=1)
    eps = np.finfo(float).eps
    n_start = np.maximum(n_start, eps)
    n_end = np.maximum(n_end, eps)
    scale = _uniform_halfspace_resistivity(config) * float(config.source_current) / (2.0 * np.pi)
    return scale * (r_end / n_end[:, None] ** 3 - r_start / n_start[:, None] ** 3)


def _biot_savart_line_h(points, start, end, *, current: float, n_quad: int = 201):
    """Return H from a finite straight wire by Gauss-Legendre line quadrature."""

    import numpy as np

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("line source endpoints must be distinct")
    tangent = axis / length
    n_quad = max(1, int(n_quad))
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * (qx + 1.0)
    weights = 0.5 * length * qw
    src_pts = p0[None, :] + s[:, None] * axis[None, :]
    dl = tangent[None, :] * weights[:, None]

    out = np.zeros((pts.shape[0], 3), dtype=float)
    for source_point, dli in zip(src_pts, dl):
        r = pts - source_point[None, :]
        norm = np.linalg.norm(r, axis=1)
        mask = norm > 0.0
        contrib = np.zeros_like(r)
        contrib[mask] = np.cross(dli, r[mask]) / norm[mask, None] ** 3
        out += contrib
    return float(current) * out / (4.0 * np.pi)


def _cell_current_density_from_debye_values(e_vals, sigma_values, delta_values=None, memory_values=None):
    """Return J = sigma_inf*E - sum(delta_sigma*chi) for cell-centered values."""

    import numpy as np

    e_vals = np.asarray(e_vals, dtype=float)
    sigma_values = np.asarray(sigma_values, dtype=float)
    currents = sigma_values[:, None] * e_vals
    if delta_values is None or memory_values is None:
        return currents
    for delta, memory in zip(delta_values, memory_values):
        currents -= np.asarray(delta, dtype=float)[:, None] * np.asarray(memory, dtype=float)
    return currents


def _initialise_debye_memories_to_field(memories, E_old) -> None:
    """A pre-ramp DC state has Debye memory variables chi=E."""

    for memory in memories:
        memory.x.array[:] = E_old.x.array
        memory.x.scatter_forward()


def _biot_savart_cell_current_h_at_receiver(
    E,
    msh,
    materials: dict[str, Any],
    config: PipelineConfig,
    *,
    debye=None,
    memories=None,
):
    """Recover receiver H from cell-centered ohmic currents with Biot-Savart."""

    import numpy as np

    centers, _radii, volumes = _cell_centers_radii_volumes(msh)
    n_cells = centers.shape[0]
    cells = np.arange(n_cells, dtype=np.int32)
    e_vals = np.asarray(E.eval(centers, cells), dtype=float).reshape(n_cells, 3)
    use_debye_current = debye is not None and memories is not None and bool(debye.get("terms", []))
    if use_debye_current:
        sigma_source = materials.get("sigma_infinity_physical", materials["sigma_infinity"])
    else:
        sigma_source = materials.get("sigma_physical", materials["sigma"])
    sigma = np.asarray(sigma_source.x.array[:n_cells], dtype=float)
    if use_debye_current:
        delta_values = [
            np.asarray(delta_fn.x.array[:n_cells], dtype=float) for delta_fn in debye["delta_functions"]
        ]
        memory_values = [
            np.asarray(memory.eval(centers, cells), dtype=float).reshape(n_cells, 3) for memory in memories
        ]
        currents = _cell_current_density_from_debye_values(e_vals, sigma, delta_values, memory_values)
    else:
        currents = _cell_current_density_from_debye_values(e_vals, sigma)
    h_values = []
    for receiver in _receiver_sampling_points(config):
        r = np.asarray(receiver, dtype=float)[None, :] - centers
        norm = np.linalg.norm(r, axis=1)
        mask = norm > 0.0
        h = np.zeros(3, dtype=float)
        if np.any(mask):
            h = np.sum(
                volumes[mask, None]
                * np.cross(currents[mask], r[mask])
                / (4.0 * np.pi * norm[mask, None] ** 3),
                axis=0,
            )
        h_values.append(h)
    return np.mean(np.vstack(h_values), axis=0)


def _biot_savart_line_h_at_receiver(config: PipelineConfig, *, current: float, n_quad: int = 201):
    """Return finite-wire H using the same point/average receiver sampling as E/dBdt."""

    import numpy as np

    return np.mean(
        _biot_savart_line_h(
            _receiver_sampling_points(config),
            np.asarray(config.source_start, dtype=float),
            np.asarray(config.source_end, dtype=float),
            current=float(current),
            n_quad=int(n_quad),
        ),
        axis=0,
    )


def _biot_savart_total_h_at_receiver(
    E,
    msh,
    materials: dict[str, Any],
    config: PipelineConfig,
    source_current: float,
    *,
    debye=None,
    memories=None,
):
    """Return receiver H from ohmic cell currents plus the impressed wire current."""

    import numpy as np

    h = _biot_savart_cell_current_h_at_receiver(
        E,
        msh,
        materials,
        config,
        debye=debye,
        memories=memories,
    )
    h += _biot_savart_line_h_at_receiver(
        config,
        current=float(source_current),
    )
    return h


def _assign_biot_receiver_hz(receiver_values: dict[str, float], h_receiver) -> None:
    """Use Biot-Savart for H while preserving instantaneous Faraday dB/dt."""

    receiver_values["Hz"] = float(h_receiver[2])


def _advance_faraday_receiver_hz(
    *,
    previous_hz: float,
    dbzdt_new: float,
    dt: float,
    mu: float = 1.2566370614359173e-6,
) -> float:
    """Backward-Euler receiver Hz update from dBz/dt = -curl(E)."""

    dt_value = float(dt)
    mu_value = float(mu)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive")
    if not math.isfinite(mu_value) or mu_value <= 0.0:
        raise ValueError("mu must be positive")
    return float(previous_hz) + dt_value * float(dbzdt_new) / mu_value


def _biot_receiver_dbdt_from_h(h_new, h_old, *, dt: float, mu: float = 1.2566370614359173e-6):
    import numpy as np

    dt_value = float(dt)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive")
    mu_value = float(mu)
    if not math.isfinite(mu_value) or mu_value <= 0.0:
        raise ValueError("mu must be positive")
    new = np.asarray(h_new, dtype=float).reshape(-1)
    old = np.asarray(h_old, dtype=float).reshape(-1)
    if new.shape != old.shape or new.size != 3:
        raise ValueError("h_new and h_old must be three-component vectors")
    if not (np.all(np.isfinite(new)) and np.all(np.isfinite(old))):
        raise ValueError("h_new and h_old must be finite")
    return mu_value * (new - old) / dt_value


def _interpolate_analytic_initial_dc_field(msh, spaces, config: PipelineConfig):
    """Interpolate analytic halfspace DC E into the Nedelec space."""

    def field(x):
        pts = x.T
        values = _analytic_halfspace_dc_electric_field(pts, config)
        return values.T

    E0 = _interpolate_vector_callable_to_nedelec_function(
        spaces,
        name="E_initial_dc_analytic_halfspace",
        field_callable=field,
    )
    log("[initial] interpolated analytic halfspace DC electric field.", comm=msh.comm)
    return E0


def _interpolate_vector_callable_to_nedelec_function(spaces, *, name: str, field_callable):
    """Interpolate a vector callable into the configured Nedelec function space."""

    from dolfinx import fem

    V = spaces["V"]
    function = fem.Function(V, name=name)
    function.interpolate(field_callable)
    function.x.scatter_forward()
    return function


def _make_nedelec_rhs_interpolator_from_samples(
    spaces,
    *,
    sample_points=None,
    name: str = "secondary_rhs_density",
):
    """Return a converter from vector samples to a Nedelec Function.

    Constant sample tables are interpolated as constant vector fields. For
    non-constant sample tables, explicit physical ``sample_points`` are needed
    so the tabulated callable can match DOLFINx interpolation coordinates.
    """

    import numpy as np

    points = None if sample_points is None else np.asarray(sample_points, dtype=float)

    def convert(samples):
        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("secondary RHS samples must have shape (n_samples, 3)")
        constant_atol = max(1.0e-14, 1.0e-12 * float(np.max(np.abs(values))))
        if np.allclose(values, values[0], rtol=1.0e-12, atol=constant_atol):
            vector = values[0].copy()

            def constant_field(x):
                return np.vstack(
                    (
                        np.full(x.shape[1], vector[0], dtype=float),
                        np.full(x.shape[1], vector[1], dtype=float),
                        np.full(x.shape[1], vector[2], dtype=float),
                    )
                )

            return _interpolate_vector_callable_to_nedelec_function(
                spaces,
                name=name,
                field_callable=constant_field,
            )
        if points is None:
            raise ValueError("sample_points are required for non-constant secondary RHS samples")
        from atem3d.primary import TabulatedVectorField

        field = TabulatedVectorField(points=points, values=values)
        return _interpolate_vector_callable_to_nedelec_function(
            spaces,
            name=name,
            field_callable=field,
        )

    return convert


def _make_nedelec_solution_sampler_at_points(msh, sample_points):
    """Return a sampler that evaluates a Nedelec Function at physical points."""

    import numpy as np

    points = np.asarray(sample_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("sample_points must have shape (n_samples, 3)")
    cells = []
    for point in points:
        point_cells = _find_cells_for_point(msh, point)
        if len(point_cells) == 0:
            raise RuntimeError(f"sample point {point.tolist()} was not found in a local cell")
        cells.append(int(point_cells[0]))
    cell_array = np.asarray(cells, dtype=np.int32)

    def sample(solution, _rhs_samples=None):
        values = np.asarray(solution.eval(points, cell_array), dtype=float)
        return values.reshape(points.shape[0], 3)

    return sample


def _solve_initial_dc_field(msh, spaces, materials, facet_tags, config: PipelineConfig):
    """Solve charge-conserving DC potential and interpolate E0=-grad(phi)."""

    mode = str(config.initial_dc_mode).strip().lower()
    if mode == "analytic_halfspace":
        return _interpolate_analytic_initial_dc_field(msh, spaces, config)
    if mode != "fem":
        raise ValueError("initial_dc_mode must be 'analytic_halfspace' or 'fem'")

    import numpy as np
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from petsc4py import PETSc

    V = spaces["V"]
    S = spaces["S"]
    phi = fem.Function(S, name="dc_phi")
    u = ufl.TrialFunction(S)
    q = ufl.TestFunction(S)
    sigma_initial = materials.get("sigma_initial", materials["sigma"])
    a = fem.form(sigma_initial * ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx)
    gauge_dofs = _scalar_potential_gauge_dofs(S)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), gauge_dofs, S)
    A = fem_petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = _build_endpoint_scalar_load(msh, S, config)
    fem_petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=max(config.max_it, 1000))
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    ksp.setFromOptions()
    ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()
    its = ksp.getIterationNumber()
    residual = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    log(
        f"[initial] solved charge-conserving DC potential with KSP its={its} "
        f"residual={residual:.3e} reason={reason}",
        comm=msh.comm,
    )
    if reason < 0:
        raise RuntimeError(f"initial DC potential solve failed, reason={reason}, residual={residual:.6e}")

    E0 = fem.Function(V, name="E_initial_dc")
    expr = fem.Expression(-ufl.grad(phi), V.element.interpolation_points(), comm=msh.comm)
    E0.interpolate(expr)
    E0.x.scatter_forward()
    ksp.destroy()
    b.destroy()
    A.destroy()
    return E0


def _solve_dc_secondary_field(
    msh,
    spaces,
    materials,
    Ep0,
    config: PipelineConfig,
    *,
    sigma_background: float,
):
    """Solve the primary-secondary DC initialization problem.

    The scalar secondary potential satisfies

    int sigma0 grad(phi_s).grad(q) dx =
    int (sigma0 - sigma_b) Ep0.grad(q) dx,

    and the secondary electric field is Es0 = -grad(phi_s).
    """

    import numpy as np
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    V = spaces["V"]
    S = spaces["S"]
    sigma0 = materials.get("sigma_initial", materials["sigma"])
    sigma_b = float(sigma_background)
    sigma_values = np.asarray(sigma0.x.array, dtype=float)
    local_contrast = float(np.max(np.abs(sigma_values - sigma_b))) if sigma_values.size else 0.0
    contrast_norm = float(msh.comm.allreduce(local_contrast, op=MPI.MAX))
    contrast_is_zero = contrast_norm <= max(1.0e-14, 1.0e-12 * abs(sigma_b))

    phi_s = fem.Function(S, name="dc_secondary_phi")
    Es0 = fem.Function(V, name="E_secondary_dc")
    if contrast_is_zero:
        Es0.x.array[:] = 0.0
        Es0.x.scatter_forward()
        phi_s.x.array[:] = 0.0
        phi_s.x.scatter_forward()
        return {
            "phi_s": phi_s,
            "Es0": Es0,
            "contrast_is_zero": True,
            "contrast_norm": contrast_norm,
            "ksp_iterations": 0,
            "ksp_reason": 0,
            "ksp_residual": 0.0,
        }

    u = ufl.TrialFunction(S)
    q = ufl.TestFunction(S)
    sigma_b_const = fem.Constant(msh, PETSc.ScalarType(sigma_b))
    a = fem.form(sigma0 * ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx)
    L = fem.form((sigma0 - sigma_b_const) * ufl.inner(Ep0, ufl.grad(q)) * ufl.dx)
    gauge_dofs = _scalar_potential_gauge_dofs(S)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), gauge_dofs, S)
    A = fem_petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = fem_petsc.assemble_vector(L)
    fem_petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=max(config.max_it, 1000))
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    ksp.setFromOptions()
    ksp.solve(b, phi_s.x.petsc_vec)
    phi_s.x.scatter_forward()
    reason = int(ksp.getConvergedReason())
    residual = float(ksp.getResidualNorm())
    its = int(ksp.getIterationNumber())
    if reason < 0:
        ksp.destroy()
        b.destroy()
        A.destroy()
        raise RuntimeError(f"DC secondary potential solve failed, reason={reason}, residual={residual:.6e}")

    expr = fem.Expression(-ufl.grad(phi_s), V.element.interpolation_points(), comm=msh.comm)
    Es0.interpolate(expr)
    Es0.x.scatter_forward()
    ksp.destroy()
    b.destroy()
    A.destroy()
    return {
        "phi_s": phi_s,
        "Es0": Es0,
        "contrast_is_zero": False,
        "contrast_norm": contrast_norm,
        "ksp_iterations": its,
        "ksp_reason": reason,
        "ksp_residual": residual,
    }


def _make_dolfinx_secondary_step_solver(
    spaces,
    operators,
    config: PipelineConfig,
    *,
    rhs_to_function=None,
    solution_to_samples=None,
):
    """Return a DOLFINx-backed secondary step solver callable.

    The returned callable matches ``atem3d.solvers.SecondarySolver``.  The
    optional hooks convert between pure sample tables and DOLFINx Functions.
    The zero-RHS path is handled without hooks and is used by zero-contrast
    primary-secondary validation.
    """

    import numpy as np
    from dolfinx import fem
    from petsc4py import PETSc

    V = spaces["V"]
    solution = fem.Function(V, name="E_secondary_step")
    solver_context = {"ksp": None}

    def solve(rhs_density, sigma_eff: float, dt: float):
        rhs_is_function = hasattr(rhs_density, "x") and hasattr(rhs_density, "function_space")
        rhs_array = None if rhs_is_function else np.asarray(rhs_density, dtype=float)
        dt_value = float(dt)
        if dt_value <= 0.0 or not math.isfinite(dt_value):
            raise ValueError("dt must be finite and positive")
        if float(sigma_eff) <= 0.0 or not math.isfinite(float(sigma_eff)):
            raise ValueError("sigma_eff must be finite and positive")
        A = _copy_and_combine_matrix(operators["K"], operators["M"], 1.0 / dt_value)
        if str(config.outer_boundary_mode).strip().lower() == "robin":
            B_robin = operators.get("B_robin")
            if B_robin is None:
                A.destroy()
                raise RuntimeError("Robin outer boundary mode requires B_robin operator")
            A.axpy(
                _outer_boundary_robin_admittance(config, dt_value),
                B_robin,
                structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
            )
            A.assemble()
        _zero_rows_columns(A, operators["bc_global"], diag=1.0)

        b = solution.x.petsc_vec.duplicate()
        if rhs_is_function:
            if float(rhs_density.x.petsc_vec.norm()) == 0.0:
                b.set(0.0)
            else:
                operators["M"].mult(rhs_density.x.petsc_vec, b)
        elif np.allclose(rhs_array, 0.0, rtol=0.0, atol=0.0):
            b.set(0.0)
        else:
            if rhs_to_function is None:
                b.destroy()
                A.destroy()
                raise ValueError("rhs_to_function is required for nonzero secondary RHS samples")
            rhs_function = rhs_to_function(rhs_array)
            operators["M"].mult(rhs_function.x.petsc_vec, b)
        b.assemble()
        _zero_rhs_entries(b, operators["bc_global"])

        if solver_context["ksp"] is None:
            solver_context["ksp"] = configure_ams_solver(A, spaces, config)
        else:
            solver_context["ksp"]["ksp"].setOperators(A)
        solution.x.array[:] = 0.0
        solver_context["ksp"]["ksp"].solve(b, solution.x.petsc_vec)
        solution.x.scatter_forward()
        reason = int(solver_context["ksp"]["ksp"].getConvergedReason())
        residual = float(solver_context["ksp"]["ksp"].getResidualNorm())
        if reason < 0:
            b.destroy()
            A.destroy()
            raise RuntimeError(f"secondary step solve failed, reason={reason}, residual={residual:.6e}")

        if solution_to_samples is not None:
            result = np.asarray(solution_to_samples(solution, rhs_array), dtype=float)
        elif (not rhs_is_function) and np.allclose(rhs_array, 0.0, rtol=0.0, atol=0.0):
            result = np.zeros_like(rhs_array)
        else:
            b.destroy()
            A.destroy()
            raise ValueError("solution_to_samples is required for nonzero secondary solves")
        b.destroy()
        A.destroy()
        return result

    return solve


def _record_primary_secondary_step_equation(
    diagnostics: dict[str, Any],
    *,
    material,
    sigma_background: float,
    dt: float,
) -> dict[str, Any]:
    from atem3d.solvers import secondary_step_equation_metadata

    metadata = secondary_step_equation_metadata(
        material=material,
        sigma_background=float(sigma_background),
        dt=float(dt),
    )
    metadata["adapter_backend"] = "dolfinx_primary_secondary"
    metadata["dt_source"] = "secondary_state_stepper_runtime_dt"
    diagnostics["primary_secondary_step_equation"] = metadata
    return metadata


def _make_dolfinx_primary_secondary_forward_adapters(
    msh,
    spaces,
    materials,
    operators,
    config: PipelineConfig,
    fem_points,
    *,
    sigma_background: float,
    debye=None,
):
    """Wire DOLFINx secondary solves into the pure primary-secondary core.

    The pure core currently passes the DC secondary callback a scalar
    contrast-weighted ``(sigma0 - sigma_b) Ep0`` sample table.  This adapter
    reconstructs ``Ep0`` from that scalar bridge for initialization, then keeps
    the no-IP transient current contrast and RHS in DOLFINx Function form so
    variable DG0 material contrast is represented in the step solve.
    """

    import numpy as np
    from dolfinx import fem

    points = np.asarray(fem_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("fem_points must have shape (n_points, 3)")
    sigma_b = float(sigma_background)
    if sigma_b <= 0.0 or not math.isfinite(sigma_b):
        raise ValueError("sigma_background must be finite and positive")
    sigma_initial = materials.get("sigma_initial", materials["sigma"])
    sigma_values = np.asarray(sigma_initial.x.array, dtype=float)
    if sigma_values.size == 0:
        raise ValueError("sigma_initial must contain local DG values")
    local_contrast = sigma_values - sigma_b
    contrast_index = int(np.argmax(np.abs(local_contrast)))
    nominal_contrast = float(local_contrast[contrast_index])
    if abs(nominal_contrast) <= max(1.0e-14, 1.0e-12 * abs(sigma_b)):
        raise ValueError("nonzero contrast is required for this adapter")

    V = spaces["V"]
    sigma = materials["sigma"]
    sigma_infinity = materials.get("sigma_infinity", sigma)
    sigma_b_const = fem.Constant(msh, float(sigma_b))
    debye_terms = tuple(debye.get("terms", ())) if debye is not None else ()
    delta_functions = tuple(debye.get("delta_functions", ())) if debye is not None else ()
    if delta_functions and len(delta_functions) != len(debye_terms):
        raise ValueError("debye delta_functions must match debye terms")
    sample_to_function = _make_nedelec_rhs_interpolator_from_samples(
        spaces,
        sample_points=points,
        name="primary_secondary_sample_field",
    )
    sample_solution = _make_nedelec_solution_sampler_at_points(msh, points)
    latest_secondary = {
        "E": fem.Function(V, name="E_secondary_latest"),
        "deltaJ": None,
        "chi": [],
        "dc_result": None,
    }
    latest_secondary["E"].x.array[:] = 0.0
    latest_secondary["E"].x.scatter_forward()

    def interpolate_vector_expression(name: str, expression):
        function = fem.Function(V, name=name)
        expr = fem.Expression(expression, V.element.interpolation_points(), comm=msh.comm)
        function.interpolate(expr)
        function.x.scatter_forward()
        return function

    def delta_expression(term, index: int):
        if index < len(delta_functions):
            return delta_functions[index]
        return float(term.delta_sigma)

    def build_initialization(Ep0_samples, material, sigma_background_value):
        from atem3d.solvers.dc_secondary import DCSecondaryInitialization

        if abs(float(sigma_background_value) - sigma_b) > max(1.0e-14, 1.0e-12 * abs(sigma_b)):
            raise ValueError("sigma_background mismatch in DOLFINx primary-secondary initializer")
        Ep0_array = np.asarray(Ep0_samples, dtype=float)
        Ep0_function = sample_to_function(Ep0_array)
        result = _solve_dc_secondary_field(
            msh,
            spaces,
            materials,
            Ep0_function,
            config,
            sigma_background=sigma_b,
        )
        latest_secondary["E"] = result["Es0"]
        latest_secondary["dc_result"] = result
        Es0_samples = sample_solution(result["Es0"], Ep0_array)
        Etotal0_samples = Ep0_array + Es0_samples
        Etotal0_expr = Ep0_function + result["Es0"]
        if material.terms:
            latest_secondary["chi"] = [
                interpolate_vector_expression(f"primary_secondary_chi_{index}_dc", Etotal0_expr)
                for index, _term in enumerate(material.terms)
            ]
            deltaJ_expr = sigma_infinity * Etotal0_expr - sigma_b_const * Ep0_function
            for index, (term, memory) in enumerate(zip(material.terms, latest_secondary["chi"])):
                deltaJ_expr = deltaJ_expr - delta_expression(term, index) * memory
        else:
            latest_secondary["chi"] = []
            deltaJ_expr = sigma * Etotal0_expr - sigma_b_const * Ep0_function
        latest_secondary["deltaJ"] = interpolate_vector_expression(
            "primary_secondary_deltaJ_dc",
            deltaJ_expr,
        )
        deltaJ0_samples = sample_solution(latest_secondary["deltaJ"], Ep0_array)
        chi0_samples = [sample_solution(memory, Ep0_array) for memory in latest_secondary["chi"]]
        return DCSecondaryInitialization(
            Ep0=Ep0_array,
            Es0=Es0_samples,
            Etotal0=Etotal0_samples,
            chi0=chi0_samples,
            deltaJ0=deltaJ0_samples,
            phi_s=None,
            contrast_is_zero=bool(result["contrast_is_zero"]),
        )

    def secondary_field_solver(contrast_weighted_Ep0):
        contrast_weighted = np.asarray(contrast_weighted_Ep0, dtype=float)
        Ep0_samples = contrast_weighted / nominal_contrast
        initialization = build_initialization(
            Ep0_samples,
            material=type("_NoIPMaterial", (), {"terms": ()})(),
            sigma_background_value=sigma_b,
        )
        return None, initialization.Es0

    def solution_to_samples(solution, rhs_samples):
        latest_secondary["E"] = solution
        return sample_solution(solution, rhs_samples)

    secondary_step_solver = _make_dolfinx_secondary_step_solver(
        spaces,
        operators,
        config,
        rhs_to_function=_make_nedelec_rhs_interpolator_from_samples(
            spaces,
            sample_points=points,
            name="secondary_rhs_density",
        ),
        solution_to_samples=solution_to_samples,
    )

    def electric_getter(_state, _Ep_new, _time_value, _dt):
        return latest_secondary["E"]

    def dbdt_getter(_state, _Ep_new, _time_value, _dt):
        return compute_dbdt(latest_secondary["E"], spaces)

    def secondary_state_stepper(state, Ep_old, Ep_new, material, sigma_background_value, dt):
        from atem3d.solvers.tdem_secondary import SecondaryState

        if abs(float(sigma_background_value) - sigma_b) > max(1.0e-14, 1.0e-12 * abs(sigma_b)):
            raise ValueError("sigma_background mismatch in DOLFINx primary-secondary adapter")
        if latest_secondary["deltaJ"] is None:
            raise RuntimeError("secondary_field_solver must run before secondary_state_stepper")
        dt_value = float(dt)
        if dt_value <= 0.0 or not math.isfinite(dt_value):
            raise ValueError("dt must be finite and positive")
        _record_primary_secondary_step_equation(
            latest_secondary,
            material=material,
            sigma_background=sigma_b,
            dt=dt_value,
        )
        Ep_new_function = sample_to_function(Ep_new)
        if material.terms and len(latest_secondary["chi"]) != len(material.terms):
            latest_secondary["chi"] = [
                sample_to_function(memory_samples)
                for memory_samples in state.chi
            ]
            if len(latest_secondary["chi"]) != len(material.terms):
                raise ValueError("state.chi must contain one memory field per Debye term")
        if material.terms:
            alpha = material.alpha(dt_value)
            beta = material.beta(dt_value)
            c_expr = (sigma_infinity - sigma_b_const) * Ep_new_function
            for index, (term, memory, alpha_i, beta_i) in enumerate(
                zip(material.terms, latest_secondary["chi"], alpha, beta)
            ):
                delta = delta_expression(term, index)
                c_expr = c_expr - delta * float(beta_i) * Ep_new_function
                c_expr = c_expr - delta * float(alpha_i) * memory
        else:
            c_expr = (sigma - sigma_b_const) * Ep_new_function
        c_new = interpolate_vector_expression("primary_secondary_c_new", c_expr)
        rhs_function = fem.Function(V, name="primary_secondary_rhs_density")
        rhs_function.x.array[:] = (latest_secondary["deltaJ"].x.array - c_new.x.array) / dt_value
        rhs_function.x.scatter_forward()
        step_sigma = material.sigma_eff(dt_value) if material.terms else material.sigma_inf
        Es_new = secondary_step_solver(rhs_function, step_sigma, dt_value)
        Etotal_new = Ep_new_function + latest_secondary["E"]
        if material.terms:
            alpha = material.alpha(dt_value)
            beta = material.beta(dt_value)
            chi_new = []
            for index, (old_memory, alpha_i, beta_i) in enumerate(zip(latest_secondary["chi"], alpha, beta)):
                chi_new.append(
                    interpolate_vector_expression(
                        f"primary_secondary_chi_{index}",
                        float(alpha_i) * old_memory + float(beta_i) * Etotal_new,
                    )
                )
            latest_secondary["chi"] = chi_new
            deltaJ_expr = sigma_infinity * Etotal_new - sigma_b_const * Ep_new_function
            for index, (term, memory) in enumerate(zip(material.terms, latest_secondary["chi"])):
                deltaJ_expr = deltaJ_expr - delta_expression(term, index) * memory
        else:
            deltaJ_expr = sigma * Etotal_new - sigma_b_const * Ep_new_function
            latest_secondary["chi"] = []
        deltaJ_new = interpolate_vector_expression("primary_secondary_deltaJ", deltaJ_expr)
        latest_secondary["deltaJ"] = deltaJ_new
        return SecondaryState(
            Es=Es_new,
            deltaJ=sample_solution(deltaJ_new, Ep_new),
            chi=[],
        )

    return {
        "secondary_state_initializer": build_initialization,
        "secondary_field_solver": secondary_field_solver,
        "secondary_step_solver": secondary_step_solver,
        "secondary_receiver_projector": _make_secondary_receiver_projector_from_evaluate_receivers(
            electric_getter,
            dbdt_getter,
            msh=msh,
            config=config,
        ),
        "secondary_state_stepper": secondary_state_stepper,
        "diagnostics": latest_secondary,
    }


def _make_dolfinx_primary_secondary_forward_operator(
    msh,
    spaces,
    materials,
    operators,
    config: PipelineConfig,
    *,
    primary,
    receiver_locations,
    components,
    material,
    sigma_background: float,
    debye=None,
    turnoff_time: float = 0.0,
    turnoff_steps: int = 1,
):
    """Build a DOLFINx-wired primary-secondary forward operator.

    This helper is the production bridge from a background primary provider to
    the pure `PrimarySecondaryForwardOperator`: it samples the provider on the
    actual local Nedelec physical interpolation points exported from the mesh,
    then wires the DOLFINx secondary initializer, stepper, and receiver
    projector.
    """

    from atem3d.solvers import PrimarySecondaryForwardOperator

    interpolation = _nedelec_interpolation_points(msh, spaces)
    fem_points = interpolation["points"]
    adapters = _make_dolfinx_primary_secondary_forward_adapters(
        msh,
        spaces,
        materials,
        operators,
        config,
        fem_points,
        sigma_background=sigma_background,
        debye=debye,
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=components,
        material=material,
        sigma_background=sigma_background,
        secondary_state_initializer=adapters["secondary_state_initializer"],
        secondary_step_solver=adapters["secondary_step_solver"],
        secondary_receiver_projector=adapters["secondary_receiver_projector"],
        secondary_state_stepper=adapters["secondary_state_stepper"],
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
        diagnostics=adapters["diagnostics"],
    )
    return {
        "operator": operator,
        "fem_points": fem_points,
        "interpolation": interpolation,
        "adapters": adapters,
        "diagnostics": adapters["diagnostics"],
    }


def _update_debye_memories(debye, memories, E_new, dt: float) -> None:
    if debye is None:
        return
    for term, chi in zip(debye["terms"], memories):
        alpha, beta = _debye_backward_euler_coefficients(term, dt)
        chi.x.array[:] = beta * E_new.x.array + alpha * chi.x.array
        chi.x.scatter_forward()


def run_fetd_forward(msh, cell_tags, facet_tags, spaces, materials, source, config: PipelineConfig, debye=None, times=None):
    """Run backward-Euler FETD forward modelling."""

    import numpy as np
    from dolfinx import fem
    from petsc4py import PETSc

    if times is None:
        times = generate_time_array(config)
    schedule = _forward_observation_schedule(times, config)
    step_times = schedule["step_times"]
    return_times = schedule["return_times"]
    output_step_indices = set(schedule["output_step_indices"])
    observation_time_by_step = {
        int(step): float(return_times[i]) for i, step in enumerate(schedule["output_step_indices"])
    }
    operators = assemble_operators(msh, spaces, materials, facet_tags, config, debye=debye)
    divergence_cleaner = None
    if str(config.divergence_cleaning).strip().lower() == "conductivity":
        if debye is not None:
            raise ValueError("conductivity divergence cleaning is currently implemented for non-polarizable runs only")
        divergence_cleaner = _build_conductivity_divergence_cleaner(spaces, operators, config)
    divergence_control = None
    divergence_control_norms = None
    if float(config.divergence_control_weight) > 0.0:
        if debye is not None:
            raise ValueError("conductivity divergence control is currently implemented for non-polarizable runs only")
        divergence_control = _build_conductivity_divergence_control_matrix(spaces, operators, config)
        divergence_control_norms = {
            "mass": _petsc_matrix_norm(operators["M"]),
            "stiffness": _petsc_matrix_norm(operators["K"]),
            "control": _petsc_matrix_norm(divergence_control),
        }
    V = spaces["V"]
    E_initial = _solve_initial_dc_field(msh, spaces, materials, facet_tags, config)
    source["initial_field_diagnostics"] = _initial_field_curl_diagnostics(E_initial, spaces)
    source_term_mode = str(config.source_term_mode).strip().lower()
    if source_term_mode == "primary_dc":
        E_old = fem.Function(V, name="E_secondary_old")
        E_old.x.array[:] = 0.0
        E_old.x.scatter_forward()
        source_time_vector = E_initial.x.petsc_vec.duplicate()
        operators["M"].mult(E_initial.x.petsc_vec, source_time_vector)
        log("[source] source_term_mode=primary_dc; solving secondary E driven by M_sigma E_dc.", comm=msh.comm)
    else:
        E_old = E_initial
        E_old.name = "E_old"
        source_time_vector = source["vector"]
    E_new = fem.Function(V, name="E_new")
    memories = [fem.Function(V, name=f"chi_{i}") for i, _ in enumerate((debye or {}).get("terms", []))]
    if memories:
        _initialise_debye_memories_to_field(memories, E_old)
    magnetic_receiver_mode = str(config.magnetic_receiver_mode).strip().lower()
    if magnetic_receiver_mode not in {"curl", "biot_current", "biot_ohmic", "faraday_integrated"}:
        raise ValueError(
            "magnetic_receiver_mode must be 'curl', 'biot_current', 'biot_ohmic', or 'faraday_integrated'"
        )
    magnetic_dbdt_mode = str(config.magnetic_dbdt_mode).strip().lower()
    if magnetic_dbdt_mode not in {"curl", "biot_rate"}:
        raise ValueError("magnetic_dbdt_mode must be 'curl' or 'biot_rate'")
    if magnetic_dbdt_mode == "biot_rate" and magnetic_receiver_mode not in {"biot_current", "biot_ohmic"}:
        raise ValueError("magnetic_dbdt_mode='biot_rate' requires a Biot magnetic_receiver_mode")
    H_old_receiver = None
    faraday_receiver_hz = None
    if magnetic_receiver_mode == "biot_current":
        H_old_receiver = _biot_savart_total_h_at_receiver(
            E_old,
            msh,
            materials,
            config,
            _source_current(0.0, config),
            debye=debye,
            memories=memories,
        )
    elif magnetic_receiver_mode == "biot_ohmic":
        H_old_receiver = _biot_savart_cell_current_h_at_receiver(E_old, msh, materials, config)
    elif magnetic_receiver_mode == "faraday_integrated":
        faraday_initial_h = _biot_savart_total_h_at_receiver(
            E_old,
            msh,
            materials,
            config,
            _source_current(0.0, config),
            debye=debye,
            memories=memories,
        )
        faraday_receiver_hz = float(faraday_initial_h[2])

    rows = []
    receiver_diagnostic_rows = []
    solver_log = []
    components = _forward_components(config)
    previous_time = 0.0
    start_step = 0
    if bool(config.resume_forward):
        checkpoint = _load_forward_checkpoint(config)
        if checkpoint["e_old"].shape != E_old.x.array.shape:
            raise ValueError(
                "forward checkpoint E state shape does not match current function space: "
                f"checkpoint={checkpoint['e_old'].shape}, current={E_old.x.array.shape}"
            )
        E_old.x.array[:] = checkpoint["e_old"]
        E_old.x.scatter_forward()
        if checkpoint["memories"].shape[0] != len(memories):
            raise ValueError(
                "forward checkpoint Debye memory count does not match current model: "
                f"checkpoint={checkpoint['memories'].shape[0]}, current={len(memories)}"
            )
        for memory, values in zip(memories, checkpoint["memories"]):
            if values.shape != memory.x.array.shape:
                raise ValueError("forward checkpoint Debye memory shape does not match current function space")
            memory.x.array[:] = values
            memory.x.scatter_forward()
        rows = checkpoint["rows"].tolist()
        receiver_diagnostic_rows = list(checkpoint["receiver_diagnostic_rows"])
        solver_log = list(checkpoint["solver_log"])
        previous_time = float(checkpoint["previous_time"])
        start_step = int(checkpoint["completed_step"]) + 1
        if magnetic_receiver_mode in {"biot_current", "biot_ohmic"}:
            h_old = checkpoint["h_old_receiver"]
            if h_old.shape == (3,) and np.all(np.isfinite(h_old)):
                H_old_receiver = h_old
            else:
                raise ValueError("forward checkpoint is missing a finite magnetic receiver state for Biot mode")
        elif magnetic_receiver_mode == "faraday_integrated" and rows:
            component_names = _forward_components(config)
            if "Hz" in component_names:
                faraday_receiver_hz = float(rows[-1][component_names.index("Hz")])
        log(
            f"[resume] loaded forward checkpoint {config.forward_checkpoint_npz()} at completed_step="
            f"{checkpoint['completed_step']} previous_time={previous_time:.6e} rows={len(rows)}",
            comm=msh.comm,
        )
    initial_rows_count = len(rows)
    stop_after_outputs = int(config.stop_after_outputs)
    solver_context = None
    time_theta = float(config.time_theta)
    time_method = str(config.time_method).strip().lower()
    E_older = None
    previous_step_dt = None
    for step, t in enumerate(step_times):
        if step < start_step:
            continue
        dt = float(t - previous_time)
        if dt <= 0.0:
            raise RuntimeError("non-positive dt encountered")
        use_bdf2 = (
            time_method == "bdf2"
            and debye is None
            and E_older is not None
            and previous_step_dt is not None
        )
        owns_M_eff = debye is not None and bool(debye["terms"])
        M_eff = operators["M"] if use_bdf2 else _matrix_for_effective_conductivity(operators, debye, dt)
        if use_bdf2:
            bdf2_coeffs = _bdf2_step_coefficients(dt, previous_step_dt)
            lhs_mass = bdf2_coeffs["lhs"]
            lhs_stiffness = 1.0
        else:
            coeffs = _theta_step_coefficients(dt, 1.0 if owns_M_eff else time_theta)
            lhs_mass = coeffs["mass_lhs"]
            lhs_stiffness = coeffs["stiffness_lhs"]
        A = _copy_and_combine_matrix(
            operators["K"],
            M_eff,
            lhs_mass,
            k_scale=lhs_stiffness,
        )
        if str(config.outer_boundary_mode).strip().lower() == "robin":
            B_robin = operators.get("B_robin")
            if B_robin is None:
                raise RuntimeError("Robin outer boundary mode requires B_robin operator")
            from petsc4py import PETSc

            A.axpy(
                _outer_boundary_robin_admittance(config, dt),
                B_robin,
                structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
            )
            A.assemble()
        divergence_control_applied = divergence_control is not None and _should_apply_divergence_control(float(t), config)
        divergence_control_stats = None
        if divergence_control_applied:
            from petsc4py import PETSc

            divergence_control_stats = _divergence_control_step_stats(
                config,
                dt=dt,
                lhs_mass=lhs_mass,
                lhs_stiffness=lhs_stiffness,
                mass_norm=float(divergence_control_norms["mass"]),
                stiffness_norm=float(divergence_control_norms["stiffness"]),
                control_norm=float(divergence_control_norms["control"]),
            )
            A.axpy(
                float(divergence_control_stats["divergence_control_applied_weight"]),
                divergence_control,
                structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
            )
            A.assemble()
        _zero_rows_columns(A, operators["bc_global"], diag=1.0)
        if solver_context is None:
            solver_context = configure_ams_solver(A, spaces, config)
        else:
            solver_context["ksp"].setOperators(A)

        if use_bdf2:
            b = E_old.x.petsc_vec.duplicate()
            operators["M"].mult(E_old.x.petsc_vec, b)
            b.scale(bdf2_coeffs["old"])
            older_term = E_old.x.petsc_vec.duplicate()
            operators["M"].mult(E_older.x.petsc_vec, older_term)
            b.axpy(bdf2_coeffs["older"], older_term)
            older_term.destroy()
        else:
            b = _assemble_history_rhs(operators, debye, memories, E_old, dt)
        if not use_bdf2 and not owns_M_eff and coeffs["stiffness_rhs"] != 0.0:
            k_old = b.duplicate()
            operators["K"].mult(E_old.x.petsc_vec, k_old)
            b.axpy(coeffs["stiffness_rhs"], k_old)
            k_old.destroy()
        avg_didt = _source_interval_average_didt(previous_time, float(t), config)
        b.axpy(float(config.source_rhs_sign) * avg_didt, source_time_vector)
        _zero_rhs_entries(b, operators["bc_global"])

        E_new.x.array[:] = 0.0
        solver_context["ksp"].solve(b, E_new.x.petsc_vec)
        E_new.x.scatter_forward()
        its = solver_context["ksp"].getIterationNumber()
        reason = solver_context["ksp"].getConvergedReason()
        residual = solver_context["ksp"].getResidualNorm()
        if reason < 0:
            raise RuntimeError(f"KSP failed at step={step}, t={t:.6e}, reason={reason}, residual={residual:.6e}")

        clean_stats = None
        if divergence_cleaner is not None and _should_apply_divergence_cleaning(float(t), config):
            clean_stats = _apply_conductivity_divergence_cleaning(divergence_cleaner, E_new, operators, config)
        _update_debye_memories(debye, memories, E_new, dt)
        if magnetic_receiver_mode == "biot_current":
            H_new_receiver = _biot_savart_total_h_at_receiver(
                E_new,
                msh,
                materials,
                config,
                _source_current(float(t), config),
                debye=debye,
                memories=memories,
            )
        elif magnetic_receiver_mode == "biot_ohmic":
            H_new_receiver = _biot_savart_cell_current_h_at_receiver(E_new, msh, materials, config)
        else:
            H_new_receiver = None

        faraday_step_dbdt = None
        faraday_step_record = None
        if magnetic_receiver_mode == "faraday_integrated":
            faraday_step_dbdt = compute_dbdt(E_new, spaces)
            faraday_step_record = evaluate_receivers(E_new, faraday_step_dbdt, msh, config)
            faraday_step_record["dBzdt_curl"] = float(faraday_step_record.get("dBzdt", np.nan))
            faraday_step_record["dBzdt_biot_rate"] = float("nan")
            faraday_receiver_hz = _advance_faraday_receiver_hz(
                previous_hz=float(faraday_receiver_hz),
                dbzdt_new=float(faraday_step_record["dBzdt"]),
                dt=dt,
            )
            faraday_step_record["Hz"] = float(faraday_receiver_hz)

        is_output = step in output_step_indices
        if is_output:
            if magnetic_receiver_mode == "faraday_integrated":
                dbdt = faraday_step_dbdt
                rec = dict(faraday_step_record)
            else:
                dbdt = compute_dbdt(E_new, spaces)
                rec = evaluate_receivers(E_new, dbdt, msh, config)
            rec["dBzdt_curl"] = float(rec.get("dBzdt", np.nan))
            rec["dBzdt_biot_rate"] = float("nan")
            if magnetic_receiver_mode in {"biot_current", "biot_ohmic"}:
                _assign_biot_receiver_hz(rec, H_new_receiver)
                biot_rate = float(_biot_receiver_dbdt_from_h(H_new_receiver, H_old_receiver, dt=dt)[2])
                rec["dBzdt_biot_rate"] = biot_rate
                if magnetic_dbdt_mode == "biot_rate":
                    rec["dBzdt"] = biot_rate
            rows.append([rec[name] for name in components])
            receiver_diagnostic_rows.extend(
                _evaluate_receiver_diagnostics(
                    E_new,
                    dbdt,
                    msh,
                    config,
                    time_obs=observation_time_by_step[step],
                    main_record=rec,
                )
            )

        if magnetic_receiver_mode in {"biot_current", "biot_ohmic"}:
            H_old_receiver = H_new_receiver
        log_item = {
            "step": step,
            "time": float(t),
            "dt": dt,
            "its": int(its),
            "residual": float(residual),
            "reason": int(reason),
            "is_output": bool(is_output),
            "time_theta": float(lhs_stiffness),
            "time_method": "bdf2" if use_bdf2 else "theta",
            "avg_didt": float(avg_didt),
            "divergence_control_applied": bool(divergence_control_applied),
            "ip_total_field_equation": _debye_total_field_step_metadata(debye, dt),
        }
        if divergence_control_stats is not None:
            log_item.update(divergence_control_stats)
        if is_output:
            log_item["observation_time"] = float(observation_time_by_step[step])
        if clean_stats is not None:
            log_item["divergence_clean_before"] = float(clean_stats["before"])
            log_item["divergence_clean_after"] = float(clean_stats["after"])
            log_item["divergence_clean_correction_norm"] = float(clean_stats["correction_norm"])
            log_item["divergence_clean_applied_correction_norm"] = float(clean_stats["applied_correction_norm"])
            log_item["divergence_clean_strength"] = float(clean_stats["strength"])
        solver_log.append(log_item)
        if is_output and msh.comm.rank == 0:
            _save_forward_partial(
                config,
                return_times,
                rows,
                components,
                solver_log,
                receiver_diagnostic_rows=receiver_diagnostic_rows,
            )
        if is_output:
            hz_text = f" Hz={rec['Hz']:.6e}" if "Hz" in rec else ""
            log(
                f"[time] step={step:04d} t_internal={t:.6e} t_obs={observation_time_by_step[step]:.6e} "
                f"dt={dt:.6e} KSP its={its} residual={residual:.3e} reason={reason} "
                f"Ex={rec['Ex']:.6e} Ey={rec['Ey']:.6e}{hz_text} dBzdt={rec['dBzdt']:.6e}",
                comm=msh.comm,
            )
            if clean_stats is not None:
                log(
                    f"[div-clean] step={step:04d} before={clean_stats['before']:.6e} "
                    f"after={clean_stats['after']:.6e} strength={clean_stats['strength']:.6g} "
                    f"correction={clean_stats['correction_norm']:.6e} "
                    f"applied={clean_stats['applied_correction_norm']:.6e} "
                    f"its={clean_stats['its']} residual={clean_stats['residual']:.3e}",
                    comm=msh.comm,
                )
        else:
            log(
                f"[time] step={step:04d} t_internal={t:.6e} dt={dt:.6e} "
                f"KSP its={its} residual={residual:.3e} reason={reason}",
                comm=msh.comm,
            )

        if time_method == "bdf2":
            if E_older is None:
                E_older = fem.Function(V, name="E_older")
            E_older.x.array[:] = E_old.x.array
            E_older.x.scatter_forward()
            previous_step_dt = dt
        E_old.x.array[:] = E_new.x.array
        E_old.x.scatter_forward()
        previous_time = float(t)
        if is_output and (bool(config.checkpoint_forward) or stop_after_outputs > 0):
            _save_forward_checkpoint(
                config,
                completed_step=step,
                previous_time=previous_time,
                E_old=E_old,
                memories=memories,
                rows=rows,
                components=components,
                solver_log=solver_log,
                h_old_receiver=H_old_receiver,
                receiver_diagnostic_rows=receiver_diagnostic_rows,
            )
        should_stop = is_output and stop_after_outputs > 0 and (len(rows) - initial_rows_count) >= stop_after_outputs
        b.destroy()
        A.destroy()
        if owns_M_eff:
            M_eff.destroy()
        if should_stop:
            log(
                f"[checkpoint] stop_after_outputs={stop_after_outputs} reached at completed_step={step}; "
                f"saved {config.forward_checkpoint_npz()} with rows={len(rows)}",
                comm=msh.comm,
            )
            break

    completed_times = _completed_return_times(return_times, rows)
    return {
        "times": completed_times,
        "data": np.asarray(rows),
        "components": components,
        "solver_log": solver_log,
        "receiver_diagnostic_rows": receiver_diagnostic_rows,
    }


def assemble_h_operators(msh, spaces: dict[str, Any], materials: dict[str, Any], facet_tags):
    """Assemble H-form curl-rho-curl and mu mass matrices."""

    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    V = spaces["V"]
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    _bc, bc_dofs, bc_global = _make_zero_tangential_bc(msh, spaces, facet_tags)
    K_form = fem.form(materials["rho"] * ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
    M_form = fem.form(materials["mu"] * ufl.inner(u, v) * ufl.dx)
    K = fem_petsc.assemble_matrix(K_form)
    K.assemble()
    M = fem_petsc.assemble_matrix(M_form)
    M.assemble()
    log("[operators:h] Assembled K_rho and M_mu matrices.", comm=msh.comm)
    return {"K": K, "M": M, "bc_dofs": bc_dofs, "bc_global": bc_global}


def assemble_h_source_vector(msh, spaces: dict[str, Any], materials: dict[str, Any], current_density):
    """Assemble int rho J_s . curl(v) dx for unit source current."""

    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    V = spaces["V"]
    v = ufl.TestFunction(V)
    form = fem.form(materials["rho"] * ufl.inner(current_density, ufl.curl(v)) * ufl.dx)
    vec = fem_petsc.assemble_vector(form)
    vec.assemble()
    log(f"[source:h] assembled curl-rho source vector norm={vec.norm():.6e}", comm=msh.comm)
    return vec


def _recover_h_electric_field(H, current_density, source_current: float, spaces: dict[str, Any], materials: dict[str, Any]):
    """Recover E = rho * (curl(H) - J_s) as a DG0 vector field."""

    import ufl
    from dolfinx import fem

    W = spaces["W"]
    E = fem.Function(W, name="E_from_H")
    expr = fem.Expression(
        materials["rho"] * (ufl.curl(H) - float(source_current) * current_density),
        W.element.interpolation_points(),
        comm=H.function_space.mesh.comm,
    )
    E.interpolate(expr)
    E.x.scatter_forward()
    return E


def _evaluate_h_receivers(E, H_new, H_old, dt: float, msh, config: PipelineConfig):
    """Evaluate Ex/Ey from recovered E plus Hz and dBz/dt from H."""

    import numpy as np
    from scipy.constants import mu_0

    cell = _find_cell_for_point(msh, config.receiver)
    if cell is None:
        raise RuntimeError(f"receiver {config.receiver} was not found in a local cell")
    point = np.asarray(config.receiver, dtype=float).reshape(1, 3)
    cells = np.asarray([cell], dtype=np.int32)
    e_val = np.asarray(E.eval(point, cells), dtype=float).reshape(-1)
    h_new = np.asarray(H_new.eval(point, cells), dtype=float).reshape(-1)
    h_old = np.asarray(H_old.eval(point, cells), dtype=float).reshape(-1)
    dbdt = mu_0 * (h_new - h_old) / float(dt)
    return {"Ex": float(e_val[0]), "Ey": float(e_val[1]), "Hz": float(h_new[2]), "dBzdt": float(dbdt[2])}


def run_h_forward(msh, cell_tags, facet_tags, spaces, materials, source, config: PipelineConfig, times=None):
    """Run a non-polarizable H-form backward-Euler forward model."""

    import numpy as np
    from dolfinx import fem

    if times is None:
        times = generate_time_array(config)
    schedule = _forward_observation_schedule(times, config)
    step_times = schedule["step_times"]
    return_times = schedule["return_times"]
    output_step_indices = set(schedule["output_step_indices"])
    observation_time_by_step = {
        int(step): float(return_times[i]) for i, step in enumerate(schedule["output_step_indices"])
    }
    if config.polarization != "none":
        raise NotImplementedError("H-formulation is currently implemented for non-polarizable validation only")

    operators = assemble_h_operators(msh, spaces, materials, facet_tags)
    current_density = source.get("current_density")
    if current_density is None:
        current_density = _build_regularized_current_density(msh, spaces, config, cell_tags)
    source_vec = assemble_h_source_vector(msh, spaces, materials, current_density)
    _zero_rhs_entries(source_vec, operators["bc_global"])

    V = spaces["V"]
    H_old = fem.Function(V, name="H_old")
    H_new = fem.Function(V, name="H_new")
    static_dt = max(float(config.t_max), float(config.ramp_off_time), 1.0) * 1.0e9
    A0 = _copy_and_combine_matrix(operators["K"], operators["M"], 1.0 / static_dt)
    _zero_rows_columns(A0, operators["bc_global"], diag=1.0)
    solver_context = configure_lu_solver(A0)
    b0 = source_vec.copy()
    b0.scale(float(config.source_current))
    _zero_rhs_entries(b0, operators["bc_global"])
    rhs0_norm = float(b0.norm())
    solver_context["ksp"].solve(b0, H_old.x.petsc_vec)
    H_old.x.scatter_forward()
    initial_stats = _validate_h_solver_state(
        "initial:h",
        solver_context["ksp"],
        H_old.x.array,
        rhs_norm=rhs0_norm,
        config=config,
    )
    log(
        f"[initial:h] solved static H with KSP its={initial_stats['its']} "
        f"residual={initial_stats['residual']:.3e} "
        f"relative_residual={initial_stats['relative_residual']:.3e} "
        f"reason={initial_stats['reason']}",
        comm=msh.comm,
    )
    b0.destroy()
    A0.destroy()

    rows = []
    solver_log = []
    components = _forward_components(config)
    previous_time = 0.0
    initial_rows_count = len(rows)
    stop_after_outputs = int(config.stop_after_outputs)
    for step, t in enumerate(step_times):
        dt = float(t - previous_time)
        if dt <= 0.0:
            raise RuntimeError("non-positive dt encountered")
        A = _copy_and_combine_matrix(operators["K"], operators["M"], 1.0 / dt)
        _zero_rows_columns(A, operators["bc_global"], diag=1.0)
        solver_context["ksp"].setOperators(A)
        b = H_old.x.petsc_vec.duplicate()
        operators["M"].mult(H_old.x.petsc_vec, b)
        b.scale(1.0 / dt)
        b.axpy(_source_current(float(t), config), source_vec)
        _zero_rhs_entries(b, operators["bc_global"])
        rhs_norm = float(b.norm())

        H_new.x.array[:] = 0.0
        solver_context["ksp"].solve(b, H_new.x.petsc_vec)
        H_new.x.scatter_forward()
        solver_stats = _validate_h_solver_state(
            f"time:h step={step} t={t:.6e}",
            solver_context["ksp"],
            H_new.x.array,
            rhs_norm=rhs_norm,
            config=config,
        )
        its = int(solver_stats["its"])
        reason = int(solver_stats["reason"])
        residual = float(solver_stats["residual"])

        is_output = step in output_step_indices
        if is_output:
            E = _recover_h_electric_field(H_new, current_density, _source_current(float(t), config), spaces, materials)
            rec = _evaluate_h_receivers(E, H_new, H_old, dt, msh, config)
            rows.append([rec[name] for name in components])
        log_item = {
            "step": step,
            "time": float(t),
            "dt": dt,
            "its": its,
            "residual": residual,
            "relative_residual": float(solver_stats["relative_residual"]),
            "reason": reason,
            "is_output": bool(is_output),
        }
        if is_output:
            log_item["observation_time"] = float(observation_time_by_step[step])
        solver_log.append(log_item)
        if is_output and msh.comm.rank == 0:
            _save_forward_partial(config, return_times, rows, components, solver_log)
        if is_output:
            log(
                f"[time:h] step={step:04d} t_internal={t:.6e} t_obs={observation_time_by_step[step]:.6e} "
                f"dt={dt:.6e} KSP its={its} residual={residual:.3e} "
                f"relative_residual={solver_stats['relative_residual']:.3e} reason={reason} "
                f"Ex={rec['Ex']:.6e} Ey={rec['Ey']:.6e} Hz={rec['Hz']:.6e} dBzdt={rec['dBzdt']:.6e}",
                comm=msh.comm,
            )
        else:
            log(
                f"[time:h] step={step:04d} t_internal={t:.6e} dt={dt:.6e} "
                f"KSP its={its} residual={residual:.3e} "
                f"relative_residual={solver_stats['relative_residual']:.3e} reason={reason}",
                comm=msh.comm,
            )

        H_old.x.array[:] = H_new.x.array
        H_old.x.scatter_forward()
        previous_time = float(t)
        should_stop = is_output and stop_after_outputs > 0 and (len(rows) - initial_rows_count) >= stop_after_outputs
        b.destroy()
        A.destroy()
        if should_stop:
            log(
                f"[time:h] stop_after_outputs={stop_after_outputs} reached at completed_step={step}; "
                f"rows={len(rows)}",
                comm=msh.comm,
            )
            break

    completed_times = _completed_return_times(return_times, rows)
    return {"times": completed_times, "data": np.asarray(rows), "components": components, "solver_log": solver_log}


def _empymod_rec_mapping(receiver, component: str):
    x, y, z = receiver
    depth = -z
    if component == "Ex":
        return [x, y, depth, 0.0, 0.0], False, -1, 1.0
    if component == "Ey":
        return [x, y, depth, 90.0, 0.0], False, -1, 1.0
    if component == "Hz":
        return [x, y, depth, 0.0, 90.0], True, -1, 1.0
    if component == "dBzdt":
        from scipy.constants import mu_0

        return [x, y, depth, 0.0, 90.0], True, 0, -mu_0
    raise ValueError(f"unsupported empymod component {component}")


def _empymod_call_kwargs(config: PipelineConfig, *, srcpts: int | None = None) -> dict[str, Any]:
    """Return optional empymod transform/source-integration keyword arguments."""

    kwargs: dict[str, Any] = {"srcpts": int(config.empymod_srcpts if srcpts is None else srcpts)}
    ht = str(config.empymod_ht).strip().lower()
    ft = str(config.empymod_ft).strip().lower()
    if ht != "dlf":
        kwargs["ht"] = ht
        if ht == "qwe":
            kwargs["htarg"] = {"rtol": 1.0e-12, "atol": 1.0e-30, "nquad": 51, "maxint": 80, "pts_per_dec": 0}
    if ft != "dlf":
        kwargs["ft"] = ft
        if ft == "qwe":
            kwargs["ftarg"] = {
                "rtol": 1.0e-10,
                "atol": 1.0e-24,
                "nquad": 31,
                "maxint": 300,
                "pts_per_dec": 30,
            }
    return kwargs


def _empymod_polarizable_layer_indices(depth: list[float], res: list[float], config: PipelineConfig) -> list[int]:
    top = float(config.cole_layer_top)
    bottom = float(config.cole_layer_bottom)
    if math.isinf(bottom):
        bottom = math.inf
    indices: list[int] = []
    for idx in range(1, len(res)):
        layer_top = float(depth[idx - 1])
        layer_bottom = float(depth[idx]) if idx < len(depth) else math.inf
        if layer_bottom > top and layer_top < bottom:
            indices.append(idx)
    return indices


def _apply_linear_ramp_average(times, values, ramp_time: float):
    """Approximate a linear ramp-off response by averaging step-off values."""

    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.integrate import cumulative_trapezoid

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if ramp_time <= 0.0 or times.size < 2:
        return values
    dense_min = max(times.min() * 0.25, 1.0e-9)
    dense_max = times.max()
    dense = np.unique(np.r_[np.logspace(math.log10(dense_min), math.log10(dense_max), max(800, 4 * times.size)), times])
    interp = PchipInterpolator(times, values, extrapolate=True)
    vals = interp(dense)
    cumulative = np.r_[0.0, cumulative_trapezoid(vals, dense)]
    cint = PchipInterpolator(dense, cumulative, extrapolate=True)
    out = np.empty_like(times)
    for i, t in enumerate(times):
        a = max(dense_min, t - ramp_time)
        b = max(dense_min, t)
        if b <= a:
            out[i] = values[i]
        else:
            out[i] = float((cint(b) - cint(a)) / (b - a))
    return out


def _ramp_average_from_dense(times, dense_times, dense_values, ramp_time: float, window: str = "ramp_start"):
    """Average dense step-off samples over a finite linear ramp interval."""

    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.integrate import cumulative_trapezoid

    times = np.asarray(times, dtype=float)
    dense_times = np.asarray(dense_times, dtype=float)
    dense_values = np.asarray(dense_values, dtype=float)
    window = str(window).strip().lower()
    if window not in {"ramp_start", "after_ramp"}:
        raise ValueError("window must be 'ramp_start' or 'after_ramp'")
    if ramp_time <= 0.0 or times.size == 0:
        return np.interp(times, dense_times, dense_values)
    order = np.argsort(dense_times)
    dense_times = dense_times[order]
    dense_values = dense_values[order]
    unique_times, unique_idx = np.unique(dense_times, return_index=True)
    dense_times = unique_times
    dense_values = dense_values[unique_idx]
    cumulative = np.r_[0.0, cumulative_trapezoid(dense_values, dense_times)]
    cint = PchipInterpolator(dense_times, cumulative, extrapolate=True)
    out = np.empty_like(times)
    dense_min = float(dense_times[0])
    for i, t in enumerate(times):
        if window == "after_ramp":
            a = max(dense_min, float(t))
            b = max(dense_min, float(t) + float(ramp_time))
        else:
            a = max(dense_min, float(t) - float(ramp_time))
            b = max(dense_min, float(t))
        if b <= a:
            out[i] = float(np.interp(float(t), dense_times, dense_values))
        else:
            out[i] = float((cint(b) - cint(a)) / (b - a))
    return out


def _reference_ramp_dense_times(times, ramp_time: float, window: str = "ramp_start"):
    """Return dense positive times needed to average early ramp-off references."""

    import numpy as np

    times = np.asarray(times, dtype=float)
    window = str(window).strip().lower()
    if window not in {"ramp_start", "after_ramp"}:
        raise ValueError("window must be 'ramp_start' or 'after_ramp'")
    if ramp_time <= 0.0 or times.size == 0:
        return times
    dense_min = min(max(float(times.min()) * 1.0e-3, 1.0e-9), float(times.min()))
    dense_max = float(times.max()) + (float(ramp_time) if window == "after_ramp" else 0.0)
    n_dense = max(1600, 8 * int(times.size))
    dense = np.logspace(math.log10(dense_min), math.log10(dense_max), n_dense)
    if window == "after_ramp":
        return np.unique(np.r_[dense, times, times + float(ramp_time)])
    return np.unique(np.r_[dense, times])


def _apply_reference_ramp_average(component: str, times, values, ramp_time: float):
    """Apply finite ramp-off averaging to empymod reference components."""

    if component in {"Ex", "Ey", "Hz", "dBzdt"}:
        return _apply_linear_ramp_average(times, values, ramp_time)
    return values


def get_empymod_reference(t_array, config: PipelineConfig, mode: str = "noip", *, srcpts: int | None = None):
    """Compute matching 1D air-earth empymod reference data."""

    import numpy as np
    import empymod

    times = np.asarray(t_array, dtype=float)
    src = [
        config.source_start[0],
        config.source_end[0],
        config.source_start[1],
        config.source_end[1],
        -config.source_start[2],
        -config.source_end[2],
    ]
    if mode == "cole-cole":
        depth, base_res = _empymod_depth_res(config)
        fit = fit_cole_cole_to_debye(config)
        polarizable_indices = _empymod_polarizable_layer_indices(depth, base_res, config)
        if not polarizable_indices:
            raise RuntimeError(
                "No empymod earth layer overlaps the configured Cole-Cole layer window "
                f"{float(config.cole_layer_top):.6g}-{float(config.cole_layer_bottom):.6g} m"
            )
        sigma = 1.0 / np.asarray(base_res, dtype=float)
        for idx in polarizable_indices:
            sigma[idx] = fit.sigma_infinity

        def func_eta(_model, context):
            freq = np.asarray(context["freq"], dtype=float)
            eta = np.tile(np.asarray(sigma, dtype=float), (freq.size, 1)).astype(complex)
            for idx in polarizable_indices:
                for term in fit.terms:
                    eta[:, idx] -= term.delta_sigma / (1.0 + 1j * 2.0 * np.pi * freq * term.tau)
            return eta, eta

        res_base = list(base_res)
        for idx in polarizable_indices:
            res_base[idx] = 1.0 / fit.sigma_infinity
        res = {"res": res_base, "func_eta": func_eta}
    else:
        depth, res = _empymod_depth_res(config)

    cols = []
    components = _forward_components(config)
    ramp_window = "after_ramp" if str(config.time_origin).strip().lower() == "after_ramp" else "ramp_start"
    dense_times = _reference_ramp_dense_times(times, config.ramp_off_time, window=ramp_window)
    empymod_kwargs = _empymod_call_kwargs(config, srcpts=srcpts)
    for component in components:
        rec, mrec, signal, factor = _empymod_rec_mapping(config.receiver, component)
        eval_times = dense_times if component in {"Ex", "Ey", "Hz", "dBzdt"} and config.ramp_off_time > 0.0 else times
        response = empymod.bipole(
            src=src,
            rec=rec,
            depth=depth,
            res=res,
            freqtime=eval_times,
            signal=signal,
            strength=config.source_current,
            mrec=mrec,
            verb=1,
            **empymod_kwargs,
        )
        values = np.asarray(response, dtype=float).reshape(-1) * factor
        if eval_times is dense_times:
            values = _ramp_average_from_dense(times, dense_times, values, config.ramp_off_time, window=ramp_window)
        else:
            values = _apply_reference_ramp_average(component, times, values, config.ramp_off_time)
        cols.append(values)
    data = np.column_stack(cols)
    print(
        f"[empymod] reference complete; finite source, strength={config.source_current:g} A, "
        f"source z={config.source_start[2]} m, receiver z={config.receiver[2]} m, "
        f"srcpts={empymod_kwargs['srcpts']}, ht={config.empymod_ht}, ft={config.empymod_ft}",
        flush=True,
    )
    return {"times": times, "data": data, "components": components}


def compute_error(fem_data, ref_data, components, floor_factor: float = 1.0e-6):
    """Compute floor-denominator relative errors per component."""

    import numpy as np

    fem_arr = np.asarray(fem_data, dtype=float)
    ref_arr = np.asarray(ref_data, dtype=float)
    if fem_arr.shape != ref_arr.shape:
        raise ValueError("FEM and reference arrays must have the same shape")
    result: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(components):
        ref = ref_arr[:, i]
        diff = fem_arr[:, i] - ref
        ref_max = float(np.max(np.abs(ref)))
        floor = max(_validation_floor(name, ref), floor_factor * ref_max)
        denom = np.maximum(np.abs(ref), floor)
        rel = np.abs(diff) / denom
        max_idx = int(np.argmax(rel))
        result[name] = {
            "floor": float(floor),
            "mean": float(np.mean(rel)),
            "median": float(np.median(rel)),
            "rms": float(np.sqrt(np.mean(rel**2))),
            "max": float(rel[max_idx]),
            "max_index": max_idx,
            "relative": rel,
            "absolute": np.abs(diff),
            "reference_max": ref_max,
            "reference_too_small": bool(ref_max == 0.0 or np.sum(np.abs(ref) > floor) < max(3, ref.size // 20)),
        }
    return result


def _validation_floor(component: str, ref_values) -> float:
    import numpy as np

    ref_values = np.asarray(ref_values, dtype=float)
    max_abs_ref = float(np.max(np.abs(ref_values))) if ref_values.size else 0.0
    if str(component).startswith("E"):
        return max(1.0e-14, 1.0e-6 * max_abs_ref)
    if str(component).startswith("H"):
        return max(1.0e-16, 1.0e-6 * max_abs_ref)
    if str(component).startswith(("dB", "B")):
        return max(1.0e-18, 1.0e-6 * max_abs_ref)
    return max(1.0e-300, 1.0e-6 * max_abs_ref)


def _write_component_csv(path: Path, times, data, components) -> None:
    import numpy as np

    arr = np.asarray(data, dtype=float)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_obs", *components])
        for time, row in zip(times, arr):
            writer.writerow([f"{float(time):.16e}", *[f"{float(value):.16e}" for value in row]])


def _robust_error_rows(times, pred_data, ref_data, components, threshold: float):
    import numpy as np

    pred = np.asarray(pred_data, dtype=float)
    ref = np.asarray(ref_data, dtype=float)
    times = np.asarray(times, dtype=float)
    if pred.shape != ref.shape:
        raise ValueError("prediction and reference arrays must have matching shapes")
    if pred.ndim != 2 or pred.shape[0] != times.size:
        raise ValueError("validation arrays must have shape (n_times, n_components)")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "pass_all_components": True,
        "failed_times": [],
        "failed_components": [],
    }
    failed_components: set[str] = set()
    failed_times: set[float] = set()
    magnetic_quantity = None
    magnetic_components: list[str] = []
    for col, component in enumerate(components):
        component = str(component)
        ref_col = ref[:, col]
        pred_col = pred[:, col]
        floor = _validation_floor(component, ref_col)
        max_abs_ref = float(np.max(np.abs(ref_col))) if ref_col.size else 0.0
        robust_values = []
        peak_values = []
        for time, pred_value, ref_value in zip(times, pred_col, ref_col):
            abs_error = float(abs(pred_value - ref_value))
            ordinary = float(abs_error / abs(ref_value)) if ref_value != 0.0 else float("inf")
            robust = float(abs_error / max(abs(float(ref_value)), floor))
            peak = float(abs_error / max(max_abs_ref, floor))
            passed = bool(robust <= threshold and peak <= threshold)
            if not passed:
                failed_components.add(component)
                failed_times.add(float(time))
            robust_values.append(robust)
            peak_values.append(peak)
            rows.append(
                {
                    "time_obs": float(time),
                    "component": component,
                    "pred": float(pred_value),
                    "ref": float(ref_value),
                    "abs_error": abs_error,
                    "ordinary_relative_error": ordinary,
                    "relative_error_with_floor": robust,
                    "peak_normalized_error": peak,
                    "pass_5pct": passed,
                }
            )
        summary[f"max_error_{component}"] = float(np.max(robust_values))
        summary[f"rms_error_{component}"] = float(np.sqrt(np.mean(np.asarray(robust_values) ** 2)))
        summary[f"max_peak_normalized_error_{component}"] = float(np.max(peak_values))
        key = component if component in {"Ex", "Ey"} else "Hz_or_dBzdt"
        if component not in {"Ex", "Ey"}:
            magnetic_quantity = component
            magnetic_components.append(component)
        summary[f"max_error_{key}"] = float(np.max(robust_values))
        summary[f"rms_error_{key}"] = float(np.sqrt(np.mean(np.asarray(robust_values) ** 2)))
        summary[f"max_peak_normalized_error_{key}"] = float(np.max(peak_values))
        summary[f"floor_{component}"] = floor
    weak_gate = check_weak_component_error_window(
        times,
        pred,
        ref,
        components,
        error_min_time=0.0,
        tolerance=float(threshold),
        weak_reference_fraction=0.1,
    )
    weak_components = set(weak_gate["weak_components"])
    physical_failed_components = sorted(
        component for component in failed_components if component not in weak_components
    )
    if not weak_gate["passed"]:
        physical_failed_components.extend(
            component for component in weak_gate["weak_components"] if component not in physical_failed_components
        )

    summary["pass_all_components"] = len(failed_components) == 0
    summary["failed_components"] = sorted(failed_components)
    summary["failed_times"] = sorted(failed_times)
    summary["physical_pass_all_components"] = len(physical_failed_components) == 0
    summary["physical_failed_components"] = sorted(physical_failed_components)
    summary["weak_component_passed"] = bool(weak_gate["passed"])
    summary["weak_components"] = list(weak_gate["weak_components"])
    summary["weak_component_primary_scale"] = float(weak_gate["primary_scale"])
    summary["weak_component_scaled_abs_error_max"] = dict(weak_gate["maxima"])
    summary["weak_component_reference_max"] = dict(weak_gate["reference_maxima"])
    summary["magnetic_quantity"] = magnetic_quantity or ""
    summary["magnetic_components"] = magnetic_components
    return rows, summary


def _write_errors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "time_obs",
        "component",
        "pred",
        "ref",
        "abs_error",
        "ordinary_relative_error",
        "relative_error_with_floor",
        "peak_normalized_error",
        "pass_5pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_validation_plots(workdir: Path, times, pred_data, ref_data, rows, components) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    pred = np.asarray(pred_data, dtype=float)
    ref = np.asarray(ref_data, dtype=float)
    n_comp = len(components)
    fig, axes = plt.subplots(1, n_comp, figsize=(5.0 * n_comp, 4.0), squeeze=False)
    for col, component in enumerate(components):
        ax = axes[0, col]
        ax.loglog(times, np.maximum(np.abs(pred[:, col]), 1.0e-300), "o-", label="FEM")
        ax.loglog(times, np.maximum(np.abs(ref[:, col]), 1.0e-300), "-", label="reference")
        ax.set_title(component)
        ax.set_xlabel("time_obs (s)")
        ax.set_ylabel(f"|{component}|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(workdir / "comparison_3comp.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, n_comp, figsize=(5.0 * n_comp, 4.0), squeeze=False)
    for col, component in enumerate(components):
        ax = axes[0, col]
        component_rows = [row for row in rows if row["component"] == component]
        robust = np.asarray([row["relative_error_with_floor"] for row in component_rows], dtype=float)
        peak = np.asarray([row["peak_normalized_error"] for row in component_rows], dtype=float)
        ax.semilogx(times, robust, "o-", label="relative_error_with_floor")
        ax.semilogx(times, peak, "s--", label="peak_normalized_error")
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1.2, label="5%")
        ax.set_title(component)
        ax.set_xlabel("time_obs (s)")
        ax.set_ylabel("error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(workdir / "error_curves_3comp.png", dpi=180)
    plt.close(fig)


def _automatic_failure_diagnostics(summary: dict[str, Any], *, magnetic_receiver_mode: str) -> dict[str, Any]:
    strict_failed = not bool(summary.get("pass_all_components", False))
    failed = not bool(summary.get("physical_pass_all_components", summary.get("pass_all_components", False)))
    acceptance_status = dict(summary.get("acceptance_status", {}))
    checks = [
        "time_step_error",
        "mesh_error",
        "boundary_error",
        "source_term_error",
        "receiver_sampling_error",
        "magnetic_recovery_error",
        "ip_memory_error",
    ]
    diagnostics = {
        "failed": failed,
        "strict_failed": strict_failed,
        "final_acceptance_passed": bool(summary.get("final_acceptance_passed", False)),
        "acceptance_blocking_reasons": list(acceptance_status.get("blocking_reasons", [])),
        "failed_components": summary.get("failed_components", []),
        "physical_failed_components": summary.get("physical_failed_components", summary.get("failed_components", [])),
        "failed_times": summary.get("failed_times", []),
        "recommended_check_order": checks,
        "magnetic_receiver_mode": magnetic_receiver_mode,
        "weak_component_passed": summary.get("weak_component_passed", True),
        "weak_components": summary.get("weak_components", []),
    }
    if failed and magnetic_receiver_mode.startswith("biot"):
        diagnostics["magnetic_recovery_note"] = (
            "Hz is recovered from Biot-Savart current; compare dBzdt before treating this as a PDE failure."
        )
    return diagnostics


def _faraday_integrated_hz_trace(times, dbzdt, *, initial_hz: float, mu: float = 1.2566370614359173e-6):
    import numpy as np

    t = np.asarray(times, dtype=float)
    rate = np.asarray(dbzdt, dtype=float)
    if t.ndim != 1 or rate.ndim != 1 or t.size != rate.size:
        raise ValueError("times and dbzdt must be one-dimensional arrays with matching length")
    if t.size == 0:
        return np.empty(0, dtype=float)
    if np.any(np.diff(t) < 0.0):
        raise ValueError("times must be sorted in nondecreasing order")
    hz = np.empty(t.size, dtype=float)
    hz[0] = float(initial_hz)
    mu_value = float(mu)
    if not math.isfinite(mu_value) or mu_value <= 0.0:
        raise ValueError("mu must be positive")
    for i in range(1, t.size):
        hz[i] = hz[i - 1] + 0.5 * (rate[i - 1] + rate[i]) * (t[i] - t[i - 1]) / mu_value
    return hz


def _faraday_rate_consistency_summary(
    times,
    hz,
    dbzdt,
    *,
    mu: float = 1.2566370614359173e-6,
) -> dict[str, Any]:
    import numpy as np

    t = np.asarray(times, dtype=float)
    h = np.asarray(hz, dtype=float)
    rate = np.asarray(dbzdt, dtype=float)
    if t.ndim != 1 or h.ndim != 1 or rate.ndim != 1 or not (t.size == h.size == rate.size):
        return {"enabled": False, "reason": "times, Hz, and dBzdt must be matching one-dimensional arrays"}
    if t.size < 2:
        return {"enabled": False, "reason": "requires at least two samples"}
    dt = np.diff(t)
    valid = dt > 0.0
    if not np.any(valid):
        return {"enabled": False, "reason": "requires at least one positive time interval"}
    mu_value = float(mu)
    if not math.isfinite(mu_value) or mu_value <= 0.0:
        return {"enabled": False, "reason": "mu must be positive"}
    hz_rate = mu_value * np.diff(h)[valid] / dt[valid]
    dbdt_avg = 0.5 * (rate[:-1][valid] + rate[1:][valid])
    abs_diff = np.abs(hz_rate - dbdt_avg)
    floor = max(float(np.max(np.abs(dbdt_avg))) * 1.0e-6, 1.0e-18) if dbdt_avg.size else 1.0e-18
    rel = abs_diff / np.maximum(np.abs(dbdt_avg), floor)
    max_index = int(np.argmax(rel)) if rel.size else 0
    t_mid = 0.5 * (t[:-1][valid] + t[1:][valid])
    return {
        "enabled": True,
        "method": "mu_dHzdt_vs_trapezoid_dBzdt",
        "sample_count": int(rel.size),
        "max_absolute_difference": float(abs_diff[max_index]) if abs_diff.size else 0.0,
        "max_relative_difference": float(rel[max_index]) if rel.size else 0.0,
        "time_mid_at_max_relative_difference": float(t_mid[max_index]) if t_mid.size else float("nan"),
        "hz_rate_at_max_relative_difference": float(hz_rate[max_index]) if hz_rate.size else float("nan"),
        "dbzdt_average_at_max_relative_difference": float(dbdt_avg[max_index]) if dbdt_avg.size else float("nan"),
    }


def _magnetic_recovery_summary(times, pred_data, components) -> dict[str, Any]:
    import numpy as np

    component_names = [str(component) for component in components]
    if "Hz" not in component_names or "dBzdt" not in component_names:
        return {"enabled": False, "reason": "requires both Hz and dBzdt components"}
    hz_idx = component_names.index("Hz")
    dbdt_idx = component_names.index("dBzdt")
    pred = np.asarray(pred_data, dtype=float)
    t = np.asarray(times, dtype=float)
    if pred.ndim != 2 or pred.shape[0] != t.size:
        return {"enabled": False, "reason": "inconsistent prediction/time shapes"}
    hz = pred[:, hz_idx]
    dbzdt = pred[:, dbdt_idx]
    if not (np.all(np.isfinite(hz)) and np.all(np.isfinite(dbzdt))):
        return {"enabled": False, "reason": "non-finite Hz or dBzdt values"}
    faraday_hz = _faraday_integrated_hz_trace(t, dbzdt, initial_hz=float(hz[0]))
    rate_consistency = _faraday_rate_consistency_summary(t, hz, dbzdt)
    diff = faraday_hz - hz
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(hz), max(float(np.max(np.abs(hz))) * 1.0e-6, 1.0e-16))
    rel = abs_diff / denom
    max_index = int(np.argmax(rel)) if rel.size else 0
    return {
        "enabled": True,
        "method": "faraday_integrated_dBzdt",
        "initial_hz": float(hz[0]) if hz.size else float("nan"),
        "max_absolute_hz_difference": float(abs_diff[max_index]) if abs_diff.size else 0.0,
        "max_relative_hz_difference": float(rel[max_index]) if rel.size else 0.0,
        "time_at_max_relative_hz_difference": float(t[max_index]) if t.size else float("nan"),
        "faraday_hz_final": float(faraday_hz[-1]) if faraday_hz.size else float("nan"),
        "reported_hz_final": float(hz[-1]) if hz.size else float("nan"),
        "rate_consistency": rate_consistency,
    }


def _divergence_cleaning_summary(solver_log) -> dict[str, Any]:
    entries = []
    for item in solver_log or []:
        if "divergence_clean_before" not in item:
            continue
        try:
            before = float(item.get("divergence_clean_before", math.nan))
        except (TypeError, ValueError):
            before = math.nan
        if not math.isfinite(before):
            continue
        entry = {
            "step": int(item.get("step", -1)),
            "time": float(item.get("time", math.nan)),
            "observation_time": float(item.get("observation_time", item.get("time", math.nan))),
            "before": before,
            "after": float(item.get("divergence_clean_after", math.nan)),
            "correction_norm": float(item.get("divergence_clean_correction_norm", math.nan)),
            "applied_correction_norm": float(item.get("divergence_clean_applied_correction_norm", math.nan)),
            "strength": float(item.get("divergence_clean_strength", math.nan)),
        }
        entries.append(entry)
    if not entries:
        return {"enabled": False, "cleaned_step_count": 0}

    def max_entry(key: str) -> dict[str, Any]:
        finite = [entry for entry in entries if math.isfinite(float(entry[key]))]
        if not finite:
            return entries[0]
        return max(finite, key=lambda entry: float(entry[key]))

    max_before = max_entry("before")
    max_correction = max_entry("correction_norm")
    max_applied = max_entry("applied_correction_norm")
    strengths = sorted({float(entry["strength"]) for entry in entries if math.isfinite(float(entry["strength"]))})
    return {
        "enabled": True,
        "cleaned_step_count": len(entries),
        "first_clean_step": int(entries[0]["step"]),
        "first_clean_time": float(entries[0]["time"]),
        "first_clean_observation_time": float(entries[0]["observation_time"]),
        "max_before": float(max_before["before"]),
        "time_at_max_before": float(max_before["time"]),
        "observation_time_at_max_before": float(max_before["observation_time"]),
        "max_after": float(max(entry["after"] for entry in entries if math.isfinite(float(entry["after"])))),
        "max_correction_norm": float(max_correction["correction_norm"]),
        "time_at_max_correction_norm": float(max_correction["time"]),
        "observation_time_at_max_correction_norm": float(max_correction["observation_time"]),
        "max_applied_correction_norm": float(max_applied["applied_correction_norm"]),
        "strength_values": strengths,
    }


def _divergence_control_summary(solver_log) -> dict[str, Any]:
    entries = []
    for item in solver_log or []:
        if not bool(item.get("divergence_control_applied", False)):
            continue
        try:
            applied_weight = float(item.get("divergence_control_applied_weight", math.nan))
        except (TypeError, ValueError):
            applied_weight = math.nan
        entry = {
            "step": int(item.get("step", -1)),
            "time": float(item.get("time", math.nan)),
            "observation_time": float(item.get("observation_time", item.get("time", math.nan))),
            "scale": str(item.get("divergence_control_scale", "")),
            "weight": float(item.get("divergence_control_weight", math.nan)),
            "applied_weight": applied_weight,
            "reference_norm": float(item.get("divergence_control_reference_norm", math.nan)),
            "matrix_norm": float(item.get("divergence_control_matrix_norm", math.nan)),
            "relative_weight": float(item.get("divergence_control_relative_weight", math.nan)),
        }
        entries.append(entry)
    if not entries:
        return {"enabled": False, "applied_step_count": 0}

    def max_finite(key: str) -> float:
        vals = [float(entry[key]) for entry in entries if math.isfinite(float(entry[key]))]
        return float(max(vals)) if vals else float("nan")

    return {
        "enabled": True,
        "applied_step_count": len(entries),
        "first_applied_step": int(entries[0]["step"]),
        "first_applied_time": float(entries[0]["time"]),
        "first_applied_observation_time": float(entries[0]["observation_time"]),
        "scale_values": sorted({str(entry["scale"]) for entry in entries if str(entry["scale"])}),
        "weight_values": sorted({float(entry["weight"]) for entry in entries if math.isfinite(float(entry["weight"]))}),
        "max_applied_weight": max_finite("applied_weight"),
        "max_relative_weight": max_finite("relative_weight"),
        "max_reference_norm": max_finite("reference_norm"),
        "max_matrix_norm": max_finite("matrix_norm"),
    }


def diagnose_source_consistency(
    config: PipelineConfig,
    *,
    source_projection_residual: float | None = None,
    source_diagnostic_inputs: dict[str, Any] | None = None,
    initial_field_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return source/waveform consistency diagnostics available without FEM matrices."""

    if source_diagnostic_inputs is not None:
        from atem3d.source_diagnostics import diagnose_source_consistency as diagnose_core

        diagnostics = diagnose_core(**dict(source_diagnostic_inputs))
        diagnostics["diagnostic_backend"] = "atem3d.source_diagnostics"
        return diagnostics

    initial = _source_current(0.0, config)
    final = _source_current(float(config.ramp_off_time), config)
    integrated = _source_interval_average_didt(0.0, float(config.ramp_off_time), config) * float(config.ramp_off_time)
    expected = final - initial
    return {
        "source_endpoint_balance_residual": source_projection_residual,
        "dc_current_conservation_residual": None,
        "initial_curl_residual": (
            float(initial_field_diagnostics["initial_curl_residual"])
            if isinstance(initial_field_diagnostics, dict)
            and initial_field_diagnostics.get("initial_curl_residual") is not None
            else None
        ),
        "waveform_integral_residual": float(abs(integrated - expected)),
        "endpoint_source_total_sum": 0.0,
        "current_initial": float(initial),
        "current_final": float(final),
        "integral_didt_dt": float(integrated),
        "expected_current_change": float(expected),
    }


def _summarize_manual_line_source_local_diagnostics(
    *,
    npts: int,
    added: int,
    missed: int,
    hit_cell_ids,
    svals,
    cell_l1_contributions: dict[int, float],
    dof_l1_contributions: dict[int, float],
    endpoint_window_fraction: float = 0.05,
) -> dict[str, Any]:
    """Summarize local support of the direct Nedelec line-source projection."""

    npts = int(npts)
    added = int(added)
    missed = int(missed)
    endpoint_window_fraction = max(0.0, min(float(endpoint_window_fraction), 0.5))
    hits = [int(cell) for cell in hit_cell_ids if cell is not None]
    cell_hit_counts: dict[int, int] = {}
    for cell in hits:
        cell_hit_counts[cell] = cell_hit_counts.get(cell, 0) + 1
    hit_count_values = list(cell_hit_counts.values())
    cell_changes = 0
    previous_cell: int | None = None
    for cell in hit_cell_ids:
        if cell is None:
            continue
        current = int(cell)
        if previous_cell is not None and current != previous_cell:
            cell_changes += 1
        previous_cell = current

    def _top_entry(values: dict[int, float]) -> tuple[int | None, float, float]:
        positive = {int(key): float(value) for key, value in values.items() if float(value) > 0.0}
        total = float(sum(positive.values()))
        if not positive or total <= 0.0:
            return None, 0.0, 0.0
        top_key, top_value = sorted(positive.items(), key=lambda item: (-item[1], item[0]))[0]
        return int(top_key), float(top_value), float(top_value / total)

    top_cell, top_cell_l1, top_cell_fraction = _top_entry(cell_l1_contributions)
    top_dof, top_dof_l1, top_dof_fraction = _top_entry(dof_l1_contributions)

    s_array = [float(value) for value in svals]
    start_cells: set[int] = set()
    end_cells: set[int] = set()
    start_points = 0
    end_points = 0
    start_missed = 0
    end_missed = 0
    for s, cell in zip(s_array, hit_cell_ids):
        in_start = s <= endpoint_window_fraction
        in_end = s >= 1.0 - endpoint_window_fraction
        if in_start:
            start_points += 1
            if cell is None:
                start_missed += 1
            else:
                start_cells.add(int(cell))
        if in_end:
            end_points += 1
            if cell is None:
                end_missed += 1
            else:
                end_cells.add(int(cell))

    cell_l1_total = float(sum(float(value) for value in cell_l1_contributions.values()))
    dof_l1_total = float(sum(float(value) for value in dof_l1_contributions.values()))
    return {
        "mode": "manual_line",
        "quadrature_points": npts,
        "added_points": added,
        "missed_points": missed,
        "missed_fraction": float(missed / npts) if npts > 0 else 0.0,
        "unique_hit_cells": int(len(cell_hit_counts)),
        "cell_hit_count_min": int(min(hit_count_values)) if hit_count_values else 0,
        "cell_hit_count_max": int(max(hit_count_values)) if hit_count_values else 0,
        "cell_hit_count_mean": float(sum(hit_count_values) / len(hit_count_values)) if hit_count_values else 0.0,
        "cell_hit_top_fraction": float(max(hit_count_values) / added) if hit_count_values and added > 0 else 0.0,
        "cell_sequence_changes": int(cell_changes),
        "cell_contribution_l1_total": cell_l1_total,
        "cell_contribution_top_cell": top_cell,
        "cell_contribution_top_l1": top_cell_l1,
        "cell_contribution_top_fraction": top_cell_fraction,
        "active_dof_count": int(len([value for value in dof_l1_contributions.values() if float(value) > 0.0])),
        "dof_contribution_l1_total": dof_l1_total,
        "dof_contribution_top_dof": top_dof,
        "dof_contribution_top_l1": top_dof_l1,
        "dof_contribution_top_fraction": top_dof_fraction,
        "endpoint_window_fraction": endpoint_window_fraction,
        "start_window_points": int(start_points),
        "start_window_unique_cells": int(len(start_cells)),
        "start_window_missed": int(start_missed),
        "end_window_points": int(end_points),
        "end_window_unique_cells": int(len(end_cells)),
        "end_window_missed": int(end_missed),
    }


def _scalar_source_balance_vector_diagnostics(endpoint, current_div, residual) -> dict[str, Any]:
    """Summarize scalar-space source balance vectors before projection."""

    import numpy as np

    endpoint = np.asarray(endpoint, dtype=float).reshape(-1)
    current_div = np.asarray(current_div, dtype=float).reshape(-1)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if endpoint.shape != current_div.shape or endpoint.shape != residual.shape:
        raise ValueError("endpoint, current_div, and residual must have the same shape")

    def _norms(prefix: str, values) -> dict[str, Any]:
        abs_values = np.abs(np.asarray(values, dtype=float))
        linf = float(abs_values.max()) if abs_values.size else 0.0
        tol = max(1.0e-14, 1.0e-12 * linf)
        l1 = float(np.sum(abs_values))
        l2 = float(np.linalg.norm(values))
        return {
            f"{prefix}_active_dofs": int(np.count_nonzero(abs_values > tol)),
            f"{prefix}_l1_norm": l1,
            f"{prefix}_l2_norm": l2,
            f"{prefix}_linf_norm": linf,
            f"{prefix}_top_abs_fraction": float(linf / l1) if l1 > 0.0 else 0.0,
        }

    endpoint_l2 = float(np.linalg.norm(endpoint))
    current_l2 = float(np.linalg.norm(current_div))
    residual_l2 = float(np.linalg.norm(residual))
    dot = float(np.dot(current_div, endpoint))
    diagnostics: dict[str, Any] = {}
    diagnostics.update(_norms("endpoint", endpoint))
    diagnostics.update(_norms("current_div", current_div))
    diagnostics.update(_norms("residual", residual))
    diagnostics["residual_l2_over_endpoint_l2"] = float(residual_l2 / endpoint_l2) if endpoint_l2 > 0.0 else 0.0
    diagnostics["current_div_l2_over_endpoint_l2"] = float(current_l2 / endpoint_l2) if endpoint_l2 > 0.0 else 0.0
    diagnostics["current_div_endpoint_dot"] = dot
    diagnostics["current_div_endpoint_alignment"] = float(dot / (current_l2 * endpoint_l2)) if current_l2 > 0.0 and endpoint_l2 > 0.0 else 0.0
    return diagnostics


def _source_projection_diagnostics_from_info(source_info) -> dict[str, Any] | None:
    if not isinstance(source_info, dict):
        return None
    diagnostics = source_info.get("projection_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    out: dict[str, Any] = {}
    for key in (
        "projection_mode",
        "applied",
        "before_residual",
        "after_residual",
        "endpoint_norm",
        "raw_source_l2_norm",
        "projected_source_l2_norm",
        "correction_l2_norm",
        "correction_l2_over_raw",
        "raw_source_l1_norm",
        "projected_source_l1_norm",
        "correction_l1_norm",
        "correction_l1_over_raw",
        "divergence_residual_reduction",
        "ksp_iterations",
        "ksp_reason",
        "ksp_residual",
    ):
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if isinstance(value, bool):
            out[key] = bool(value)
        elif isinstance(value, str):
            out[key] = str(value)
        elif isinstance(value, int):
            out[key] = int(value)
        elif value is None:
            out[key] = None
        else:
            out[key] = float(value)
    scalar_balance = diagnostics.get("scalar_balance")
    if isinstance(scalar_balance, dict):
        out["scalar_balance"] = json.loads(json.dumps(scalar_balance, allow_nan=False, default=float))
    return out


def _source_local_projection_diagnostics_from_info(source_info) -> dict[str, Any] | None:
    if not isinstance(source_info, dict):
        return None
    diagnostics = source_info.get("local_projection_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return json.loads(json.dumps(diagnostics, allow_nan=False, default=float))


def _source_line_orientation_diagnostics_from_info(source_info) -> dict[str, Any] | None:
    local = _source_local_projection_diagnostics_from_info(source_info)
    if not isinstance(local, dict):
        return None
    orientation = local.get("line_orientation")
    if not isinstance(orientation, dict):
        return None
    return json.loads(json.dumps(orientation, allow_nan=False, default=float))


def _source_consistency_inputs_from_info(source_info) -> dict[str, Any] | None:
    if not isinstance(source_info, dict):
        return None
    diagnostics = source_info.get("consistency_diagnostic_inputs")
    if not isinstance(diagnostics, dict):
        return None
    return dict(diagnostics)


def _source_initial_field_diagnostics_from_info(source_info) -> dict[str, Any] | None:
    if not isinstance(source_info, dict):
        return None
    diagnostics = source_info.get("initial_field_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return json.loads(json.dumps(diagnostics, allow_nan=False, default=float))


def write_validation_artifacts(
    times,
    pred_data,
    ref_data,
    components,
    config: PipelineConfig,
    *,
    case_type: str,
    reference_type: str,
    source_info=None,
    receiver_diagnostic_rows=None,
    solver_log=None,
    validation_scope: str = "smoke",
) -> dict[str, Any]:
    """Write P2 validation CSV/JSON/plot artifacts for a three-component run."""

    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    threshold = float(config.error_tolerance)
    rows, summary = _robust_error_rows(times, pred_data, ref_data, components, threshold)
    summary.update(
        {
            "case_type": str(case_type),
            "reference_type": str(reference_type),
            "relative_error_threshold": threshold,
            "validation_scope": str(validation_scope),
        }
    )
    from atem3d.validation_3comp import validation_acceptance_status

    acceptance_status = validation_acceptance_status(
        times,
        components,
        summary,
        case_type=str(case_type),
        reference_type=str(reference_type),
        threshold=threshold,
        validation_scope=str(validation_scope),
    )
    summary["acceptance_status"] = acceptance_status
    summary["full_window_covered"] = bool(acceptance_status["full_window_covered"])
    summary["required_components_present"] = bool(acceptance_status["required_components_present"])
    summary["final_acceptance_passed"] = bool(acceptance_status["final_acceptance_passed"])
    _write_component_csv(workdir / "predictions.csv", times, pred_data, components)
    _write_component_csv(workdir / "reference_empymod_or_1d.csv", times, ref_data, components)
    _write_errors_csv(workdir / "errors.csv", rows)
    _write_receiver_diagnostics_csv(config, receiver_diagnostic_rows)
    _plot_receiver_diagnostics(config, receiver_diagnostic_rows)
    receiver_reference_rows = _receiver_reference_error_rows(
        receiver_diagnostic_rows,
        times,
        ref_data,
        components,
        threshold=threshold,
    )
    _write_receiver_reference_error_artifacts(workdir, receiver_reference_rows)
    (workdir / "error_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    diagnostics = _automatic_failure_diagnostics(
        summary,
        magnetic_receiver_mode=str(config.magnetic_receiver_mode),
    )
    diagnostics["acceptance_status"] = acceptance_status
    diagnostics["validation_failure"] = dict(diagnostics)
    source_projection = _source_projection_diagnostics_from_info(source_info)
    source_consistency_inputs = _source_consistency_inputs_from_info(source_info)
    initial_field_diagnostics = _source_initial_field_diagnostics_from_info(source_info)
    diagnostics["source_consistency"] = diagnose_source_consistency(
        config,
        source_projection_residual=source_projection.get("after_residual") if source_projection else None,
        source_diagnostic_inputs=source_consistency_inputs,
        initial_field_diagnostics=initial_field_diagnostics,
    )
    if source_projection is not None:
        diagnostics["source_projection"] = source_projection
    if initial_field_diagnostics is not None:
        diagnostics["initial_field"] = initial_field_diagnostics
    source_local_projection = _source_local_projection_diagnostics_from_info(source_info)
    if source_local_projection is not None:
        diagnostics["source_local_projection"] = source_local_projection
    source_line_orientation = _source_line_orientation_diagnostics_from_info(source_info)
    if source_line_orientation is not None:
        diagnostics["source_line_orientation"] = source_line_orientation
    diagnostics["receiver_sampling"] = _receiver_diagnostic_summary(
        receiver_diagnostic_rows,
        threshold=float(config.error_tolerance),
    )
    diagnostics["receiver_vs_reference"] = _receiver_reference_summary(
        receiver_diagnostic_rows,
        times,
        ref_data,
        components,
        threshold=float(config.error_tolerance),
    )
    diagnostics["magnetic_recovery"] = _magnetic_recovery_summary(times, pred_data, components)
    diagnostics["divergence_cleaning"] = _divergence_cleaning_summary(solver_log)
    diagnostics["divergence_control"] = _divergence_control_summary(solver_log)
    (workdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (workdir / "run_config_resolved.yaml").write_text(_resolved_config_yaml(config), encoding="utf-8")
    _write_validation_plots(workdir, times, pred_data, ref_data, rows, components)
    return summary


def _resolved_config_yaml(config: PipelineConfig) -> str:
    values = {
        "source_start": list(config.source_start),
        "source_end": list(config.source_end),
        "source_current": float(config.source_current),
        "source_projection_mode": str(config.source_projection_mode),
        "receiver": list(config.receiver),
        "receiver_mesh_size": float(config.receiver_mesh_size),
        "receiver_anchor_mesh_size": float(config.receiver_anchor_mesh_size),
        "receiver_refinement_radius": float(config.receiver_refinement_radius),
        "ramp_off_time": float(config.ramp_off_time),
        "time_origin": str(config.time_origin),
        "t_min": float(config.t_min),
        "t_max": float(config.t_max),
        "time_growth": float(config.time_growth),
        "observation_times": list(config.observation_times),
        "min_steps_during_turnoff": int(config.min_steps_during_turnoff),
        "min_steps_before_first_observation": int(config.min_steps_before_first_observation),
        "components": "runtime",
        "outer_boundary_mode": str(config.outer_boundary_mode),
        "outer_boundary_robin_scale": float(config.outer_boundary_robin_scale),
        "magnetic_receiver_mode": str(config.magnetic_receiver_mode),
        "magnetic_dbdt_mode": str(config.magnetic_dbdt_mode),
        "divergence_cleaning": str(config.divergence_cleaning),
        "divergence_cleaning_strength": float(config.divergence_cleaning_strength),
        "divergence_cleaning_t_obs_min": float(config.divergence_cleaning_t_obs_min),
        "divergence_control_weight": float(config.divergence_control_weight),
        "divergence_control_t_obs_min": float(config.divergence_control_t_obs_min),
        "divergence_control_scale": str(config.divergence_control_scale),
        "polarization": str(config.polarization),
    }
    lines = []
    for key, value in values.items():
        if isinstance(value, list):
            rendered = "[" + ", ".join(f"{float(item):.16g}" for item in value) + "]"
        elif isinstance(value, str):
            rendered = value
        else:
            rendered = f"{value:.16g}" if isinstance(value, float) else str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def write_source_only_diagnostics(
    config: PipelineConfig,
    env: dict[str, Any],
    source_info,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write source-only diagnostics without running the transient forward solve."""

    runtime = {} if runtime is None else dict(runtime)
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    source_projection = _source_projection_diagnostics_from_info(source_info)
    source_local_projection = _source_local_projection_diagnostics_from_info(source_info)
    source_line_orientation = _source_line_orientation_diagnostics_from_info(source_info)
    source_consistency_inputs = _source_consistency_inputs_from_info(source_info)
    initial_field_diagnostics = _source_initial_field_diagnostics_from_info(source_info)
    diagnostics: dict[str, Any] = {
        "source_only": True,
        "source_mode": str(source_info.get("mode")) if isinstance(source_info, dict) else None,
        "source_consistency": diagnose_source_consistency(
            config,
            source_projection_residual=source_projection.get("after_residual") if source_projection else None,
            source_diagnostic_inputs=source_consistency_inputs,
            initial_field_diagnostics=initial_field_diagnostics,
        ),
        "runtime": {str(key): float(value) for key, value in runtime.items() if isinstance(value, (int, float))},
    }
    if source_projection is not None:
        diagnostics["source_projection"] = source_projection
    if source_local_projection is not None:
        diagnostics["source_local_projection"] = source_local_projection
    if source_line_orientation is not None:
        diagnostics["source_line_orientation"] = source_line_orientation
    if initial_field_diagnostics is not None:
        diagnostics["initial_field"] = initial_field_diagnostics
    (workdir / "source_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (workdir / "run_config_resolved.yaml").write_text(_resolved_config_yaml(config), encoding="utf-8")

    lines = ["SOTEM source-only diagnostics", "============================", ""]
    lines.append(f"python: {env.get('python', sys.executable)}")
    lines.append(f"source mode: {diagnostics['source_mode']}")
    lines.append(f"source projection mode: {config.source_projection_mode}")
    if source_projection is not None:
        lines.append(
            "source projection: "
            f"mode={source_projection.get('projection_mode', config.source_projection_mode)}; "
            f"applied={source_projection.get('applied')}; "
            f"before={float(source_projection.get('before_residual', math.nan)):.6e}; "
            f"after={float(source_projection.get('after_residual', math.nan)):.6e}; "
            f"endpoint_norm={float(source_projection.get('endpoint_norm', math.nan)):.6e}"
        )
        scalar_balance = source_projection.get("scalar_balance")
        if isinstance(scalar_balance, dict):
            lines.append(
                "source scalar balance: "
                f"residual_active_dofs={int(scalar_balance.get('residual_active_dofs', 0))}; "
                f"residual_l2/endpoint_l2={float(scalar_balance.get('residual_l2_over_endpoint_l2', math.nan)):.6g}; "
                f"residual_top_fraction={float(scalar_balance.get('residual_top_abs_fraction', math.nan)):.6g}; "
                f"current_endpoint_alignment={float(scalar_balance.get('current_div_endpoint_alignment', math.nan)):.6g}"
            )
    if source_local_projection is not None:
        lines.append(
            "source local projection: "
            f"quadrature={int(source_local_projection.get('quadrature_points', 0))}; "
            f"added={int(source_local_projection.get('added_points', 0))}; "
            f"missed={int(source_local_projection.get('missed_points', 0))}; "
            f"unique_cells={int(source_local_projection.get('unique_hit_cells', 0))}"
        )
    if source_line_orientation is not None:
        lines.append(
            "source line orientation: "
            f"length={float(source_line_orientation.get('source_length_m', math.nan)):.6g} m; "
            f"weight_sum={float(source_line_orientation.get('quadrature_weight_sum_m', math.nan)):.6g} m; "
            f"cos={float(source_line_orientation.get('orientation_cosine', math.nan)):.6g}; "
            f"parallel_error={float(source_line_orientation.get('relative_parallel_length_error', math.nan)):.6g}; "
            f"transverse={float(source_line_orientation.get('transverse_residual_m', math.nan)):.6g} m; "
            f"monotonic={bool(source_line_orientation.get('s_parameter_monotonic', False))}; "
            f"reversed={bool(source_line_orientation.get('reversed_orientation', False))}"
        )
    if runtime:
        lines.append("")
        lines.append("runtime:")
        for key, value in sorted(runtime.items()):
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {float(value):.6g} s")
    (workdir / "source_diagnostics_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diagnostics


def compute_horizontal_electric_error(fem_data, ref_data, floor_factor: float = 1.0e-6):
    """Compute relative error of the horizontal electric vector (Ex, Ey)."""

    import numpy as np

    fem_arr = np.asarray(fem_data, dtype=float)
    ref_arr = np.asarray(ref_data, dtype=float)
    if fem_arr.shape != ref_arr.shape or fem_arr.shape[1] < 2:
        raise ValueError("FEM and reference arrays must have matching Ex/Ey columns")
    ref_norm = np.linalg.norm(ref_arr[:, :2], axis=1)
    diff_norm = np.linalg.norm(fem_arr[:, :2] - ref_arr[:, :2], axis=1)
    floor = floor_factor * float(np.max(ref_norm))
    if floor == 0.0:
        floor = 1.0e-300
    rel = diff_norm / np.maximum(ref_norm, floor)
    max_idx = int(np.argmax(rel))
    return {
        "floor": float(floor),
        "mean": float(np.mean(rel)),
        "median": float(np.median(rel)),
        "rms": float(np.sqrt(np.mean(rel**2))),
        "max": float(rel[max_idx]),
        "max_index": max_idx,
        "relative": rel,
    }


def compute_windowed_error_metrics(times, fem_data, ref_data, components, error_min_time: float = 0.0):
    """Compute component and horizontal-E errors for samples at or after error_min_time."""

    import numpy as np

    times = np.asarray(times, dtype=float)
    mask = times >= float(error_min_time)
    if not np.any(mask):
        raise ValueError("error_min_time excludes all samples")
    fem_arr = np.asarray(fem_data, dtype=float)[mask]
    ref_arr = np.asarray(ref_data, dtype=float)[mask]
    return {
        "time_count": int(np.count_nonzero(mask)),
        "time_min": float(times[mask][0]),
        "time_max": float(times[mask][-1]),
        "errors": compute_error(fem_arr, ref_arr, components),
        "horizontal_electric": compute_horizontal_electric_error(fem_arr, ref_arr),
    }


def check_physical_error_window(times, fem_data, ref_data, components, error_min_time: float = 0.0, tolerance: float = 0.05):
    """Check Ex, dBzdt, and Eh_vector max errors over a fixed time window."""

    metrics = compute_windowed_error_metrics(times, fem_data, ref_data, components, error_min_time=error_min_time)
    errors = metrics["errors"]
    maxima = {
        "Ex": float(errors["Ex"]["max"]),
        "dBzdt": float(errors["dBzdt"]["max"]),
        "Eh_vector": float(metrics["horizontal_electric"]["max"]),
    }
    return {
        "passed": bool(all(value <= float(tolerance) for value in maxima.values())),
        "tolerance": float(tolerance),
        "time_min": float(metrics["time_min"]),
        "time_max": float(metrics["time_max"]),
        "time_count": int(metrics["time_count"]),
        "maxima": maxima,
    }


def check_weak_component_error_window(
    times,
    fem_data,
    ref_data,
    components,
    error_min_time: float = 0.0,
    tolerance: float = 0.05,
    weak_reference_fraction: float = 0.05,
):
    """Check weak horizontal E components with an absolute error scaled by the main horizontal field."""

    import numpy as np

    times = np.asarray(times, dtype=float)
    mask = times >= float(error_min_time)
    if not np.any(mask):
        raise ValueError("error_min_time excludes all samples")
    components = [str(comp) for comp in components]
    fem_arr = np.asarray(fem_data, dtype=float)[mask]
    ref_arr = np.asarray(ref_data, dtype=float)[mask]
    if fem_arr.shape != ref_arr.shape or fem_arr.shape[1] != len(components):
        raise ValueError("FEM data, reference data, and component names have inconsistent shapes")
    if "Ex" not in components or "Ey" not in components:
        return {
            "passed": True,
            "tolerance": float(tolerance),
            "weak_reference_fraction": float(weak_reference_fraction),
            "primary_scale": 0.0,
            "weak_components": [],
            "maxima": {},
            "reference_maxima": {},
        }

    ix = components.index("Ex")
    iy = components.index("Ey")
    ref_horizontal = np.sqrt(ref_arr[:, ix] ** 2 + ref_arr[:, iy] ** 2)
    primary_scale = float(np.max(np.abs(ref_horizontal)))
    if primary_scale <= 0.0:
        primary_scale = float(np.max(np.abs(ref_arr[:, [ix, iy]])))
    if primary_scale <= 0.0:
        primary_scale = 1.0

    weak_components = []
    maxima = {}
    reference_maxima = {}
    for comp, idx in (("Ex", ix), ("Ey", iy)):
        ref_max = float(np.max(np.abs(ref_arr[:, idx])))
        if ref_max <= float(weak_reference_fraction) * primary_scale:
            weak_components.append(comp)
            reference_maxima[comp] = ref_max
            maxima[comp] = float(np.max(np.abs(fem_arr[:, idx] - ref_arr[:, idx])) / primary_scale)

    return {
        "passed": bool(all(value <= float(tolerance) for value in maxima.values())),
        "tolerance": float(tolerance),
        "weak_reference_fraction": float(weak_reference_fraction),
        "primary_scale": primary_scale,
        "weak_components": weak_components,
        "maxima": maxima,
        "reference_maxima": reference_maxima,
    }


def find_physical_error_passing_window(times, fem_data, ref_data, components, tolerance: float = 0.05):
    """Find the earliest time from which Ex, dBzdt, and Eh_vector max errors pass tolerance."""

    import numpy as np

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    fem_arr = np.asarray(fem_data, dtype=float)
    ref_arr = np.asarray(ref_data, dtype=float)
    if fem_arr.shape != ref_arr.shape or fem_arr.shape[0] != times.size:
        raise ValueError("times, FEM data, and reference data have inconsistent shapes")

    for start in range(times.size):
        window = check_physical_error_window(times, fem_arr, ref_arr, components, error_min_time=float(times[start]), tolerance=tolerance)
        if window["passed"]:
            return {
                "time_min": float(window["time_min"]),
                "time_max": float(window["time_max"]),
                "time_count": int(window["time_count"]),
                "start_index": int(start),
                "tolerance": float(tolerance),
                "maxima": window["maxima"],
            }
    return None


def plot_verification(times, fem_data, ref_data, errors, components, config: PipelineConfig) -> None:
    """Save FEM-vs-empymod comparison and error curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ncols = max(1, len(components))
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        fem_abs = np.abs(np.asarray(fem_data[:, i], dtype=float))
        ref_abs = np.abs(np.asarray(ref_data[:, i], dtype=float))
        fem_abs = np.where(fem_abs > 0.0, fem_abs, np.nan)
        ref_abs = np.where(ref_abs > 0.0, ref_abs, np.nan)
        ax.loglog(times, fem_abs, "o-", ms=2, lw=1, label="FEM")
        ax.loglog(times, ref_abs, "-", lw=1.5, label="empymod")
        ax.set_title(comp)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"|{comp}|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        errax = axes[1, i]
        errax.semilogx(times, errors[comp]["relative"], "r-", lw=1)
        errax.axhline(config.error_tolerance, color="k", ls="--", lw=0.8)
        errax.set_xlabel("time (s)")
        errax.set_ylabel("floor relative error")
        errax.grid(True, which="both", alpha=0.3)
    fig.suptitle("3D FETD air-earth verification against empymod")
    fig.savefig(config.output_png(), dpi=180)
    plt.close(fig)
    print(f"[plot] saved {config.output_png()}", flush=True)


def write_report(
    config: PipelineConfig,
    env: dict[str, str],
    fem_result,
    ref_result,
    errors,
    source_info,
    debye=None,
    reference_audit_errors=None,
    runtime=None,
) -> None:
    """Write a text report with diagnostics and optimization guidance."""

    lines: list[str] = []
    lines.append("SOTEM FETD verification report")
    lines.append("=" * 34)
    lines.append(f"python: {sys.executable}")
    lines.append("environment:")
    for key in sorted(env):
        lines.append(f"  {key}: {env[key]}")
    if runtime is not None:
        lines.append("")
        lines.append("model runtime:")
        runtime_labels = [
            ("total_seconds", "total"),
            ("mesh_seconds", "mesh"),
            ("setup_seconds", "setup"),
            ("forward_seconds", "forward solve"),
            ("reference_seconds", "empymod reference"),
            ("postprocess_seconds", "postprocess/report"),
        ]
        for key, label in runtime_labels:
            if key in runtime:
                lines.append(f"  {label}: {float(runtime[key]):.3f} s")
    lines.append("")
    lines.append("geometry:")
    lines.append(f"  x,y extent: +/- {config.x_extent:g} m")
    lines.append(f"  air height: {config.air_height:g} m; earth depth: {config.earth_depth:g} m")
    lines.append(f"  source: {config.source_start} -> {config.source_end}, I={config.source_current:g} A")
    lines.append(f"  receiver: {config.receiver}")
    try:
        model = validate_model_consistency(config)
        lines.append(
            "  survey geometry: "
            f"source_length={model['source_length']:.6g} m, "
            f"inline_from_source_start={model['inline_distance_from_source_start']:.6g} m, "
            f"parallel_offset={model['parallel_offset']:.6g} m"
        )
        lines.append(
            "  model audit: "
            f"reference_mode={model['reference_mode']}, "
            f"source_depth=({model['source_depth_start']:.6g}, {model['source_depth_end']:.6g}) m, "
            f"receiver_depth={model['receiver_depth']:.6g} m, "
            f"sigma_earth={model['sigma_earth']:.6e} S/m, "
            f"time_origin={model['time_origin']}, "
            f"reference_ramp_window={model['reference_ramp_window']}"
        )
    except ValueError as exc:
        lines.append(f"  model audit: INVALID ({exc})")
    lines.append(f"  source mode: {source_info['mode']}")
    source_projection = _source_projection_diagnostics_from_info(source_info)
    if source_projection is not None:
        lines.append(
            "  source projection: "
            f"mode={source_projection.get('projection_mode', config.source_projection_mode)}; "
            f"applied={source_projection.get('applied')}; "
            f"before={float(source_projection.get('before_residual', math.nan)):.6e}; "
            f"after={float(source_projection.get('after_residual', math.nan)):.6e}; "
            f"endpoint_norm={float(source_projection.get('endpoint_norm', math.nan)):.6e}; "
            f"correction_l2/raw={float(source_projection.get('correction_l2_over_raw', math.nan)):.6g}; "
            f"correction_l1/raw={float(source_projection.get('correction_l1_over_raw', math.nan)):.6g}"
        )
        scalar_balance = source_projection.get("scalar_balance")
        if isinstance(scalar_balance, dict):
            lines.append(
                "  source scalar balance: "
                f"residual_active_dofs={int(scalar_balance.get('residual_active_dofs', 0))}; "
                f"residual_l2/endpoint_l2={float(scalar_balance.get('residual_l2_over_endpoint_l2', math.nan)):.6g}; "
                f"residual_top_fraction={float(scalar_balance.get('residual_top_abs_fraction', math.nan)):.6g}; "
                f"current_endpoint_alignment={float(scalar_balance.get('current_div_endpoint_alignment', math.nan)):.6g}"
            )
    source_local_projection = _source_local_projection_diagnostics_from_info(source_info)
    if source_local_projection is not None:
        lines.append(
            "  source local projection: "
            f"quadrature={int(source_local_projection.get('quadrature_points', 0))}; "
            f"added={int(source_local_projection.get('added_points', 0))}; "
            f"missed={int(source_local_projection.get('missed_points', 0))}; "
            f"unique_cells={int(source_local_projection.get('unique_hit_cells', 0))}; "
            f"top_cell_fraction={float(source_local_projection.get('cell_contribution_top_fraction', math.nan)):.6g}; "
            f"top_dof_fraction={float(source_local_projection.get('dof_contribution_top_fraction', math.nan)):.6g}"
        )
        if "integration_mode" in source_local_projection:
            lines.append(
                "  source line integration: "
                f"mode={source_local_projection.get('integration_mode')}; "
                f"segments={int(source_local_projection.get('segment_count', 0))}; "
                f"segment_length_total={float(source_local_projection.get('segment_total_length', 0.0)):.6g} m; "
                "quadrature_per_segment[min/mean/max]="
                f"{float(source_local_projection.get('quadrature_points_per_segment_min', 0.0)):.6g}/"
                f"{float(source_local_projection.get('quadrature_points_per_segment_mean', 0.0)):.6g}/"
                f"{float(source_local_projection.get('quadrature_points_per_segment_max', 0.0)):.6g}"
            )
    source_line_orientation = _source_line_orientation_diagnostics_from_info(source_info)
    if source_line_orientation is not None:
        lines.append(
            "  source line orientation: "
            f"length={float(source_line_orientation.get('source_length_m', math.nan)):.6g} m; "
            f"weight_sum={float(source_line_orientation.get('quadrature_weight_sum_m', math.nan)):.6g} m; "
            f"cos={float(source_line_orientation.get('orientation_cosine', math.nan)):.6g}; "
            f"parallel_error={float(source_line_orientation.get('relative_parallel_length_error', math.nan)):.6g}; "
            f"transverse={float(source_line_orientation.get('transverse_residual_m', math.nan)):.6g} m; "
            f"monotonic={bool(source_line_orientation.get('s_parameter_monotonic', False))}; "
            f"reversed={bool(source_line_orientation.get('reversed_orientation', False))}"
        )
    lines.append(f"  source RHS sign: {config.source_rhs_sign:g}")
    lines.append(f"  source projection mode: {config.source_projection_mode}")
    lines.append(f"  source quadrature points: {config.source_quadrature_points if int(config.source_quadrature_points) > 0 else 'auto'}")
    lines.append(f"  source term mode: {config.source_term_mode}")
    lines.append(f"  formulation: {config.formulation}")
    lines.append(f"  Nedelec order: {config.nedelec_order}")
    lines.append(f"  initial DC mode: {config.initial_dc_mode}")
    lines.append(f"  magnetic receiver mode: {config.magnetic_receiver_mode}")
    lines.append(f"  magnetic dBdt mode: {config.magnetic_dbdt_mode}")
    lines.append(
        f"  outer boundary mode: {config.outer_boundary_mode}; "
        f"robin scale={float(config.outer_boundary_robin_scale):.6g}"
    )
    lines.append(
        f"  receiver type: {config.receiver_type}; "
        f"average radius={float(config.receiver_average_radius):.6g} m"
    )
    lines.append(
        f"  receiver mesh size: {float(config.receiver_mesh_size):.6g} m; "
        f"anchor mesh size={_receiver_anchor_mesh_size(config):.6g} m; "
        f"refinement radius={float(config.receiver_refinement_radius):.6g} m"
    )
    lines.append(f"  receiver evaluation mode: {config.receiver_evaluation_mode}")
    receiver_summary = _receiver_diagnostic_summary(
        fem_result.get("receiver_diagnostic_rows"),
        threshold=float(config.error_tolerance),
    )
    if receiver_summary.get("enabled"):
        lines.append(
            "  receiver diagnostics: "
            f"baseline={receiver_summary['baseline_receiver_type']}; "
            f"types={','.join(receiver_summary['receiver_types'])}; "
            f"time_count={receiver_summary['time_count']}; "
            f"sampling_issue_suspected={receiver_summary['receiver_sampling_issue_suspected']}"
        )
    lines.append(
        f"  divergence cleaning: {config.divergence_cleaning}; "
        f"strength={float(config.divergence_cleaning_strength):.6g}; "
        f"t_obs_min={float(config.divergence_cleaning_t_obs_min):.6g} s"
    )
    lines.append(
        f"  divergence control: weight={float(config.divergence_control_weight):.6g}; "
        f"t_obs_min={float(config.divergence_control_t_obs_min):.6g} s; "
        f"scale={str(config.divergence_control_scale)}"
    )
    lines.append(
        f"  checkpoint forward: {bool(config.checkpoint_forward)}; resume forward: {bool(config.resume_forward)}; "
        f"stop after outputs: {int(config.stop_after_outputs)}"
    )
    lines.append(
        f"  memory budget: {float(config.memory_limit_gb):.3g} GB; "
        f"safety fraction: {float(config.memory_safety_fraction):.3g}"
    )
    diffusion_audit = _diffusion_refinement_audit(config)
    lines.append(
        "  late diffusion audit: "
        f"Lmax(t_max)={diffusion_audit['diffusion_length']:.6g} m; "
        f"recommended radius/depth>={diffusion_audit['recommended_radius']:.6g} m; "
        f"refinement box radius={diffusion_audit['box_radius']:.6g} m; "
        f"refinement box depth={diffusion_audit['box_depth']:.6g} m; "
        f"refinement mesh_size={diffusion_audit['mesh_size']:.6g} m; "
        f"refinement_underresolved={diffusion_audit['underresolved']}; "
        f"domain radius={diffusion_audit['domain_horizontal_radius']:.6g} m; "
        f"domain depth={diffusion_audit['domain_depth']:.6g} m; "
        f"domain_underresolved={diffusion_audit['domain_underresolved']}"
    )
    sponge = _sponge_diagnostics(config)
    lines.append(
        "  transient sponge: "
        f"enabled={sponge['enabled']}; "
        f"strength={sponge['strength']:.6e} S/m; "
        f"thickness={sponge['thickness']:.6g} m; "
        f"power={sponge['power']:.6g}; "
        f"apply_to_initial={sponge['apply_to_initial']}; "
        f"sides={','.join(sponge['sides']) if sponge['sides'] else 'none'}"
    )
    lines.append(f"  time method: {config.time_method}")
    lines.append(f"  time origin: {config.time_origin}")
    lines.append(f"  time theta: {config.time_theta:g}")
    lines.append(f"  ramp-off time: {config.ramp_off_time:g} s")
    lines.append(
        f"  time samples: t_min={config.t_min:g} s; t_max={config.t_max:g} s; "
        f"ramp_solver_t_min={config.ramp_solver_t_min:g} s; "
        f"min_steps_during_turnoff={config.min_steps_during_turnoff}; "
        f"min_steps_before_first_observation={config.min_steps_before_first_observation}"
    )
    lines.append(f"  empymod srcpts: {config.empymod_srcpts}; ht={config.empymod_ht}; ft={config.empymod_ft}")
    if int(config.reference_audit_srcpts) > 0:
        lines.append(f"  reference audit srcpts: {config.reference_audit_srcpts}")
    if float(config.error_min_time) > 0.0:
        lines.append(f"  error metrics start time: {config.error_min_time:g} s")
    layer_depths, layer_resistivities = _normalise_layer_model(config)
    if layer_depths:
        lines.append(
            f"  rho_air={config.rho_air:g} ohm m; "
            f"layer_depths={_format_float_list(layer_depths)} m; "
            f"layer_resistivities={_format_float_list(layer_resistivities)} ohm m"
        )
    else:
        lines.append(f"  rho_air={config.rho_air:g} ohm m; rho_earth={layer_resistivities[0]:g} ohm m")
    if debye is not None:
        fit = debye["fit"]
        lines.append("")
        lines.append(f"Cole-Cole Debye fit relative L2: {fit.relative_l2:.6e}")
        for i, term in enumerate(fit.terms):
            lines.append(f"  term {i}: A={term.delta_sigma:.6e} S/m, tau={term.tau:.6e} s")

    lines.append("")
    lines.append("error metrics with floor denominator denom=max(abs(F_ref), 1e-6*max(abs(F_ref))):")
    max_error = 0.0
    for comp, values in errors.items():
        max_error = max(max_error, float(values["max"]))
        idx = int(values["max_index"])
        lines.append(
            f"  {comp}: mean={values['mean']:.6e}, median={values['median']:.6e}, "
            f"RMS={values['rms']:.6e}, max={values['max']:.6e} at t={fem_result['times'][idx]:.6e} s, "
            f"floor={values['floor']:.6e}"
        )
        if values["reference_too_small"]:
            lines.append(f"    note: {comp} reference is very small over much of the window; max relative error may be unrepresentative.")

    horizontal_e_error = compute_horizontal_electric_error(fem_result["data"], ref_result["data"])
    idx = int(horizontal_e_error["max_index"])
    lines.append(
        f"  Eh_vector: mean={horizontal_e_error['mean']:.6e}, median={horizontal_e_error['median']:.6e}, "
        f"RMS={horizontal_e_error['rms']:.6e}, max={horizontal_e_error['max']:.6e} "
        f"at t={fem_result['times'][idx]:.6e} s, floor={horizontal_e_error['floor']:.6e}"
    )
    if receiver_summary.get("enabled"):
        lines.append("")
        lines.append("receiver sampling diagnostics:")
        for receiver_type, comparison in receiver_summary["comparisons"].items():
            lines.append(f"  {receiver_type} vs {receiver_summary['baseline_receiver_type']}:")
            for component, metrics in comparison.items():
                lines.append(
                    f"    {component}: max_rel_diff={metrics['max_relative_difference']:.6e} "
                    f"at t={metrics['time_at_max_relative_difference']:.6e} s; "
                    f"mean_rel_diff={metrics['mean_relative_difference']:.6e}; "
                    f"max_abs_diff={metrics['max_absolute_difference']:.6e}"
                )
    receiver_reference = _receiver_reference_summary(
        fem_result.get("receiver_diagnostic_rows"),
        fem_result["times"],
        ref_result["data"],
        fem_result["components"],
        threshold=float(config.error_tolerance),
    )
    if receiver_reference.get("enabled"):
        lines.append("")
        lines.append("receiver vs reference diagnostics:")
        for receiver_type, comparison in receiver_reference["comparisons"].items():
            lines.append(f"  {receiver_type} vs {receiver_reference['baseline_receiver_type']}:")
            for component, metrics in comparison.items():
                lines.append(
                    f"    {component}: candidate_max={metrics['candidate_max_relative_error']:.6e}; "
                    f"baseline_max={metrics['baseline_max_relative_error']:.6e}; "
                    f"improves={metrics['improves_over_baseline']}"
                )
    magnetic_summary = _magnetic_recovery_summary(
        fem_result["times"],
        fem_result["data"],
        fem_result["components"],
    )
    if magnetic_summary.get("enabled"):
        lines.append("")
        lines.append("magnetic recovery diagnostics:")
        lines.append(
            f"  {magnetic_summary['method']}: max_rel_hz_diff={magnetic_summary['max_relative_hz_difference']:.6e} "
            f"at t={magnetic_summary['time_at_max_relative_hz_difference']:.6e} s; "
            f"max_abs_hz_diff={magnetic_summary['max_absolute_hz_difference']:.6e}; "
            f"faraday_hz_final={magnetic_summary['faraday_hz_final']:.6e}; "
            f"reported_hz_final={magnetic_summary['reported_hz_final']:.6e}"
        )

    passing_window = find_physical_error_passing_window(
        fem_result["times"],
        fem_result["data"],
        ref_result["data"],
        fem_result["components"],
        tolerance=float(config.error_tolerance),
    )
    configured_error_start = float(config.error_min_time) if float(config.error_min_time) > 0.0 else float(fem_result["times"][0])
    configured_physical_window = check_physical_error_window(
        fem_result["times"],
        fem_result["data"],
        ref_result["data"],
        fem_result["components"],
        error_min_time=configured_error_start,
        tolerance=float(config.error_tolerance),
    )
    configured_weak_window = check_weak_component_error_window(
        fem_result["times"],
        fem_result["data"],
        ref_result["data"],
        fem_result["components"],
        error_min_time=configured_error_start,
        tolerance=float(config.error_tolerance),
        weak_reference_fraction=float(config.weak_component_reference_fraction),
    )
    lines.append("")
    if passing_window is None:
        lines.append(
            f"strict physical-error passing window: none found for Ex, dBzdt, and Eh_vector max <= {config.error_tolerance:.3g}"
        )
    else:
        maxima = passing_window["maxima"]
        lines.append(
            "strict physical-error passing window "
            f"(Ex, dBzdt, Eh_vector max <= {passing_window['tolerance']:.3g}): "
            f"t >= {passing_window['time_min']:.6e} s "
            f"(n={passing_window['time_count']}, range {passing_window['time_min']:.6e}-{passing_window['time_max']:.6e} s)"
        )
        lines.append(
            f"  maxima: Ex={maxima['Ex']:.6e}, dBzdt={maxima['dBzdt']:.6e}, Eh_vector={maxima['Eh_vector']:.6e}"
        )

    lines.append("")
    lines.append(
        "weak horizontal-component gate "
        f"(reference max <= {config.weak_component_reference_fraction:.3g} * max(|Eh_ref|); "
        "absolute error scaled by max(|Eh_ref|)):"
    )
    lines.append(
        f"  passed={configured_weak_window['passed']}; "
        f"primary_scale={configured_weak_window['primary_scale']:.6e}; "
        f"weak_components={','.join(configured_weak_window['weak_components']) if configured_weak_window['weak_components'] else 'none'}"
    )
    for comp in configured_weak_window["weak_components"]:
        lines.append(
            f"  {comp}: ref_max={configured_weak_window['reference_maxima'][comp]:.6e}, "
            f"scaled_abs_error_max={configured_weak_window['maxima'][comp]:.6e}, "
            f"tolerance={configured_weak_window['tolerance']:.6e}"
        )

    if float(config.error_min_time) > 0.0:
        window = compute_windowed_error_metrics(
            fem_result["times"],
            fem_result["data"],
            ref_result["data"],
            fem_result["components"],
            error_min_time=float(config.error_min_time),
        )
        lines.append("")
        lines.append(
            f"windowed error metrics for t >= {config.error_min_time:.6e} s "
            f"(n={window['time_count']}, actual range {window['time_min']:.6e}-{window['time_max']:.6e} s):"
        )
        for comp, values in window["errors"].items():
            lines.append(
                f"  {comp}: mean={values['mean']:.6e}, median={values['median']:.6e}, "
                f"RMS={values['rms']:.6e}, max={values['max']:.6e}, floor={values['floor']:.6e}"
            )
        eh = window["horizontal_electric"]
        lines.append(
            f"  Eh_vector: mean={eh['mean']:.6e}, median={eh['median']:.6e}, "
            f"RMS={eh['rms']:.6e}, max={eh['max']:.6e}, floor={eh['floor']:.6e}"
        )

    if reference_audit_errors is not None:
        lines.append("")
        lines.append(
            "empymod reference self-audit "
            f"(primary srcpts={config.empymod_srcpts}, audit srcpts={config.reference_audit_srcpts}):"
        )
        for comp, values in reference_audit_errors.items():
            idx = int(values["max_index"])
            lines.append(
                f"  {comp}: mean={values['mean']:.6e}, median={values['median']:.6e}, "
                f"RMS={values['rms']:.6e}, max={values['max']:.6e} at t={ref_result['times'][idx]:.6e} s, "
                f"floor={values['floor']:.6e}"
            )
            if values["reference_too_small"]:
                lines.append(f"    note: {comp} audit denominator is small; weak-component reference may be transform/source-integration sensitive.")

    lines.append("")
    lines.append("solver log:")
    for item in fem_result["solver_log"]:
        div_clean_text = ""
        if "divergence_clean_before" in item:
            div_clean_text = (
                f" div_clean_before={float(item['divergence_clean_before']):.6e}"
                f" div_clean_after={float(item.get('divergence_clean_after', math.nan)):.6e}"
                f" div_clean_correction={float(item.get('divergence_clean_correction_norm', math.nan)):.6e}"
                f" div_clean_applied={float(item.get('divergence_clean_applied_correction_norm', math.nan)):.6e}"
                f" div_clean_strength={float(item.get('divergence_clean_strength', math.nan)):.6g}"
            )
        if item.get("is_output", True):
            lines.append(
                f"  step={item['step']:04d} t_internal={item['time']:.6e} "
                f"t_obs={item.get('observation_time', item['time']):.6e} dt={item['dt']:.6e} "
                f"its={item['its']} residual={item['residual']:.6e} reason={item['reason']}{div_clean_text}"
            )
        else:
            lines.append(
                f"  step={item['step']:04d} t_internal={item['time']:.6e} dt={item['dt']:.6e} "
                f"its={item['its']} residual={item['residual']:.6e} reason={item['reason']}{div_clean_text}"
            )

    if max_error > config.error_tolerance:
        lines.append("")
        if configured_physical_window["passed"]:
            weak_components = [
                comp
                for comp, values in errors.items()
                if float(values["max"]) > float(config.error_tolerance)
                and comp not in {"Ex", "dBzdt"}
            ]
            lines.append("[metric note] Physical response metrics pass in the configured window.")
            lines.append(
                "  - The physical gate uses Ex, dBzdt, and Eh_vector; Eh_vector avoids singular "
                "relative percentages for near-zero transverse electric components."
            )
            if weak_components:
                lines.append(f"  - Component relative error exceeds tolerance for weak components: {', '.join(weak_components)}.")
            if configured_weak_window["passed"]:
                lines.append(
                    "  - The weak horizontal-component absolute gate also passes when scaled by max(|Eh_ref|)."
                )
            else:
                lines.append(
                    "  - The weak horizontal-component absolute gate does not pass; refine mesh/receiver "
                    "evaluation before accepting this run."
                )
        elif passing_window is not None:
            lines.append("[optimization note] The configured window does not fully pass the physical gate.")
            lines.append(
                f"  - Configured error start: t >= {configured_error_start:.6e} s; "
                f"earliest passing start: t >= {passing_window['time_min']:.6e} s."
            )
            lines.append("  - If earlier time coverage is required, inspect ramp-off discretization, initial field, and early time steps.")
        else:
            lines.append("[optimization note] The configured run exceeds the physical gate. Check likely causes:")
            lines.append("1. 3D FEM geometry and empymod air-earth geometry are inconsistent.")
            lines.append("2. Source normalization is inconsistent.")
            lines.append("3. empymod finite-length source integration needs more srcpts.")
            lines.append("4. Source or receiver lies exactly on the z=0 interface.")
            lines.append("5. Local mesh refinement around source/receiver is insufficient.")
            lines.append("6. Outer boundary is too close.")
            lines.append("7. Ramp-off waveform discretization is too coarse.")
            lines.append("8. dBz/dt extraction mode is inconsistent.")
            lines.append("9. Relative error is amplified where the reference value is near zero.")
            lines.append("")
            lines.append("Second-run suggestions:")
            lines.append("  - Use --force-mesh and reduce source/receiver local SizeMin.")
            lines.append("  - Increase --x-extent/--y-extent/--earth-depth/--air-height to audit boundary effects.")
            lines.append("  - Increase --wire-radius or refine the source region to improve source-volume integration.")
            lines.append("  - Reduce --time-growth or shorten the first-stage window to audit ramp-off discretization.")

    if False and max_error > config.error_tolerance:
        lines.append("")
        if configured_physical_window["passed"]:
            weak_components = [
                comp
                for comp, values in errors.items()
                if float(values["max"]) > float(config.error_tolerance)
                and comp not in {"Ex", "dBzdt"}
            ]
            lines.append("【指标提示】物理验收指标已在当前误差窗口内通过；仍有弱分量相对误差超过阈值。")
            lines.append(
                "  - 物理指标采用 Ex、dBzdt 和 Eh_vector；Eh_vector 避免单个近零横向分量放大相对误差。"
            )
            if weak_components:
                lines.append(f"  - 弱分量相对误差超过阈值: {', '.join(weak_components)}。")
            lines.append("  - 若验收要求逐分量相对误差，请为近零参考分量同时设置绝对误差或矢量范数口径。")
        elif passing_window is not None:
            lines.append("【优化提示】当前误差窗口未全通过；严格物理指标需要从更晚时间开始统计。")
            lines.append(
                f"  - 当前配置统计起点: t >= {configured_error_start:.6e} s；"
                f"最早通过起点: t >= {passing_window['time_min']:.6e} s。"
            )
            lines.append("  - 如必须覆盖更早时间，请继续检查 ramp-off 离散、初始场和早期步长。")
        else:
            lines.append("【优化提示】当前误差超标，请检查以下可能原因：")
            lines.append("1. 3D FEM 几何与 empymod 的 air-earth 几何不一致；")
            lines.append("2. 源项归一化不一致；")
            lines.append("3. empymod 有限长源未设置 srcpts>=3；")
            lines.append("4. 源或接收点落在 z=0 界面；")
            lines.append("5. 网格局部加密不足；")
            lines.append("6. 外边界太近；")
            lines.append("7. ramp-off 波形离散不够平滑；")
            lines.append("8. dBz/dt 提取方式不一致；")
            lines.append("9. 相对误差在参考值接近零处被放大。")
            lines.append("")
            lines.append("二次运行建议：")
            lines.append("  - 使用 --force-mesh 重新生成网格，并减小源/接收点附近 SizeMin；")
            lines.append("  - 增大 --x-extent/--y-extent/--earth-depth/--air-height 检查边界影响；")
            lines.append("  - 增大 --wire-radius 或细化源附近网格，确认正则化体源体积分辨充分；")
            lines.append("  - 减小 --time-growth 或缩短第一阶段时间窗核对 ramp-off 离散。")

    config.output_report().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] saved {config.output_report()}", flush=True)


def _save_npz(config: PipelineConfig, fem_result, ref_result, errors) -> None:
    import numpy as np

    payload = {
        "times": fem_result["times"],
        "fem": fem_result["data"],
        "empymod": ref_result["data"],
        "components": np.asarray(fem_result["components"]),
    }
    for comp, values in errors.items():
        payload[f"error_relative_{comp}"] = values["relative"]
        payload[f"error_absolute_{comp}"] = values["absolute"]
    weak_window = check_weak_component_error_window(
        fem_result["times"],
        fem_result["data"],
        ref_result["data"],
        fem_result["components"],
        error_min_time=float(config.error_min_time) if float(config.error_min_time) > 0.0 else float(fem_result["times"][0]),
        tolerance=float(config.error_tolerance),
        weak_reference_fraction=float(config.weak_component_reference_fraction),
    )
    payload["weak_component_names"] = np.asarray(weak_window["weak_components"])
    payload["weak_component_scaled_abs_error_max"] = np.asarray(
        [weak_window["maxima"][comp] for comp in weak_window["weak_components"]],
        dtype=float,
    )
    payload["weak_component_reference_max"] = np.asarray(
        [weak_window["reference_maxima"][comp] for comp in weak_window["weak_components"]],
        dtype=float,
    )
    payload["weak_component_primary_scale"] = np.asarray([weak_window["primary_scale"]], dtype=float)
    payload["weak_component_passed"] = np.asarray([weak_window["passed"]], dtype=bool)
    np.savez(config.output_npz(), **payload)
    print(f"[data] saved {config.output_npz()}", flush=True)


def _receiver_diagnostic_payload(receiver_diagnostic_rows):
    import numpy as np

    rows = list(receiver_diagnostic_rows or [])
    if not rows:
        return {
            "receiver_diagnostic_times": np.empty(0, dtype=float),
            "receiver_diagnostic_types": np.asarray([], dtype="<U1"),
            "receiver_diagnostic_radii": np.empty(0, dtype=float),
            "receiver_diagnostic_values": np.empty((0, 4), dtype=float),
            "receiver_diagnostic_dbdt_curl": np.empty(0, dtype=float),
            "receiver_diagnostic_dbdt_biot_rate": np.empty(0, dtype=float),
            "receiver_diagnostic_sample_counts": np.empty(0, dtype=int),
            "receiver_diagnostic_candidate_count_min": np.empty(0, dtype=int),
            "receiver_diagnostic_candidate_count_max": np.empty(0, dtype=int),
            "receiver_diagnostic_candidate_count_mean": np.empty(0, dtype=float),
            "receiver_diagnostic_multi_candidate_sample_count": np.empty(0, dtype=int),
            "receiver_diagnostic_candidate_center_distance_min": np.empty(0, dtype=float),
            "receiver_diagnostic_candidate_center_distance_max": np.empty(0, dtype=float),
            "receiver_diagnostic_candidate_center_distance_mean": np.empty(0, dtype=float),
            "receiver_diagnostic_selected_center_distance_mean": np.empty(0, dtype=float),
            "receiver_diagnostic_selected_center_distance_max": np.empty(0, dtype=float),
            "receiver_diagnostic_candidate_center_z_min": np.empty(0, dtype=float),
            "receiver_diagnostic_candidate_center_z_max": np.empty(0, dtype=float),
            "receiver_diagnostic_selected_center_z_mean": np.empty(0, dtype=float),
        }
    return {
        "receiver_diagnostic_times": np.asarray([float(row["time_obs"]) for row in rows], dtype=float),
        "receiver_diagnostic_types": np.asarray([str(row["receiver_type"]) for row in rows]),
        "receiver_diagnostic_radii": np.asarray([float(row.get("radius", np.nan)) for row in rows], dtype=float),
        "receiver_diagnostic_values": np.asarray(
            [
                [
                    float(row.get("Ex", np.nan)),
                    float(row.get("Ey", np.nan)),
                    float(row.get("Hz", np.nan)),
                    float(row.get("dBzdt", np.nan)),
                ]
                for row in rows
            ],
            dtype=float,
        ),
        "receiver_diagnostic_dbdt_curl": np.asarray(
            [float(row.get("dBzdt_curl", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_dbdt_biot_rate": np.asarray(
            [float(row.get("dBzdt_biot_rate", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_sample_counts": np.asarray(
            [int(row.get("sample_count", 0)) for row in rows], dtype=int
        ),
        "receiver_diagnostic_candidate_count_min": np.asarray(
            [int(row.get("candidate_count_min", 0)) for row in rows], dtype=int
        ),
        "receiver_diagnostic_candidate_count_max": np.asarray(
            [int(row.get("candidate_count_max", 0)) for row in rows], dtype=int
        ),
        "receiver_diagnostic_candidate_count_mean": np.asarray(
            [float(row.get("candidate_count_mean", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_multi_candidate_sample_count": np.asarray(
            [int(row.get("multi_candidate_sample_count", 0)) for row in rows], dtype=int
        ),
        "receiver_diagnostic_candidate_center_distance_min": np.asarray(
            [float(row.get("candidate_center_distance_min", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_candidate_center_distance_max": np.asarray(
            [float(row.get("candidate_center_distance_max", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_candidate_center_distance_mean": np.asarray(
            [float(row.get("candidate_center_distance_mean", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_selected_center_distance_mean": np.asarray(
            [float(row.get("selected_center_distance_mean", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_selected_center_distance_max": np.asarray(
            [float(row.get("selected_center_distance_max", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_candidate_center_z_min": np.asarray(
            [float(row.get("candidate_center_z_min", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_candidate_center_z_max": np.asarray(
            [float(row.get("candidate_center_z_max", np.nan)) for row in rows], dtype=float
        ),
        "receiver_diagnostic_selected_center_z_mean": np.asarray(
            [float(row.get("selected_center_z_mean", np.nan)) for row in rows], dtype=float
        ),
    }


def _receiver_diagnostic_rows_from_payload(payload) -> list[dict[str, Any]]:
    import numpy as np

    if "receiver_diagnostic_times" not in payload.files:
        return []
    times = np.asarray(payload["receiver_diagnostic_times"], dtype=float)
    types = [str(item) for item in np.asarray(payload["receiver_diagnostic_types"]).tolist()]
    radii = np.asarray(payload["receiver_diagnostic_radii"], dtype=float)
    values = np.asarray(payload["receiver_diagnostic_values"], dtype=float)
    dbdt_curl = np.asarray(payload["receiver_diagnostic_dbdt_curl"], dtype=float) if "receiver_diagnostic_dbdt_curl" in payload.files else np.full(times.size, np.nan, dtype=float)
    dbdt_biot_rate = np.asarray(payload["receiver_diagnostic_dbdt_biot_rate"], dtype=float) if "receiver_diagnostic_dbdt_biot_rate" in payload.files else np.full(times.size, np.nan, dtype=float)
    sample_counts = np.asarray(payload["receiver_diagnostic_sample_counts"], dtype=int) if "receiver_diagnostic_sample_counts" in payload.files else np.zeros(times.size, dtype=int)
    candidate_min = np.asarray(payload["receiver_diagnostic_candidate_count_min"], dtype=int) if "receiver_diagnostic_candidate_count_min" in payload.files else np.zeros(times.size, dtype=int)
    candidate_max = np.asarray(payload["receiver_diagnostic_candidate_count_max"], dtype=int) if "receiver_diagnostic_candidate_count_max" in payload.files else np.zeros(times.size, dtype=int)
    candidate_mean = np.asarray(payload["receiver_diagnostic_candidate_count_mean"], dtype=float) if "receiver_diagnostic_candidate_count_mean" in payload.files else np.full(times.size, np.nan, dtype=float)
    multi_candidate = np.asarray(payload["receiver_diagnostic_multi_candidate_sample_count"], dtype=int) if "receiver_diagnostic_multi_candidate_sample_count" in payload.files else np.zeros(times.size, dtype=int)
    distance_min = np.asarray(payload["receiver_diagnostic_candidate_center_distance_min"], dtype=float) if "receiver_diagnostic_candidate_center_distance_min" in payload.files else np.full(times.size, np.nan, dtype=float)
    distance_max = np.asarray(payload["receiver_diagnostic_candidate_center_distance_max"], dtype=float) if "receiver_diagnostic_candidate_center_distance_max" in payload.files else np.full(times.size, np.nan, dtype=float)
    distance_mean = np.asarray(payload["receiver_diagnostic_candidate_center_distance_mean"], dtype=float) if "receiver_diagnostic_candidate_center_distance_mean" in payload.files else np.full(times.size, np.nan, dtype=float)
    selected_distance_mean = np.asarray(payload["receiver_diagnostic_selected_center_distance_mean"], dtype=float) if "receiver_diagnostic_selected_center_distance_mean" in payload.files else np.full(times.size, np.nan, dtype=float)
    selected_distance_max = np.asarray(payload["receiver_diagnostic_selected_center_distance_max"], dtype=float) if "receiver_diagnostic_selected_center_distance_max" in payload.files else np.full(times.size, np.nan, dtype=float)
    center_z_min = np.asarray(payload["receiver_diagnostic_candidate_center_z_min"], dtype=float) if "receiver_diagnostic_candidate_center_z_min" in payload.files else np.full(times.size, np.nan, dtype=float)
    center_z_max = np.asarray(payload["receiver_diagnostic_candidate_center_z_max"], dtype=float) if "receiver_diagnostic_candidate_center_z_max" in payload.files else np.full(times.size, np.nan, dtype=float)
    selected_z_mean = np.asarray(payload["receiver_diagnostic_selected_center_z_mean"], dtype=float) if "receiver_diagnostic_selected_center_z_mean" in payload.files else np.full(times.size, np.nan, dtype=float)
    rows = []
    for i, time_obs in enumerate(times):
        rows.append(
            {
                "time_obs": float(time_obs),
                "receiver_type": types[i] if i < len(types) else "",
                "radius": float(radii[i]) if i < radii.size else float("nan"),
                "Ex": float(values[i, 0]) if i < values.shape[0] else float("nan"),
                "Ey": float(values[i, 1]) if i < values.shape[0] else float("nan"),
                "Hz": float(values[i, 2]) if i < values.shape[0] else float("nan"),
                "dBzdt": float(values[i, 3]) if i < values.shape[0] else float("nan"),
                "dBzdt_curl": float(dbdt_curl[i]) if i < dbdt_curl.size else float("nan"),
                "dBzdt_biot_rate": float(dbdt_biot_rate[i]) if i < dbdt_biot_rate.size else float("nan"),
                "sample_count": int(sample_counts[i]) if i < sample_counts.size else 0,
                "candidate_count_min": int(candidate_min[i]) if i < candidate_min.size else 0,
                "candidate_count_max": int(candidate_max[i]) if i < candidate_max.size else 0,
                "candidate_count_mean": float(candidate_mean[i]) if i < candidate_mean.size else float("nan"),
                "multi_candidate_sample_count": int(multi_candidate[i]) if i < multi_candidate.size else 0,
                "candidate_center_distance_min": float(distance_min[i]) if i < distance_min.size else float("nan"),
                "candidate_center_distance_max": float(distance_max[i]) if i < distance_max.size else float("nan"),
                "candidate_center_distance_mean": float(distance_mean[i]) if i < distance_mean.size else float("nan"),
                "selected_center_distance_mean": float(selected_distance_mean[i]) if i < selected_distance_mean.size else float("nan"),
                "selected_center_distance_max": float(selected_distance_max[i]) if i < selected_distance_max.size else float("nan"),
                "candidate_center_z_min": float(center_z_min[i]) if i < center_z_min.size else float("nan"),
                "candidate_center_z_max": float(center_z_max[i]) if i < center_z_max.size else float("nan"),
                "selected_center_z_mean": float(selected_z_mean[i]) if i < selected_z_mean.size else float("nan"),
            }
        )
    return rows


def _write_receiver_diagnostics_csv(config: PipelineConfig, receiver_diagnostic_rows) -> None:
    rows = list(receiver_diagnostic_rows or [])
    path = config.receiver_diagnostics_csv()
    if not rows:
        if path.exists():
            path.unlink()
        png_path = config.receiver_diagnostics_png()
        if png_path.exists():
            png_path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_obs",
        "receiver_type",
        "radius",
        "Ex",
        "Ey",
        "Hz",
        "dBzdt",
        "dBzdt_curl",
        "dBzdt_biot_rate",
        "sample_count",
        "candidate_count_min",
        "candidate_count_max",
        "candidate_count_mean",
        "multi_candidate_sample_count",
        "candidate_center_distance_min",
        "candidate_center_distance_max",
        "candidate_center_distance_mean",
        "selected_center_distance_mean",
        "selected_center_distance_max",
        "candidate_center_z_min",
        "candidate_center_z_max",
        "selected_center_z_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time_obs": float(row.get("time_obs", math.nan)),
                    "receiver_type": str(row.get("receiver_type", "")),
                    "radius": float(row.get("radius", math.nan)),
                    "Ex": float(row.get("Ex", math.nan)),
                    "Ey": float(row.get("Ey", math.nan)),
                    "Hz": float(row.get("Hz", math.nan)),
                    "dBzdt": float(row.get("dBzdt", math.nan)),
                    "dBzdt_curl": float(row.get("dBzdt_curl", math.nan)),
                    "dBzdt_biot_rate": float(row.get("dBzdt_biot_rate", math.nan)),
                    "sample_count": int(row.get("sample_count", 0)),
                    "candidate_count_min": int(row.get("candidate_count_min", 0)),
                    "candidate_count_max": int(row.get("candidate_count_max", 0)),
                    "candidate_count_mean": float(row.get("candidate_count_mean", math.nan)),
                    "multi_candidate_sample_count": int(row.get("multi_candidate_sample_count", 0)),
                    "candidate_center_distance_min": float(row.get("candidate_center_distance_min", math.nan)),
                    "candidate_center_distance_max": float(row.get("candidate_center_distance_max", math.nan)),
                    "candidate_center_distance_mean": float(row.get("candidate_center_distance_mean", math.nan)),
                    "selected_center_distance_mean": float(row.get("selected_center_distance_mean", math.nan)),
                    "selected_center_distance_max": float(row.get("selected_center_distance_max", math.nan)),
                    "candidate_center_z_min": float(row.get("candidate_center_z_min", math.nan)),
                    "candidate_center_z_max": float(row.get("candidate_center_z_max", math.nan)),
                    "selected_center_z_mean": float(row.get("selected_center_z_mean", math.nan)),
                }
            )


def _plot_receiver_diagnostics(config: PipelineConfig, receiver_diagnostic_rows) -> None:
    rows = list(receiver_diagnostic_rows or [])
    path = config.receiver_diagnostics_png()
    if not rows:
        if path.exists():
            path.unlink()
        return

    import numpy as np

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    components = ["Ex", "Ey", "dBzdt"]
    if any(math.isfinite(float(row.get("Hz", math.nan))) for row in rows):
        components.insert(2, "Hz")
    receiver_types = []
    for row in rows:
        receiver_type = str(row.get("receiver_type", ""))
        if receiver_type and receiver_type not in receiver_types:
            receiver_types.append(receiver_type)

    fig, axes = plt.subplots(len(components), 1, figsize=(7.2, 2.4 * len(components) + 1.0), sharex=True)
    axes_arr = np.atleast_1d(axes)
    markers = ["o", "s", "^", "d", "x"]
    for ax, component in zip(axes_arr, components):
        for i, receiver_type in enumerate(receiver_types):
            subset = [row for row in rows if str(row.get("receiver_type", "")) == receiver_type]
            subset.sort(key=lambda row: float(row.get("time_obs", math.nan)))
            times = [float(row.get("time_obs", math.nan)) for row in subset]
            values = [float(row.get(component, math.nan)) for row in subset]
            finite_pairs = [(t, value) for t, value in zip(times, values) if math.isfinite(t) and math.isfinite(value)]
            if not finite_pairs:
                continue
            ax.plot(
                [item[0] for item in finite_pairs],
                [item[1] for item in finite_pairs],
                marker=markers[i % len(markers)],
                label=receiver_type,
            )
        finite_times = [float(row.get("time_obs", math.nan)) for row in rows if math.isfinite(float(row.get("time_obs", math.nan)))]
        if len(set(finite_times)) > 1 and min(finite_times) > 0.0:
            ax.set_xscale("log")
        ax.set_ylabel(component)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    axes_arr[-1].set_xlabel("t_obs (s)")
    fig.suptitle("Receiver diagnostics: point and averaged receivers")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_forward_partial(config: PipelineConfig, times, rows, components, solver_log, receiver_diagnostic_rows=None) -> None:
    import numpy as np

    path = config.forward_partial_npz()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_arr = np.asarray(rows, dtype=float)
    if rows_arr.size == 0:
        rows_arr = np.empty((0, len(components)), dtype=float)
    else:
        rows_arr = rows_arr.reshape(-1, len(components))
    n_rows = rows_arr.shape[0]
    output_logs = [item for item in solver_log if item.get("is_output", True)][-n_rows:]
    payload = {
        "times": _completed_return_times(times, rows_arr),
        "fem": rows_arr,
        "components": np.asarray(components),
        "solver_steps": np.asarray([int(item.get("step", -1)) for item in output_logs], dtype=int),
        "solver_iterations": np.asarray([int(item.get("its", -1)) for item in output_logs], dtype=int),
        "solver_residuals": np.asarray([float(item.get("residual", np.nan)) for item in output_logs], dtype=float),
        "solver_reasons": np.asarray([int(item.get("reason", 0)) for item in output_logs], dtype=int),
        "solver_divergence_clean_before": np.asarray(
            [float(item.get("divergence_clean_before", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_clean_after": np.asarray(
            [float(item.get("divergence_clean_after", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_clean_correction_norm": np.asarray(
            [float(item.get("divergence_clean_correction_norm", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_clean_applied_correction_norm": np.asarray(
            [float(item.get("divergence_clean_applied_correction_norm", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_clean_strength": np.asarray(
            [float(item.get("divergence_clean_strength", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_control_applied": np.asarray(
            [bool(item.get("divergence_control_applied", False)) for item in output_logs],
            dtype=bool,
        ),
        "solver_divergence_control_scale": np.asarray(
            [str(item.get("divergence_control_scale", "")) for item in output_logs],
        ),
        "solver_divergence_control_weight": np.asarray(
            [float(item.get("divergence_control_weight", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_control_applied_weight": np.asarray(
            [float(item.get("divergence_control_applied_weight", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_control_reference_norm": np.asarray(
            [float(item.get("divergence_control_reference_norm", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_control_matrix_norm": np.asarray(
            [float(item.get("divergence_control_matrix_norm", np.nan)) for item in output_logs],
            dtype=float,
        ),
        "solver_divergence_control_relative_weight": np.asarray(
            [float(item.get("divergence_control_relative_weight", np.nan)) for item in output_logs],
            dtype=float,
        ),
    }
    if receiver_diagnostic_rows:
        payload.update(_receiver_diagnostic_payload(receiver_diagnostic_rows))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
    tmp_path.replace(path)
    _write_receiver_diagnostics_csv(config, receiver_diagnostic_rows)
    _plot_receiver_diagnostics(config, receiver_diagnostic_rows)


def _completed_return_times(return_times, rows):
    import numpy as np

    rows_arr = np.asarray(rows, dtype=float)
    if rows_arr.size == 0:
        n_rows = 0
    else:
        n_rows = int(rows_arr.reshape(rows_arr.shape[0], -1).shape[0])
    return np.asarray(return_times, dtype=float)[:n_rows]


def _solver_log_to_arrays(solver_log):
    import numpy as np

    return {
        "solver_steps": np.asarray([int(item.get("step", -1)) for item in solver_log], dtype=int),
        "solver_times": np.asarray([float(item.get("time", np.nan)) for item in solver_log], dtype=float),
        "solver_observation_times": np.asarray(
            [float(item.get("observation_time", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_dt": np.asarray([float(item.get("dt", np.nan)) for item in solver_log], dtype=float),
        "solver_iterations": np.asarray([int(item.get("its", -1)) for item in solver_log], dtype=int),
        "solver_residuals": np.asarray([float(item.get("residual", np.nan)) for item in solver_log], dtype=float),
        "solver_reasons": np.asarray([int(item.get("reason", 0)) for item in solver_log], dtype=int),
        "solver_is_output": np.asarray([bool(item.get("is_output", False)) for item in solver_log], dtype=bool),
        "solver_time_theta": np.asarray([float(item.get("time_theta", np.nan)) for item in solver_log], dtype=float),
        "solver_divergence_clean_before": np.asarray(
            [float(item.get("divergence_clean_before", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_clean_after": np.asarray(
            [float(item.get("divergence_clean_after", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_clean_correction_norm": np.asarray(
            [float(item.get("divergence_clean_correction_norm", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_clean_applied_correction_norm": np.asarray(
            [float(item.get("divergence_clean_applied_correction_norm", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_clean_strength": np.asarray(
            [float(item.get("divergence_clean_strength", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_control_applied": np.asarray(
            [bool(item.get("divergence_control_applied", False)) for item in solver_log],
            dtype=bool,
        ),
        "solver_divergence_control_scale": np.asarray(
            [str(item.get("divergence_control_scale", "")) for item in solver_log],
        ),
        "solver_divergence_control_weight": np.asarray(
            [float(item.get("divergence_control_weight", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_control_applied_weight": np.asarray(
            [float(item.get("divergence_control_applied_weight", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_control_reference_norm": np.asarray(
            [float(item.get("divergence_control_reference_norm", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_control_matrix_norm": np.asarray(
            [float(item.get("divergence_control_matrix_norm", np.nan)) for item in solver_log],
            dtype=float,
        ),
        "solver_divergence_control_relative_weight": np.asarray(
            [float(item.get("divergence_control_relative_weight", np.nan)) for item in solver_log],
            dtype=float,
        ),
    }


def _solver_log_from_arrays(payload) -> list[dict[str, Any]]:
    import numpy as np

    if "solver_steps" not in payload.files:
        return []
    steps = np.asarray(payload["solver_steps"], dtype=int)
    times = np.asarray(payload["solver_times"], dtype=float) if "solver_times" in payload.files else np.full(steps.size, np.nan)
    observation_times = (
        np.asarray(payload["solver_observation_times"], dtype=float)
        if "solver_observation_times" in payload.files
        else np.full(steps.size, np.nan)
    )
    dts = np.asarray(payload["solver_dt"], dtype=float) if "solver_dt" in payload.files else np.full(steps.size, np.nan)
    iterations = (
        np.asarray(payload["solver_iterations"], dtype=int)
        if "solver_iterations" in payload.files
        else np.full(steps.size, -1)
    )
    residuals = (
        np.asarray(payload["solver_residuals"], dtype=float)
        if "solver_residuals" in payload.files
        else np.full(steps.size, np.nan)
    )
    reasons = np.asarray(payload["solver_reasons"], dtype=int) if "solver_reasons" in payload.files else np.zeros(steps.size, dtype=int)
    is_output = (
        np.asarray(payload["solver_is_output"], dtype=bool)
        if "solver_is_output" in payload.files
        else np.zeros(steps.size, dtype=bool)
    )
    time_theta = (
        np.asarray(payload["solver_time_theta"], dtype=float)
        if "solver_time_theta" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_before = (
        np.asarray(payload["solver_divergence_clean_before"], dtype=float)
        if "solver_divergence_clean_before" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_after = (
        np.asarray(payload["solver_divergence_clean_after"], dtype=float)
        if "solver_divergence_clean_after" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_correction = (
        np.asarray(payload["solver_divergence_clean_correction_norm"], dtype=float)
        if "solver_divergence_clean_correction_norm" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_applied = (
        np.asarray(payload["solver_divergence_clean_applied_correction_norm"], dtype=float)
        if "solver_divergence_clean_applied_correction_norm" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_strength = (
        np.asarray(payload["solver_divergence_clean_strength"], dtype=float)
        if "solver_divergence_clean_strength" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_control_applied = (
        np.asarray(payload["solver_divergence_control_applied"], dtype=bool)
        if "solver_divergence_control_applied" in payload.files
        else np.zeros(steps.size, dtype=bool)
    )
    div_control_scale = (
        np.asarray(payload["solver_divergence_control_scale"]).astype(str)
        if "solver_divergence_control_scale" in payload.files
        else np.full(steps.size, "", dtype="<U1")
    )
    div_control_weight = (
        np.asarray(payload["solver_divergence_control_weight"], dtype=float)
        if "solver_divergence_control_weight" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_control_applied_weight = (
        np.asarray(payload["solver_divergence_control_applied_weight"], dtype=float)
        if "solver_divergence_control_applied_weight" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_control_reference_norm = (
        np.asarray(payload["solver_divergence_control_reference_norm"], dtype=float)
        if "solver_divergence_control_reference_norm" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_control_matrix_norm = (
        np.asarray(payload["solver_divergence_control_matrix_norm"], dtype=float)
        if "solver_divergence_control_matrix_norm" in payload.files
        else np.full(steps.size, np.nan)
    )
    div_control_relative_weight = (
        np.asarray(payload["solver_divergence_control_relative_weight"], dtype=float)
        if "solver_divergence_control_relative_weight" in payload.files
        else np.full(steps.size, np.nan)
    )
    out: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        item = {
            "step": int(step),
            "time": float(times[i]) if i < times.size else float("nan"),
            "dt": float(dts[i]) if i < dts.size else float("nan"),
            "its": int(iterations[i]) if i < iterations.size else -1,
            "residual": float(residuals[i]) if i < residuals.size else float("nan"),
            "reason": int(reasons[i]) if i < reasons.size else 0,
            "is_output": bool(is_output[i]) if i < is_output.size else False,
            "time_theta": float(time_theta[i]) if i < time_theta.size else float("nan"),
        }
        if i < observation_times.size and np.isfinite(observation_times[i]):
            item["observation_time"] = float(observation_times[i])
        if i < div_before.size and np.isfinite(div_before[i]):
            item["divergence_clean_before"] = float(div_before[i])
        if i < div_after.size and np.isfinite(div_after[i]):
            item["divergence_clean_after"] = float(div_after[i])
        if i < div_correction.size and np.isfinite(div_correction[i]):
            item["divergence_clean_correction_norm"] = float(div_correction[i])
        if i < div_applied.size and np.isfinite(div_applied[i]):
            item["divergence_clean_applied_correction_norm"] = float(div_applied[i])
        if i < div_strength.size and np.isfinite(div_strength[i]):
            item["divergence_clean_strength"] = float(div_strength[i])
        if i < div_control_applied.size:
            item["divergence_control_applied"] = bool(div_control_applied[i])
        if i < div_control_scale.size and str(div_control_scale[i]):
            item["divergence_control_scale"] = str(div_control_scale[i])
        if i < div_control_weight.size and np.isfinite(div_control_weight[i]):
            item["divergence_control_weight"] = float(div_control_weight[i])
        if i < div_control_applied_weight.size and np.isfinite(div_control_applied_weight[i]):
            item["divergence_control_applied_weight"] = float(div_control_applied_weight[i])
        if i < div_control_reference_norm.size and np.isfinite(div_control_reference_norm[i]):
            item["divergence_control_reference_norm"] = float(div_control_reference_norm[i])
        if i < div_control_matrix_norm.size and np.isfinite(div_control_matrix_norm[i]):
            item["divergence_control_matrix_norm"] = float(div_control_matrix_norm[i])
        if i < div_control_relative_weight.size and np.isfinite(div_control_relative_weight[i]):
            item["divergence_control_relative_weight"] = float(div_control_relative_weight[i])
        out.append(item)
    return out


def _save_forward_checkpoint(
    config: PipelineConfig,
    *,
    completed_step: int,
    previous_time: float,
    E_old,
    memories,
    rows,
    components,
    solver_log,
    h_old_receiver=None,
    receiver_diagnostic_rows=None,
) -> None:
    import numpy as np

    path = config.forward_checkpoint_npz()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_arr = np.asarray(rows, dtype=float)
    if rows_arr.size == 0:
        rows_arr = np.empty((0, len(components)), dtype=float)
    else:
        rows_arr = rows_arr.reshape(-1, len(components))
    memory_arrays = [np.asarray(memory.x.array, dtype=float).copy() for memory in memories]
    payload = {
        "completed_step": np.asarray(int(completed_step), dtype=int),
        "previous_time": np.asarray(float(previous_time), dtype=float),
        "e_old": np.asarray(E_old.x.array, dtype=float).copy(),
        "memories": np.asarray(memory_arrays, dtype=float)
        if memory_arrays
        else np.empty((0, E_old.x.array.size), dtype=float),
        "rows": rows_arr,
        "components": np.asarray(components),
        "h_old_receiver": np.asarray(h_old_receiver, dtype=float)
        if h_old_receiver is not None
        else np.full(3, np.nan, dtype=float),
    }
    if receiver_diagnostic_rows:
        payload.update(_receiver_diagnostic_payload(receiver_diagnostic_rows))
    payload.update(_solver_log_to_arrays(solver_log))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
    tmp_path.replace(path)


def _load_forward_checkpoint(config: PipelineConfig):
    import numpy as np

    path = config.forward_checkpoint_npz()
    if not path.is_file():
        raise FileNotFoundError(f"forward checkpoint file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    required = {"completed_step", "previous_time", "e_old", "memories", "rows", "components"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"forward checkpoint {path} is missing keys: {', '.join(missing)}")
    return {
        "completed_step": int(np.asarray(payload["completed_step"]).item()),
        "previous_time": float(np.asarray(payload["previous_time"]).item()),
        "e_old": np.asarray(payload["e_old"], dtype=float),
        "memories": np.asarray(payload["memories"], dtype=float),
        "rows": np.asarray(payload["rows"], dtype=float),
        "components": [str(item) for item in np.asarray(payload["components"]).tolist()],
        "solver_log": _solver_log_from_arrays(payload),
        "h_old_receiver": np.asarray(payload["h_old_receiver"], dtype=float)
        if "h_old_receiver" in payload.files
        else np.full(3, np.nan, dtype=float),
        "receiver_diagnostic_rows": _receiver_diagnostic_rows_from_payload(payload),
    }


def _load_forward_partial(config: PipelineConfig):
    import numpy as np

    path = config.forward_partial_npz()
    if not path.is_file():
        raise FileNotFoundError(f"partial forward file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    required = {"times", "fem", "components"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"partial forward file {path} is missing keys: {', '.join(missing)}")
    times = np.asarray(payload["times"], dtype=float)
    data = np.asarray(payload["fem"], dtype=float)
    components = [str(item) for item in np.asarray(payload["components"]).tolist()]
    if times.ndim != 1:
        raise ValueError("partial forward times must be one-dimensional")
    if data.ndim != 2:
        raise ValueError("partial forward fem data must be two-dimensional")
    if data.shape[0] != times.size:
        raise ValueError("partial forward times and fem rows have inconsistent lengths")
    if data.shape[1] != len(components):
        raise ValueError("partial forward component count does not match fem columns")
    solver_log = []
    steps = np.asarray(payload["solver_steps"], dtype=int) if "solver_steps" in payload.files else np.full(times.size, -1)
    iterations = (
        np.asarray(payload["solver_iterations"], dtype=int)
        if "solver_iterations" in payload.files
        else np.full(times.size, -1)
    )
    residuals = (
        np.asarray(payload["solver_residuals"], dtype=float)
        if "solver_residuals" in payload.files
        else np.full(times.size, np.nan)
    )
    reasons = np.asarray(payload["solver_reasons"], dtype=int) if "solver_reasons" in payload.files else np.zeros(times.size, dtype=int)
    div_before = (
        np.asarray(payload["solver_divergence_clean_before"], dtype=float)
        if "solver_divergence_clean_before" in payload.files
        else np.full(times.size, np.nan)
    )
    div_after = (
        np.asarray(payload["solver_divergence_clean_after"], dtype=float)
        if "solver_divergence_clean_after" in payload.files
        else np.full(times.size, np.nan)
    )
    div_correction = (
        np.asarray(payload["solver_divergence_clean_correction_norm"], dtype=float)
        if "solver_divergence_clean_correction_norm" in payload.files
        else np.full(times.size, np.nan)
    )
    div_applied = (
        np.asarray(payload["solver_divergence_clean_applied_correction_norm"], dtype=float)
        if "solver_divergence_clean_applied_correction_norm" in payload.files
        else np.full(times.size, np.nan)
    )
    div_strength = (
        np.asarray(payload["solver_divergence_clean_strength"], dtype=float)
        if "solver_divergence_clean_strength" in payload.files
        else np.full(times.size, np.nan)
    )
    div_control_applied = (
        np.asarray(payload["solver_divergence_control_applied"], dtype=bool)
        if "solver_divergence_control_applied" in payload.files
        else np.zeros(times.size, dtype=bool)
    )
    div_control_scale = (
        np.asarray(payload["solver_divergence_control_scale"]).astype(str)
        if "solver_divergence_control_scale" in payload.files
        else np.full(times.size, "", dtype="<U1")
    )
    div_control_weight = (
        np.asarray(payload["solver_divergence_control_weight"], dtype=float)
        if "solver_divergence_control_weight" in payload.files
        else np.full(times.size, np.nan)
    )
    div_control_applied_weight = (
        np.asarray(payload["solver_divergence_control_applied_weight"], dtype=float)
        if "solver_divergence_control_applied_weight" in payload.files
        else np.full(times.size, np.nan)
    )
    div_control_reference_norm = (
        np.asarray(payload["solver_divergence_control_reference_norm"], dtype=float)
        if "solver_divergence_control_reference_norm" in payload.files
        else np.full(times.size, np.nan)
    )
    div_control_matrix_norm = (
        np.asarray(payload["solver_divergence_control_matrix_norm"], dtype=float)
        if "solver_divergence_control_matrix_norm" in payload.files
        else np.full(times.size, np.nan)
    )
    div_control_relative_weight = (
        np.asarray(payload["solver_divergence_control_relative_weight"], dtype=float)
        if "solver_divergence_control_relative_weight" in payload.files
        else np.full(times.size, np.nan)
    )
    for i, t in enumerate(times):
        item = {
            "step": int(steps[i]) if i < steps.size else -1,
            "time": float(t),
            "observation_time": float(t),
            "dt": float("nan"),
            "its": int(iterations[i]) if i < iterations.size else -1,
            "residual": float(residuals[i]) if i < residuals.size else float("nan"),
            "reason": int(reasons[i]) if i < reasons.size else 0,
            "is_output": True,
        }
        if i < div_before.size and np.isfinite(div_before[i]):
            item["divergence_clean_before"] = float(div_before[i])
        if i < div_after.size and np.isfinite(div_after[i]):
            item["divergence_clean_after"] = float(div_after[i])
        if i < div_correction.size and np.isfinite(div_correction[i]):
            item["divergence_clean_correction_norm"] = float(div_correction[i])
        if i < div_applied.size and np.isfinite(div_applied[i]):
            item["divergence_clean_applied_correction_norm"] = float(div_applied[i])
        if i < div_strength.size and np.isfinite(div_strength[i]):
            item["divergence_clean_strength"] = float(div_strength[i])
        if i < div_control_applied.size:
            item["divergence_control_applied"] = bool(div_control_applied[i])
        if i < div_control_scale.size and str(div_control_scale[i]):
            item["divergence_control_scale"] = str(div_control_scale[i])
        if i < div_control_weight.size and np.isfinite(div_control_weight[i]):
            item["divergence_control_weight"] = float(div_control_weight[i])
        if i < div_control_applied_weight.size and np.isfinite(div_control_applied_weight[i]):
            item["divergence_control_applied_weight"] = float(div_control_applied_weight[i])
        if i < div_control_reference_norm.size and np.isfinite(div_control_reference_norm[i]):
            item["divergence_control_reference_norm"] = float(div_control_reference_norm[i])
        if i < div_control_matrix_norm.size and np.isfinite(div_control_matrix_norm[i]):
            item["divergence_control_matrix_norm"] = float(div_control_matrix_norm[i])
        if i < div_control_relative_weight.size and np.isfinite(div_control_relative_weight[i]):
            item["divergence_control_relative_weight"] = float(div_control_relative_weight[i])
        solver_log.append(item)
    return {
        "times": times,
        "data": data,
        "components": components,
        "solver_log": solver_log,
        "receiver_diagnostic_rows": _receiver_diagnostic_rows_from_payload(payload),
    }


def postprocess_saved_forward(config: PipelineConfig, env: dict[str, str], *, ref_mode: str = "noip", runtime=None):
    """Postprocess an existing forward_partial.npz without rerunning FEM."""

    runtime = {} if runtime is None else dict(runtime)
    t0 = time.perf_counter()
    fem_result = _load_forward_partial(config)
    runtime.setdefault("forward_seconds", 0.0)
    runtime["setup_seconds"] = runtime.get("setup_seconds", 0.0)
    t_ref = time.perf_counter()
    ref_result = get_empymod_reference(fem_result["times"], config, mode=ref_mode)
    errors = compute_error(fem_result["data"], ref_result["data"], fem_result["components"])
    reference_audit_errors = None
    if int(config.reference_audit_srcpts) > 0 and int(config.reference_audit_srcpts) != int(config.empymod_srcpts):
        audit_ref_result = get_empymod_reference(
            fem_result["times"],
            config,
            mode=ref_mode,
            srcpts=int(config.reference_audit_srcpts),
        )
        reference_audit_errors = compute_error(ref_result["data"], audit_ref_result["data"], ref_result["components"])
    runtime["reference_seconds"] = time.perf_counter() - t_ref
    t_post = time.perf_counter()
    _save_npz(config, fem_result, ref_result, errors)
    plot_verification(fem_result["times"], fem_result["data"], ref_result["data"], errors, fem_result["components"], config)
    write_validation_artifacts(
        fem_result["times"],
        fem_result["data"],
        ref_result["data"],
        fem_result["components"],
        config,
        case_type="ip" if ref_mode == "cole-cole" else "noip",
        reference_type="empymod",
        source_info={"mode": f"postprocess_partial/{config.source_mode}"},
        receiver_diagnostic_rows=fem_result.get("receiver_diagnostic_rows"),
        solver_log=fem_result.get("solver_log"),
    )
    runtime["postprocess_seconds"] = time.perf_counter() - t_post
    runtime["total_seconds"] = time.perf_counter() - t0
    write_report(
        config,
        env,
        fem_result,
        ref_result,
        errors,
        {"mode": f"postprocess_partial/{config.source_mode}"},
        reference_audit_errors=reference_audit_errors,
        runtime=runtime,
    )
    return {
        "fem_result": fem_result,
        "ref_result": ref_result,
        "errors": errors,
        "reference_audit_errors": reference_audit_errors,
        "runtime": runtime,
    }


def _parse_float_csv(value: str | None, name: str) -> tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return ()
    items: list[float] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw == "":
            raise argparse.ArgumentTypeError(f"{name} contains an empty item")
        try:
            items.append(float(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} contains a non-numeric item: {raw!r}") from exc
    return tuple(items)


def _parse_string_csv(value: str | None) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return ()
    return tuple(raw.strip() for raw in str(value).split(",") if raw.strip())


def main(argv: list[str] | None = None) -> int:
    run_t0 = time.perf_counter()
    runtime: dict[str, float] = {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force-mesh", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--source-only", action="store_true", help="Assemble mesh/materials/source diagnostics and exit before time stepping.")
    parser.add_argument("--postprocess-partial", action="store_true", help="Postprocess workdir/forward_partial.npz without rerunning FEM.")
    parser.add_argument("--checkpoint-forward", action="store_true", help="Save forward_checkpoint.npz at each output time for long-run restart.")
    parser.add_argument("--resume-forward", action="store_true", help="Resume E-form forward modelling from workdir/forward_checkpoint.npz.")
    parser.add_argument("--stop-after-outputs", type=int, default=0, help="Stop after N newly completed output times and save a forward checkpoint; 0 disables segmented stopping.")
    parser.add_argument("--memory-limit-gb", type=float, default=32.0, help="Configured workstation memory budget for mesh/solver preflight.")
    parser.add_argument("--memory-safety-fraction", type=float, default=0.95, help="Fraction of memory-limit-gb allowed for the solver estimate.")
    parser.add_argument("--check-env-only", action="store_true")
    parser.add_argument("--no-install", action="store_true", help="Do not pip install missing non-core packages.")
    parser.add_argument("--source-mode", choices=["auto", "line", "manual_line", "regularized"], default="auto")
    parser.add_argument(
        "--source-projection-mode",
        choices=["charge_conserving", "raw"],
        default="charge_conserving",
        help="Use raw only for diagnostics; charge_conserving enforces endpoint balance.",
    )
    parser.add_argument("--source-rhs-sign", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--source-term-mode", choices=["impressed_current", "primary_dc"], default="impressed_current")
    parser.add_argument("--formulation", choices=["e", "h"], default="e")
    parser.add_argument("--initial-dc-mode", choices=["fem", "analytic_halfspace"], default="fem")
    parser.add_argument(
        "--magnetic-receiver-mode",
        choices=["curl", "biot_current", "biot_ohmic", "faraday_integrated"],
        default="curl",
    )
    parser.add_argument("--magnetic-dbdt-mode", choices=["curl", "biot_rate"], default="curl")
    parser.add_argument("--outer-boundary-mode", choices=["pec", "natural", "robin"], default="pec")
    parser.add_argument("--outer-boundary-robin-scale", type=float, default=1.0)
    parser.add_argument("--receiver-type", choices=["point", "volume_average", "disk_average"], default="point")
    parser.add_argument("--receiver-average-radius", type=float, default=2.0)
    parser.add_argument(
        "--receiver-diagnostic-types",
        default="",
        help="Comma-separated receiver diagnostics to write separately, e.g. point,disk_average.",
    )
    parser.add_argument(
        "--receiver-evaluation-mode",
        choices=["first_cell", "mean", "median", "nearest_center", "shallowest"],
        default="median",
    )
    parser.add_argument("--divergence-cleaning", choices=["none", "conductivity"], default="none")
    parser.add_argument(
        "--divergence-cleaning-strength",
        type=float,
        default=1.0,
        help="Fraction of the conductivity divergence-cleaning correction to apply; 1 keeps the existing full projection.",
    )
    parser.add_argument(
        "--divergence-cleaning-t-obs-min",
        type=float,
        default=0.0,
        help="Only apply conductivity divergence cleaning after this post-ramp observation time in seconds.",
    )
    parser.add_argument(
        "--divergence-control-weight",
        type=float,
        default=0.0,
        help="Weight for implicit conductivity weak-divergence control; 0 disables this diagnostic term.",
    )
    parser.add_argument(
        "--divergence-control-t-obs-min",
        type=float,
        default=0.0,
        help="Only add implicit divergence-control after this post-ramp observation time in seconds.",
    )
    parser.add_argument(
        "--divergence-control-scale",
        choices=["absolute", "mass", "stiffness", "lhs"],
        default="absolute",
        help="Scale divergence-control weight as an absolute matrix coefficient or as a relative mass/stiffness/LHS fraction.",
    )
    parser.add_argument("--polarization", choices=["none", "cole-cole"], default="none")
    parser.add_argument("--cole-layer-top", type=float, default=0.0, help="Top depth of the polarizable Cole-Cole interval in meters.")
    parser.add_argument(
        "--cole-layer-bottom",
        type=float,
        default=float("inf"),
        help="Bottom depth of the polarizable Cole-Cole interval in meters; default applies Cole-Cole to all earth cells.",
    )
    parser.add_argument("--cole-rho0", type=float, default=100.0)
    parser.add_argument("--cole-m", type=float, default=0.2)
    parser.add_argument("--cole-tau", type=float, default=0.1)
    parser.add_argument("--cole-c", type=float, default=0.6)
    parser.add_argument("--cole-n-terms", type=int, default=10)
    parser.add_argument("--cole-f-min", type=float, default=1.0e-2)
    parser.add_argument("--cole-f-max", type=float, default=1.0e5)
    parser.add_argument("--cole-n-freq", type=int, default=96)
    parser.add_argument("--rho-air", type=float, default=1.0e8)
    parser.add_argument("--rho-earth", type=float, default=100.0)
    parser.add_argument("--source-start-x", type=float, default=-500.0)
    parser.add_argument("--source-start-y", type=float, default=200.0)
    parser.add_argument("--source-start-z", type=float, default=-0.1)
    parser.add_argument("--source-end-x", type=float, default=500.0)
    parser.add_argument("--source-end-y", type=float, default=200.0)
    parser.add_argument("--source-end-z", type=float, default=-0.1)
    parser.add_argument("--source-current", type=float, default=10.0)
    parser.add_argument("--ramp-off-time", type=float, default=1.0e-5)
    parser.add_argument(
        "--observation-times",
        default="",
        help="Comma-separated explicit output times; empty uses the geometric growth grid.",
    )
    parser.add_argument("--receiver-x", type=float, default=0.0)
    parser.add_argument("--receiver-y", type=float, default=-300.0)
    parser.add_argument("--receiver-z", type=float, default=-0.1)
    parser.add_argument("--source-mesh-size", type=float, default=5.0)
    parser.add_argument("--source-refinement-radius", type=float, default=100.0)
    parser.add_argument("--source-quadrature-points", type=int, default=0, help="Override manual line-source Gauss points; 0 keeps the automatic rule.")
    parser.add_argument("--receiver-mesh-size", type=float, default=10.0)
    parser.add_argument(
        "--receiver-anchor-mesh-size",
        type=float,
        default=0.0,
        help="Optional local receiver anchoring mesh size in metres; 0 reuses --receiver-mesh-size.",
    )
    parser.add_argument("--receiver-refinement-radius", type=float, default=60.0)
    parser.add_argument("--diffusion-refinement-factor", type=float, default=0.0, help="If >0, expand the 80 m late-diffusion refinement box to factor*sqrt(2*rho_max*t_max/mu).")
    parser.add_argument("--diffusion-refinement-mesh-size", type=float, default=80.0)
    parser.add_argument("--sponge-strength", type=float, default=0.0, help="Transient outer-shell conductivity increment in S/m; 0 disables the sponge.")
    parser.add_argument("--sponge-thickness", type=float, default=0.0, help="Outer-shell sponge thickness in metres for the unstructured DOLFINx mesh.")
    parser.add_argument("--sponge-power", type=float, default=2.0, help="Polynomial ramp power for the transient sponge conductivity.")
    parser.add_argument("--sponge-apply-to-initial", action="store_true", help="Also apply the sponge to the DC/on-time initial solve; normally leave this off.")
    parser.add_argument("--sponge-sides", default=",".join(SPONGE_ALL_SIDES), help="Comma-separated active sponge sides: x_min,x_max,y_min,y_max,z_min,z_max.")
    parser.add_argument("--nedelec-order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--expected-source-length", type=float, default=1000.0)
    parser.add_argument("--expected-parallel-offset", type=float, default=500.0)
    parser.add_argument("--wire-radius", type=float, default=2.5)
    parser.add_argument("--time-growth", type=float, default=1.05)
    parser.add_argument("--time-method", choices=["theta", "bdf2"], default="theta")
    parser.add_argument("--time-theta", type=float, default=1.0, help="Theta-method weight for non-polarizable E-form time stepping; 1.0=BE, 0.5=Crank-Nicolson.")
    parser.add_argument("--time-origin", choices=["after_ramp", "ramp_start"], default="after_ramp")
    parser.add_argument("--empymod-srcpts", type=int, default=5)
    parser.add_argument("--empymod-ht", choices=["dlf", "qwe", "quad"], default="dlf")
    parser.add_argument("--empymod-ft", choices=["dlf", "sin", "cos", "qwe", "fftlog", "fft"], default="dlf")
    parser.add_argument("--reference-audit-srcpts", type=int, default=0)
    parser.add_argument("--max-it", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--error-min-time", type=float, default=0.0)
    parser.add_argument(
        "--weak-component-reference-fraction",
        type=float,
        default=0.1,
        help="Horizontal E component is treated as weak if its reference max is below this fraction of max(|Eh_ref|).",
    )
    parser.add_argument("--t-min", type=float, default=1.0e-6)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--ramp-solver-t-min", type=float, default=1.0e-6)
    parser.add_argument("--min-steps-during-turnoff", type=int, default=10)
    parser.add_argument(
        "--min-steps-before-first-observation",
        type=int,
        default=1,
        help="Internal solve steps from ramp-off end to the first after-ramp observation; 1 keeps only the output step.",
    )
    parser.add_argument("--x-extent", type=float, default=25_000.0)
    parser.add_argument("--y-extent", type=float, default=25_000.0)
    parser.add_argument("--air-height", type=float, default=10_000.0)
    parser.add_argument("--earth-depth", type=float, default=25_000.0)
    parser.add_argument("--layer-depths", default="", help="Comma-separated earth-interface depths below z=0, e.g. 2000,2200.")
    parser.add_argument("--layer-resistivities", default="", help="Comma-separated earth-layer resistivities, e.g. 200,20,200.")
    args = parser.parse_args(argv)

    config = PipelineConfig(
        workdir=args.workdir,
        force_mesh=args.force_mesh,
        checkpoint_forward=args.checkpoint_forward,
        resume_forward=args.resume_forward,
        stop_after_outputs=args.stop_after_outputs,
        source_only=args.source_only,
        memory_limit_gb=args.memory_limit_gb,
        memory_safety_fraction=args.memory_safety_fraction,
        source_mode=args.source_mode,
        source_projection_mode=args.source_projection_mode,
        source_rhs_sign=args.source_rhs_sign,
        source_term_mode=args.source_term_mode,
        formulation=args.formulation,
        initial_dc_mode=args.initial_dc_mode,
        magnetic_receiver_mode=args.magnetic_receiver_mode,
        magnetic_dbdt_mode=args.magnetic_dbdt_mode,
        outer_boundary_mode=args.outer_boundary_mode,
        outer_boundary_robin_scale=args.outer_boundary_robin_scale,
        receiver_type=args.receiver_type,
        receiver_average_radius=args.receiver_average_radius,
        receiver_diagnostic_types=_parse_string_csv(args.receiver_diagnostic_types),
        receiver_evaluation_mode=args.receiver_evaluation_mode,
        divergence_cleaning=args.divergence_cleaning,
        divergence_cleaning_strength=args.divergence_cleaning_strength,
        divergence_cleaning_t_obs_min=args.divergence_cleaning_t_obs_min,
        divergence_control_weight=args.divergence_control_weight,
        divergence_control_t_obs_min=args.divergence_control_t_obs_min,
        divergence_control_scale=args.divergence_control_scale,
        polarization=args.polarization,
        cole_layer_top=args.cole_layer_top,
        cole_layer_bottom=args.cole_layer_bottom,
        cole_rho0=args.cole_rho0,
        cole_m=args.cole_m,
        cole_tau=args.cole_tau,
        cole_c=args.cole_c,
        cole_n_terms=args.cole_n_terms,
        cole_f_min=args.cole_f_min,
        cole_f_max=args.cole_f_max,
        cole_n_freq=args.cole_n_freq,
        rho_air=args.rho_air,
        rho_earth=args.rho_earth,
        source_start=(args.source_start_x, args.source_start_y, args.source_start_z),
        source_end=(args.source_end_x, args.source_end_y, args.source_end_z),
        source_current=args.source_current,
        ramp_off_time=args.ramp_off_time,
        observation_times=_parse_float_csv(args.observation_times, "--observation-times"),
        receiver=(args.receiver_x, args.receiver_y, args.receiver_z),
        source_mesh_size=args.source_mesh_size,
        source_refinement_radius=args.source_refinement_radius,
        source_quadrature_points=args.source_quadrature_points,
        receiver_mesh_size=args.receiver_mesh_size,
        receiver_anchor_mesh_size=args.receiver_anchor_mesh_size,
        receiver_refinement_radius=args.receiver_refinement_radius,
        diffusion_refinement_factor=args.diffusion_refinement_factor,
        diffusion_refinement_mesh_size=args.diffusion_refinement_mesh_size,
        sponge_strength=args.sponge_strength,
        sponge_thickness=args.sponge_thickness,
        sponge_power=args.sponge_power,
        sponge_apply_to_initial=args.sponge_apply_to_initial,
        sponge_sides=_parse_string_csv(args.sponge_sides),
        nedelec_order=args.nedelec_order,
        expected_source_length=args.expected_source_length,
        expected_parallel_offset=args.expected_parallel_offset,
        wire_radius=args.wire_radius,
        time_growth=args.time_growth,
        time_method=args.time_method,
        time_theta=args.time_theta,
        time_origin=args.time_origin,
        min_steps_during_turnoff=args.min_steps_during_turnoff,
        min_steps_before_first_observation=args.min_steps_before_first_observation,
        empymod_srcpts=args.empymod_srcpts,
        empymod_ht=args.empymod_ht,
        empymod_ft=args.empymod_ft,
        reference_audit_srcpts=args.reference_audit_srcpts,
        max_it=args.max_it,
        rtol=args.rtol,
        atol=args.atol,
        error_min_time=args.error_min_time,
        weak_component_reference_fraction=args.weak_component_reference_fraction,
        t_min=args.t_min,
        t_max=args.t_max,
        ramp_solver_t_min=args.ramp_solver_t_min,
        x_extent=args.x_extent,
        y_extent=args.y_extent,
        air_height=args.air_height,
        earth_depth=args.earth_depth,
        layer_depths=_parse_float_csv(args.layer_depths, "--layer-depths"),
        layer_resistivities=_parse_float_csv(args.layer_resistivities, "--layer-resistivities"),
    )
    config.workdir.mkdir(parents=True, exist_ok=True)
    try:
        model = validate_model_consistency(config)
    except ValueError as exc:
        raise SystemExit(f"[model] {exc}") from None
    try:
        config.formulation = validate_formulation(config)
    except ValueError as exc:
        raise SystemExit(f"[formulation] {exc}") from None
    print(
        "[model] "
        f"source_length={model['source_length']:.6g} m; "
        f"inline_from_source_start={model['inline_distance_from_source_start']:.6g} m; "
        f"parallel_offset={model['parallel_offset']:.6g} m; "
        f"reference_mode={model['reference_mode']}; "
        f"time_origin={model['time_origin']}; "
        f"source_depth=({model['source_depth_start']:.6g}, {model['source_depth_end']:.6g}) m; "
        f"receiver_depth={model['receiver_depth']:.6g} m; "
        f"sponge_enabled={model['sponge']['enabled']}",
        flush=True,
    )
    env = check_environment(install_missing=not args.no_install, require_core=not args.postprocess_partial)
    if args.check_env_only:
        return 0
    if args.postprocess_partial:
        ref_mode = "cole-cole" if config.polarization == "cole-cole" else "noip"
        postprocess_saved_forward(config, env, ref_mode=ref_mode, runtime={"mesh_seconds": 0.0})
        return 0

    from mpi4py import MPI

    if MPI.COMM_WORLD.size != 1:
        raise SystemExit("This verification script currently requires serial execution for point receiver extraction.")

    t0 = time.perf_counter()
    generate_verification_mesh(config)
    runtime["mesh_seconds"] = time.perf_counter() - t0
    if args.mesh_only:
        return 0
    t0 = time.perf_counter()
    msh, cell_tags, facet_tags = load_mesh(config)
    spaces = build_function_spaces(msh, config)
    materials = assign_materials(msh, cell_tags, spaces, config)

    debye = None
    ref_mode = "noip"
    if config.polarization == "cole-cole":
        fit = fit_cole_cole_to_debye(config)
        debye = _build_debye_materials(msh, cell_tags, spaces, fit, config)
        materials["sigma_infinity"].x.array[:] = materials["sigma"].x.array
        _assign_dg0_by_cell(materials["sigma_infinity"], debye["polarizable_cells"], fit.sigma_infinity)
        materials["sigma_infinity"].x.scatter_forward()
        materials["sigma_infinity_physical"].x.array[:] = materials["sigma_infinity"].x.array
        materials["sigma_infinity_physical"].x.scatter_forward()
        ref_mode = "cole-cole"
    else:
        print("[polarization] disabled: running non-polarizable air-earth stage-1 validation.", flush=True)
    apply_transient_sponge(msh, materials, config)

    source = build_source(msh, spaces, config, cell_tags)
    if config.formulation == "h":
        source["current_density"] = _build_regularized_current_density(msh, spaces, config, cell_tags)
    runtime["setup_seconds"] = time.perf_counter() - t0
    if config.source_only:
        if msh.comm.rank == 0:
            runtime["total_seconds"] = time.perf_counter() - run_t0
            write_source_only_diagnostics(config, env, source, runtime=runtime)
        return 0
    times = generate_time_array(config)
    t0 = time.perf_counter()
    if config.formulation == "h":
        fem_result = run_h_forward(msh, cell_tags, facet_tags, spaces, materials, source, config, times=times)
    else:
        fem_result = run_fetd_forward(msh, cell_tags, facet_tags, spaces, materials, source, config, debye=debye, times=times)
    runtime["forward_seconds"] = time.perf_counter() - t0
    completed_times = fem_result["times"]
    t0 = time.perf_counter()
    ref_result = get_empymod_reference(completed_times, config, mode=ref_mode)
    errors = compute_error(fem_result["data"], ref_result["data"], fem_result["components"])
    reference_audit_errors = None
    if int(config.reference_audit_srcpts) > 0 and int(config.reference_audit_srcpts) != int(config.empymod_srcpts):
        audit_ref_result = get_empymod_reference(completed_times, config, mode=ref_mode, srcpts=int(config.reference_audit_srcpts))
        reference_audit_errors = compute_error(
            ref_result["data"],
            audit_ref_result["data"],
            ref_result["components"],
        )
    runtime["reference_seconds"] = time.perf_counter() - t0
    if msh.comm.rank == 0:
        t0 = time.perf_counter()
        _save_npz(config, fem_result, ref_result, errors)
        plot_verification(completed_times, fem_result["data"], ref_result["data"], errors, fem_result["components"], config)
        write_validation_artifacts(
            completed_times,
            fem_result["data"],
            ref_result["data"],
            fem_result["components"],
            config,
            case_type="ip" if ref_mode == "cole-cole" else "noip",
            reference_type="empymod",
            source_info=source,
            receiver_diagnostic_rows=fem_result.get("receiver_diagnostic_rows"),
            solver_log=fem_result.get("solver_log"),
        )
        runtime["postprocess_seconds"] = time.perf_counter() - t0
        runtime["total_seconds"] = time.perf_counter() - run_t0
        write_report(
            config,
            env,
            fem_result,
            ref_result,
            errors,
            source,
            debye=debye,
            reference_audit_errors=reference_audit_errors,
            runtime=runtime,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
