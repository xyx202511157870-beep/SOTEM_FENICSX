"""Corrected-model validation runner orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.solvers import secondary_step_equation_metadata
from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)
from atem3d.waveforms import summarize_internal_time_grid

ResponseRunner = Callable[[dict], np.ndarray]
DiagnosticsBuilder = Callable[[dict, np.ndarray, np.ndarray], dict]


def run_corrected_model_validation(
    case_spec: dict,
    *,
    forward_runner: ResponseRunner | None = None,
    reference_runner: ResponseRunner | None = None,
    diagnostics_builder: DiagnosticsBuilder | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Run one corrected-model case and write validation artifacts.

    The orchestration layer is intentionally pure Python. Heavy backends such
    as DOLFINx and empymod are reached only by the injected/default runners.
    """

    spec = deepcopy(dict(case_spec))
    if output_dir is not None:
        spec["output_dir"] = str(output_dir)
    times = np.asarray(spec["observation_times"], dtype=float)
    components = [str(value) for value in spec["components"]]
    forward = forward_runner or _default_forward_runner
    reference = reference_runner or _default_reference_runner
    forward_t0 = time.perf_counter()
    predictions = _validate_response_table(forward(spec), times, components, "forward_runner")
    forward_runtime = time.perf_counter() - forward_t0
    reference_t0 = time.perf_counter()
    reference_values = _validate_response_table(reference(spec), times, components, "reference_runner")
    reference_runtime = time.perf_counter() - reference_t0
    material = _material_from_case_spec(spec)
    runtime_seconds = {
        "forward": float(forward_runtime),
        "reference": float(reference_runtime),
    }
    diagnostics = {
        **dict(spec.get("diagnostics", {})),
        "runner": spec.get("runner", {}),
        "source_start": spec.get("source_start"),
        "source_end": spec.get("source_end"),
        "receiver": spec.get("receiver"),
        "runtime_seconds": runtime_seconds,
    }
    diagnostics["primary_secondary_step_equation"] = (
        _primary_secondary_step_equation_metadata_for_case(spec, times)
    )
    if diagnostics_builder is not None:
        diagnostics.update(dict(diagnostics_builder(spec, predictions, reference_values)))
    artifact_t0 = time.perf_counter()
    case = ThreeComponentValidationInput(
        output_dir=spec["output_dir"],
        times=times,
        predictions=predictions,
        reference=reference_values,
        component_names=components,
        case_type=str(spec["case_type"]),
        reference_type=str(spec.get("reference_type", "empymod")),
        magnetic_quantity=str(spec.get("magnetic_quantity", components[-1])),
        diagnostics=diagnostics,
        resolved_config=spec,
        material=material,
        validation_scope=str(spec.get("validation_scope", "smoke")),
    )
    summary = write_three_component_validation_artifacts(case)
    schematic_info = _write_model_schematic_if_possible(spec)
    runtime_seconds["artifact_total"] = float(time.perf_counter() - artifact_t0)
    _update_runtime_diagnostics(
        Path(spec["output_dir"]),
        runtime_seconds,
        model_schematic=schematic_info,
    )
    return summary


def run_corrected_model_convergence_validation(
    case_spec: dict,
    *,
    output_dir: str | Path | None = None,
    forward_runner: ResponseRunner | None = None,
    reference_runner: ResponseRunner | None = None,
    primary_runner: ResponseRunner | None = None,
    secondary_effect_atol: float = 1.0e-15,
) -> dict:
    """Run a coarse-vs-refined DOLFINx convergence validation artifact set.

    This is a diagnostic validation path for nonzero secondary fields where a
    one-dimensional empymod background is not a physical reference for the
    three-dimensional anomaly. It deliberately writes ``reference_type`` as
    ``dolfinx_refined`` so final task-book acceptance remains blocked until an
    empymod/1D final-reference case is used.
    """

    prediction_spec, refined_spec = _build_dolfinx_refined_reference_specs(
        case_spec,
        output_dir=output_dir,
    )
    runner = forward_runner or _default_forward_runner
    reference = reference_runner or runner
    primary = primary_runner or _default_reference_runner

    def run_refined_reference(_prediction_case_spec: dict) -> np.ndarray:
        return reference(refined_spec)

    def build_secondary_effect_diagnostics(
        prediction_case_spec: dict,
        predictions: np.ndarray,
        refined_reference: np.ndarray,
    ) -> dict:
        times = np.asarray(prediction_case_spec["observation_times"], dtype=float)
        components = [str(value) for value in prediction_case_spec["components"]]
        primary_t0 = time.perf_counter()
        primary_values = _validate_response_table(
            primary(prediction_case_spec),
            times,
            components,
            "primary_runner",
        )
        primary_runtime = time.perf_counter() - primary_t0
        return {
            "secondary_effect_diagnostic": _secondary_effect_diagnostic(
                predictions,
                refined_reference,
                primary_values,
                components,
                reference_type=str(prediction_case_spec.get("reference_type", "dolfinx_refined")),
                nonzero_atol=float(secondary_effect_atol),
                primary_runtime_s=float(primary_runtime),
            )
        }

    return run_corrected_model_validation(
        prediction_spec,
        forward_runner=runner,
        reference_runner=run_refined_reference,
        diagnostics_builder=build_secondary_effect_diagnostics,
    )


def _secondary_effect_diagnostic(
    predictions: np.ndarray,
    refined_reference: np.ndarray,
    primary_values: np.ndarray,
    component_names: list[str],
    *,
    reference_type: str,
    nonzero_atol: float,
    primary_runtime_s: float,
) -> dict:
    prediction_delta = np.asarray(predictions, dtype=float) - np.asarray(primary_values, dtype=float)
    reference_delta = np.asarray(refined_reference, dtype=float) - np.asarray(primary_values, dtype=float)
    coarse_fine_delta = np.asarray(predictions, dtype=float) - np.asarray(refined_reference, dtype=float)
    prediction_by_component = _max_abs_by_component(prediction_delta, component_names)
    reference_by_component = _max_abs_by_component(reference_delta, component_names)
    coarse_fine_by_component = _max_abs_by_component(coarse_fine_delta, component_names)
    prediction_nonzero = any(value > float(nonzero_atol) for value in prediction_by_component.values())
    reference_nonzero = any(value > float(nonzero_atol) for value in reference_by_component.values())
    return {
        "reference_type": str(reference_type),
        "primary_reference_type": "empymod_or_1d_background",
        "component_names": [str(name) for name in component_names],
        "nonzero_atol": float(nonzero_atol),
        "primary_background_runtime_s": float(primary_runtime_s),
        "max_abs_prediction_minus_primary_by_component": prediction_by_component,
        "max_abs_reference_minus_primary_by_component": reference_by_component,
        "max_abs_prediction_minus_reference_by_component": coarse_fine_by_component,
        "prediction_secondary_effect_nonzero": bool(prediction_nonzero),
        "reference_secondary_effect_nonzero": bool(reference_nonzero),
        "secondary_effect_nonzero": bool(prediction_nonzero and reference_nonzero),
    }


def _max_abs_by_component(values: np.ndarray, component_names: list[str]) -> dict[str, float]:
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[1] != len(component_names):
        raise ValueError("secondary-effect diagnostic values must have shape (n_times, n_components)")
    return {
        str(component): float(np.max(np.abs(table[:, index]))) if table.shape[0] else 0.0
        for index, component in enumerate(component_names)
    }


def _build_dolfinx_refined_reference_specs(
    case_spec: dict,
    *,
    output_dir: str | Path | None = None,
) -> tuple[dict, dict]:
    prediction_spec = deepcopy(dict(case_spec))
    if output_dir is not None:
        prediction_spec["output_dir"] = str(output_dir)
    prediction_spec["reference_type"] = "dolfinx_refined"
    reference_spec = deepcopy(prediction_spec)
    convergence_cfg = dict(prediction_spec.get("convergence_reference", {}))
    reference_overrides = {key: value for key, value in convergence_cfg.items() if key != "metadata"}
    reference_spec = _deep_merge_dict(reference_spec, reference_overrides)
    reference_spec["reference_type"] = "dolfinx_refined"
    reference_spec.setdefault("runner", {}).update({"reference_role": "dolfinx_refined"})
    prediction_cells = list(dict(prediction_spec.get("dolfinx_forward", {})).get("cells", []))
    reference_cells = list(dict(reference_spec.get("dolfinx_forward", {})).get("cells", []))
    diagnostics = dict(prediction_spec.get("diagnostics", {}))
    diagnostics["convergence_reference"] = {
        "reference_type": "dolfinx_refined",
        "prediction_cells": prediction_cells,
        "reference_cells": reference_cells,
        "overrides": reference_overrides,
    }
    marker_preflight = _leakage_marker_preflight(prediction_spec, reference_spec)
    if marker_preflight:
        diagnostics["leakage_marker_preflight"] = marker_preflight
    prediction_spec["diagnostics"] = diagnostics
    return prediction_spec, reference_spec


def _leakage_marker_preflight(prediction_spec: dict, reference_spec: dict) -> dict:
    prediction_forward = dict(prediction_spec.get("dolfinx_forward", {}))
    reference_forward = dict(reference_spec.get("dolfinx_forward", {}))
    if "leakage_channel" not in prediction_forward or "leakage_channel" not in reference_forward:
        return {}
    from atem3d.materials.material_map import leakage_channel_marker_diagnostics

    return {
        "prediction": _leakage_marker_preflight_for_forward(
            prediction_forward,
            leakage_channel_marker_diagnostics,
        ),
        "reference": _leakage_marker_preflight_for_forward(
            reference_forward,
            leakage_channel_marker_diagnostics,
        ),
    }


def _leakage_marker_preflight_for_forward(forward_cfg: dict, diagnostics_func) -> dict:
    leakage = dict(forward_cfg["leakage_channel"])
    return diagnostics_func(
        domain_min=forward_cfg["domain_min"],
        domain_max=forward_cfg["domain_max"],
        cells=forward_cfg["cells"],
        channel_points=leakage["points"],
        radius=float(leakage["radius"]),
        min_marked_cells=int(leakage.get("min_marked_cells", 0)),
    )


def _deep_merge_dict(base: dict, overrides: dict) -> dict:
    merged = deepcopy(dict(base))
    for key, value in dict(overrides).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _default_forward_runner(case_spec: dict) -> np.ndarray:
    return _run_dolfinx_primary_secondary_forward(case_spec)


def dolfinx_backend_status() -> dict:
    """Return import diagnostics for the default DOLFINx forward backend."""

    runtime_modules = [
        "numpy",
        "dolfinx.fem",
        "dolfinx.mesh",
        "mpi4py.MPI",
        "ufl",
        "basix",
        "petsc4py",
    ]
    test_modules = ["pytest"]
    required_modules = list(runtime_modules)
    checks: dict[str, dict[str, object]] = {}
    for module_name in runtime_modules + test_modules:
        try:
            module = importlib.import_module(module_name)
            checks[module_name] = {
                "available": True,
                "module_file": getattr(module, "__file__", None),
            }
        except ImportError as exc:
            checks[module_name] = {
                "available": False,
                "error": str(exc),
            }
    missing = [name for name in runtime_modules if not bool(checks[name]["available"])]
    missing_test = [name for name in test_modules if not bool(checks[name]["available"])]
    available = not missing
    if available:
        message = "DOLFINx forward backend is available."
    else:
        message = (
            "DOLFINx forward backend is unavailable: missing "
            + ", ".join(missing)
            + ". Use a complete FEniCSx/DOLFINx Python environment before "
            "running the default corrected-model forward backend."
        )
    return {
        "available": available,
        "required_modules": required_modules,
        "runtime_modules": runtime_modules,
        "test_modules": test_modules,
        "missing_modules": missing,
        "missing_test_modules": missing_test,
        "checks": checks,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "message": message,
    }


def _import_dolfinx_backend():
    status = dolfinx_backend_status()
    if not bool(status["available"]):
        raise ImportError(str(status["message"]))
    from dolfinx import fem, mesh
    from mpi4py import MPI

    return fem, mesh, MPI


def _turnoff_time_from_case_spec(case_spec: dict) -> float:
    value = float(case_spec.get("ramp_off_time", case_spec.get("ramp_off_time_s", 0.0)))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("ramp_off_time must be finite and nonnegative")
    return value


def _turnoff_steps_from_case_spec(case_spec: dict) -> int:
    value = int(case_spec.get("turnoff_steps", case_spec.get("min_steps_during_turnoff", 1)))
    if value < 1:
        raise ValueError("turnoff_steps must be positive")
    return value


def _primary_secondary_step_equation_metadata_for_case(case_spec: dict, times: np.ndarray) -> dict:
    forward_material = _forward_material_from_case_spec(case_spec)
    sigma_background = float(case_spec.get("sigma_background", _background_sigma_from_case_spec(case_spec)))
    secondary_material = _secondary_material_for_forward(
        case_spec,
        forward_material,
        sigma_background,
    )
    time_array = np.asarray(times, dtype=float)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("observation times must be a non-empty 1D array")
    turnoff_time = _turnoff_time_from_case_spec(case_spec)
    turnoff_steps = _turnoff_steps_from_case_spec(case_spec)
    first_observation_time = float(time_array[0])
    first_output_internal_time = turnoff_time + first_observation_time
    first_internal_step_dt = turnoff_time / float(turnoff_steps) if turnoff_time > 0.0 else first_observation_time
    metadata = secondary_step_equation_metadata(
        material=secondary_material,
        sigma_background=sigma_background,
        dt=first_internal_step_dt,
    )
    metadata["dt_source"] = "turnoff_grid_first_step"
    metadata["turnoff_time_s"] = turnoff_time
    metadata["turnoff_steps"] = turnoff_steps
    metadata["first_observation_time_s"] = first_observation_time
    metadata["first_output_internal_time_s"] = first_output_internal_time
    metadata["first_internal_step_dt_s"] = first_internal_step_dt
    metadata["internal_time_grid"] = summarize_internal_time_grid(
        time_array,
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
    )
    metadata["primary_includes_ip_background"] = bool(
        dict(case_spec.get("dolfinx_forward", {})).get("primary_includes_ip_background", True)
    )
    if forward_material.terms and not secondary_material.terms:
        metadata["secondary_material_reason"] = "ip_primary_background_included"
        metadata["original_ip_material"] = secondary_step_equation_metadata(
            material=forward_material,
            sigma_background=sigma_background,
            dt=first_internal_step_dt,
        )
    else:
        metadata["secondary_material_reason"] = "case_material"
    return metadata


def _run_dolfinx_primary_secondary_forward(case_spec: dict) -> np.ndarray:
    from atem3d.solvers import PrimarySecondaryForwardOperator
    fem, mesh, MPI = _import_dolfinx_backend()

    sp = _load_sotem_pipeline_module()
    forward_cfg = dict(case_spec.get("dolfinx_forward", {}))
    domain_min = np.asarray(
        forward_cfg.get("domain_min", [-600.0, -400.0, -50.0]),
        dtype=float,
    )
    domain_max = np.asarray(
        forward_cfg.get("domain_max", [600.0, 300.0, 50.0]),
        dtype=float,
    )
    cells = [int(value) for value in forward_cfg.get("cells", [1, 1, 1])]
    terrain_runtime = _terrain_mesh_runtime_config(
        forward_cfg,
        output_dir=Path(case_spec.get("output_dir", ".")),
    )
    config = sp.PipelineConfig(
        workdir=Path(terrain_runtime["workdir"]),
        msh_name=str(terrain_runtime["msh_name"]),
        receiver=tuple(float(value) for value in case_spec["receiver"]),
        receiver_evaluation_mode=str(forward_cfg.get("receiver_evaluation_mode", "first_cell")),
        outer_boundary_mode=str(forward_cfg.get("outer_boundary_mode", "natural")),
        ksp_type=str(forward_cfg.get("ksp_type", "cg")),
        rtol=float(forward_cfg.get("rtol", 1.0e-8)),
        atol=float(forward_cfg.get("atol", 1.0e-10)),
        max_it=int(forward_cfg.get("max_it", 200)),
    )
    msh, facet_tags, mesh_runtime = _build_dolfinx_forward_mesh(
        sp,
        mesh,
        MPI,
        config,
        domain_min=domain_min,
        domain_max=domain_max,
        cells=cells,
        terrain_runtime=terrain_runtime,
    )
    case_spec.setdefault("diagnostics", {})["mesh_runtime"] = mesh_runtime
    spaces = sp.build_function_spaces(msh, config)
    material = _forward_material_from_case_spec(case_spec)
    sigma_background = float(case_spec.get("sigma_background", _background_sigma_from_case_spec(case_spec)))
    secondary_material = _secondary_material_for_forward(case_spec, material, sigma_background)
    turnoff_time = _turnoff_time_from_case_spec(case_spec)
    turnoff_steps = _turnoff_steps_from_case_spec(case_spec)
    primary = _primary_provider_for_case_spec(case_spec)
    interpolation = sp._nedelec_interpolation_points(msh, spaces)
    fem_points = interpolation["points"]
    receiver_locations = np.asarray([case_spec["receiver"]], dtype=float)
    components = tuple(str(value) for value in case_spec["components"])
    forward_diagnostics = case_spec.setdefault("diagnostics", {})
    forward_diagnostics["primary_provider_mode"] = str(forward_cfg.get("primary_provider_mode", "empymod"))

    if "leakage_channel" in forward_cfg:
        return _run_dolfinx_leakage_channel_forward(
            sp,
            msh,
            spaces,
            config,
            case_spec,
            forward_cfg,
            primary=primary,
            receiver_locations=receiver_locations,
            components=components,
            sigma_background=sigma_background,
            facet_tags=facet_tags,
        )

    if _is_zero_secondary_material(secondary_material, sigma_background):
        operator = PrimarySecondaryForwardOperator(
            primary=primary,
            fem_points=fem_points,
            receiver_locations=receiver_locations,
            components=components,
            material=secondary_material,
            sigma_background=sigma_background,
            secondary_receiver_projector=sp._make_dolfinx_zero_secondary_receiver_projector(
                msh,
                spaces,
                config,
            ),
            contrast_atol=float(forward_cfg.get("contrast_atol", 1.0e-12)),
            turnoff_time=turnoff_time,
            turnoff_steps=turnoff_steps,
            diagnostics=forward_diagnostics,
        )
        return operator.forward(np.asarray(case_spec["observation_times"], dtype=float))

    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = secondary_material.sigma_inf
    sigma.x.scatter_forward()
    sigma_initial = fem.Function(spaces["Q"], name="sigma_initial")
    sigma_initial.x.array[:] = secondary_material.sigma0
    sigma_initial.x.scatter_forward()
    sigma_infinity = fem.Function(spaces["Q"], name="sigma_infinity")
    sigma_infinity.x.array[:] = secondary_material.sigma_inf
    sigma_infinity.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma,
        "sigma_initial": sigma_initial,
        "sigma_infinity": sigma_infinity,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=facet_tags, config=config)
    built = sp._make_dolfinx_primary_secondary_forward_operator(
        msh,
        spaces,
        materials,
        operators,
        config,
        primary=primary,
        receiver_locations=receiver_locations,
        components=components,
        material=secondary_material,
        sigma_background=sigma_background,
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
    )
    values = built["operator"].forward(np.asarray(case_spec["observation_times"], dtype=float))
    _copy_dolfinx_forward_diagnostics_to_case_spec(case_spec, built["diagnostics"])
    return values


def _run_dolfinx_leakage_channel_forward(
    sp,
    msh,
    spaces,
    config,
    case_spec: dict,
    forward_cfg: dict,
    *,
    primary,
    receiver_locations: np.ndarray,
    components: tuple[str, ...],
    sigma_background: float,
    facet_tags,
) -> np.ndarray:
    from atem3d.materials.material_map import (
        CellMaterialMap,
        apply_leakage_channel_marker_with_diagnostics,
    )

    leakage_cfg = dict(forward_cfg["leakage_channel"])
    centers = sp._cell_centers(msh)
    background_marker = int(leakage_cfg.get("background_marker", 1))
    leakage_marker = int(leakage_cfg.get("leakage_marker", 7))
    markers = np.full(centers.shape[0], background_marker, dtype=int)
    marker_result = apply_leakage_channel_marker_with_diagnostics(
        markers,
        centers,
        channel_points=np.asarray(leakage_cfg["points"], dtype=float),
        radius=float(leakage_cfg["radius"]),
        leakage_marker=leakage_marker,
        min_marked_cells=int(leakage_cfg.get("min_marked_cells", 0)),
    )
    markers = marker_result.markers
    case_spec.setdefault("diagnostics", {})["leakage_marker_runtime"] = marker_result.diagnostics
    if not np.any(markers == leakage_marker):
        raise ValueError("leakage_channel did not mark any mesh cells")
    material_map = CellMaterialMap(
        markers=markers,
        materials={
            background_marker: PronyConductivity.no_ip(float(sigma_background)),
            leakage_marker: _leakage_material_from_config(leakage_cfg),
        },
    )
    times = np.asarray(case_spec["observation_times"], dtype=float)
    turnoff_time = _turnoff_time_from_case_spec(case_spec)
    turnoff_steps = _turnoff_steps_from_case_spec(case_spec)
    first_internal_step_dt = turnoff_time / float(turnoff_steps) if turnoff_time > 0.0 else float(times[0])
    built_materials = sp._make_dolfinx_materials_from_cell_material_map(
        msh,
        spaces,
        material_map,
        dt=float(first_internal_step_dt),
    )
    operators = sp.assemble_operators(
        msh,
        spaces,
        built_materials["materials"],
        facet_tags=facet_tags,
        config=config,
        debye=built_materials["debye"],
    )
    built_operator = sp._make_dolfinx_primary_secondary_forward_operator(
        msh,
        spaces,
        built_materials["materials"],
        operators,
        config,
        primary=primary,
        receiver_locations=receiver_locations,
        components=components,
        material=built_materials["representative_material"],
        sigma_background=float(sigma_background),
        debye=built_materials["debye"],
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
    )
    values = built_operator["operator"].forward(times)
    _copy_dolfinx_forward_diagnostics_to_case_spec(case_spec, built_operator["diagnostics"])
    return values


def _terrain_mesh_runtime_config(forward_cfg: dict, *, output_dir: str | Path) -> dict:
    terrain_cfg = dict(dict(forward_cfg).get("terrain_mesh", {}))
    output = Path(output_dir)
    mode = str(terrain_cfg.get("mode", "box_create_box"))
    if mode == "box_create_box":
        return {
            "mode": "box_create_box",
            "mesh_size": 0.0,
            "msh_name": "",
            "workdir": str(output),
            "mesh_path": "",
        }
    if mode != "small_gmsh_terrain_leakage":
        raise ValueError("terrain_mesh mode must be 'box_create_box' or 'small_gmsh_terrain_leakage'")
    mesh_size = float(terrain_cfg.get("mesh_size", 0.65))
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("terrain_mesh mesh_size must be finite and positive")
    msh_name = str(terrain_cfg.get("msh_name", "terrain_leakage.msh"))
    if not msh_name or Path(msh_name).is_absolute() or Path(msh_name).name != msh_name:
        raise ValueError("terrain_mesh msh_name must be a simple relative file name")
    return {
        "mode": mode,
        "mesh_size": mesh_size,
        "msh_name": msh_name,
        "workdir": str(output),
        "mesh_path": str(output / msh_name),
    }


def _build_dolfinx_forward_mesh(
    sp,
    mesh_module,
    MPI,
    config,
    *,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    cells: list[int],
    terrain_runtime: dict,
):
    mode = str(terrain_runtime.get("mode", "box_create_box"))
    if mode == "small_gmsh_terrain_leakage":
        terrain_info = sp._write_small_gmsh_terrain_leakage_mesh(
            Path(str(terrain_runtime["mesh_path"])),
            mesh_size=float(terrain_runtime["mesh_size"]),
        )
        msh, _cell_tags, facet_tags = sp.load_mesh(config)
        return msh, facet_tags, {
            "mode": "gmsh",
            "terrain_mesh": {
                **dict(terrain_runtime),
                **dict(terrain_info),
            },
        }
    msh = mesh_module.create_box(MPI.COMM_WORLD, [domain_min, domain_max], cells)
    return msh, None, {
        "mode": "create_box",
        "terrain_mesh": None,
        "box": {
            "domain_min": [float(value) for value in domain_min],
            "domain_max": [float(value) for value in domain_max],
            "cells": [int(value) for value in cells],
        },
    }


def _copy_dolfinx_forward_diagnostics_to_case_spec(
    case_spec: dict,
    forward_diagnostics: dict,
) -> None:
    diagnostics = case_spec.setdefault("diagnostics", {})
    for key in (
        "primary_secondary_internal_time_grid",
        "primary_secondary_step_equation",
        "dc_result",
    ):
        if key not in forward_diagnostics:
            continue
        value = forward_diagnostics[key]
        if _is_json_serializable(value):
            diagnostics[key] = deepcopy(value)


def _is_json_serializable(value) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def _leakage_material_from_config(leakage_cfg: dict) -> PronyConductivity:
    if "sigma" in leakage_cfg:
        return PronyConductivity.no_ip(float(leakage_cfg["sigma"]))
    delta = list(leakage_cfg.get("delta_sigma_list", []))
    tau = list(leakage_cfg.get("tau_list", []))
    if len(delta) != len(tau):
        raise ValueError("leakage_channel delta_sigma_list and tau_list must have the same length")
    return PronyConductivity(
        sigma_inf=float(leakage_cfg["sigma_inf"]),
        terms=[
            DebyeTerm(delta_sigma=float(delta_i), tau=float(tau_i))
            for delta_i, tau_i in zip(delta, tau)
        ],
    )


def _load_sotem_pipeline_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    module_name = "atem3d_dolfinx_sotem_pipeline"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DOLFINx pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _forward_material_from_case_spec(case_spec: dict) -> PronyConductivity:
    forward_cfg = dict(case_spec.get("dolfinx_forward", {}))
    if str(case_spec.get("case_type")) == "noip" and "secondary_sigma" in forward_cfg:
        return PronyConductivity.no_ip(float(forward_cfg["secondary_sigma"]))
    ip_material = _material_from_case_spec(case_spec)
    if ip_material is not None:
        return ip_material
    material = dict(case_spec.get("material") or {})
    if "sigma" in material:
        return PronyConductivity.no_ip(float(material["sigma"]))
    if "resistivity" in material:
        return PronyConductivity.no_ip(1.0 / float(material["resistivity"]))
    primary = dict(case_spec.get("empymod_primary", {}))
    resistivities = primary.get("resistivities") or ()
    if resistivities:
        return PronyConductivity.no_ip(1.0 / float(resistivities[-1]))
    raise ValueError("no-IP corrected-model material must define sigma or resistivity")


def _background_sigma_from_case_spec(case_spec: dict) -> float:
    material = dict(case_spec.get("material") or {})
    if "sigma0" in material:
        return float(material["sigma0"])
    if "sigma" in material:
        return float(material["sigma"])
    if "resistivity" in material:
        return 1.0 / float(material["resistivity"])
    primary = dict(case_spec.get("empymod_primary", {}))
    resistivities = primary.get("resistivities") or ()
    if resistivities:
        return 1.0 / float(resistivities[-1])
    raise ValueError("corrected-model case must define a background conductivity")


def _is_zero_secondary_material(material: PronyConductivity, sigma_background: float) -> bool:
    return (
        not material.terms
        and abs(float(material.sigma_inf) - float(sigma_background)) <= 1.0e-12
    )


def _default_reference_runner(case_spec: dict) -> np.ndarray:
    from atem3d.primary import EmpymodPrimaryProvider

    components = [str(value) for value in case_spec["components"]]
    times = np.asarray(case_spec["observation_times"], dtype=float)
    receiver = np.asarray([case_spec["receiver"]], dtype=float)
    provider = EmpymodPrimaryProvider(
        config=_empymod_primary_config_for_case_spec(case_spec),
        empymod_kwargs=dict(case_spec.get("empymod_kwargs", {})),
    )
    rows = []
    for time in times:
        e_values = provider.get_receiver_E(float(time), receiver)[0]
        dbdt_values = provider.get_receiver_dBdt(float(time), receiver)[0]
        values = {
            "Ex": e_values[0],
            "Ey": e_values[1],
            "Ez": e_values[2],
            "dBxdt": dbdt_values[0],
            "dBydt": dbdt_values[1],
            "dBzdt": dbdt_values[2],
        }
        rows.append([values[name] for name in components])
    return np.asarray(rows, dtype=float)


def _primary_provider_for_case_spec(case_spec: dict):
    forward_cfg = dict(case_spec.get("dolfinx_forward", {}))
    mode = str(forward_cfg.get("primary_provider_mode", "empymod"))
    if mode == "empymod":
        from atem3d.primary import EmpymodPrimaryProvider

        return EmpymodPrimaryProvider(
            config=_empymod_primary_config_for_case_spec(case_spec),
            empymod_kwargs=dict(case_spec.get("empymod_kwargs", {})),
        )
    if mode == "constant":
        return _ConstantPrimaryProvider(
            electric=_constant_primary_vector(forward_cfg, "constant_primary_E", [1.0, 0.0, 0.0]),
            dbdt=_constant_primary_vector(forward_cfg, "constant_primary_dBdt", [0.0, 0.0, 0.0]),
        )
    raise ValueError("primary_provider_mode must be 'empymod' or 'constant'")


def _constant_primary_vector(forward_cfg: dict, key: str, default: list[float]) -> tuple[float, float, float]:
    values = np.asarray(forward_cfg.get(key, default), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must be a finite 3-vector")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class _ConstantPrimaryProvider:
    electric: tuple[float, float, float]
    dbdt: tuple[float, float, float]

    def get_Ep_on_V(self, _t: float, V) -> np.ndarray:
        return self._tile(self.electric, V, "points")

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        return self._tile(self.electric, V, "points")

    def get_receiver_E(self, _t: float, receivers) -> np.ndarray:
        return self._tile(self.electric, receivers, "receivers")

    def get_receiver_dBdt(self, _t: float, receivers) -> np.ndarray:
        return self._tile(self.dbdt, receivers, "receivers")

    @staticmethod
    def _tile(vector: tuple[float, float, float], points, name: str) -> np.ndarray:
        from atem3d.primary.base import as_points

        pts = as_points(points, name)
        return np.tile(np.asarray(vector, dtype=float), (pts.shape[0], 1))


def _empymod_primary_config_for_case_spec(case_spec: dict) -> dict:
    from atem3d.empymod_compare import make_debye_resistivity_model

    provider_config = dict(case_spec["empymod_primary"])
    material = _material_from_case_spec(case_spec)
    if material is None:
        return provider_config
    layer_count = len(provider_config["resistivities"])
    provider_config["resistivity"] = 1.0 / float(material.sigma0)
    provider_config["resistivities"] = make_debye_resistivity_model(
        [material.sigma_inf] * layer_count,
        [
            {"delta_sigma": term.delta_sigma, "tau": term.tau}
            for term in material.terms
        ],
    )
    return provider_config


def _secondary_material_for_forward(
    case_spec: dict,
    material: PronyConductivity,
    sigma_background: float,
) -> PronyConductivity:
    forward_cfg = dict(case_spec.get("dolfinx_forward", {}))
    primary_includes_ip = bool(forward_cfg.get("primary_includes_ip_background", True))
    if (
        str(case_spec.get("case_type")) == "ip"
        and primary_includes_ip
        and abs(material.sigma0 - float(sigma_background)) <= 1.0e-12
    ):
        return PronyConductivity.no_ip(float(sigma_background))
    return material


def _validate_response_table(values, times: np.ndarray, components: list[str], runner_name: str) -> np.ndarray:
    table = np.asarray(values, dtype=float)
    expected_shape = (times.size, len(components))
    if table.shape != expected_shape:
        raise ValueError(f"{runner_name} returned shape {table.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{runner_name} returned non-finite values")
    return table


def _write_model_schematic_if_possible(case_spec: dict) -> dict | None:
    required = ("source_start", "source_end", "receiver", "output_dir")
    if not all(name in case_spec for name in required):
        return None
    from atem3d.model_schematic import write_model_schematic

    return write_model_schematic(case_spec, Path(case_spec["output_dir"]) / "model_schematic.png")


def _update_runtime_diagnostics(
    output_dir: Path,
    runtime_seconds: dict[str, float],
    *,
    model_schematic: dict | None = None,
) -> None:
    import json

    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["runtime_seconds"] = dict(runtime_seconds)
    if model_schematic is not None:
        diagnostics["model_schematic"] = dict(model_schematic)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")


def _material_from_case_spec(case_spec: dict) -> PronyConductivity | None:
    if str(case_spec.get("case_type")) != "ip":
        return None
    material = dict(case_spec.get("material") or {})
    if not material:
        return None
    if "terms" in material:
        terms = [
            DebyeTerm(delta_sigma=float(term["delta_sigma"]), tau=float(term["tau"]))
            for term in material["terms"]
        ]
    else:
        delta = material.get("delta_sigma_list", [])
        tau = material.get("tau_list", [])
        if len(delta) != len(tau):
            raise ValueError("IP material delta_sigma_list and tau_list must have the same length")
        terms = [
            DebyeTerm(delta_sigma=float(delta_i), tau=float(tau_i))
            for delta_i, tau_i in zip(delta, tau)
        ]
    return PronyConductivity(sigma_inf=float(material["sigma_inf"]), terms=terms)
