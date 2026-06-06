"""Corrected-model validation runner orchestration."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)

ResponseRunner = Callable[[dict], np.ndarray]


def run_corrected_model_validation(
    case_spec: dict,
    *,
    forward_runner: ResponseRunner | None = None,
    reference_runner: ResponseRunner | None = None,
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
        diagnostics={
            **dict(spec.get("diagnostics", {})),
            "runner": spec.get("runner", {}),
            "source_start": spec.get("source_start"),
            "source_end": spec.get("source_end"),
            "receiver": spec.get("receiver"),
            "runtime_seconds": runtime_seconds,
        },
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

    def run_refined_reference(_prediction_case_spec: dict) -> np.ndarray:
        return reference(refined_spec)

    return run_corrected_model_validation(
        prediction_spec,
        forward_runner=runner,
        reference_runner=run_refined_reference,
    )


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


def _run_dolfinx_primary_secondary_forward(case_spec: dict) -> np.ndarray:
    from atem3d.primary import EmpymodPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import fem, mesh
    from mpi4py import MPI

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
    msh = mesh.create_box(MPI.COMM_WORLD, [domain_min, domain_max], cells)
    config = sp.PipelineConfig(
        receiver=tuple(float(value) for value in case_spec["receiver"]),
        receiver_evaluation_mode=str(forward_cfg.get("receiver_evaluation_mode", "first_cell")),
        outer_boundary_mode=str(forward_cfg.get("outer_boundary_mode", "natural")),
        ksp_type=str(forward_cfg.get("ksp_type", "cg")),
        rtol=float(forward_cfg.get("rtol", 1.0e-8)),
        atol=float(forward_cfg.get("atol", 1.0e-10)),
        max_it=int(forward_cfg.get("max_it", 200)),
    )
    spaces = sp.build_function_spaces(msh, config)
    material = _forward_material_from_case_spec(case_spec)
    sigma_background = float(case_spec.get("sigma_background", _background_sigma_from_case_spec(case_spec)))
    secondary_material = _secondary_material_for_forward(case_spec, material, sigma_background)
    primary = EmpymodPrimaryProvider(
        config=_empymod_primary_config_for_case_spec(case_spec),
        empymod_kwargs=dict(case_spec.get("empymod_kwargs", {})),
    )
    interpolation = sp._nedelec_interpolation_points(msh, spaces)
    fem_points = interpolation["points"]
    receiver_locations = np.asarray([case_spec["receiver"]], dtype=float)
    components = tuple(str(value) for value in case_spec["components"])

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
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
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
    )
    return built["operator"].forward(np.asarray(case_spec["observation_times"], dtype=float))


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
) -> np.ndarray:
    from atem3d.materials.material_map import (
        CellMaterialMap,
        apply_leakage_channel_marker,
    )

    leakage_cfg = dict(forward_cfg["leakage_channel"])
    centers = sp._cell_centers(msh)
    background_marker = int(leakage_cfg.get("background_marker", 1))
    leakage_marker = int(leakage_cfg.get("leakage_marker", 7))
    markers = np.full(centers.shape[0], background_marker, dtype=int)
    markers = apply_leakage_channel_marker(
        markers,
        centers,
        channel_points=np.asarray(leakage_cfg["points"], dtype=float),
        radius=float(leakage_cfg["radius"]),
        leakage_marker=leakage_marker,
    )
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
    built_materials = sp._make_dolfinx_materials_from_cell_material_map(
        msh,
        spaces,
        material_map,
        dt=float(times[0]),
    )
    operators = sp.assemble_operators(
        msh,
        spaces,
        built_materials["materials"],
        facet_tags=None,
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
    )
    return built_operator["operator"].forward(times)


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
