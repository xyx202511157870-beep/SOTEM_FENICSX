"""Corrected-model validation runner orchestration."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
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
    predictions = _validate_response_table(forward(spec), times, components, "forward_runner")
    reference_values = _validate_response_table(reference(spec), times, components, "reference_runner")
    material = _material_from_case_spec(spec)
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
            "runner": spec.get("runner", {}),
            "source_start": spec.get("source_start"),
            "source_end": spec.get("source_end"),
            "receiver": spec.get("receiver"),
        },
        resolved_config=spec,
        material=material,
        validation_scope=str(spec.get("validation_scope", "smoke")),
    )
    return write_three_component_validation_artifacts(case)


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
    sigma_background = float(case_spec.get("sigma_background", material.sigma0))
    secondary_material = _secondary_material_for_forward(case_spec, material, sigma_background)
    primary = EmpymodPrimaryProvider(
        config=_empymod_primary_config_for_case_spec(case_spec),
        empymod_kwargs=dict(case_spec.get("empymod_kwargs", {})),
    )
    interpolation = sp._nedelec_interpolation_points(msh, spaces)
    fem_points = interpolation["points"]
    receiver_locations = np.asarray([case_spec["receiver"]], dtype=float)
    components = tuple(str(value) for value in case_spec["components"])

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
