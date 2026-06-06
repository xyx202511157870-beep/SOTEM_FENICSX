"""Corrected-model validation runner orchestration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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
    raise NotImplementedError(
        "DOLFINx corrected-model primary-secondary forward runner is not wired yet"
    )


def _default_reference_runner(case_spec: dict) -> np.ndarray:
    from atem3d.primary import EmpymodPrimaryProvider

    components = [str(value) for value in case_spec["components"]]
    times = np.asarray(case_spec["observation_times"], dtype=float)
    receiver = np.asarray([case_spec["receiver"]], dtype=float)
    provider = EmpymodPrimaryProvider(
        config=dict(case_spec["empymod_primary"]),
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
