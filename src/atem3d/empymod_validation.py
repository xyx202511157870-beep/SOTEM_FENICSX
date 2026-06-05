"""In-memory empymod validation helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .config import build_simulation
from .empymod_compare import (
    EmpymodSurvey,
    build_empymod_survey_from_config,
    make_debye_resistivity_model_from_config,
    run_empymod_reference,
)
from .metrics import (
    component_diagnostics,
    component_group_summary,
    component_pass_report,
    summarize_errors,
)


@dataclass(frozen=True)
class EmpymodValidationResult:
    """Numerical-vs-empymod validation data and report metrics."""

    times: np.ndarray
    numerical: np.ndarray
    reference: np.ndarray
    component_names: list[str]
    components: dict[str, dict[str, float | bool | str | None]]
    diagnostics: dict[str, dict[str, float]]
    metadata: dict[str, Any]
    tolerance: float | None = None
    absolute_tolerance: float | None = None

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""

        relative_linf_max = max(
            values["relative_linf"] for values in self.components.values()
        )
        absolute_linf_max = max(
            values["absolute_linf"] for values in self.components.values()
        )
        passed_values = [values["passed"] for values in self.components.values()]
        passed = None if any(value is None for value in passed_values) else all(passed_values)
        return {
            "n_times": int(self.times.size),
            "n_components": int(self.numerical.shape[1]),
            "tolerance": None if self.tolerance is None else float(self.tolerance),
            "absolute_tolerance": (
                None
                if self.absolute_tolerance is None
                else float(self.absolute_tolerance)
            ),
            "relative_linf_max": float(relative_linf_max),
            "absolute_linf_max": float(absolute_linf_max),
            "passed": passed,
            "components": self.components,
            "component_groups": component_group_summary(self.components),
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EmpymodValidationSweepResult:
    """A set of named empymod validation cases run from one base config."""

    cases: dict[str, EmpymodValidationResult]
    overrides: dict[str, dict[str, Any]]

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable sweep report."""

        case_reports = {
            name: {
                "overrides": self.overrides[name],
                "report": result.to_report(),
            }
            for name, result in self.cases.items()
        }
        return {
            "n_cases": len(self.cases),
            "passed": _combined_passed(
                case["report"]["passed"] for case in case_reports.values()
            ),
            "cases": case_reports,
        }


def run_empymod_validation(
    config: dict[str, Any],
    *,
    depths: list[float],
    resistivities: list[float],
    signal: int | None = -1,
    use_config_ip: bool = False,
    positive_times_only: bool = True,
    skip_positive_times: int = 0,
    time_min: float | None = None,
    time_max: float | None = None,
    data_only: bool = False,
    tolerance: float | None = None,
    absolute_tolerance: float | None = None,
    runner: Callable[[dict[str, Any]], Any] | None = None,
    reference_runner: Callable[[EmpymodSurvey], np.ndarray] | None = None,
    empymod_kwargs: dict[str, Any] | None = None,
    empymod_strength: float | None = None,
    output_path: str | Path | None = None,
) -> EmpymodValidationResult:
    """Run a config and compare its receiver data to empymod in memory."""

    if skip_positive_times < 0:
        raise ValueError("skip_positive_times must be nonnegative")
    if time_min is not None and time_max is not None and time_min > time_max:
        raise ValueError("time_min must be <= time_max")
    if tolerance is not None and tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if absolute_tolerance is not None and absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be nonnegative")
    run = runner or (lambda cfg: _run_config(cfg, data_only=data_only))
    result = run(config)
    times = np.asarray(result.times, dtype=float)
    numerical = np.asarray(result.data, dtype=float)
    if numerical.shape[0] != times.size:
        raise ValueError("result.data row count must match result.times")

    mask = np.ones(times.shape, dtype=bool)
    if positive_times_only:
        mask = times > 0.0
    if skip_positive_times:
        selected = np.flatnonzero(mask)
        mask[selected[:skip_positive_times]] = False
    time_atol = _time_selection_atol(times, time_min, time_max)
    if time_min is not None:
        mask &= times >= float(time_min) - time_atol
    if time_max is not None:
        mask &= times <= float(time_max) + time_atol
    if not np.any(mask):
        raise ValueError("time selection produced no samples")

    survey, names = build_empymod_survey_from_config(
        config,
        times=times[mask],
        depths=depths,
        resistivities=resistivities,
        signal=signal,
    )
    reference_resistivities = (
        make_debye_resistivity_model_from_config(config, depths)
        if use_config_ip
        else survey.resistivities
    )
    reference_strength = (
        survey.strength if empymod_strength is None else float(empymod_strength)
    )
    survey = EmpymodSurvey(
        source_start=survey.source_start,
        source_end=survey.source_end,
        receiver_locations=survey.receiver_locations,
        components=survey.components,
        times=survey.times,
        depths=survey.depths,
        resistivities=reference_resistivities,
        strength=reference_strength,
        signal=survey.signal,
        receiver_components=survey.receiver_components,
        coordinate_system=survey.coordinate_system,
    )
    if reference_runner is None:
        reference = run_empymod_reference(survey, **(empymod_kwargs or {}))
    else:
        reference = reference_runner(survey)
    reference = np.asarray(reference, dtype=float)
    numerical_window = numerical[mask]
    summary = summarize_errors(numerical_window, reference, names)
    components = component_pass_report(
        summary,
        relative_tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    diagnostics = component_diagnostics(numerical_window, reference, names)
    formulation = str(config.get("formulation", "eb")).lower()
    metadata = {
        "empymod": {
            "depths": [float(depth) for depth in depths],
            "resistivities": [float(resistivity) for resistivity in resistivities],
            "signal": signal,
            "use_config_ip": bool(use_config_ip),
            "coordinate_system": survey.coordinate_system,
            "strength": float(survey.strength),
            "skip_positive_times": int(skip_positive_times),
            "time_window": {
                "min": None if time_min is None else float(time_min),
                "max": None if time_max is None else float(time_max),
            },
            "data_only": bool(data_only),
        },
        "atem3d": {
            "formulation": formulation,
            "initial_magnetic_field": str(config.get("initial_magnetic_field", "ampere")),
            "magnetic_receiver_mode": _metadata_magnetic_receiver_mode(config, formulation),
            "magnetic_recovery_subdivisions": int(
                config.get("magnetic_recovery_subdivisions", 1)
            ),
            "magnetic_recovery_polarization_scale": _metadata_scale(
                config.get("magnetic_recovery_polarization_scale", 1.0)
            ),
            "magnetic_recovery_initial_polarization_scale": float(
                config.get("magnetic_recovery_initial_polarization_scale", 0.0)
            ),
            "magnetic_recovery_source_primary_delta6": bool(
                config.get("magnetic_recovery_source_primary_delta6", False)
            ),
            "magnetic_recovery_source_primary_delta6_basis": str(
                config.get("magnetic_recovery_source_primary_delta6_basis", "wire")
            ),
            "magnetic_recovery_source_history": _metadata_source_history(
                config.get("magnetic_recovery_source_history")
            ),
        },
    }
    validation = EmpymodValidationResult(
        times=times[mask],
        numerical=numerical_window,
        reference=reference,
        component_names=names,
        components=components,
        diagnostics=diagnostics,
        metadata=metadata,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(validation.to_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return validation


def run_empymod_validation_sweep(
    config: dict[str, Any],
    cases: list[dict[str, Any]] | dict[str, dict[str, Any]],
    *,
    output_path: str | Path | None = None,
    **validation_kwargs,
) -> EmpymodValidationSweepResult:
    """Run named validation cases by deep-merging overrides into a base config."""

    normalized_cases = _normalize_sweep_cases(cases)
    results: dict[str, EmpymodValidationResult] = {}
    overrides_by_name: dict[str, dict[str, Any]] = {}
    for name, overrides in normalized_cases:
        case_config = _deep_merge(config, overrides)
        results[name] = run_empymod_validation(case_config, **validation_kwargs)
        overrides_by_name[name] = deepcopy(overrides)

    sweep = EmpymodValidationSweepResult(results, overrides_by_name)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sweep.to_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return sweep


def _run_config(config: dict[str, Any], *, data_only: bool):
    simulation = build_simulation(config)
    if data_only and hasattr(simulation, "run_data_only"):
        return simulation.run_data_only()
    return simulation.run()


def _metadata_magnetic_receiver_mode(config: Mapping[str, Any], formulation: str) -> str:
    if "magnetic_receiver_mode" in config:
        return str(config["magnetic_receiver_mode"])
    return "stored_h" if formulation == "hj" else "stored_b"


def _combined_passed(values) -> bool | None:
    states = list(values)
    if any(value is False for value in states):
        return False
    if any(value is None for value in states):
        return None
    return True


def _time_selection_atol(times: np.ndarray, *bounds: float | None) -> float:
    finite_values = [float(np.max(np.abs(times)))] if times.size else [0.0]
    finite_values.extend(abs(float(value)) for value in bounds if value is not None)
    return 100.0 * np.finfo(float).eps * max(1.0, *finite_values)


def _normalize_sweep_cases(
    cases: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(cases, Mapping):
        iterable = [
            {"name": str(name), "overrides": overrides}
            for name, overrides in cases.items()
        ]
    else:
        iterable = list(cases)
    if not iterable:
        raise ValueError("sweep cases must not be empty")

    normalized: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, case in enumerate(iterable):
        if not isinstance(case, Mapping):
            raise ValueError("each sweep case must be a mapping")
        name = str(case.get("name", f"case_{index}"))
        if not name:
            raise ValueError("sweep case name must not be empty")
        if name in seen:
            raise ValueError(f"duplicate sweep case name: {name}")
        overrides = case.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("sweep case overrides must be a mapping")
        normalized.append((name, dict(overrides)))
        seen.add(name)
    return normalized


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    _merge_into(merged, overrides)
    return merged


def _metadata_scale(value: Any) -> float | str | list[float]:
    if isinstance(value, str):
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return [float(item) for item in array]
    return float(array)


def _metadata_source_history(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("magnetic_recovery_source_history must be a mapping")
    if "terms" in value:
        terms = value["terms"]
        if not isinstance(terms, list) or not terms:
            raise ValueError("magnetic_recovery_source_history.terms must be a nonempty list")
        return {
            "term_count": len(terms),
            "terms": [_metadata_source_history(term) for term in terms],
        }
    kind = str(value.get("kind", "prescribed_source_moments"))
    if kind == "initial_polarization_source_moments":
        return {
            "diagnostic_only": True,
            "kind": kind,
            "requires_ip": _metadata_source_history_requires_ip(kind),
            "source_moment_degrees": [
                int(degree) for degree in value.get("source_moment_degrees", [0, 2])
            ],
            "receiver_matrix": str(value.get("receiver_matrix", "auto")),
            "projection": str(value.get("projection", "receiver_l2")),
        }
    if kind == "charge_conserving_initial_polarization_source_moments":
        return {
            "diagnostic_only": True,
            "kind": kind,
            "requires_ip": _metadata_source_history_requires_ip(kind),
            "source_moment_degrees": [
                int(degree) for degree in value.get("source_moment_degrees", [0, 2])
            ],
            "receiver_matrix": str(value.get("receiver_matrix", "auto")),
            "projection": str(value.get("projection", "receiver_l2")),
        }
    if kind == "driven_recovery_source_moments":
        coefficients = value.get("coefficients", [])
        metadata = {
            "diagnostic_only": True,
            "kind": kind,
            "requires_ip": _metadata_source_history_requires_ip(kind),
            "driver_tau": float(value["driver_tau"]),
            "source_moment_degrees": [
                int(degree) for degree in value.get("source_moment_degrees", [0, 2])
            ],
            "coefficient_count": len(coefficients),
            "receiver_matrix": str(value.get("receiver_matrix", "auto")),
        }
        if "response_taus" in value:
            metadata["response_taus"] = [
                float(response_tau) for response_tau in value["response_taus"]
            ]
        else:
            metadata["response_tau"] = float(value["response_tau"])
        return metadata
    if kind == "source_diffusion_kernel_source_moments":
        coefficients = value.get("coefficients")
        has_normalized = "normalized_amplitude" in value
        metadata = {
            "diagnostic_only": True,
            "kind": kind,
            "requires_ip": _metadata_source_history_requires_ip(kind),
            "tau_multiplier": float(value.get("tau_multiplier", 1.0)),
            "amplitude_time": float(value.get("amplitude_time", 0.0)),
            "basis_kind": str(value.get("basis_kind", "continuous")),
            "source_moment_degrees": [
                int(degree) for degree in value.get("source_moment_degrees", [0])
            ],
            "receiver_matrix": str(value.get("receiver_matrix", "auto")),
        }
        if has_normalized:
            metadata["normalized_amplitude"] = float(value["normalized_amplitude"])
            metadata["coefficient_count"] = 1
        elif coefficients is None:
            metadata["amplitude"] = float(value["amplitude"])
            metadata["coefficient_count"] = 1
        else:
            metadata["coefficient_count"] = len(coefficients)
        return metadata
    coefficients = value.get("coefficients", [])
    return {
        "diagnostic_only": True,
        "kind": kind,
        "requires_ip": _metadata_source_history_requires_ip(kind),
        "tau": float(value["tau"]),
        "max_order": int(value.get("max_order", 1)),
        "source_moment_degrees": [
            int(degree) for degree in value.get("source_moment_degrees", [0, 2])
        ],
        "coefficient_count": len(coefficients),
        "receiver_matrix": str(value.get("receiver_matrix", "auto")),
    }


def _metadata_source_history_requires_ip(kind: str) -> bool:
    return str(kind).strip().lower() != "source_diffusion_kernel_source_moments"


def _merge_into(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _merge_into(target[key], value)
        else:
            target[key] = deepcopy(value)
