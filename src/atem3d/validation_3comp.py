"""Three-component validation artifact writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json

import numpy as np

from atem3d.materials.prony import PronyConductivity
from atem3d.metrics import robust_component_errors
from atem3d.yaml_io import safe_dump_yaml

REQUIRED_TIME_MIN = 1.0e-5
REQUIRED_TIME_MAX = 1.0
FINAL_ACCEPTANCE_SCOPE = "corrected_model_full"
FINAL_ACCEPTANCE_REFERENCE_TYPES = {"empymod", "1d"}
DIAGNOSTIC_REFERENCE_TYPES = {
    "dolfinx_refined",
    "self_convergence",
    "manufactured",
    "published_response_curve",
}
SUPPORTED_REFERENCE_TYPES = FINAL_ACCEPTANCE_REFERENCE_TYPES | DIAGNOSTIC_REFERENCE_TYPES


@dataclass(frozen=True)
class ThreeComponentValidationInput:
    """Input table for no-IP/IP three-component validation artifacts."""

    output_dir: str | Path
    times: np.ndarray
    predictions: np.ndarray
    reference: np.ndarray
    component_names: list[str]
    case_type: str
    reference_type: str
    magnetic_quantity: str
    threshold: float = 0.05
    diagnostics: dict = field(default_factory=dict)
    resolved_config: dict = field(default_factory=dict)
    material: PronyConductivity | None = None
    validation_scope: str = "smoke"


def write_three_component_validation_artifacts(case: ThreeComponentValidationInput) -> dict:
    """Write validation CSV, JSON, and plot artifacts."""

    output_dir = Path(case.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times, predictions, reference, component_names = _validated_arrays(case)
    rows, summary = robust_component_errors(
        times,
        predictions,
        reference,
        component_names,
        threshold=case.threshold,
    )
    summary = _augment_summary(case, summary)
    acceptance_status = validation_acceptance_status(
        times,
        component_names,
        summary,
        case_type=case.case_type,
        reference_type=case.reference_type,
        threshold=case.threshold,
        validation_scope=case.validation_scope,
    )
    summary["acceptance_status"] = acceptance_status
    summary["full_window_covered"] = bool(acceptance_status["full_window_covered"])
    summary["required_components_present"] = bool(acceptance_status["required_components_present"])
    summary["final_acceptance_passed"] = bool(acceptance_status["final_acceptance_passed"])

    _write_response_csv(output_dir / "predictions.csv", times, predictions, component_names)
    _write_response_csv(output_dir / "reference_empymod_or_1d.csv", times, reference, component_names)
    _write_error_csv(output_dir / "errors.csv", rows)
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnostics = dict(case.diagnostics)
    diagnostics.setdefault("case_type", case.case_type)
    diagnostics.setdefault("reference_type", case.reference_type)
    diagnostics.setdefault("components", component_names)
    diagnostics["acceptance_status"] = acceptance_status
    diagnostics["validation_failure"] = _automatic_failure_diagnostics(
        summary,
        diagnostics,
        rows=rows,
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "run_config_resolved.yaml").write_text(
        safe_dump_yaml(_resolved_config(case, component_names), sort_keys=True),
        encoding="utf-8",
    )
    _plot_comparison(
        output_dir / "comparison_3comp.png",
        times,
        predictions,
        reference,
        component_names,
    )
    _plot_errors(
        output_dir / "error_curves_3comp.png",
        rows,
        component_names,
        threshold=case.threshold,
    )
    return summary


def _resolved_config(case: ThreeComponentValidationInput, component_names: list[str]) -> dict:
    values = dict(case.resolved_config)
    values.update(
        {
            "case_type": case.case_type,
            "reference_type": case.reference_type,
            "magnetic_quantity": case.magnetic_quantity,
            "component_names": component_names,
            "relative_error_threshold": float(case.threshold),
            "validation_scope": case.validation_scope,
        }
    )
    if case.material is not None:
        values["material"] = {
            "sigma_inf": float(case.material.sigma_inf),
            "sigma0": float(case.material.sigma0),
            "terms": [
                {"delta_sigma": float(term.delta_sigma), "tau": float(term.tau)}
                for term in case.material.terms
            ],
        }
    return values


def _validated_arrays(
    case: ThreeComponentValidationInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    times = np.asarray(case.times, dtype=float)
    predictions = np.asarray(case.predictions, dtype=float)
    reference = np.asarray(case.reference, dtype=float)
    component_names = [str(name) for name in case.component_names]
    magnetic_quantity = str(case.magnetic_quantity)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(times <= 0.0):
        raise ValueError("times must be positive observation times")
    if np.min(times) > REQUIRED_TIME_MIN or np.max(times) < REQUIRED_TIME_MAX:
        raise ValueError("validation times must cover 1e-5 s to 1 s")
    if predictions.shape != reference.shape:
        raise ValueError("predictions and reference must have the same shape")
    if predictions.shape != (times.size, len(component_names)):
        raise ValueError("responses must have shape (n_times, n_components)")
    if case.reference_type == "published_response_curve":
        if len(component_names) < 2:
            raise ValueError("published_response_curve validation requires at least two components")
    elif len(component_names) < 3:
        raise ValueError("at least three components are required")
    if case.case_type not in {"noip", "ip"}:
        raise ValueError("case_type must be 'noip' or 'ip'")
    if case.reference_type not in SUPPORTED_REFERENCE_TYPES:
        names = ", ".join(sorted(SUPPORTED_REFERENCE_TYPES))
        raise ValueError(f"reference_type must be one of: {names}")
    if magnetic_quantity not in {"Hz", "dBzdt"}:
        raise ValueError("magnetic_quantity must be 'Hz' or 'dBzdt'")
    if magnetic_quantity not in component_names:
        raise ValueError("magnetic_quantity must be present in component_names")
    return times, predictions, reference, component_names


def validation_acceptance_status(
    times,
    component_names,
    summary: dict,
    *,
    case_type: str,
    reference_type: str,
    threshold: float,
    validation_scope: str = "smoke",
    required_time_min: float = REQUIRED_TIME_MIN,
    required_time_max: float = REQUIRED_TIME_MAX,
) -> dict:
    """Return the final-acceptance gate status for a validation table."""

    times = np.asarray(times, dtype=float)
    component_names = [str(name) for name in component_names]
    time_min = float(np.min(times)) if times.size else float("nan")
    time_max = float(np.max(times)) if times.size else float("nan")
    full_window_covered = bool(
        times.size > 0 and time_min <= float(required_time_min) and time_max >= float(required_time_max)
    )
    electric_present = {"Ex", "Ey"}.issubset(component_names)
    magnetic_present = any(name in component_names for name in ("Hz", "dBzdt"))
    required_components_present = bool(electric_present and magnetic_present)
    scope_is_final = str(validation_scope) == FINAL_ACCEPTANCE_SCOPE
    case_type_ok = case_type in {"noip", "ip"}
    reference_type_supported = reference_type in SUPPORTED_REFERENCE_TYPES
    reference_type_ok = reference_type in FINAL_ACCEPTANCE_REFERENCE_TYPES
    threshold_ok = float(threshold) <= 0.05
    physical_gate_passed = bool(summary.get("physical_pass_all_components", summary.get("pass_all_components", False)))
    strict_gate_passed = bool(summary.get("pass_all_components", False))

    blocking_reasons: list[str] = []
    if not scope_is_final:
        blocking_reasons.append("validation_scope_not_corrected_model_full")
    if not full_window_covered:
        blocking_reasons.append("time_window_not_covered")
    if not required_components_present:
        blocking_reasons.append("required_components_missing")
    if not case_type_ok:
        blocking_reasons.append("case_type_invalid")
    if not reference_type_supported:
        blocking_reasons.append("reference_type_invalid")
    elif not reference_type_ok:
        blocking_reasons.append("reference_type_not_final_acceptance")
    if not threshold_ok:
        blocking_reasons.append("threshold_above_5pct")
    if not physical_gate_passed:
        blocking_reasons.append("physical_error_gate_failed")

    final_acceptance_passed = bool(
        scope_is_final
        and full_window_covered
        and required_components_present
        and case_type_ok
        and reference_type_ok
        and threshold_ok
        and physical_gate_passed
    )
    return {
        "validation_scope": str(validation_scope),
        "final_acceptance_scope": FINAL_ACCEPTANCE_SCOPE,
        "time_min": time_min,
        "time_max": time_max,
        "required_time_min": float(required_time_min),
        "required_time_max": float(required_time_max),
        "full_window_covered": full_window_covered,
        "electric_components_present": bool(electric_present),
        "magnetic_component_present": bool(magnetic_present),
        "required_components_present": required_components_present,
        "case_type_ok": bool(case_type_ok),
        "reference_type_ok": bool(reference_type_ok),
        "reference_type_supported": bool(reference_type_supported),
        "final_acceptance_reference_types": sorted(FINAL_ACCEPTANCE_REFERENCE_TYPES),
        "diagnostic_reference_types": sorted(DIAGNOSTIC_REFERENCE_TYPES),
        "threshold_requirement_met": bool(threshold_ok),
        "strict_error_gate_passed": strict_gate_passed,
        "physical_error_gate_passed": physical_gate_passed,
        "final_acceptance_passed": final_acceptance_passed,
        "blocking_reasons": blocking_reasons,
    }


def _augment_summary(case: ThreeComponentValidationInput, summary: dict) -> dict:
    values = dict(summary)
    values["case_type"] = case.case_type
    values["reference_type"] = case.reference_type
    values["magnetic_quantity"] = case.magnetic_quantity
    values["validation_scope"] = case.validation_scope
    _bind_declared_magnetic_aliases(values, str(case.magnetic_quantity))
    if case.material is not None:
        terms = list(case.material.terms)
        sum_delta = float(sum(term.delta_sigma for term in terms))
        values.update(
            {
                "sigma0": float(case.material.sigma0),
                "sigma_inf": float(case.material.sigma_inf),
                "sum_delta_sigma": sum_delta,
                "tau_list": [float(term.tau) for term in terms],
                "delta_sigma_list": [float(term.delta_sigma) for term in terms],
                "prony_dc_constraint_error": float(
                    case.material.sigma_inf - sum_delta - case.material.sigma0
                ),
            }
        )
    return values


def _bind_declared_magnetic_aliases(summary: dict, magnetic_quantity: str) -> None:
    for prefix in ("max_error", "rms_error", "max_peak_normalized_error"):
        source_key = f"{prefix}_{magnetic_quantity}"
        alias_key = f"{prefix}_Hz_or_dBzdt"
        if source_key in summary:
            summary[alias_key] = summary[source_key]


def _automatic_failure_diagnostics(
    summary: dict,
    diagnostics: dict | None = None,
    *,
    rows: np.ndarray | None = None,
) -> dict:
    failed = not bool(summary.get("physical_pass_all_components", summary.get("pass_all_components", False)))
    acceptance_status = dict(summary.get("acceptance_status", {}))
    diagnostics = dict(diagnostics or {})
    reason_codes = list(acceptance_status.get("blocking_reasons", []))
    if failed and "physical_error_gate_failed" not in reason_codes:
        reason_codes.append("physical_error_gate_failed")
    if not bool(summary.get("pass_all_components", False)) and "strict_error_gate_failed" not in reason_codes:
        reason_codes.append("strict_error_gate_failed")
    failure = {
        "failed": failed,
        "strict_failed": not bool(summary.get("pass_all_components", False)),
        "final_acceptance_passed": bool(summary.get("final_acceptance_passed", False)),
        "acceptance_blocking_reasons": list(acceptance_status.get("blocking_reasons", [])),
        "reason_codes": reason_codes,
        "failed_components": list(summary.get("failed_components", [])),
        "physical_failed_components": list(
            summary.get("physical_failed_components", summary.get("failed_components", []))
        ),
        "failed_times": list(summary.get("failed_times", [])),
        "recommended_check_order": [
            "time_step_error",
            "mesh_error",
            "boundary_error",
            "source_term_error",
            "receiver_sampling_error",
            "magnetic_recovery_error",
            "ip_memory_error",
        ],
        "checks": _task_book_failure_checks(summary, diagnostics, failed=failed),
        "weak_component_passed": bool(summary.get("weak_component_passed", True)),
        "weak_components": list(summary.get("weak_components", [])),
    }
    convergence = _convergence_failure_diagnostic(summary, diagnostics, rows=rows)
    if convergence:
        failure["convergence_diagnostic"] = convergence
    return failure


def _convergence_failure_diagnostic(
    summary: dict,
    diagnostics: dict,
    *,
    rows: np.ndarray | None,
) -> dict:
    reference_type = str(summary.get("reference_type", ""))
    if reference_type not in DIAGNOSTIC_REFERENCE_TYPES:
        return {}
    convergence = dict(diagnostics.get("convergence_reference", {}))
    physical_components = [
        str(component)
        for component in summary.get(
            "physical_failed_components",
            summary.get("failed_components", []),
        )
    ]
    failed_times = _physical_failed_times_from_rows(rows, physical_components)
    if not failed_times:
        failed_times = [float(value) for value in summary.get("failed_times", [])]
    acceptance = dict(summary.get("acceptance_status", {}))
    time_min = float(acceptance.get("time_min", min(failed_times) if failed_times else 0.0))
    time_max = float(acceptance.get("time_max", max(failed_times) if failed_times else time_min))
    if time_min > 0.0 and time_max > time_min:
        time_split = float(np.sqrt(time_min * time_max))
    else:
        time_split = float(0.5 * (time_min + time_max))
    early_failed = [time for time in failed_times if time <= time_split]
    late_failed = [time for time in failed_times if time > time_split]
    if late_failed and not early_failed:
        band = "late_time"
    elif early_failed and late_failed:
        band = "broadband"
    elif early_failed:
        band = "early_time"
    else:
        band = "none"
    return {
        "reference_type": reference_type,
        "failed_time_band": band,
        "time_split": time_split,
        "failed_times": failed_times,
        "early_failed_times": early_failed,
        "late_failed_times": late_failed,
        "physical_failed_components": physical_components,
        "prediction_cells": list(convergence.get("prediction_cells", [])),
        "reference_cells": list(convergence.get("reference_cells", [])),
        "recommended_action": _convergence_recommended_action(band),
    }


def _physical_failed_times_from_rows(rows: np.ndarray | None, physical_components: list[str]) -> list[float]:
    if rows is None or not physical_components:
        return []
    components = set(physical_components)
    times = {
        float(row["time_obs"])
        for row in rows
        if str(row["component"]) in components and not bool(row["pass_5pct"])
    }
    return sorted(times)


def _convergence_recommended_action(failed_time_band: str) -> str:
    if failed_time_band == "late_time":
        return (
            "run boundary/domain-size sweeps and refine the late-time diffusion region "
            "before increasing source/receiver local resolution"
        )
    if failed_time_band == "early_time":
        return (
            "refine source, receiver, and leakage-channel cells and audit source loading "
            "before running larger-domain late-time sweeps"
        )
    if failed_time_band == "broadband":
        return "run mesh, time-step, and boundary sweeps because failures span early and late times"
    return "inspect convergence artifacts; no physical failed times were isolated"


def _task_book_failure_checks(summary: dict, diagnostics: dict, *, failed: bool) -> dict[str, dict]:
    base_status = "needs_evaluation" if failed else "not_required"
    source_evidence = {}
    source_evidence.update(dict(diagnostics.get("source_consistency", {})))
    source_evidence.update(dict(diagnostics.get("source_projection", {})))
    receiver_evidence = dict(diagnostics.get("receiver_sampling", {}))
    receiver_evidence.update(
        {"receiver_vs_reference": dict(diagnostics.get("receiver_vs_reference", {}))}
    )
    magnetic_evidence = {
        "magnetic_quantity": str(summary.get("magnetic_quantity", "")),
        "magnetic_components": list(summary.get("magnetic_components", [])),
        "magnetic_receiver_mode": diagnostics.get(
            "magnetic_receiver_mode",
            diagnostics.get("magnetic_dbdt_mode", ""),
        ),
    }
    ip_evidence = {
        key: summary[key]
        for key in (
            "sigma0",
            "sigma_inf",
            "sum_delta_sigma",
            "tau_list",
            "delta_sigma_list",
            "prony_dc_constraint_error",
        )
        if key in summary
    }
    return {
        "time_step_error": _diagnostic_check(
            status=base_status,
            evidence={
                "failed_times": list(summary.get("failed_times", [])),
                "time_min": dict(summary.get("acceptance_status", {})).get("time_min"),
                "time_max": dict(summary.get("acceptance_status", {})).get("time_max"),
            },
            recommended_action="rerun with denser internal time steps and compare error reduction",
        ),
        "mesh_error": _diagnostic_check(
            status=base_status,
            evidence=dict(diagnostics.get("mesh", diagnostics.get("mesh_diagnostics", {}))),
            recommended_action="refine source, receiver, anomaly, and skin-depth controlled cells",
        ),
        "boundary_error": _diagnostic_check(
            status=base_status,
            evidence=dict(diagnostics.get("boundary", diagnostics.get("boundary_diagnostics", {}))),
            recommended_action="run domain-size and Robin/absorbing-boundary parameter sweeps",
        ),
        "source_term_error": _diagnostic_check(
            status=base_status,
            evidence=source_evidence,
            recommended_action="inspect endpoint balance, DC conservation, initial curl, and waveform integral residuals",
        ),
        "receiver_sampling_error": _diagnostic_check(
            status=base_status,
            evidence=receiver_evidence,
            recommended_action="compare point receiver against volume or disk average receiver outputs",
        ),
        "magnetic_recovery_error": _diagnostic_check(
            status=base_status,
            evidence=magnetic_evidence,
            recommended_action="compare Biot-Savart, Faraday-integrated H, and curl-derived dB/dt recovery",
        ),
        "ip_memory_error": _diagnostic_check(
            status=base_status if str(summary.get("case_type", "")) == "ip" else "not_applicable",
            evidence=(
                ip_evidence
                if str(summary.get("case_type", "")) == "ip"
                else {"case_type": summary.get("case_type", "")}
            ),
            recommended_action="verify chi0, sigma0, Prony DC constraint, and delta_sigma=0 no-IP degeneration",
        ),
    }


def _diagnostic_check(*, status: str, evidence: dict, recommended_action: str) -> dict:
    return {
        "status": status,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def _write_response_csv(
    path: Path,
    times: np.ndarray,
    values: np.ndarray,
    component_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_obs", *component_names])
        for time, row in zip(times, values):
            writer.writerow([float(time), *[float(value) for value in row]])


def _write_error_csv(path: Path, rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows.dtype.names)
        for row in rows:
            writer.writerow([_csv_scalar(row[name]) for name in rows.dtype.names])


def _csv_scalar(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plot_comparison(
    path: Path,
    times: np.ndarray,
    predictions: np.ndarray,
    reference: np.ndarray,
    component_names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(component_names), 1, figsize=(7, 2.4 * len(component_names)))
    axes = np.atleast_1d(axes)
    for index, (axis, component) in enumerate(zip(axes, component_names)):
        ref_plot = np.maximum(np.abs(reference[:, index]), np.finfo(float).tiny)
        pred_plot = np.maximum(np.abs(predictions[:, index]), np.finfo(float).tiny)
        axis.loglog(times, ref_plot, label="reference")
        axis.loglog(times, pred_plot, "--", label="prediction")
        axis.set_ylabel(component)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("time_obs (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_errors(
    path: Path,
    rows: np.ndarray,
    component_names: list[str],
    *,
    threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    for component in component_names:
        mask = rows["component"] == component
        error_plot = np.maximum(rows["relative_error_with_floor"][mask], np.finfo(float).tiny)
        axis.loglog(
            rows["time_obs"][mask],
            error_plot,
            marker="o",
            label=component,
        )
    axis.axhline(threshold, color="k", linestyle=":", linewidth=1.5, label=f"{threshold:g}")
    axis.set_xlabel("time_obs (s)")
    axis.set_ylabel("relative_error_with_floor")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
