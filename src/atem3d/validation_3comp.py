"""Three-component validation artifact writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json

import numpy as np

from atem3d.materials.prony import PronyConductivity
from atem3d.metrics import robust_component_errors

REQUIRED_TIME_MIN = 1.0e-5
REQUIRED_TIME_MAX = 1.0


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
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "run_config_resolved.yaml").write_text(
        json.dumps(_resolved_config(case, component_names), indent=2, sort_keys=True),
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
    if len(component_names) < 3:
        raise ValueError("at least three components are required")
    if case.case_type not in {"noip", "ip"}:
        raise ValueError("case_type must be 'noip' or 'ip'")
    if case.reference_type not in {"empymod", "1d"}:
        raise ValueError("reference_type must be 'empymod' or '1d'")
    return times, predictions, reference, component_names


def _augment_summary(case: ThreeComponentValidationInput, summary: dict) -> dict:
    values = dict(summary)
    values["case_type"] = case.case_type
    values["reference_type"] = case.reference_type
    values["magnetic_quantity"] = case.magnetic_quantity
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
