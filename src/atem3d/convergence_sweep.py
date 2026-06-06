"""Summaries for corrected-model convergence sweep artifact directories."""

from __future__ import annotations

from pathlib import Path
import csv
import json


def write_convergence_sweep_report(
    run_dirs,
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
) -> dict:
    """Read convergence validation runs and write CSV/JSON sweep summaries."""

    paths = [Path(path) for path in run_dirs]
    if not paths:
        raise ValueError("at least one convergence run directory is required")
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels length must match run_dirs length")

    runs = [
        _summarize_run(path, labels[index] if labels is not None else path.name)
        for index, path in enumerate(paths)
    ]
    best = min(runs, key=lambda item: float(item["max_physical_error"]))
    summary = {
        "run_count": len(runs),
        "runs": runs,
        "best_by_max_physical_error": dict(best),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "convergence_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_sweep_csv(output / "convergence_sweep_summary.csv", runs)
    return summary


def _summarize_run(run_dir: Path, label: str) -> dict:
    summary = _read_json(run_dir / "error_summary.json")
    diagnostics = _read_json(run_dir / "diagnostics.json")
    failure = dict(diagnostics.get("validation_failure", {}))
    convergence = dict(failure.get("convergence_diagnostic", {}))
    marker = dict(diagnostics.get("leakage_marker_preflight", {}))
    marker_prediction = dict(marker.get("prediction", {}))
    marker_reference = dict(marker.get("reference", {}))
    prediction_marker_count = int(marker_prediction.get("leakage_cell_count", 0))
    reference_marker_count = int(marker_reference.get("leakage_cell_count", 0))
    secondary = dict(diagnostics.get("secondary_effect_diagnostic", {}))
    prediction_secondary = dict(secondary.get("max_abs_prediction_minus_primary_by_component", {}))
    reference_secondary = dict(secondary.get("max_abs_reference_minus_primary_by_component", {}))
    runtime = dict(diagnostics.get("runtime_seconds", {}))
    physical_failed = [
        str(component)
        for component in summary.get(
            "physical_failed_components",
            summary.get("failed_components", []),
        )
    ]
    max_physical = _max_physical_error(summary, physical_failed)
    return {
        "label": str(label),
        "run_dir": str(run_dir),
        "case_type": str(summary.get("case_type", "")),
        "reference_type": str(summary.get("reference_type", "")),
        "final_acceptance_passed": bool(summary.get("final_acceptance_passed", False)),
        "physical_pass_all_components": bool(summary.get("physical_pass_all_components", False)),
        "physical_failed_components": physical_failed,
        "failed_time_band": str(convergence.get("failed_time_band", "")),
        "prediction_cells": list(convergence.get("prediction_cells", [])),
        "reference_cells": list(convergence.get("reference_cells", [])),
        "prediction_initial_leakage_cell_count": int(
            marker_prediction.get("initial_leakage_cell_count", prediction_marker_count)
        ),
        "reference_initial_leakage_cell_count": int(
            marker_reference.get("initial_leakage_cell_count", reference_marker_count)
        ),
        "prediction_leakage_cell_count": prediction_marker_count,
        "reference_leakage_cell_count": reference_marker_count,
        "prediction_marker_fallback_used": bool(marker_prediction.get("fallback_used", False)),
        "reference_marker_fallback_used": bool(marker_reference.get("fallback_used", False)),
        "prediction_marker_fallback_added_cell_count": int(
            marker_prediction.get("fallback_added_cell_count", 0)
        ),
        "reference_marker_fallback_added_cell_count": int(
            marker_reference.get("fallback_added_cell_count", 0)
        ),
        "prediction_min_marked_cells": int(marker_prediction.get("min_marked_cells", 0)),
        "reference_min_marked_cells": int(marker_reference.get("min_marked_cells", 0)),
        "leakage_cell_count_ratio": _leakage_cell_count_ratio(
            prediction_marker_count,
            reference_marker_count,
        ),
        "leakage_marker_issue": _leakage_marker_issue(
            marker_prediction,
            prediction_marker_count,
            reference_marker_count,
        ),
        "prediction_nearest_channel_distance_m": float(
            marker_prediction.get("nearest_channel_distance_m", 0.0)
        ),
        "reference_nearest_channel_distance_m": float(
            marker_reference.get("nearest_channel_distance_m", 0.0)
        ),
        "prediction_secondary_effect_nonzero": bool(
            secondary.get("prediction_secondary_effect_nonzero", False)
        ),
        "reference_secondary_effect_nonzero": bool(
            secondary.get("reference_secondary_effect_nonzero", False)
        ),
        "secondary_effect_nonzero": bool(secondary.get("secondary_effect_nonzero", False)),
        "max_prediction_secondary_effect_Ex": float(prediction_secondary.get("Ex", 0.0)),
        "max_prediction_secondary_effect_Ey": float(prediction_secondary.get("Ey", 0.0)),
        "max_prediction_secondary_effect_dBzdt": float(prediction_secondary.get("dBzdt", 0.0)),
        "max_reference_secondary_effect_Ex": float(reference_secondary.get("Ex", 0.0)),
        "max_reference_secondary_effect_Ey": float(reference_secondary.get("Ey", 0.0)),
        "max_reference_secondary_effect_dBzdt": float(reference_secondary.get("dBzdt", 0.0)),
        "max_error_Ex": float(summary.get("max_error_Ex", 0.0)),
        "max_error_Ey": float(summary.get("max_error_Ey", 0.0)),
        "max_error_dBzdt": float(summary.get("max_error_dBzdt", summary.get("max_error_Hz_or_dBzdt", 0.0))),
        "max_physical_error": float(max_physical),
        "runtime_forward_s": float(runtime.get("forward", 0.0)),
        "runtime_reference_s": float(runtime.get("reference", 0.0)),
    }


def _max_physical_error(summary: dict, physical_failed: list[str]) -> float:
    components = physical_failed or ["Ex", "Ey", "dBzdt"]
    values = []
    for component in components:
        if component == "dBzdt":
            value = summary.get("max_error_dBzdt", summary.get("max_error_Hz_or_dBzdt", 0.0))
        else:
            value = summary.get(f"max_error_{component}", 0.0)
        values.append(float(value))
    return max(values) if values else 0.0


def _leakage_cell_count_ratio(prediction_count: int, reference_count: int) -> float:
    if reference_count <= 0:
        return 0.0
    return float(prediction_count / reference_count)


def _leakage_marker_issue(
    marker_prediction: dict,
    prediction_count: int,
    reference_count: int,
) -> str:
    if prediction_count <= 0:
        return "prediction_unmarked"
    if bool(marker_prediction.get("fallback_used", False)):
        return "fallback_used"
    if reference_count > 0 and prediction_count / reference_count < 0.5:
        return "coarse_underrepresented"
    return ""


def _write_sweep_csv(path: Path, runs: list[dict]) -> None:
    columns = [
        "label",
        "run_dir",
        "case_type",
        "reference_type",
        "final_acceptance_passed",
        "physical_pass_all_components",
        "physical_failed_components",
        "failed_time_band",
        "prediction_cells",
        "reference_cells",
        "prediction_initial_leakage_cell_count",
        "reference_initial_leakage_cell_count",
        "prediction_leakage_cell_count",
        "reference_leakage_cell_count",
        "prediction_marker_fallback_used",
        "reference_marker_fallback_used",
        "prediction_marker_fallback_added_cell_count",
        "reference_marker_fallback_added_cell_count",
        "prediction_min_marked_cells",
        "reference_min_marked_cells",
        "leakage_cell_count_ratio",
        "leakage_marker_issue",
        "prediction_nearest_channel_distance_m",
        "reference_nearest_channel_distance_m",
        "prediction_secondary_effect_nonzero",
        "reference_secondary_effect_nonzero",
        "secondary_effect_nonzero",
        "max_prediction_secondary_effect_Ex",
        "max_prediction_secondary_effect_Ey",
        "max_prediction_secondary_effect_dBzdt",
        "max_reference_secondary_effect_Ex",
        "max_reference_secondary_effect_Ey",
        "max_reference_secondary_effect_dBzdt",
        "max_error_Ex",
        "max_error_Ey",
        "max_error_dBzdt",
        "max_physical_error",
        "runtime_forward_s",
        "runtime_reference_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["physical_failed_components"] = ";".join(run["physical_failed_components"])
            row["prediction_cells"] = ";".join(str(value) for value in run["prediction_cells"])
            row["reference_cells"] = ";".join(str(value) for value in run["reference_cells"])
            writer.writerow(row)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
