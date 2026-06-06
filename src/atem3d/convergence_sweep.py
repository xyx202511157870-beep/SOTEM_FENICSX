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
