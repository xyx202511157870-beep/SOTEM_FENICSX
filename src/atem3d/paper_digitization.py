"""Published-paper response curve digitization templates."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil

import numpy as np

from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)


PAPER_CURVE_TARGETS = [
    {
        "figure": "Fig. 2",
        "model_key": "accuracy_benchmark_layer",
        "component": "Ex",
        "suggested_curve_labels": ["paper_3d_model", "paper_1d_analytical"],
    },
    {
        "figure": "Fig. 3",
        "model_key": "accuracy_benchmark_layer",
        "component": "Hz",
        "suggested_curve_labels": ["paper_3d_model", "paper_1d_analytical"],
    },
    {
        "figure": "Fig. 7",
        "model_key": "layered_polarization_model",
        "component": "Ex",
        "suggested_curve_labels": ["paper_ip", "paper_noip"],
    },
    {
        "figure": "Fig. 8",
        "model_key": "layered_polarization_model",
        "component": "Hz",
        "suggested_curve_labels": ["paper_ip", "paper_noip"],
    },
    {
        "figure": "Fig. 12",
        "model_key": "three_dimensional_polarized_body",
        "component": "Ex",
        "suggested_curve_labels": ["paper_ip", "paper_noip"],
    },
    {
        "figure": "Fig. 15",
        "model_key": "three_dimensional_polarized_body",
        "component": "Hz",
        "suggested_curve_labels": ["paper_ip", "paper_noip"],
    },
]


def write_published_paper_digitization_template(paper_spec: dict, output_dir: str | Path) -> dict:
    """Write a long-form CSV template for digitizing published response curves."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_name = "paper_curve_digitization_template.csv"
    manifest_name = "paper_curve_digitization_manifest.json"
    targets = _targets_from_spec(paper_spec)
    _write_template_csv(output / csv_name, targets)
    manifest = {
        "source_article_id": str(
            dict(paper_spec.get("published_reference", {})).get("article_id", "")
        ),
        "source_title": str(dict(paper_spec.get("published_reference", {})).get("title", "")),
        "template_csv": csv_name,
        "format": {
            "columns": ["figure", "model_key", "component", "curve_label", "time_obs", "value", "notes"],
            "time_obs": "seconds after current switch-off",
            "value": "digitized published response value in the plotted component units",
        },
        "targets": targets,
    }
    (output / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def write_published_paper_curve_artifacts(
    *,
    predictions_csv: str | Path,
    digitized_csv: str | Path,
    output_dir: str | Path,
    case_type: str,
    curve_label: str,
    component_figures: dict[str, str] | None = None,
) -> dict:
    """Write validation artifacts comparing predictions to digitized paper curves."""

    times, component_names, predictions = _read_prediction_csv(predictions_csv)
    figures = dict(component_figures or {"Ex": "Fig. 12", "Hz": "Fig. 15"})
    selected_components = [name for name in component_names if name in figures]
    if not selected_components:
        raise ValueError("predictions_csv must contain at least one component listed in component_figures")
    selected_indices = [component_names.index(name) for name in selected_components]
    selected_predictions = predictions[:, selected_indices]
    reference = _read_digitized_reference(
        digitized_csv,
        times=times,
        components=selected_components,
        curve_label=curve_label,
        component_figures=figures,
    )
    diagnostics = {
        "published_response_curve": {
            "curve_label": str(curve_label),
            "component_figures": {name: figures[name] for name in selected_components},
            "digitized_csv": str(Path(digitized_csv)),
            "predictions_csv": str(Path(predictions_csv)),
        }
    }
    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=output_dir,
            times=times,
            predictions=selected_predictions,
            reference=reference,
            component_names=selected_components,
            case_type=case_type,
            reference_type="published_response_curve",
            magnetic_quantity="Hz" if "Hz" in selected_components else selected_components[-1],
            diagnostics=diagnostics,
            resolved_config={
                "published_response_curve": diagnostics["published_response_curve"],
            },
            validation_scope="published_paper_reproduction_target",
        )
    )
    _write_published_curve_alias_artifacts(Path(output_dir))
    return summary


def _write_published_curve_alias_artifacts(output_dir: Path) -> None:
    """Write paper-reproduction filenames promised by the paper target spec."""

    aliases = {
        "comparison_3comp.png": "paper_response_overlay.png",
        "error_curves_3comp.png": "paper_relative_error_curves.png",
        "diagnostics.json": "runtime_diagnostics.json",
    }
    for source_name, alias_name in aliases.items():
        source = Path(output_dir) / source_name
        alias = Path(output_dir) / alias_name
        if not source.exists():
            raise FileNotFoundError(f"missing source artifact for paper alias: {source_name}")
        shutil.copyfile(source, alias)


def _targets_from_spec(paper_spec: dict) -> list[dict]:
    available_figures = set(dict(paper_spec.get("paper_response_targets", {})).get("candidate_overlay_figures", []))
    if not available_figures:
        return [dict(target) for target in PAPER_CURVE_TARGETS]
    return [
        dict(target)
        for target in PAPER_CURVE_TARGETS
        if str(target["figure"]) in available_figures
    ]


def _write_template_csv(path: Path, targets: list[dict]) -> None:
    columns = ["figure", "model_key", "component", "curve_label", "time_obs", "value", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for target in targets:
            for label in target["suggested_curve_labels"]:
                writer.writerow(
                    {
                        "figure": target["figure"],
                        "model_key": target["model_key"],
                        "component": target["component"],
                        "curve_label": label,
                        "time_obs": "",
                        "value": "",
                        "notes": "digitize from published plot",
                    }
                )


def _read_prediction_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_obs" not in reader.fieldnames:
            raise ValueError("predictions_csv must contain time_obs")
        component_names = [name for name in reader.fieldnames if name != "time_obs"]
        rows = list(reader)
    if not rows:
        raise ValueError("predictions_csv must contain at least one row")
    times = np.asarray([float(row["time_obs"]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[name]) for name in component_names] for row in rows],
        dtype=float,
    )
    return times, component_names, values


def _read_digitized_reference(
    path: str | Path,
    *,
    times: np.ndarray,
    components: list[str],
    curve_label: str,
    component_figures: dict[str, str],
) -> np.ndarray:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"figure", "component", "curve_label", "time_obs", "value"}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"digitized_csv missing columns: {missing}")
        records = list(reader)
    table: dict[tuple[str, float], float] = {}
    for row in records:
        component = str(row["component"])
        if component not in components:
            continue
        if str(row["curve_label"]) != str(curve_label):
            continue
        if str(row["figure"]) != str(component_figures[component]):
            continue
        table[(component, float(row["time_obs"]))] = float(row["value"])
    columns = []
    for component in components:
        values = []
        for time in times:
            key = _matching_digitized_key(table, component, float(time))
            if key is None:
                raise ValueError(
                    f"digitized_csv missing {curve_label} {component} value at time {float(time):.17g}"
                )
            values.append(table[key])
        columns.append(values)
    return np.column_stack(columns)


def _matching_digitized_key(
    table: dict[tuple[str, float], float],
    component: str,
    time: float,
) -> tuple[str, float] | None:
    for key_component, key_time in table:
        if key_component == component and np.isclose(key_time, time, rtol=1.0e-10, atol=1.0e-15):
            return (key_component, key_time)
    return None
