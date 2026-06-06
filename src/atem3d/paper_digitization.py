"""Published-paper response curve digitization templates."""

from __future__ import annotations

from pathlib import Path
import csv
import json


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
