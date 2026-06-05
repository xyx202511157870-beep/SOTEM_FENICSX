"""Audit no-IP source-diffusion amplitude scales against geometry and time gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recovery driven-trace amplitude scalars with simple "
            "source-geometry scales and first-gate driven-response normalization."
        )
    )
    parser.add_argument("audits", nargs="+", type=Path)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument(
        "--case-keys",
        nargs="+",
        help="Case keys when --geometry-report is a sweep report with a cases object",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    geometry_payload = _load_json(args.geometry_report)
    cases = _geometry_cases(
        geometry_payload,
        report_count=len(args.audits),
        case_keys=args.case_keys,
    )
    items = [
        _audit_item(audit_path, case_key=case_key, geometry=geometry)
        for audit_path, (case_key, geometry) in zip(args.audits, cases)
    ]
    report = {
        "diagnostic_only": True,
        "geometry_report": str(args.geometry_report),
        "audits": [str(path) for path in args.audits],
        "items": items,
        "summary": _summary(items),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    best = min(items, key=lambda item: item["fit_relative_l2"])
    print(
        "best fit_l2="
        f"{best['fit_relative_l2']:.6g}, "
        "scalar/(pi/2 segments)="
        f"{best['compact_optimal_scalar_over_pi_half_segments']:.6g}"
        + (
            ", scalar/first_response_inverse="
            f"{best['compact_optimal_scalar_over_first_response_inverse']:.6g}"
            if "compact_optimal_scalar_over_first_response_inverse" in best
            else ""
        )
    )
    print(f"wrote {args.output}")
    return 0


def _audit_item(
    audit_path: Path,
    *,
    case_key: str | None,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    audit = _load_json(audit_path)
    audit_item = _single_audit_item(audit)
    geometry_scale = _geometry_scale(geometry)
    compact_scalar = float(audit_item["compact_optimal_scalar"])
    fit_amplitude = _first_or_none(audit_item.get("fit_amplitude_over_normalization"))
    compact_amplitude = _first_or_none(
        audit_item.get("compact_amplitude_over_normalization")
    )
    first_compact = _first_or_none(
        audit_item.get("first_selected_compact_over_normalization")
    )
    output = {
        "audit_report": str(audit_path),
        "geometry_case": case_key,
        "fit_relative_l2": float(audit_item.get("fit_relative_l2", math.nan)),
        "fit_amplitude_over_normalization": fit_amplitude,
        "compact_amplitude_over_normalization": compact_amplitude,
        "compact_optimal_scalar": compact_scalar,
        "compact_relative_l2": (
            None
            if "compact_relative_l2" not in audit_item
            else float(audit_item["compact_relative_l2"])
        ),
        **geometry_scale,
        "compact_optimal_scalar_over_pi_half_segments": (
            compact_scalar / geometry_scale["pi_over_2_segment_scale"]
        ),
    }
    if fit_amplitude is not None:
        output["fit_amplitude_over_pi_half_segments"] = (
            fit_amplitude / geometry_scale["pi_over_2_segment_scale"]
        )
    _add_peak_error_gate(output, audit_item, prefix="compact_optimal")
    first_gate_scalar = audit_item.get("compact_first_gate_scalar")
    if first_gate_scalar is not None:
        first_gate_scalar = float(first_gate_scalar)
        output["compact_first_gate_scalar"] = first_gate_scalar
        output["compact_first_gate_scalar_over_pi_half_segments"] = (
            first_gate_scalar / geometry_scale["pi_over_2_segment_scale"]
        )
        if "compact_first_gate_relative_l2" in audit_item:
            output["compact_first_gate_relative_l2"] = float(
                audit_item["compact_first_gate_relative_l2"]
            )
        first_gate_amplitude = _first_or_none(
            audit_item.get("compact_first_gate_amplitude_over_normalization")
        )
        if first_gate_amplitude is not None:
            output["compact_first_gate_amplitude_over_normalization"] = (
                first_gate_amplitude
            )
        _add_peak_error_gate(output, audit_item, prefix="compact_first_gate")
    if (
        compact_amplitude is not None
        and compact_amplitude != 0.0
        and first_compact is not None
    ):
        first_response = first_compact / compact_amplitude
        output["first_selected_response_per_compact_amplitude"] = first_response
        if first_response != 0.0:
            inverse = 1.0 / first_response
            output["first_selected_response_inverse"] = inverse
            output["compact_optimal_scalar_over_first_response_inverse"] = (
                compact_scalar / inverse
            )
            if first_gate_scalar is not None:
                output["compact_first_gate_scalar_over_first_response_inverse"] = (
                    first_gate_scalar / inverse
                )
    return output


def _geometry_scale(geometry: dict[str, Any]) -> dict[str, Any]:
    source = geometry.get("source")
    source_vector = geometry.get("source_vector")
    if not isinstance(source, dict) or not isinstance(source_vector, dict):
        raise ValueError("geometry report must contain source and source_vector objects")
    source_length = float(source["length_m"])
    if source_length <= 0.0:
        raise ValueError("source.length_m must be positive")
    midpoint_cell = source.get("midpoint_cell")
    if not isinstance(midpoint_cell, dict):
        raise ValueError("source.midpoint_cell is required")
    widths = np.asarray(midpoint_cell["widths_m"], dtype=float)
    if widths.shape != (3,) or np.any(widths <= 0.0):
        raise ValueError("source.midpoint_cell.widths_m must contain 3 positive values")
    orientation, axis = _dominant_orientation(source_vector)
    along_width = float(widths[axis])
    segments = source_length / along_width
    pi_half_segments = 0.5 * math.pi * segments
    active_count = _active_count(source_vector, orientation)
    active_minus_one = max(active_count - 1, 0)
    return {
        "dominant_orientation": orientation,
        "source_length_m": source_length,
        "along_cell_width_m": along_width,
        "source_length_over_along_cell_width": segments,
        "pi_over_2_segment_scale": pi_half_segments,
        "active_count": int(active_count),
        "active_count_minus_one": int(active_minus_one),
        "pi_over_2_active_count_minus_one": (
            0.5 * math.pi * active_minus_one if active_minus_one else None
        ),
    }


def _dominant_orientation(source_vector: dict[str, Any]) -> tuple[str, int]:
    counts = source_vector.get("active_count_by_orientation")
    if not isinstance(counts, dict):
        raise ValueError("source_vector.active_count_by_orientation is required")
    axis_by_orientation = {"x": 0, "y": 1, "z": 2}
    values = {
        orientation: int(counts.get(orientation, 0))
        for orientation in ("x", "y", "z")
    }
    orientation = max(("x", "y", "z"), key=lambda key: values[key])
    if values[orientation] <= 0:
        raise ValueError("source vector must have at least one active orientation")
    return orientation, axis_by_orientation[orientation]


def _active_count(source_vector: dict[str, Any], orientation: str) -> int:
    if "active_count" in source_vector:
        return int(source_vector["active_count"])
    counts = source_vector["active_count_by_orientation"]
    return int(counts[orientation])


def _single_audit_item(audit: dict[str, Any]) -> dict[str, Any]:
    items = audit.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("driven audit must contain exactly one item")
    item = items[0]
    if not isinstance(item, dict):
        raise ValueError("driven audit item must be an object")
    if "compact_optimal_scalar" not in item:
        raise ValueError("driven audit item must contain compact_optimal_scalar")
    return item


def _geometry_cases(
    geometry_payload: dict[str, Any],
    *,
    report_count: int,
    case_keys: list[str] | None,
) -> list[tuple[str | None, dict[str, Any]]]:
    cases = geometry_payload.get("cases")
    if isinstance(cases, dict):
        if case_keys is None:
            raise ValueError("--case-keys is required for a geometry sweep report")
        if len(case_keys) != int(report_count):
            raise ValueError("--case-keys count must match the audit report count")
        missing = [key for key in case_keys if key not in cases]
        if missing:
            raise ValueError(f"geometry case keys not found: {missing}")
        return [(key, cases[key]) for key in case_keys]
    if case_keys is not None:
        raise ValueError("--case-keys requires a geometry report with cases")
    return [(None, geometry_payload) for _ in range(int(report_count))]


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = np.asarray(
        [item["compact_optimal_scalar_over_pi_half_segments"] for item in items],
        dtype=float,
    )
    fit_l2 = np.asarray([item["fit_relative_l2"] for item in items], dtype=float)
    summary = {
        "case_count": len(items),
        "compact_scalar_over_pi_half_segments_mean": float(np.mean(ratios)),
        "compact_scalar_over_pi_half_segments_std": float(np.std(ratios)),
        "fit_relative_l2_mean": float(np.mean(fit_l2)),
        "fit_relative_l2_max": float(np.max(fit_l2)),
    }
    first_response_ratios = np.asarray(
        [
            item["compact_optimal_scalar_over_first_response_inverse"]
            for item in items
            if "compact_optimal_scalar_over_first_response_inverse" in item
        ],
        dtype=float,
    )
    if first_response_ratios.size:
        summary.update(
            {
                "compact_scalar_over_first_response_inverse_mean": float(
                    np.mean(first_response_ratios)
                ),
                "compact_scalar_over_first_response_inverse_std": float(
                    np.std(first_response_ratios)
                ),
            }
        )
    first_gate_ratios = np.asarray(
        [
            item["compact_first_gate_scalar_over_first_response_inverse"]
            for item in items
            if "compact_first_gate_scalar_over_first_response_inverse" in item
        ],
        dtype=float,
    )
    if first_gate_ratios.size:
        summary.update(
            {
                "compact_first_gate_scalar_over_first_response_inverse_mean": (
                    float(np.mean(first_gate_ratios))
                ),
                "compact_first_gate_scalar_over_first_response_inverse_std": (
                    float(np.std(first_gate_ratios))
                ),
            }
        )
    first_gate_geometry_ratios = np.asarray(
        [
            item["compact_first_gate_scalar_over_pi_half_segments"]
            for item in items
            if "compact_first_gate_scalar_over_pi_half_segments" in item
        ],
        dtype=float,
    )
    if first_gate_geometry_ratios.size:
        summary.update(
            {
                "compact_first_gate_scalar_over_pi_half_segments_mean": float(
                    np.mean(first_gate_geometry_ratios)
                ),
                "compact_first_gate_scalar_over_pi_half_segments_std": float(
                    np.std(first_gate_geometry_ratios)
                ),
            }
        )
    first_gate_l2 = np.asarray(
        [
            item["compact_first_gate_relative_l2"]
            for item in items
            if "compact_first_gate_relative_l2" in item
        ],
        dtype=float,
    )
    if first_gate_l2.size:
        summary.update(
            {
                "compact_first_gate_relative_l2_mean": float(np.mean(first_gate_l2)),
                "compact_first_gate_relative_l2_max": float(np.max(first_gate_l2)),
            }
        )
    return summary


def _add_peak_error_gate(
    output: dict[str, Any],
    audit_item: dict[str, Any],
    *,
    prefix: str,
) -> None:
    times = audit_item.get("selected_times")
    fractions = audit_item.get(f"{prefix}_time_error_fraction")
    if times is None or fractions is None:
        return
    times_array = np.asarray(times, dtype=float).reshape(-1)
    fraction_array = np.asarray(fractions, dtype=float).reshape(-1)
    if times_array.size == 0 or times_array.shape != fraction_array.shape:
        raise ValueError(
            f"{prefix}_time_error_fraction must match selected_times"
        )
    index = int(np.argmax(fraction_array))
    output[f"{prefix}_peak_error_time_s"] = float(times_array[index])
    output[f"{prefix}_peak_error_fraction"] = float(fraction_array[index])


def _first_or_none(values) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    return float(array[0])


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
