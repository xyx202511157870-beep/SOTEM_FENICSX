"""Audit whether validation-trace errors are dominated by time-axis warping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an affine reference-time map for numerical/reference samples in "
            "an ATEM3D validation JSON report."
        )
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--scale-count", type=int, default=91)
    parser.add_argument("--shift-min", type=float, default=-5.0e-5)
    parser.add_argument("--shift-max", type=float, default=5.0e-5)
    parser.add_argument("--shift-count", type=int, default=101)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.scale_count <= 0:
        parser.error("--scale-count must be positive")
    if args.shift_count <= 0:
        parser.error("--shift-count must be positive")
    if args.min_samples <= 0:
        parser.error("--min-samples must be positive")
    if args.scale_min > args.scale_max:
        parser.error("--scale-min must be <= --scale-max")
    if args.shift_min > args.shift_max:
        parser.error("--shift-min must be <= --shift-max")

    payload = _load_json(args.report)
    samples = _samples(payload)
    component_names = args.components or sorted(samples)
    scales = np.linspace(args.scale_min, args.scale_max, args.scale_count)
    shifts = np.linspace(args.shift_min, args.shift_max, args.shift_count)
    output = {
        "diagnostic_only": True,
        "report": str(args.report),
        "time_map": "reference_time = scale * numerical_time + shift",
        "search": {
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "scale_count": int(args.scale_count),
            "shift_min": float(args.shift_min),
            "shift_max": float(args.shift_max),
            "shift_count": int(args.shift_count),
            "min_samples": int(args.min_samples),
        },
        "components": {},
    }

    for name in component_names:
        if name not in samples:
            raise ValueError(f"component {name!r} is not present in report samples")
        output["components"][name] = _audit_component(
            samples[name],
            scales=scales,
            shifts=shifts,
            min_samples=int(args.min_samples),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    for name, item in output["components"].items():
        print(
            f"{name}: base_l2={item['base_relative_l2']:.6g} "
            f"best_l2={item['best_relative_l2']:.6g} "
            f"scale={item['best_scale']:.8g} shift={item['best_shift']:.8g}"
        )
    print(f"wrote {args.output}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("report JSON root must be an object")
    return payload


def _samples(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    samples = payload.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("report must contain a non-empty samples object")
    return samples


def _audit_component(
    rows: list[dict[str, Any]],
    *,
    scales: np.ndarray,
    shifts: np.ndarray,
    min_samples: int,
) -> dict[str, Any]:
    times, numerical, reference = _sample_arrays(rows)
    order = np.argsort(times)
    times = times[order]
    numerical = numerical[order]
    reference = reference[order]
    base_relative_l2 = _relative_l2(numerical, reference)

    best = {
        "best_relative_l2": np.inf,
        "best_scale": np.nan,
        "best_shift": np.nan,
        "best_sample_count": 0,
    }
    for scale in scales:
        for shift in shifts:
            mapped_times = float(scale) * times + float(shift)
            mapped_reference = np.interp(
                mapped_times,
                times,
                reference,
                left=np.nan,
                right=np.nan,
            )
            valid = np.isfinite(mapped_reference)
            if int(np.count_nonzero(valid)) < min_samples:
                continue
            relative_l2 = _relative_l2(numerical[valid], mapped_reference[valid])
            if relative_l2 < best["best_relative_l2"]:
                best = {
                    "best_relative_l2": float(relative_l2),
                    "best_scale": float(scale),
                    "best_shift": float(shift),
                    "best_sample_count": int(np.count_nonzero(valid)),
                }

    if not np.isfinite(best["best_relative_l2"]):
        raise ValueError("no affine time map retained enough overlapping samples")
    improvement = (
        float(base_relative_l2 / best["best_relative_l2"])
        if best["best_relative_l2"] > 0.0
        else np.inf
    )
    return {
        "base_relative_l2": float(base_relative_l2),
        **best,
        "improvement_factor": improvement,
        "sample_count": int(times.size),
        "time_min": float(times[0]),
        "time_max": float(times[-1]),
    }


def _sample_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("component samples must be non-empty")
    times = np.asarray([row["time"] for row in rows], dtype=float)
    numerical = np.asarray([row["numerical"] for row in rows], dtype=float)
    reference = np.asarray([row["reference"] for row in rows], dtype=float)
    if not (times.ndim == numerical.ndim == reference.ndim == 1):
        raise ValueError("time, numerical, and reference samples must be 1D")
    if not (times.size == numerical.size == reference.size):
        raise ValueError("time, numerical, and reference samples must have equal length")
    if times.size < 2:
        raise ValueError("at least two samples are required")
    return times, numerical, reference


def _relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(numerical - reference))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
