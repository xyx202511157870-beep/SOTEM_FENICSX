"""CLI for magnetic receiver recovery decomposition diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import build_simulation, load_config
from .magnetic_recovery_decomposition import (
    align_decomposition_samples_to_validation,
    magnetic_recovery_decomposition_at_time,
    summarize_decompositions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ATEM3D config and decompose Biot magnetic receiver recovery "
            "into ohmic, Debye-memory, initial-memory, source, and source-history terms."
        )
    )
    parser.add_argument("config", type=Path, help="ATEM3D YAML configuration")
    parser.add_argument("--time-indices", nargs="+", type=int)
    parser.add_argument("--include-t0", action="store_true")
    parser.add_argument("--time-min", type=float)
    parser.add_argument("--time-max", type=float)
    parser.add_argument(
        "--validation-report",
        type=Path,
        help="Optional empymod validation JSON with per-component samples.",
    )
    parser.add_argument("--validation-time-atol", type=float, default=1.0e-12)
    parser.add_argument("--include-fields", action="store_true")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.time_min is not None and args.time_max is not None and args.time_min > args.time_max:
        parser.error("--time-min must be <= --time-max")
    if args.validation_time_atol < 0.0:
        parser.error("--validation-time-atol must be nonnegative")

    config = load_config(args.config)
    simulation = build_simulation(config)
    result = simulation.run()
    indices = _selected_time_indices(
        result.times,
        requested=args.time_indices,
        include_t0=bool(args.include_t0),
        time_min=args.time_min,
        time_max=args.time_max,
    )
    samples = [
        magnetic_recovery_decomposition_at_time(
            simulation,
            result,
            time_index=index,
            include_fields=bool(args.include_fields),
        )
        for index in indices
    ]
    formulation = "eb" if hasattr(result, "b") else "hj"
    output = {
        "diagnostic_only": True,
        "config": str(args.config),
        "formulation": formulation,
        "magnetic_receiver_mode": str(
            config.get(
                "magnetic_receiver_mode",
                "stored_h" if formulation == "hj" else "stored_b",
            )
        ),
        "magnetic_recovery_subdivisions": int(
            config.get("magnetic_recovery_subdivisions", 1)
        ),
        "time_indices": [int(index) for index in indices],
        "times": [float(result.times[index]) for index in indices],
        "summary": summarize_decompositions(samples),
        "samples": samples,
    }
    if args.validation_report is not None:
        validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
        output["validation_report"] = str(args.validation_report)
        output["validation_alignment"] = align_decomposition_samples_to_validation(
            samples,
            validation,
            time_atol=float(args.validation_time_atol),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = output["summary"]
    print(
        "relative_l2="
        f"{summary['total_recomputed_relative_l2_to_numerical']:.6g} "
        f"samples={summary['sample_count']}"
    )
    if "validation_alignment" in output:
        best = output["validation_alignment"]["best_single_term"]
        print(
            "best_residual_term="
            f"{best['name']} scalar={best['best_scalar']:.6g} "
            f"rel_l2={best['relative_l2_after_best_scalar']:.6g}"
        )
    print(f"wrote {args.output}")
    return 0


def _selected_time_indices(
    times,
    *,
    requested: list[int] | None,
    include_t0: bool,
    time_min: float | None,
    time_max: float | None,
) -> list[int]:
    times = np.asarray(times, dtype=float)
    if requested is not None:
        indices = [int(index) for index in requested]
        if any(index < 0 or index >= times.size for index in indices):
            raise ValueError("--time-indices contains an out-of-range index")
    else:
        mask = np.ones(times.shape, dtype=bool)
        if not include_t0:
            mask &= times > 0.0
        if time_min is not None:
            mask &= times >= float(time_min)
        if time_max is not None:
            mask &= times <= float(time_max)
        indices = [int(index) for index in np.flatnonzero(mask)]
    if not include_t0:
        indices = [index for index in indices if times[index] > 0.0]
    if not indices:
        raise ValueError("time selection produced no samples")
    return indices


if __name__ == "__main__":
    raise SystemExit(main())
