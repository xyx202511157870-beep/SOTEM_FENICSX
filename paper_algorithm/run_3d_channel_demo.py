#!/usr/bin/env python3
"""Run a 3-D tortuous conductive-path coarse/fine demonstration."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for path in (SRC, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atem3d.corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_leakage_channel_case_specs,
)
from atem3d.corrected_model_runner import (
    _default_reference_runner,
    run_corrected_model_convergence_validation,
)


CHANNEL_POINTS = [
    [-350.0, -100.0, -60.0],
    [-120.0, -30.0, -120.0],
    [80.0, 60.0, -180.0],
    [260.0, -40.0, -100.0],
]


def _profile(name: str) -> dict:
    if name == "pilot":
        return {
            "coarse_cells": [20, 12, 6],
            "fine_cells": [30, 18, 9],
            "coarse_rtol": 1.0e-7,
            "fine_rtol": 1.0e-8,
            "max_it": 600,
        }
    if name == "full":
        return {
            "coarse_cells": [40, 24, 12],
            "fine_cells": [60, 36, 18],
            "coarse_rtol": 1.0e-8,
            "fine_rtol": 1.0e-9,
            "max_it": 1200,
        }
    raise ValueError("profile must be pilot or full")


def _parse_receivers(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("receivers must contain at least one x coordinate")
    return values


def _write_background_csv(
    path: Path,
    times: np.ndarray,
    components: list[str],
    values: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *components])
        for time_value, row in zip(times, values):
            writer.writerow([f"{float(time_value):.17g}", *[f"{float(v):.17g}" for v in row]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--receiver-x", type=_parse_receivers, default=(-150.0, 0.0, 150.0))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = _profile(args.profile)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg = CorrectedModelValidationConfig(
        ramp_off_time=5.0e-6,
        turnoff_steps=10,
        observation_time_min=1.0e-5,
        observation_time_max=1.0e-1,
        n_observation_times=31,
        components=("Ex", "Ey", "dBzdt"),
        magnetic_quantity="dBzdt",
    )
    base = build_corrected_leakage_channel_case_specs(
        args.output_root,
        config=cfg,
    )["noip"]

    index: dict[str, object] = {
        "profile": args.profile,
        "channel_points_m": CHANNEL_POINTS,
        "channel_radius_m": 40.0,
        "background_resistivity_ohm_m": 100.0,
        "channel_resistivity_ohm_m": 10.0,
        "source_start_m": list(cfg.source_start),
        "source_end_m": list(cfg.source_end),
        "receiver_x_m": list(args.receiver_x),
        "cases": {},
    }

    for receiver_x in args.receiver_x:
        label = f"receiver_x_{receiver_x:g}m".replace("-", "m").replace(".", "p")
        output_dir = args.output_root / label
        if (output_dir / "validation_summary.json").is_file() and not args.force:
            print(f"[skip] {label}")
            index["cases"][label] = {"output_dir": str(output_dir), "skipped": True}
            continue

        spec = deepcopy(base)
        spec["receiver"] = [float(receiver_x), -300.0, -0.1]
        spec["output_dir"] = str(output_dir)
        spec["reference_type"] = "dolfinx_refined"
        spec["validation_scope"] = "paper_algorithm_3d_tortuous_conductive_path"
        spec["diagnostics"] = {
            "paper_case": {
                "role": "three_dimensional_demonstration_not_final_reference",
                "profile": args.profile,
            }
        }
        spec["dolfinx_forward"] = {
            "domain_min": [-600.0, -350.0, -350.0],
            "domain_max": [600.0, 350.0, 0.0],
            "cells": settings["coarse_cells"],
            "receiver_evaluation_mode": "nearest_center",
            "outer_boundary_mode": "natural",
            "ksp_type": "cg",
            "rtol": settings["coarse_rtol"],
            "atol": 1.0e-11,
            "max_it": settings["max_it"],
            "primary_provider_mode": "empymod",
            "leakage_channel": {
                "points": CHANNEL_POINTS,
                "radius": 40.0,
                "min_marked_cells": 4,
                "sigma": 0.1,
            },
        }
        spec["convergence_reference"] = {
            "dolfinx_forward": {
                "cells": settings["fine_cells"],
                "rtol": settings["fine_rtol"],
                "atol": 1.0e-12,
                "max_it": int(settings["max_it"] * 1.5),
            }
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        times = np.asarray(spec["observation_times"], dtype=float)
        components = [str(item) for item in spec["components"]]
        background = _default_reference_runner(spec)
        _write_background_csv(
            output_dir / "background_primary.csv",
            times,
            components,
            background,
        )

        print(f"[run] {label}")
        summary = run_corrected_model_convergence_validation(spec)
        index["cases"][label] = {
            "output_dir": str(output_dir),
            "receiver": spec["receiver"],
            "summary": summary,
        }

    (args.output_root / "case_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
