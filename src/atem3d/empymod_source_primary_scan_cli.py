"""CLI for empymod-only source-primary tau-transfer scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .empymod_source_primary_scan import run_empymod_source_primary_tau_scan


def main(argv: list[str] | None = None, **runner_overrides) -> int:
    parser = argparse.ArgumentParser(
        description="Run an empymod-only Debye tau scan and fit source-primary kernels."
    )
    parser.add_argument("config", type=Path, help="ATEM3D YAML config")
    parser.add_argument("--depths", nargs="+", type=float, required=True)
    parser.add_argument("--resistivities", nargs="+", type=float, required=True)
    parser.add_argument(
        "--delta-sigma",
        nargs="+",
        type=float,
        required=True,
        help="Debye delta_sigma for each empymod layer",
    )
    parser.add_argument("--tau-values", nargs="+", type=float, required=True)
    parser.add_argument(
        "--kernel-factors",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    parser.add_argument("--component-prefix", default="Hz")
    parser.add_argument("--signal", type=int, default=-1)
    parser.add_argument("--srcpts", type=int, default=51)
    parser.add_argument("--recpts", type=int, default=1)
    parser.add_argument("--include-t0", action="store_true")
    parser.add_argument("--skip-positive-times", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run_empymod_source_primary_tau_scan(
        config,
        depths=args.depths,
        resistivities=args.resistivities,
        delta_sigma=args.delta_sigma,
        tau_values=args.tau_values,
        kernel_factors=args.kernel_factors,
        component_prefix=args.component_prefix,
        signal=args.signal,
        positive_times_only=not args.include_t0,
        skip_positive_times=args.skip_positive_times,
        empymod_kwargs={"srcpts": args.srcpts, "recpts": args.recpts},
        **runner_overrides,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for tau, item in report["tau_values"].items():
        print(
            f"tau={tau}: best={item['best_key']} "
            f"relative_l2={item['best_relative_l2']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
