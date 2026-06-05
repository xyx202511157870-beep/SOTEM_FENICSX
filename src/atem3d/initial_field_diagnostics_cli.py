"""Command-line diagnostics for step-off initial magnetic fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .initial_field_diagnostics import run_initial_field_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose ATEM3D initial magnetic fields.")
    parser.add_argument("result", type=Path, help="ATEM3D HDF5 output with config_yaml metadata")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["ampere", "biot_savart_wire", "zero"],
        help="Initial magnetic modes to evaluate",
    )
    parser.add_argument(
        "--empymod-depths",
        nargs="+",
        type=float,
        default=None,
        help="Optional empymod layer depths for a low-frequency reference",
    )
    parser.add_argument(
        "--empymod-resistivities",
        nargs="+",
        type=float,
        default=None,
        help="Optional empymod layer resistivities for a low-frequency reference",
    )
    parser.add_argument(
        "--empymod-frequency",
        type=float,
        default=1.0e-6,
        help="Frequency used for the optional empymod reference",
    )
    parser.add_argument("--srcpts", type=int, default=51, help="empymod source integration points")
    parser.add_argument("--recpts", type=int, default=1, help="empymod receiver integration points")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/initial_field_diagnostics.json"))
    args = parser.parse_args(argv)

    report = run_initial_field_diagnostics(
        args.result,
        args.modes,
        empymod_depths=args.empymod_depths,
        empymod_resistivities=args.empymod_resistivities,
        empymod_frequency=args.empymod_frequency,
        srcpts=args.srcpts,
        recpts=args.recpts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for mode in report["modes"]:
        print(
            f"{mode['mode']}: ampere_relative_residual="
            f"{mode['ampere_relative_residual']:.6e}; "
            f"divergence_norm={mode['divergence_norm']:.6e}"
        )
    if "empymod_frequency_reference" in report:
        ref = report["empymod_frequency_reference"]
        print(f"empymod frequency reference: f={ref['frequency']:.6e} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
