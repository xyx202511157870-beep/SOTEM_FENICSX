"""Command-line entry points for ATEM3D examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import build_simulation, load_config
from .io import save_result_hdf5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a grounded-wire TDEM-IP simulation.")
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/result.h5"))
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Store receiver data without full field histories",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    simulation = build_simulation(config)
    if args.data_only:
        if not hasattr(simulation, "run_data_only"):
            parser.error("--data-only is only available for simulations with run_data_only()")
        result = simulation.run_data_only()
    else:
        result = simulation.run()
    save_result_hdf5(args.output, result, config)
    print(f"wrote {args.output}")
    print(f"time nodes: {result.times.size}; receivers: {result.data.shape[1]}")
    if args.data_only:
        print("field histories: not saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
