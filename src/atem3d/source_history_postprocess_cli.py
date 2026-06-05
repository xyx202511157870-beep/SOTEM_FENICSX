"""CLI for source-history correction of HDF5 receiver data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .source_history_postprocess import (
    load_config_from_result,
    postprocess_source_history_receiver_data,
    save_postprocessed_result_hdf5,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a configured source-history magnetic receiver correction to "
            "existing HDF5 receiver data."
        )
    )
    parser.add_argument("input", type=Path, help="Input HDF5 result with times/data")
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML configuration containing magnetic_recovery_source_history. "
            "If omitted, use the config_yaml stored in the input result."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config_from_result(args.input)
    result = postprocess_source_history_receiver_data(args.input, config)
    save_postprocessed_result_hdf5(
        args.output,
        result,
        config,
        input_path=args.input,
    )
    print(f"wrote {args.output}")
    print(f"time nodes: {result.times.size}; receivers: {result.data.shape[1]}")
    print(
        "source-history magnetic receivers: "
        f"{len(result.magnetic_receiver_indices)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
