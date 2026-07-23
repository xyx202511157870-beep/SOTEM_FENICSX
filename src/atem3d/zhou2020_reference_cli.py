"""CLI for the strict Zhou 2020 empymod reference workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .zhou2020_reference import run_reference_sweep


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--srcpts", type=_int_list, default=(3, 5, 9, 17))
    parser.add_argument(
        "--surface-offsets",
        type=_float_list,
        default=(0.0, 0.05, 0.1, 0.2),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run_reference_sweep(
        case_path=args.case,
        provenance_path=args.provenance,
        output_dir=args.output_dir,
        srcpts_values=args.srcpts,
        surface_offsets_m=args.surface_offsets,
    )
    print(f"status={result['status']} output_dir={args.output_dir.resolve()}")
    return 0 if result["status"] == "reference_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
