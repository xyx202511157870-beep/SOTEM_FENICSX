#!/usr/bin/env python3
"""Aggregate fail-closed SimPEG/FEniCSx seepage verification gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_matrix_aggregation import build_open3d_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    summary = build_open3d_summary(
        args.output_root, require_pass=args.require_pass
    )
    path = args.output_root / "verification_summary_open3d.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
