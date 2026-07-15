#!/usr/bin/env python3
"""Generate solver inputs/results for the seepage-channel benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_channel_validation import (  # noqa: E402
    save_empymod_background,
    sha256_file,
    simpeg_payload_from_h5,
    validate_result_payload,
)


SIMPEG_CONFIGS = {
    "background": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_background.yaml",
    "channel": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_channel.yaml",
}


def _run_empymod(output_root: Path, *, srcpts: int) -> Path:
    output = output_root / "empymod_background.npz"
    save_empymod_background(output, srcpts=srcpts)
    with np.load(output, allow_pickle=False) as stored:
        validate_result_payload("empymod", stored)
    return output


def _run_simpeg(output_root: Path, *, case: str) -> Path:
    config_path = SIMPEG_CONFIGS[case]
    output_h5 = output_root / f"simpeg_{case}.h5"
    output_npz = output_root / f"simpeg_{case}.npz"
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "atem3d.cli",
            "run",
            str(config_path),
            "--output",
            str(output_h5),
            "--data-only",
        ],
        cwd=ROOT,
        check=True,
    )
    payload = simpeg_payload_from_h5(output_h5)
    np.savez_compressed(output_npz, **payload)
    provenance = {
        "method": "SimPEG",
        "case": case,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "result_h5": output_h5.name,
        "result_h5_sha256": sha256_file(output_h5),
        "normalized_npz": output_npz.name,
        "value_shape": list(np.asarray(payload["values"]).shape),
        "coordinate_adapter": "physical_z_down_to_internal_z_up",
    }
    (output_root / f"simpeg_{case}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="solver", required=True)

    empymod_parser = subparsers.add_parser("empymod")
    empymod_parser.add_argument("--output-root", type=Path, required=True)
    empymod_parser.add_argument("--srcpts", type=int, default=129)

    simpeg_parser = subparsers.add_parser("simpeg")
    simpeg_parser.add_argument("--case", choices=tuple(SIMPEG_CONFIGS), required=True)
    simpeg_parser.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if args.solver == "empymod":
        output = _run_empymod(output_root, srcpts=args.srcpts)
    else:
        output = _run_simpeg(output_root, case=args.case)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
