"""Tiny CLI for the ROADS-Debye-MVP empymod layered smoke path.

Example::

    python paper_receiver_adaptive_debye_mvp/scripts/smoke_empymod_layered.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from atem3d.adaptive_debye_mvp.layered_forward import (
    BlockedBySoftwareOrResourcesError,
    compute_layered_response,
    default_smoke_case,
    response_to_json_dict,
)
from atem3d.adaptive_debye_mvp.reference_audit import (
    annotate_reference_type,
    run_reference_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_receiver_adaptive_debye_mvp/smoke_output"),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="compute the W0 point response only",
    )
    args = parser.parse_args(argv)
    case = default_smoke_case()
    point = (case["receivers"][0],)
    try:
        response = compute_layered_response(
            case["materials"]["exact_m0p2"],
            case["geometry"],
            case["waveforms"]["W0"],
            point,
            case["times"],
            case["transform"],
        )
    except BlockedBySoftwareOrResourcesError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_audit:
        payload = response
        audit = None
    else:
        audit = run_reference_audit(
            case["materials"]["exact_m0p2"],
            case["geometry"],
            case["waveforms"]["W0"],
            case["receivers"][:2],
            case["times"],
            case["transform"],
            baseline=response,
            disk_receiver_index=1,
            disk_axes=("z",),
        )
        payload = annotate_reference_type(response, audit)
        (args.output_dir / "reference_audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    (args.output_dir / "layered_response_W0_point.json").write_text(
        json.dumps(response_to_json_dict(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "empymod_version": payload["provenance"]["empymod_version"],
                "reference_type": payload["reference_type"],
                "hashes": payload["hashes"],
                "audit_reference_type": None if audit is None else audit["reference_type"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
