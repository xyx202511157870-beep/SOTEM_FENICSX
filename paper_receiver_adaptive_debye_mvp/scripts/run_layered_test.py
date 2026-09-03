#!/usr/bin/env python3
"""Flow 4: independent layered test. Refuses unless L1 passed. Never starts 3-D."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.guards import ThreeDNotAuthorizedError, refuse_3d_before_l2
from atem3d.adaptive_debye_mvp.io import read_json, write_json


def main() -> int:
    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    l1 = generated / "FLOW3_STATUS.json"
    if not l1.is_file() or str(read_json(l1).get("status")) != "L1_FROZEN":
        write_json(
            generated / "FLOW4_STATUS.json",
            {"status": "REFUSED", "reason": "L1 not frozen", "three_d_run": False},
        )
        raise SystemExit("Flow 4 refused: L1 not frozen")
    try:
        refuse_3d_before_l2(generated)
    except ThreeDNotAuthorizedError:
        # Expected: Flow 4 is the authorization step, not a 3-D run.
        pass
    write_json(
        generated / "FLOW4_STATUS.json",
        {
            "status": "NOT_EXECUTED_PENDING_L0",
            "note": "Independent test is only executed after L0 and L1 pass.",
            "three_d_run": False,
        },
    )
    raise SystemExit("Flow 4 is gated; this runner does not start 3-D")


if __name__ == "__main__":
    raise SystemExit(main())
