#!/usr/bin/env python3
"""Restart official Flow 4 workers immediately if they die. No long idle sleeps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "generated" / "receiver_adaptive_debye_mvp"
FLOW4 = GEN / "flow4_independent_test"
TEST_IDS = [f"TE{index:02d}" for index in range(1, 11)]
NEEDLE = "run_layered_test.py"
CHECK_SEC = 45


def _official(case_id: str) -> bool:
    path = FLOW4 / f"case_{case_id}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance") or {}
    if provenance.get("selector_read") is False or provenance.get("l2_evaluated") is False:
        return False
    if payload.get("point_only") is not False:
        return False
    return "pilot_case_result" in str(payload.get("schema") or "")


def _l2_done() -> bool:
    if all(_official(case_id) for case_id in TEST_IDS):
        return True
    if (GEN / "LAYERED_GATE_PASSED.json").is_file():
        return True
    if (GEN / "STOP_LAYERED_SELECTOR_FAILED.json").is_file():
        return True
    return False


def _alive() -> bool:
    result = subprocess.run(["pgrep", "-f", NEEDLE], capture_output=True, text=True)
    return result.returncode == 0


def _restart() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ROADS_WORKERS": str(max(4, os.cpu_count() or 4)),
        }
    )
    args = [
        sys.executable,
        str(REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_layered_test.py"),
    ]
    print("[watch] restarting official L2 workers from cache", flush=True)
    subprocess.Popen(args, cwd=str(REPO), env=env, start_new_session=True)


def _consume() -> None:
    script = REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "consume_sibling_case_json.py"
    subprocess.run([sys.executable, str(script)], cwd=str(REPO), check=False)


def main() -> int:
    while True:
        if _l2_done():
            print("[watch] L2 official JSON or gate file present", flush=True)
            return 0
        if not _alive():
            _restart()
        _consume()
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
