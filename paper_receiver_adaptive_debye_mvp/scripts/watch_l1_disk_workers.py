#!/usr/bin/env python3
"""Restart local L1 disk workers immediately if they die. No long idle sleeps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FLOW3 = REPO / "generated" / "receiver_adaptive_debye_mvp" / "flow3_selector"
LOCAL_IDS = ["TR01", "TR02", "TR03", "TR04"]
WORKER_NEEDLE = "run_selector_cases.py --split train --stage disks --case-ids TR01,TR02,TR03,TR04"
CHECK_SEC = 45


def _point_only(case_id: str) -> bool:
    path = FLOW3 / f"case_{case_id}.json"
    if not path.is_file():
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return payload.get("point_only") is not False


def _alive() -> bool:
    result = subprocess.run(["pgrep", "-f", WORKER_NEEDLE], capture_output=True, text=True)
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
        str(REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_selector_cases.py"),
        "--split",
        "train",
        "--stage",
        "disks",
        "--case-ids",
        ",".join(LOCAL_IDS),
        "--forced-ids-json",
        str(FLOW3 / "forced_disk_ids.json"),
        "--official-variant",
        "S1",
    ]
    print("[watch] restarting local TR01-04 disk workers from cache", flush=True)
    subprocess.Popen(args, cwd=str(REPO), env=env, start_new_session=True)


def _consume() -> None:
    script = REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "consume_sibling_case_json.py"
    subprocess.run([sys.executable, str(script)], cwd=str(REPO), check=False)


def main() -> int:
    while True:
        missing = [case_id for case_id in LOCAL_IDS if _point_only(case_id)]
        if not missing:
            print("[watch] TR01-04 disk JSON complete", flush=True)
            return 0
        if not _alive():
            _restart()
        _consume()
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
