#!/usr/bin/env python3
"""After the live point stage finishes, start PG05-PG08 disks only.

Does not start while the live 4-core point job is still writing PG05-PG08
point cache. Sends SIGTERM to the live parent only after all four point
caches are complete, so this VM does not begin PG01-PG04 disks.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "generated" / "receiver_adaptive_debye_mvp" / "cache"
FLOW2 = REPO / "generated" / "receiver_adaptive_debye_mvp" / "flow2_oracle_gap"
LOG = FLOW2 / "watch_pg0508.log"
WANTED = ("PG05", "PG06", "PG07", "PG08")
LIVE_PARENT = int(os.environ.get("ROADS_LIVE_PARENT", "6962"))
# 180 candidates + exact + noip, three waveforms each
COMPLETE_POINT_FILES = 182 * 3


def _log(message: str) -> None:
    print(message, flush=True)
    FLOW2.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _point_counts() -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not CACHE.is_dir():
        return {case: 0 for case in WANTED}
    for path in CACHE.glob("*.npz"):
        parts = path.stem.split(":")
        if len(parts) >= 3 and parts[0] in WANTED and parts[2] == "point":
            counts[parts[0]] += 1
    return {case: int(counts[case]) for case in WANTED}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    _log(f"[watch] waiting for point caches {WANTED}; live_parent={LIVE_PARENT}")
    while True:
        counts = _point_counts()
        _log(f"[watch] point_keys={counts}")
        if all(counts[case] >= COMPLETE_POINT_FILES for case in WANTED):
            break
        time.sleep(15)

    if _alive(LIVE_PARENT):
        _log(f"[watch] point caches complete; SIGTERM {LIVE_PARENT} to skip all-8 disks")
        try:
            os.kill(LIVE_PARENT, signal.SIGTERM)
        except OSError as exc:
            _log(f"[watch] SIGTERM failed: {exc}")
        time.sleep(2)
        if _alive(LIVE_PARENT):
            try:
                os.kill(LIVE_PARENT, signal.SIGKILL)
            except OSError:
                pass
    else:
        _log("[watch] live parent already gone")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ROADS_WORKERS": str(os.cpu_count() or 4),
        }
    )
    command = [
        sys.executable,
        str(REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_oracle_gap.py"),
        "--case-ids",
        "PG05,PG06,PG07,PG08",
        "--skip-l0",
    ]
    _log(f"[watch] starting disks {' '.join(command)}")
    return int(subprocess.call(command, cwd=str(REPO), env=env))


if __name__ == "__main__":
    raise SystemExit(main())
