#!/usr/bin/env python3
"""Drive official 8-case L0 disks after the live point job, then assemble.

Does not start while the live 4-core point workers are still saturating the VM.
PG05-PG08 disks run first. PG01-PG04 disks run if sibling JSON is still
point-only or missing. Official L0 requires disk-average tasks on all 8 cases.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "generated" / "receiver_adaptive_debye_mvp" / "cache"
FLOW2 = REPO / "generated" / "receiver_adaptive_debye_mvp" / "flow2_oracle_gap"
LOG = FLOW2 / "finish_official_l0.log"
COMPLETE_POINT_FILES = 182 * 3
LIVE_PARENT = int(os.environ.get("ROADS_LIVE_PARENT", "6962"))


def _log(message: str) -> None:
    print(message, flush=True)
    FLOW2.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pids_matching(needle: str) -> list[int]:
    try:
        output = subprocess.check_output(["pgrep", "-f", needle], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in output.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _oracle_gap_busy() -> bool:
    """True if another run_oracle_gap or the PG05-08 watcher still holds cores."""

    for needle in ("run_oracle_gap.py", "watch_pg0508_disks.py"):
        if _pids_matching(needle):
            return True
    return False


def _point_files(case_id: str) -> int:
    count = 0
    if not CACHE.is_dir():
        return 0
    for path in CACHE.glob(f"{case_id}:*.npz"):
        parts = path.stem.split(":")
        if len(parts) >= 3 and parts[2] == "point":
            count += 1
    return count


def _case_ready(case_id: str) -> bool:
    path = FLOW2 / f"{case_id}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("point_only"):
        return False
    tasks = payload.get("tasks") or []
    return any(str(item.get("receiver_id", "")).startswith("disk_") for item in tasks)


def _run(case_ids: str) -> int:
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
        case_ids,
        "--skip-l0",
    ]
    _log(f"[finish] {' '.join(command)}")
    return int(subprocess.call(command, cwd=str(REPO), env=env))


def _assemble() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    command = [
        sys.executable,
        str(REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_oracle_gap.py"),
        "--assemble-l0",
    ]
    _log("[finish] assembling official L0")
    return int(subprocess.call(command, cwd=str(REPO), env=env))


def main() -> int:
    _log("[finish] waiting for PG05-PG08 point caches; will not oversubscribe live workers")
    while True:
        counts = {f"PG{index:02d}": _point_files(f"PG{index:02d}") for index in range(5, 9)}
        _log(f"[finish] point_files={counts} live={_alive(LIVE_PARENT)}")
        if all(counts[case] >= COMPLETE_POINT_FILES for case in counts):
            break
        time.sleep(20)

    # Watcher SIGTERMs the live 8-case parent and starts PG05-08 disks.
    # Do not launch a second 4-worker job on the same four cores.
    if not all(_case_ready(f"PG{index:02d}") for index in range(5, 9)):
        _log("[finish] waiting for watcher/live to finish or release cores before PG05-08 disks")
        while _oracle_gap_busy() and not all(_case_ready(f"PG{index:02d}") for index in range(5, 9)):
            time.sleep(20)
            _log(
                f"[finish] pg0508_ready={all(_case_ready(f'PG{index:02d}') for index in range(5, 9))} "
                f"busy={_oracle_gap_busy()} live={_alive(LIVE_PARENT)}"
            )
        if not all(_case_ready(f"PG{index:02d}") for index in range(5, 9)):
            code = _run("PG05,PG06,PG07,PG08")
            if code != 0:
                return code

    if not all(_case_ready(f"PG{index:02d}") for index in range(1, 5)):
        _log("[finish] sibling PG01-PG04 disk JSON missing; computing disks here")
        code = _run("PG01,PG02,PG03,PG04")
        if code != 0:
            return code

    return _assemble()


if __name__ == "__main__":
    raise SystemExit(main())
