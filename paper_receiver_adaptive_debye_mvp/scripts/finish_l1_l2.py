#!/usr/bin/env python3
"""Drive Flow 3 then Flow 4 after L0. Never starts 3-D.

Honors ROADS_LOCAL_CASE_IDS so this VM can compute a subset while sibling
agents write the remaining case_TR*/case_VA*/case_TE* JSON. After local
work, waits for the full 12+6 (then 10 test) set and consumes sibling
branches via consume_sibling_case_json.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "generated" / "receiver_adaptive_debye_mvp"
FLOW3 = GEN / "flow3_selector"
FLOW4 = GEN / "flow4_independent_test"
LOG = FLOW3 / "finish_l1_l2.log"
TRAIN_IDS = [f"TR{index:02d}" for index in range(1, 13)]
VAL_IDS = [f"VA{index:02d}" for index in range(1, 7)]
TEST_IDS = [f"TE{index:02d}" for index in range(1, 11)]


def _log(message: str) -> None:
    print(message, flush=True)
    FLOW3.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _env() -> dict[str, str]:
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
    return env


def _run(args: list[str]) -> int:
    _log("[l1l2] " + " ".join(args))
    return int(subprocess.call(args, cwd=str(REPO), env=_env()))


def _local_ids() -> set[str]:
    raw = os.environ.get("ROADS_LOCAL_CASE_IDS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _case_ok(directory: Path, case_id: str, stage: str) -> bool:
    path = directory / f"case_{case_id}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not payload.get("tasks"):
        return False
    if stage == "points":
        return True
    return payload.get("point_only") is False


def _missing(directory: Path, ids: list[str], stage: str) -> list[str]:
    return [case_id for case_id in ids if not _case_ok(directory, case_id, stage)]


def _consume_siblings() -> None:
    script = REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "consume_sibling_case_json.py"
    if script.is_file():
        _run([sys.executable, str(script)])


def _write_progress(stage: str, missing: list[str]) -> None:
    write = {
        "status": "L1_IN_PROGRESS" if not (GEN / "FLOW3_STATUS.json").is_file() else None,
        "stage": stage,
        "missing": missing,
        "three_d_run": False,
    }
    path = FLOW3 / "PROGRESS.json"
    path.write_text(json.dumps({k: v for k, v in write.items() if v is not None}, indent=2) + "\n", encoding="utf-8")


def _run_split(split: str, ids: list[str], stage: str, extra: list[str] | None = None) -> int:
    missing = _missing(FLOW3, ids, stage)
    local = _local_ids()
    todo = [case_id for case_id in missing if not local or case_id in local]
    if not todo:
        _log(f"[l1l2] no local {split} {stage} work (missing={missing} local={sorted(local)})")
        return 0
    args = [
        sys.executable,
        str(REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_selector_cases.py"),
        "--split",
        split,
        "--stage",
        stage,
        "--case-ids",
        ",".join(todo),
    ]
    if extra:
        args.extend(extra)
    return _run(args)


def _wait_for(directory: Path, ids: list[str], stage: str, *, idle_takeover_min: int = 5) -> int:
    last = -1
    idle_min = 0
    while True:
        _consume_siblings()
        missing = _missing(directory, ids, stage)
        _write_progress(f"wait_{stage}", missing)
        _log(f"[l1l2] waiting {stage}: have {len(ids) - len(missing)}/{len(ids)} missing={missing}")
        if not missing:
            return 0
        local = _local_ids()
        local_missing = [case_id for case_id in missing if not local or case_id in local]
        if local_missing and directory == FLOW3:
            extra = ["--forced-ids-json", str(FLOW3 / "forced_disk_ids.json")] if stage == "disks" else None
            train_local = [case_id for case_id in local_missing if case_id.startswith("TR")]
            val_local = [case_id for case_id in local_missing if case_id.startswith("VA")]
            if train_local:
                code = _run_split("train", TRAIN_IDS, stage, extra)
                if code != 0:
                    return code
            if val_local:
                code = _run_split("validation", VAL_IDS, stage, extra)
                if code != 0:
                    return code
            continue
        have = len(ids) - len(missing)
        if have == last:
            idle_min += 1
        else:
            idle_min = 0
            last = have
        if idle_min >= idle_takeover_min:
            _log(f"[l1l2] sibling idle {idle_min} min; taking remaining {missing}")
            if directory == FLOW3:
                train_missing = [case_id for case_id in missing if case_id.startswith("TR")]
                val_missing = [case_id for case_id in missing if case_id.startswith("VA")]
                extra = ["--forced-ids-json", str(FLOW3 / "forced_disk_ids.json")] if stage == "disks" else None
                if train_missing:
                    os.environ.pop("ROADS_LOCAL_CASE_IDS", None)
                    code = _run_split("train", TRAIN_IDS, stage, extra)
                    if code != 0:
                        return code
                if val_missing:
                    os.environ.pop("ROADS_LOCAL_CASE_IDS", None)
                    code = _run_split("validation", VAL_IDS, stage, extra)
                    if code != 0:
                        return code
            else:
                return 0
            idle_min = 0
            continue
        time.sleep(60)


def main() -> int:
    py = sys.executable
    train = REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "train_selector.py"
    test = REPO / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_layered_test.py"

    code = _run_split("train", TRAIN_IDS, "points")
    if code != 0:
        return code
    code = _run_split("validation", VAL_IDS, "points")
    if code != 0:
        return code
    code = _wait_for(FLOW3, TRAIN_IDS + VAL_IDS, "points")
    if code != 0:
        return code
    code = _run([py, str(train), "--stage", "rank-points"])
    if code != 0:
        return code
    forced = FLOW3 / "forced_disk_ids.json"
    extra = ["--forced-ids-json", str(forced)]
    code = _run_split("train", TRAIN_IDS, "disks", extra)
    if code != 0:
        return code
    code = _run_split("validation", VAL_IDS, "disks", extra)
    if code != 0:
        return code
    code = _wait_for(FLOW3, TRAIN_IDS + VAL_IDS, "disks")
    if code != 0:
        return code
    code = _run([py, str(train), "--stage", "freeze"])
    if code != 0:
        return code
    return _run([py, str(test)])


if __name__ == "__main__":
    raise SystemExit(main())
