#!/usr/bin/env python3
"""Assemble official TE JSON from cache and run Gate L2 when disks are ready.

Does not reselect. Does not start 3-D. Safe to run alongside warm_l2_forced.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.io import read_json
from atem3d.adaptive_debye_mvp.oracle_gap import evaluate_pilot_case, write_case_result
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, TEST_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


GEN = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
CACHE = GEN / "cache"
FLOW4 = GEN / "flow4_independent_test"
TEST_IDS = tuple(f"TE{index:02d}" for index in range(1, 11))


def _forced_gate() -> dict[int, list[str]]:
    l1 = read_json(GEN / "FLOW3_STATUS.json")
    selected = {int(key): str(value) for key, value in dict(l1["selected"]).items()}
    spectral = {int(key): str(value) for key, value in dict(l1.get("spectral_selected") or {}).items()}
    return {
        int(poles): list(dict.fromkeys([selected[int(poles)], spectral[int(poles)]]))
        for poles in K_PILOT
    }


def _cache_ok(case_id: str, model_id: str, waveform_id: str) -> bool:
    from atem3d.adaptive_debye_mvp.case_bridge import load_cached_response

    key = f"{case_id}:{model_id}:{waveform_id}"
    path = CACHE / f"{key}.npz"
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    return load_cached_response(CACHE, key) is not None


def _official(case_id: str) -> bool:
    path = FLOW4 / f"case_{case_id}.json"
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    provenance = payload.get("provenance") or {}
    if provenance.get("selector_read") is not True or provenance.get("l2_evaluated") is not True:
        return False
    if payload.get("schema") != "atem3d.adaptive_debye_mvp.pilot_case_result.v1":
        return False
    if payload.get("point_only") is not False:
        return False
    recs = {task.get("receiver_id") for task in payload.get("tasks") or []}
    waves = {task.get("waveform_id") for task in payload.get("tasks") or []}
    return {"point", "disk_1.0", "disk_4.0", "tilted_coil"} <= recs and "W3" in waves


def _case_ready(case_id: str, allowed: list[str]) -> bool:
    for candidate_id in ("exact", "noip", *allowed):
        for waveform_id in TEST_WAVEFORMS:
            if not _cache_ok(case_id, f"{candidate_id}:point", waveform_id):
                return False
            if not _cache_ok(case_id, f"{candidate_id}:disk", waveform_id):
                return False
    return True


def _disk_progress(allowed: list[str]) -> tuple[int, int]:
    needed = 0
    have = 0
    for case_id in TEST_IDS:
        for candidate_id in ("exact", "noip", *allowed):
            for waveform_id in TEST_WAVEFORMS:
                needed += 1
                if _cache_ok(case_id, f"{candidate_id}:disk", waveform_id):
                    have += 1
    return have, needed


def _assemble_case(case_id: str, forced: dict[int, list[str]], official_variant: str) -> None:
    allowed = sorted({item for ids in forced.values() for item in ids})
    cases = {item.case_id: item for item in cases_for_split("independent_test", generate_all_cases())}
    started = time.time()
    print(f"[finish-l2] assemble {case_id}", flush=True)
    result = evaluate_pilot_case(
        cases[case_id],
        waveform_ids=TEST_WAVEFORMS,
        cache_dir=CACHE,
        k_values=K_PILOT,
        official_variant=official_variant,
        include_disks=True,
        forced_disk_ids=forced,
        include_tilted=True,
        allowed_candidate_ids=allowed,
    )
    result["waveform_ids"] = list(TEST_WAVEFORMS)
    result["k_values"] = list(K_PILOT)
    result["receiver_ids"] = ["point", "disk_1.0", "disk_4.0", "tilted_coil"]
    result["provenance"] = {
        "transform_label": "smoke_fast_lagged_dlf",
        "roads_workers": os.environ.get("ROADS_WORKERS"),
        "wall_seconds": time.time() - started,
        "stage": "disks",
        "reselection": False,
        "selector_read": True,
        "l2_evaluated": True,
        "allowed_candidate_ids": allowed,
    }
    FLOW4.mkdir(parents=True, exist_ok=True)
    write_case_result(FLOW4 / f"case_{result['case_id']}.json", result)
    print(
        f"[finish-l2] wrote {case_id} point_only={result.get('point_only')} "
        f"official={_official(case_id)}",
        flush=True,
    )


def _run_gate() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_layered_test.py"), "--assemble-l2"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    return int(proc.returncode)


def main() -> int:
    l1 = read_json(GEN / "FLOW3_STATUS.json")
    if str(l1.get("status")) != "L1_FROZEN":
        raise SystemExit("L1 is not frozen")
    if (GEN / "LAYERED_GATE_PASSED.json").is_file() or (GEN / "STOP_LAYERED_SELECTOR_FAILED.json").is_file():
        print("[finish-l2] gate artifact already present", flush=True)
        return 0
    official_variant = str(l1.get("official_spectral_variant") or "S1")
    forced = _forced_gate()
    allowed = sorted({item for ids in forced.values() for item in ids})
    while True:
        have, needed = _disk_progress(allowed)
        official = [case_id for case_id in TEST_IDS if _official(case_id)]
        print(f"[finish-l2] disks {have}/{needed} official {official}", flush=True)
        for case_id in TEST_IDS:
            if _official(case_id):
                continue
            if _case_ready(case_id, allowed):
                _assemble_case(case_id, forced, official_variant)
        official = [case_id for case_id in TEST_IDS if _official(case_id)]
        if len(official) == 10:
            print("[finish-l2] 10 official TE JSON ready; assembling L2", flush=True)
            return _run_gate()
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
