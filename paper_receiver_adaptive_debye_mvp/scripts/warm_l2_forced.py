#!/usr/bin/env python3
"""Fill official L2 cache for frozen P-R/B2/B1 only, then assemble TE JSON.

No test reselection. Shares CPUs across leftover disk (and missing point)
waveforms, then writes official independent-test case JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.case_bridge import (
    case_geometry,
    case_time_grid,
    case_waveform,
    debye_material,
    disk_receivers,
    evaluation_transform,
    exact_pelton_material,
    forward_response,
    load_cached_response,
    nonpolarizable_material,
    point_receivers,
)
from atem3d.adaptive_debye_mvp.io import read_json
from atem3d.adaptive_debye_mvp.oracle_gap import evaluate_pilot_case, fit_case_candidates, write_case_result
from atem3d.adaptive_debye_mvp.protocol_constants import B1_LOG_UNIFORM_TEMPLATE, K_PILOT, TEST_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases, instantiate_candidates


GEN = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
CACHE = GEN / "cache"
FLOW3 = GEN / "flow3_selector"
FLOW4 = GEN / "flow4_independent_test"
TEST_IDS = tuple(f"TE{index:02d}" for index in range(1, 11))


def _workers() -> int:
    requested = max(4, int(os.cpu_count() or 4))
    env = os.environ.get("ROADS_WORKERS")
    if env:
        requested = max(4, int(env))
    return requested


def _forced(*, include_b1: bool = True) -> dict[int, list[str]]:
    l1 = read_json(GEN / "FLOW3_STATUS.json")
    selected = {int(key): str(value) for key, value in dict(l1["selected"]).items()}
    spectral = {int(key): str(value) for key, value in dict(l1.get("spectral_selected") or {}).items()}
    forced: dict[int, list[str]] = {}
    for poles in K_PILOT:
        ids = [selected[int(poles)], spectral[int(poles)]]
        if include_b1:
            ids.append(B1_LOG_UNIFORM_TEMPLATE.format(K=int(poles)))
        forced[int(poles)] = list(dict.fromkeys(ids))
    return forced


def _cache_ok(case_id: str, model_id: str, waveform_id: str) -> bool:
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance") or {}
    if provenance.get("selector_read") is not True or provenance.get("l2_evaluated") is not True:
        return False
    if payload.get("point_only") is not False:
        return False
    recs = {task.get("receiver_id") for task in payload.get("tasks") or []}
    waves = {task.get("waveform_id") for task in payload.get("tasks") or []}
    return {"point", "disk_1.0", "disk_4.0", "tilted_coil"} <= recs and "W3" in waves


_WORKER_CTX: dict | None = None


def _init_worker(case_ids: tuple[str, ...]) -> None:
    global _WORKER_CTX
    cases = {item.case_id: item for item in generate_all_cases() if item.case_id in set(case_ids)}
    ctx = {}
    for case_id, case in cases.items():
        candidates = tuple(spec for spec in instantiate_candidates(case) if spec.K in K_PILOT)
        fits = {record.candidate_id: record for record in fit_case_candidates(case, candidates)}
        ctx[case_id] = {
            "case": case,
            "fits": fits,
            "points": point_receivers(case),
            "disks": disk_receivers(case),
            "times": case_time_grid(),
            "geometry": case_geometry(case),
            "transform": evaluation_transform(),
        }
    _WORKER_CTX = ctx


def _warm_one(task: tuple[str, str, str, str]) -> str:
    case_id, candidate_id, kind, waveform_id = task
    ctx = _WORKER_CTX[case_id]
    case = ctx["case"]
    if candidate_id == "exact":
        material = exact_pelton_material(case)
    elif candidate_id == "noip":
        material = nonpolarizable_material(case)
    else:
        record = ctx["fits"][candidate_id]
        material = debye_material(case, record.fit, candidate_id=candidate_id)
    receivers = ctx["disks"] if kind == "disk" else ctx["points"]
    model_id = f"{candidate_id}:{kind}"
    forward_response(
        material,
        ctx["geometry"],
        case_waveform(waveform_id),
        receivers,
        ctx["times"],
        ctx["transform"],
        cache_dir=CACHE,
        cache_key=f"{case_id}:{model_id}:{waveform_id}",
    )
    return f"{case_id}:{model_id}:{waveform_id}"


def _assemble(forced: dict[int, list[str]], official_variant: str) -> None:
    allowed = sorted({item for ids in forced.values() for item in ids})
    cases = {item.case_id: item for item in cases_for_split("independent_test", generate_all_cases())}
    FLOW4.mkdir(parents=True, exist_ok=True)
    for case_id in TEST_IDS:
        if _official(case_id):
            print(f"[l2warm] assemble skip {case_id}", flush=True)
            continue
        started = time.time()
        print(f"[l2warm] assemble {case_id}", flush=True)
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
        write_case_result(FLOW4 / f"case_{result['case_id']}.json", result)
        print(f"[l2warm] wrote {case_id} point_only={result.get('point_only')}", flush=True)


def main() -> int:
    l1 = read_json(GEN / "FLOW3_STATUS.json")
    if str(l1.get("status")) != "L1_FROZEN":
        raise SystemExit("L1 is not frozen")
    official_variant = str(l1.get("official_spectral_variant") or "S1")
    forced_all = _forced(include_b1=True)
    forced_gate = _forced(include_b1=False)
    print(f"[l2warm] forced_gate={forced_gate}", flush=True)
    for kind in ("point", "disk"):
        allowed = sorted({item for ids in (forced_all if kind == "point" else forced_gate).values() for item in ids})
        pending: list[tuple[str, str, str, str]] = []
        for case_id in TEST_IDS:
            if _official(case_id):
                continue
            for candidate_id in ("exact", "noip", *allowed):
                for waveform_id in TEST_WAVEFORMS:
                    if not _cache_ok(case_id, f"{candidate_id}:{kind}", waveform_id):
                        pending.append((case_id, candidate_id, kind, waveform_id))
        print(f"[l2warm] {kind} pending={len(pending)} workers={_workers()}", flush=True)
        if not pending:
            continue
        workers = min(_workers(), len(pending))
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(TEST_IDS,),
        ) as pool:
            futures = {pool.submit(_warm_one, task): task for task in pending}
            for future in as_completed(futures):
                key = future.result()
                done += 1
                print(f"[l2warm] {kind} {done}/{len(pending)} {key}", flush=True)
    _assemble(forced_gate, official_variant)
    missing = [case_id for case_id in TEST_IDS if not _official(case_id)]
    if missing:
        print(f"[l2warm] still missing official TE JSON: {missing}", flush=True)
        return 1
    print("[l2warm] official TE01-TE10 ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
