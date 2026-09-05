#!/usr/bin/env python3
"""Fill remaining official Flow 3 disk cache files, then assemble case JSON.

Shares all CPUs across leftover (case, template, waveform) disk forwards
instead of pinning one case to one worker. Cache hits are skipped.
"""

from __future__ import annotations

import argparse
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
    forward_response,
    load_cached_response,
)
from atem3d.adaptive_debye_mvp.oracle_gap import (
    evaluate_pilot_case,
    fit_case_candidates,
    write_case_result,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, TRAIN_VAL_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import generate_all_cases, instantiate_candidates


GEN = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
CACHE = GEN / "cache"
FLOW3 = GEN / "flow3_selector"
DEFAULT_CASES = ("TR01", "TR03", "TR04")


def _workers() -> int:
    requested = max(4, int(os.cpu_count() or 4))
    env = os.environ.get("ROADS_WORKERS")
    if env:
        requested = max(4, int(env))
    return requested


def _forced() -> dict[int, list[str]]:
    path = FLOW3 / "forced_disk_ids.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): list(value) for key, value in payload.items()}


def _cache_ok(case_id: str, model_id: str, waveform_id: str) -> bool:
    key = f"{case_id}:{model_id}:{waveform_id}"
    path = CACHE / f"{key}.npz"
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    return load_cached_response(CACHE, key) is not None


def _shortlist_for_case(case, *, official_variant: str, forced: dict[int, list[str]]) -> dict[int, set[str]]:
    result = evaluate_pilot_case(
        case,
        waveform_ids=TRAIN_VAL_WAVEFORMS,
        cache_dir=CACHE,
        k_values=K_PILOT,
        official_variant=official_variant,
        include_disks=False,
        forced_disk_ids=forced,
        include_tilted=False,
    )
    return {int(key): set(value) for key, value in (result.get("disk_shortlist") or {}).items()}


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
            "receivers": disk_receivers(case),
            "times": case_time_grid(),
            "geometry": case_geometry(case),
            "transform": evaluation_transform(),
        }
    _WORKER_CTX = ctx


def _warm_one(task: tuple[str, str, str]) -> str:
    case_id, candidate_id, waveform_id = task
    ctx = _WORKER_CTX[case_id]
    record = ctx["fits"][candidate_id]
    material = debye_material(ctx["case"], record.fit, candidate_id=candidate_id)
    forward_response(
        material,
        ctx["geometry"],
        case_waveform(waveform_id),
        ctx["receivers"],
        ctx["times"],
        ctx["transform"],
        cache_dir=CACHE,
        cache_key=f"{case_id}:{candidate_id}:disk:{waveform_id}",
    )
    return f"{case_id}:{candidate_id}:disk:{waveform_id}"


def _official_ready(case_id: str) -> bool:
    path = FLOW3 / f"case_{case_id}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("point_only") is not False:
        return False
    if "pilot_case_result" not in str(payload.get("schema") or ""):
        return False
    n_disk = sum(
        1
        for task in payload.get("tasks") or []
        if str(task.get("receiver_id") or "").startswith("disk_")
    )
    return n_disk >= 180


def _assemble(case_ids: tuple[str, ...], official_variant: str, forced: dict[int, list[str]]) -> None:
    cases = {item.case_id: item for item in generate_all_cases()}
    for case_id in case_ids:
        if _official_ready(case_id):
            print(f"[warm] assemble skip {case_id} already official", flush=True)
            continue
        started = time.time()
        print(f"[warm] assemble {case_id}", flush=True)
        result = evaluate_pilot_case(
            cases[case_id],
            waveform_ids=TRAIN_VAL_WAVEFORMS,
            cache_dir=CACHE,
            k_values=K_PILOT,
            official_variant=official_variant,
            include_disks=True,
            forced_disk_ids=forced,
            include_tilted=False,
        )
        result["waveform_ids"] = list(TRAIN_VAL_WAVEFORMS)
        result["k_values"] = list(K_PILOT)
        result["receiver_ids"] = ["point", "disk_1.0", "disk_4.0"]
        result["provenance"] = {
            "transform_label": "smoke_fast_lagged_dlf",
            "roads_workers": os.environ.get("ROADS_WORKERS"),
            "wall_seconds": time.time() - started,
            "stage": "disks",
        }
        write_case_result(FLOW3 / f"case_{result['case_id']}.json", result)
        print(f"[warm] wrote {case_id} point_only={result.get('point_only')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASES))
    parser.add_argument("--official-variant", default="S1")
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    case_ids = tuple(item.strip() for item in args.case_ids.split(",") if item.strip())
    forced = _forced()
    if args.assemble_only:
        _assemble(case_ids, args.official_variant, forced)
        return 0 if all(_official_ready(case_id) for case_id in case_ids) else 1

    cases = {item.case_id: item for item in generate_all_cases() if item.case_id in set(case_ids)}
    pending: list[tuple[str, str, str]] = []
    for case_id in case_ids:
        if _official_ready(case_id):
            print(f"[warm] {case_id} already official disk JSON", flush=True)
            continue
        print(f"[warm] ranking shortlist {case_id} from point cache", flush=True)
        shortlists = _shortlist_for_case(
            cases[case_id],
            official_variant=args.official_variant,
            forced=forced,
        )
        for poles, ids in shortlists.items():
            print(f"[warm] {case_id} K={poles} n={len(ids)} {sorted(ids)}", flush=True)
            for candidate_id in sorted(ids):
                for waveform_id in TRAIN_VAL_WAVEFORMS:
                    if not _cache_ok(case_id, f"{candidate_id}:disk", waveform_id):
                        pending.append((case_id, candidate_id, waveform_id))
    print(f"[warm] pending disk waveforms={len(pending)} workers={_workers()}", flush=True)
    if pending:
        workers = min(_workers(), len(pending))
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(case_ids,),
        ) as pool:
            futures = {pool.submit(_warm_one, task): task for task in pending}
            for future in as_completed(futures):
                key = future.result()
                done += 1
                print(f"[warm] {done}/{len(pending)} {key}", flush=True)
    _assemble(case_ids, args.official_variant, forced)
    missing = [case_id for case_id in case_ids if not _official_ready(case_id)]
    if missing:
        print(f"[warm] still missing official JSON: {missing}", flush=True)
        return 1
    print("[warm] TR01/TR03/TR04 official disk JSON ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
