#!/usr/bin/env python3
"""Evaluate train/validation cases for Flow 3 (points then disks)."""

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

from atem3d.adaptive_debye_mvp.guards import assert_case_ids_split_safe, assert_split_readable
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError
from atem3d.adaptive_debye_mvp.oracle_gap import evaluate_pilot_case, write_case_result
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, TRAIN_VAL_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def case_json_satisfies(path: Path, stage: str) -> bool:
    """True if an existing case JSON already covers this stage.

    Disk JSON also satisfies the point stage so a later point pass cannot
    overwrite official disk tasks.
    """

    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not payload.get("case_id") or not payload.get("tasks"):
        return False
    if not payload.get("point_only", True):
        return True
    return stage == "points"


def _workers(n_cases: int) -> int:
    requested = max(4, int(os.cpu_count() or 4))
    env = os.environ.get("ROADS_WORKERS")
    if env:
        requested = max(4, int(env))
    return max(1, min(requested, n_cases))


def _evaluate(case, *, cache_dir, include_disks, forced_disk_ids, official_variant):
    started = time.time()
    result = evaluate_pilot_case(
        case,
        waveform_ids=TRAIN_VAL_WAVEFORMS,
        cache_dir=cache_dir,
        k_values=K_PILOT,
        official_variant=official_variant,
        include_disks=include_disks,
        forced_disk_ids=forced_disk_ids,
        include_tilted=False,
    )
    result["waveform_ids"] = list(TRAIN_VAL_WAVEFORMS)
    result["k_values"] = list(K_PILOT)
    result["receiver_ids"] = ["point"] if not include_disks else ["point", "disk_1.0", "disk_4.0"]
    result["provenance"] = {
        "transform_label": "smoke_fast_lagged_dlf",
        "roads_workers": os.environ.get("ROADS_WORKERS"),
        "wall_seconds": time.time() - started,
        "stage": "disks" if include_disks else "points",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--stage", choices=("points", "disks", "both"), default="both")
    parser.add_argument("--forced-ids-json", default="")
    parser.add_argument("--official-variant", default="")
    args = parser.parse_args()

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    cache_dir = generated / "cache"
    out_dir = generated / "flow3_selector"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list(cases_for_split(args.split, generate_all_cases()))
    for case in cases:
        assert_split_readable(case.split, stage="flow3")
    wanted = _parse_case_ids(args.case_ids)
    if wanted:
        assert_case_ids_split_safe(wanted, stage="flow3")
        cases = [case for case in cases if case.case_id in set(wanted)]
    assert_case_ids_split_safe([case.case_id for case in cases], stage="flow3")

    forced: dict[int, list[str]] | None = None
    if args.forced_ids_json:
        payload = json.loads(Path(args.forced_ids_json).read_text(encoding="utf-8"))
        forced = {int(key): list(value) for key, value in payload.items()}
    official_variant = args.official_variant or None
    workers = _workers(len(cases))
    stages = ("points", "disks") if args.stage == "both" else (args.stage,)

    def _log(message: str) -> None:
        print(message, flush=True)
        with (out_dir / "flow3.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    _log(f"[flow3] split={args.split} n={len(cases)} workers={workers} stages={stages}")

    try:
        for stage in stages:
            include_disks = stage == "disks"
            if include_disks and forced is None:
                forced_path = out_dir / "forced_disk_ids.json"
                if forced_path.is_file():
                    payload = json.loads(forced_path.read_text(encoding="utf-8"))
                    forced = {int(key): list(value) for key, value in payload.items()}
            pending = [
                case
                for case in cases
                if not case_json_satisfies(out_dir / f"case_{case.case_id}.json", stage)
            ]
            skipped = [case.case_id for case in cases if case not in pending]
            if skipped:
                _log(f"[flow3] {stage} skip existing {skipped}")
            if not pending:
                continue
            stage_workers = max(1, min(workers, len(pending)))
            if stage_workers == 1:
                for case in pending:
                    result = _evaluate(
                        case,
                        cache_dir=cache_dir,
                        include_disks=include_disks,
                        forced_disk_ids=forced,
                        official_variant=official_variant,
                    )
                    write_case_result(out_dir / f"case_{result['case_id']}.json", result)
                    _log(f"[flow3] {stage} finished {result['case_id']}")
            else:
                with ProcessPoolExecutor(max_workers=stage_workers) as pool:
                    futures = {
                        pool.submit(
                            _evaluate,
                            case,
                            cache_dir=str(cache_dir),
                            include_disks=include_disks,
                            forced_disk_ids=forced,
                            official_variant=official_variant,
                        ): case.case_id
                        for case in pending
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        write_case_result(out_dir / f"case_{result['case_id']}.json", result)
                        _log(f"[flow3] {stage} finished {result['case_id']}")
    except BlockedBySoftwareOrResourcesError as exc:
        _log(f"BLOCKED_BY_SOFTWARE_OR_RESOURCES {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
