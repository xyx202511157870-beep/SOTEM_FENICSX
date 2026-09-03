#!/usr/bin/env python3
"""Flow 2: cheapest oracle-gap falsification on the 8 pilot cases."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.case_bridge import evaluation_transform, load_cached_response
from atem3d.adaptive_debye_mvp.guards import assert_split_readable
from atem3d.adaptive_debye_mvp.io import read_json, write_json
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError
from atem3d.adaptive_debye_mvp.oracle_gap import (
    evaluate_l0,
    evaluate_pilot_case,
    execute_forward_unit,
    load_case_results,
    plan_disk_forward_units,
    plan_point_forward_units,
    write_case_result,
    write_oracle_gap_artifacts,
    write_pilot_case_result,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, PILOT_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


def _markdown_report(l0: dict, n_cases: int) -> str:
    lines = [
        "# LAYERED_ORACLE_GAP_REPORT",
        "",
        f"- L0 status: `{l0['status']}`",
        f"- passed A: `{l0['passed_A']}`",
        f"- passed B: `{l0['passed_B']}`",
        f"- n pilot cases: `{n_cases}`",
        f"- best same-K OR/B2 median ratio: `{l0['best_same_k_median_ratio']}` at K=`{l0['best_same_k']}`",
        f"- bootstrap 95% CI: `[{l0['bootstrap_ci_low']}, {l0['bootstrap_ci_high']}]`",
        f"- win rate: `{l0['win_rate']}`",
        f"- median K_qual_B2 - K_qual_OR: `{l0['median_k_qual_diff']}`",
        f"- nonnegative K_qual rate: `{l0['nonnegative_k_qual_rate']}`",
        "",
        "3-D was not run.",
        "",
    ]
    return "\n".join(lines)


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _pin_thread_env() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))


def _git_info() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_cache_identity(cache_dir: Path, transform, empymod_version: str) -> None:
    manifest_path = cache_dir / "CACHE_MANIFEST.json"
    expected = {
        "transform_label": transform.label,
        "ft_pts_per_dec": transform.ft_pts_per_dec,
        "empymod_version": empymod_version,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"cache manifest {key}={existing.get(key)!r} != {value!r}; refuse to reuse"
                )
        return
    if any(cache_dir.glob("*.npz")):
        raise RuntimeError("cache has .npz files but no CACHE_MANIFEST.json")
    write_json(manifest_path, expected)


def _warm_units(
    units: list[dict],
    *,
    cache_dir: Path,
    workers: int,
    log,
) -> dict[str, float | int]:
    pending: list[dict] = []
    seen: set[str] = set()
    already = 0
    for unit in units:
        key = str(unit["cache_key"])
        if key in seen:
            continue
        seen.add(key)
        if (cache_dir / f"{key}.npz").is_file():
            already += 1
            continue
        pending.append(unit)
    log(f"[flow2] warm {len(pending)}/{len(seen)} units ({already} already cached)")
    stats: dict[str, float | int] = {
        "unique": len(seen),
        "submitted": len(pending),
        "already_cached": already,
        "computed": 0,
        "seconds": 0.0,
    }
    if not pending:
        return stats
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(execute_forward_unit, unit, str(cache_dir)): unit["cache_key"] for unit in pending
        }
        try:
            for future in as_completed(futures):
                info = future.result()
                stats["computed"] = int(stats["computed"]) + 1
                stats["seconds"] = float(stats["seconds"]) + float(info["seconds"])
                log(
                    f"[flow2] unit done {info['cache_key']} {info['seconds']:.1f}s "
                    f"from_cache={info['from_cache']}"
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return stats


def _evaluate_cases(
    cases,
    *,
    include_disks: bool,
    label: str,
    cache_dir: Path,
    k_values: tuple[int, ...],
    workers: int,
    log,
    write_legacy_json: bool,
    flow2: Path,
) -> list:
    log(f"[flow2] stage={label} include_disks={include_disks}")
    stage_results = []
    if len(cases) == 1 or workers == 1:
        for case in cases:
            log(f"[flow2] {label} {case.case_id}")
            result = evaluate_pilot_case(
                case,
                waveform_ids=PILOT_WAVEFORMS,
                cache_dir=cache_dir,
                k_values=k_values,
                include_disks=include_disks,
            )
            if write_legacy_json:
                write_case_result(flow2 / f"{result['case_id']}.json", result)
            stage_results.append(result)
        return stage_results
    with ProcessPoolExecutor(max_workers=min(workers, len(cases))) as pool:
        futures = {
            pool.submit(
                evaluate_pilot_case,
                case,
                waveform_ids=PILOT_WAVEFORMS,
                cache_dir=str(cache_dir),
                k_values=k_values,
                include_disks=include_disks,
            ): case.case_id
            for case in cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            result = future.result()
            if write_legacy_json:
                write_case_result(flow2 / f"{result['case_id']}.json", result)
            log(f"[flow2] {label} finished {case_id}")
            stage_results.append(result)
    stage_results.sort(key=lambda item: item["case_id"])
    return stage_results


def _survey_hash(cache_dir: Path, case_id: str, kind: str) -> str:
    loaded = load_cached_response(cache_dir, f"{case_id}:exact:{kind}:W0")
    if loaded is None:
        return ""
    return str(loaded["hashes"]["shared_survey_hash"])


def _run_subset(
    *,
    cases,
    cache_dir: Path,
    flow2: Path,
    k_values: tuple[int, ...],
    workers: int,
    points_only: bool,
    log,
) -> int:
    import empymod
    import numpy
    import scipy

    started = time.perf_counter()
    transform = evaluation_transform()
    try:
        _assert_cache_identity(cache_dir, transform, empymod.__version__)
    except RuntimeError as exc:
        write_json(
            flow2 / f"subset_{'_'.join(case.case_id for case in cases)}_manifest.json",
            {
                "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
                "reason": str(exc),
                "three_d_run": False,
            },
        )
        print("BLOCKED_BY_SOFTWARE_OR_RESOURCES", flush=True)
        return 2

    point_units = [unit for case in cases for unit in plan_point_forward_units(case, k_values=k_values)]
    log(f"[flow2] planned {len(point_units)} point units across {len(cases)} cases")
    point_stats = _warm_units(point_units, cache_dir=cache_dir, workers=workers, log=log)
    point_results = _evaluate_cases(
        cases,
        include_disks=False,
        label="points",
        cache_dir=cache_dir,
        k_values=k_values,
        workers=workers,
        log=log,
        write_legacy_json=False,
        flow2=flow2,
    )
    disk_units: list[dict] = []
    results = point_results
    if not points_only:
        point_by_id = {item["case_id"]: item for item in point_results}
        for case in cases:
            result = point_by_id[case.case_id]
            shortlist_ids = {
                candidate_id
                for ids in (result.get("disk_shortlist") or {}).values()
                for candidate_id in ids
            }
            disk_units.extend(plan_disk_forward_units(case, shortlist_ids))
        log(f"[flow2] planned {len(disk_units)} disk units")
        disk_stats = _warm_units(disk_units, cache_dir=cache_dir, workers=workers, log=log)
        results = _evaluate_cases(
            cases,
            include_disks=True,
            label="disks",
            cache_dir=cache_dir,
            k_values=k_values,
            workers=workers,
            log=log,
            write_legacy_json=False,
            flow2=flow2,
        )
    else:
        disk_stats = {"unique": 0, "submitted": 0, "already_cached": 0, "computed": 0, "seconds": 0.0}

    wall_seconds = time.perf_counter() - started
    git = _git_info()
    by_id = {case.case_id: case for case in cases}
    written: dict[str, str] = {}
    for result in results:
        case = by_id[result["case_id"]]
        provenance = {
            "transform_label": transform.label,
            "ft_pts_per_dec": transform.ft_pts_per_dec,
            "empymod_version": empymod.__version__,
            "shared_survey_hash": {
                "point": _survey_hash(cache_dir, case.case_id, "point"),
                "disk": "" if points_only else _survey_hash(cache_dir, case.case_id, "disk"),
            },
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "roads_workers": workers,
            "wall_seconds": wall_seconds,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "three_d_run": False,
        }
        path = flow2 / f"case_{case.case_id}.json"
        write_pilot_case_result(
            path,
            result,
            case=case,
            provenance=provenance,
            k_values=k_values,
            waveform_ids=PILOT_WAVEFORMS,
        )
        written[path.name] = _file_sha256(path)
        log(f"[flow2] wrote {path.name}")

    cache_files = len(list(cache_dir.glob("*.npz")))
    ids = [case.case_id for case in cases]
    write_json(
        flow2 / f"subset_{'_'.join(ids)}_manifest.json",
        {
            "status": "SUBSET_COMPLETE",
            "case_ids": ids,
            "official_l0_gate_evaluated": False,
            "three_d_run": False,
            "n_forward_units": {
                "point": int(point_stats["unique"]),
                "disk": int(disk_stats["unique"]),
                "total": int(point_stats["unique"]) + int(disk_stats["unique"]),
            },
            "warm_stats": {"point": point_stats, "disk": disk_stats},
            "cache_files": cache_files,
            "wall_seconds": wall_seconds,
            "environment": {
                "empymod": empymod.__version__,
                "scipy": scipy.__version__,
                "numpy": numpy.__version__,
                "python": sys.version.split()[0],
                "cpu_count": os.cpu_count(),
                "roads_workers": workers,
            },
            "git": git,
            "sha256": written,
        },
    )
    log("[flow2] subset mode: official L0 gate not evaluated")
    return 0


def main() -> int:
    _pin_thread_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--case-ids",
        default="",
        help="comma-separated pilot case ids, e.g. PG03,PG04; subset mode writes only per-case JSON",
    )
    parser.add_argument("--k", default=",".join(str(value) for value in K_PILOT))
    parser.add_argument("--points-only", action="store_true", help="skip official disk averages")
    parser.add_argument("--skip-l0", action="store_true", help="write case JSON only; do not apply L0")
    parser.add_argument("--assemble-l0", action="store_true", help="load existing PG*.json and apply L0")
    args = parser.parse_args()
    k_values = tuple(int(item) for item in args.k.split(",") if item.strip())
    wanted = _parse_case_ids(args.case_ids)

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    cache_dir = generated / "cache"
    flow2 = generated / "flow2_oracle_gap"
    flow2.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        print(message, flush=True)
        with (flow2 / "flow2.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    if args.assemble_l0:
        paths = sorted(flow2.glob("PG*.json"))
        results = load_case_results(paths)
        if wanted:
            results = [item for item in results if item["case_id"] in set(wanted)]
        if len(results) != 8:
            _log(f"[flow2] assemble-l0 found {len(results)} cases, need 8")
            return 3
        l0 = evaluate_l0(results)
        write_oracle_gap_artifacts(flow2, results, l0)
        (flow2 / "LAYERED_ORACLE_GAP_REPORT.md").write_text(_markdown_report(l0, len(results)), encoding="utf-8")
        write_json(generated / "FLOW2_STATUS.json", {"status": l0["status"], "passed": l0["passed"], "l0": l0})
        if not l0["passed"]:
            write_json(
                generated / "STOP_REASON.json",
                {
                    "status": "STOP_LAYERED_NO_ACTIONABLE_GAP",
                    "gate": "L0",
                    "reason": "oracle gap A and B both failed on the frozen pilot set",
                    "l0": l0,
                    "three_d_run": False,
                },
            )
        _log(l0["status"])
        return 0

    cases = list(cases_for_split("pilot_gap", generate_all_cases()))
    for case in cases:
        assert_split_readable(case.split, stage="flow2")
    if wanted:
        by_id = {case.case_id: case for case in cases}
        unknown = [case_id for case_id in wanted if case_id not in by_id]
        if unknown:
            parser.error(f"unknown pilot case ids {unknown}; pilot ids are {sorted(by_id)}")
        cases = [by_id[case_id] for case_id in wanted]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    subset_mode = bool(wanted) or args.max_cases > 0
    if subset_mode:
        workers = max(1, int(os.environ.get("ROADS_WORKERS", str(os.cpu_count() or 1))))
    else:
        workers = max(1, min(int(os.environ.get("ROADS_WORKERS", "4")), len(cases)))
    _log(f"[flow2] {len(cases)} cases, K={k_values}, workers={workers}, case_ids={[c.case_id for c in cases]}")

    if subset_mode:
        try:
            return _run_subset(
                cases=cases,
                cache_dir=cache_dir,
                flow2=flow2,
                k_values=k_values,
                workers=workers,
                points_only=args.points_only,
                log=_log,
            )
        except BlockedBySoftwareOrResourcesError as exc:
            write_json(
                flow2 / f"subset_{'_'.join(case.case_id for case in cases)}_manifest.json",
                {
                    "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
                    "reason": str(exc),
                    "three_d_run": False,
                },
            )
            print("BLOCKED_BY_SOFTWARE_OR_RESOURCES", flush=True)
            return 2

    def _run_cases(*, include_disks: bool, label: str) -> list:
        return _evaluate_cases(
            cases,
            include_disks=include_disks,
            label=label,
            cache_dir=cache_dir,
            k_values=k_values,
            workers=workers,
            log=_log,
            write_legacy_json=True,
            flow2=flow2,
        )

    try:
        point_results = _run_cases(include_disks=False, label="points")
        point_l0 = evaluate_l0(point_results)
        write_json(
            flow2 / "L0_point_only_preview.json",
            {"note": "point receivers only; not the official L0", **point_l0},
        )
        _log(f"[flow2] point-only preview {point_l0['status']} median_ratio={point_l0['best_same_k_median_ratio']}")
        results = point_results
        if not args.points_only:
            results = _run_cases(include_disks=True, label="disks")
    except BlockedBySoftwareOrResourcesError as exc:
        write_json(
            generated / "STOP_REASON.json",
            {
                "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
                "gate": "L0",
                "reason": str(exc),
                "three_d_run": False,
            },
        )
        write_json(
            generated / "FLOW2_STATUS.json",
            {"status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES", "passed": False},
        )
        print("BLOCKED_BY_SOFTWARE_OR_RESOURCES", flush=True)
        return 2

    if args.skip_l0 or len(results) != 8 or args.points_only:
        _log(f"[flow2] wrote {len(results)} case JSON files; L0 not finalized")
        return 0

    l0 = evaluate_l0(results)
    write_oracle_gap_artifacts(flow2, results, l0)
    (flow2 / "LAYERED_ORACLE_GAP_REPORT.md").write_text(_markdown_report(l0, len(cases)), encoding="utf-8")
    if not l0["passed"]:
        write_json(
            generated / "STOP_REASON.json",
            {
                "status": "STOP_LAYERED_NO_ACTIONABLE_GAP",
                "gate": "L0",
                "reason": "oracle gap A and B both failed on the frozen pilot set",
                "l0": l0,
                "three_d_run": False,
            },
        )
    write_json(generated / "FLOW2_STATUS.json", {"status": l0["status"], "passed": l0["passed"], "l0": l0})
    print(l0["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
