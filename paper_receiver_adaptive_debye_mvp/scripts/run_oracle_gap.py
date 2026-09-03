#!/usr/bin/env python3
"""Flow 2: cheapest oracle-gap falsification on the 8 pilot cases."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402  — after thread-env pin


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.case_bridge import (  # noqa: E402
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
from atem3d.adaptive_debye_mvp.guards import assert_split_readable  # noqa: E402
from atem3d.adaptive_debye_mvp.io import read_csv, read_json, sha256_hex, to_record, write_json  # noqa: E402
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError  # noqa: E402
from atem3d.adaptive_debye_mvp.oracle_gap import (  # noqa: E402
    _group_ratio,
    evaluate_l0,
    evaluate_pilot_case,
    fit_case_candidates,
    pick_oracle_best,
    pick_spectral_best,
    qualifying_k,
    reduce_case_error,
    write_oracle_gap_artifacts,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, PILOT_WAVEFORMS  # noqa: E402
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases, instantiate_candidates  # noqa: E402


CASE_RESULT_SCHEMA = "roads_debye_mvp.flow2_case_result.v1"


def parse_case_ids(text: str) -> tuple[str, ...]:
    """Parse a comma-separated case-id list, preserving order and dropping blanks."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in str(text).split(","):
        case_id = item.strip()
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        ordered.append(case_id)
    return tuple(ordered)


def select_pilot_cases(case_ids: tuple[str, ...] = ()) -> list:
    """Return ``pilot_gap`` cases, optionally filtered by ``--case-ids``."""

    cases = list(cases_for_split("pilot_gap", generate_all_cases()))
    if not case_ids:
        return cases
    known = {case.case_id: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in known]
    if missing:
        raise SystemExit(f"unknown or non-pilot case ids: {', '.join(missing)}")
    return [known[case_id] for case_id in case_ids]


def _pin_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _file_sha256(path: Path) -> str:
    return sha256_hex(path.read_text(encoding="utf-8"))


def _git_field(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "missing"
    return str(getattr(module, "__version__", "present"))


def _ensure_cache_identity(cache_dir: Path) -> dict:
    transform = evaluation_transform()
    identity = {"label": transform.label, **transform.hash_payload()}
    path = cache_dir / "CACHE_IDENTITY.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = read_json(path)
        if existing != identity:
            raise RuntimeError(
                f"cache identity mismatch in {path}: existing={existing} wanted={identity}"
            )
        return identity
    write_json(path, identity)
    return identity


def _assert_case_hashes(cases, registry_path: Path) -> None:
    rows = {row["case_id"]: row for row in read_csv(registry_path)}
    for case in cases:
        expected = str(rows[case.case_id]["case_hash"])
        actual = case.case_hash()
        if actual != expected:
            raise SystemExit(f"case_hash mismatch for {case.case_id}: {actual} != {expected}")


def _fit_one_case(case, k_values: tuple[int, ...]) -> tuple[str, list]:
    _pin_threads()
    candidates = tuple(spec for spec in instantiate_candidates(case) if spec.K in k_values)
    return case.case_id, fit_case_candidates(case, candidates)


def _warm_forward_unit(payload: dict) -> dict:
    """Compute one cached ``forward_response`` unit. Safe for ProcessPoolExecutor."""

    _pin_threads()
    case = payload["case"]
    waveform_id = payload["waveform_id"]
    receiver_kind = payload["receiver_kind"]
    cache_dir = payload["cache_dir"]
    kind = payload["kind"]
    if kind == "exact":
        material = exact_pelton_material(case)
        model_id = f"exact:{receiver_kind}"
    elif kind == "noip":
        material = nonpolarizable_material(case)
        model_id = f"noip:{receiver_kind}"
    else:
        material = debye_material(case, payload["fit"], candidate_id=payload["candidate_id"])
        model_id = f"{payload['candidate_id']}:{receiver_kind}"
    receivers = disk_receivers(case) if receiver_kind == "disk" else point_receivers(case)
    cache_key = f"{case.case_id}:{model_id}:{waveform_id}"
    if load_cached_response(cache_dir, cache_key) is not None:
        return {"cache_key": cache_key, "hit": True, "case_id": case.case_id}
    forward_response(
        material,
        case_geometry(case),
        case_waveform(waveform_id),
        receivers,
        case_time_grid(),
        evaluation_transform(),
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    return {"cache_key": cache_key, "hit": False, "case_id": case.case_id}


def _point_units(case, fits, waveform_ids: tuple[str, ...], cache_dir: Path) -> list[dict]:
    units = []
    for waveform_id in waveform_ids:
        units.append(
            {
                "case": case,
                "kind": "exact",
                "receiver_kind": "point",
                "waveform_id": waveform_id,
                "cache_dir": str(cache_dir),
            }
        )
        units.append(
            {
                "case": case,
                "kind": "noip",
                "receiver_kind": "point",
                "waveform_id": waveform_id,
                "cache_dir": str(cache_dir),
            }
        )
    for record in fits:
        if not record.valid:
            continue
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case": case,
                    "kind": "candidate",
                    "candidate_id": record.candidate_id,
                    "fit": record.fit,
                    "receiver_kind": "point",
                    "waveform_id": waveform_id,
                    "cache_dir": str(cache_dir),
                }
            )
    return units


def _disk_units(
    case,
    fits,
    waveform_ids: tuple[str, ...],
    cache_dir: Path,
    *,
    candidate_ids: set[str] | None,
    include_exact_noip: bool,
) -> list[dict]:
    units = []
    if include_exact_noip:
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case": case,
                    "kind": "exact",
                    "receiver_kind": "disk",
                    "waveform_id": waveform_id,
                    "cache_dir": str(cache_dir),
                }
            )
            units.append(
                {
                    "case": case,
                    "kind": "noip",
                    "receiver_kind": "disk",
                    "waveform_id": waveform_id,
                    "cache_dir": str(cache_dir),
                }
            )
    wanted = candidate_ids
    for record in fits:
        if not record.valid:
            continue
        if wanted is not None and record.candidate_id not in wanted:
            continue
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case": case,
                    "kind": "candidate",
                    "candidate_id": record.candidate_id,
                    "fit": record.fit,
                    "receiver_kind": "disk",
                    "waveform_id": waveform_id,
                    "cache_dir": str(cache_dir),
                }
            )
    return units


def _run_pool(units: list[dict], workers: int, log, *, label: str) -> dict[str, int]:
    if not units:
        log(f"[flow2] {label}: 0 units")
        return {"hit": 0, "miss": 0, "n": 0}
    hits = 0
    misses = 0
    done = 0
    log(f"[flow2] {label}: {len(units)} units workers={workers}")
    if workers == 1:
        for unit in units:
            result = _warm_forward_unit(unit)
            hits += int(result["hit"])
            misses += int(not result["hit"])
            done += 1
            if done == 1 or done == len(units) or done % 20 == 0:
                log(f"[flow2] {label} {done}/{len(units)} last={result['cache_key']} hit={result['hit']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_warm_forward_unit, unit) for unit in units]
            for future in as_completed(futures):
                result = future.result()
                hits += int(result["hit"])
                misses += int(not result["hit"])
                done += 1
                if done == 1 or done == len(units) or done % 20 == 0:
                    log(
                        f"[flow2] {label} {done}/{len(units)} "
                        f"last={result['cache_key']} hit={result['hit']}"
                    )
    log(f"[flow2] {label} done hits={hits} misses={misses}")
    return {"hit": hits, "miss": misses, "n": len(units)}


def disk_shortlists_from_point_result(result: dict, *, disk_shortlist: int = 2) -> dict[int, list[str]]:
    """Mirror ``evaluate_pilot_case`` shortlist selection from a point-only result."""

    fits = result["fits"]
    tasks = result["tasks"]
    variant = result["official_variant"]
    by_k: dict[int, list] = {}
    for record in fits:
        by_k.setdefault(record.K, []).append(record)
    shortlists: dict[int, list[str]] = {}
    for poles, records in by_k.items():
        point_map: dict[str, list] = {}
        for record in records:
            if not record.valid:
                continue
            rows = [
                item
                for item in tasks
                if item.candidate_id == record.candidate_id and item.K == poles
            ]
            if rows:
                point_map[record.candidate_id] = rows
        if not point_map:
            continue
        spectral = pick_spectral_best(records, variant)
        oracle = pick_oracle_best(point_map, records)
        ranked = sorted(point_map, key=lambda cid: (reduce_case_error(point_map[cid]), cid))
        shortlists[poles] = sorted(
            {
                spectral.candidate_id,
                oracle.candidate_id,
                *ranked[: int(disk_shortlist)],
            }
        )
    return shortlists


def _fit_json(record) -> dict:
    tau = []
    if record.fit is not None:
        tau = [float(value) for value in record.fit.tau_grid]
    return {
        "candidate_id": record.candidate_id,
        "K": int(record.K),
        "valid": bool(record.valid),
        "spectral_error_s0": float(record.spectral_error_s0),
        "spectral_error_s1": float(record.spectral_error_s1),
        "condition_number": float(record.condition_number),
        "relative_dc_error": float(record.relative_dc_error),
        "optimizer_success": bool(record.optimizer_success),
        "tau_grid": tau,
    }


def _case_groups(choices, tasks) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for choice in choices:
        or_tasks = [item for item in tasks if item.candidate_id == choice.oracle_id and item.K == choice.K]
        b2_tasks = [item for item in tasks if item.candidate_id == choice.spectral_id and item.K == choice.K]
        waveform = {
            waveform_id: _group_ratio(or_tasks, b2_tasks, lambda item, wid=waveform_id: item.waveform_id == wid)
            for waveform_id in PILOT_WAVEFORMS
        }
        receiver = {
            "point": _group_ratio(or_tasks, b2_tasks, lambda item: item.receiver_id == "point"),
            "disk_1.0": _group_ratio(or_tasks, b2_tasks, lambda item: item.receiver_id == "disk_1.0"),
            "disk_4.0": _group_ratio(or_tasks, b2_tasks, lambda item: item.receiver_id == "disk_4.0"),
            "disk": _group_ratio(
                or_tasks, b2_tasks, lambda item: item.receiver_id in {"disk_1.0", "disk_4.0"}
            ),
        }
        channel = {
            "H": _group_ratio(or_tasks, b2_tasks, lambda item: np.isfinite(item.h_p95)),
            "dBdt": _group_ratio(or_tasks, b2_tasks, lambda item: np.isfinite(item.dbdt_p95)),
        }
        ip_ratio = (
            float(choice.ip_or / choice.ip_b2)
            if choice.ip_b2 > 0.0
            else float("nan")
        )
        groups[str(choice.K)] = {
            "waveform": waveform,
            "receiver": receiver,
            "channel": channel,
            "ip_ratio": ip_ratio,
        }
    return groups


def build_case_result_payload(
    result: dict,
    *,
    case,
    k_values: tuple[int, ...],
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
    disk_shortlists: dict[int, list[str]] | None = None,
    provenance: dict | None = None,
) -> dict:
    """Compact per-case JSON that ``evaluate_l0`` can rehydrate from."""

    choices = result["choices"]
    tasks = result["tasks"]
    fits = result.get("fits") or []
    shortlists = disk_shortlists or disk_shortlists_from_point_result(result)
    point_ids = sorted({item.candidate_id for item in tasks if item.receiver_id == "point"})
    disk_ids = sorted(
        {item.candidate_id for item in tasks if item.receiver_id in {"disk_1.0", "disk_4.0"}}
    )
    k_qual_b2 = qualifying_k(choices, oracle=False)
    k_qual_or = qualifying_k(choices, oracle=True)
    payload = {
        "schema": CASE_RESULT_SCHEMA,
        "case_id": result["case_id"],
        "split": case.split,
        "case_hash": case.case_hash(),
        "official_variant": result["official_variant"],
        "k_values": list(k_values),
        "waveform_ids": list(waveform_ids),
        "receiver_ids": ["point", "disk_1.0", "disk_4.0"],
        "channel_groups": {"H": ["Hx", "Hy", "Hz"], "dBdt": ["dBxdt", "dBydt", "dBzdt"]},
        "ip_metric": "ip_increment_nrmse",
        "point_only": bool(result.get("point_only", False)),
        "choices": [to_record(item) for item in choices],
        "tasks": [to_record(item) for item in tasks],
        "fits": [_fit_json(item) for item in fits],
        "k_qual": {
            "b2": k_qual_b2,
            "or": k_qual_or,
            "diff_b2_minus_or": (
                float(k_qual_b2 - k_qual_or)
                if k_qual_b2 == k_qual_b2 and k_qual_or == k_qual_or
                else float("nan")
            ),
        },
        "groups": _case_groups(choices, tasks),
        "disk_shortlists": {str(key): value for key, value in sorted(shortlists.items())},
        "disk_shortlist": {str(key): value for key, value in sorted(shortlists.items())},
        "candidate_ids_evaluated": {"point": point_ids, "disk": disk_ids},
        "three_d_run": False,
        "l0_gate_evaluated": False,
    }
    if provenance:
        payload["provenance"] = provenance
    return payload


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


def _write_blocked(flow2: Path, cases, reason: str, *, per_case: bool) -> None:
    generated = flow2.parent
    payload = {
        "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
        "gate": "L0",
        "reason": str(reason),
        "three_d_run": False,
        "case_ids": [case.case_id for case in cases],
    }
    if per_case:
        for case in cases:
            write_json(flow2 / f"case_{case.case_id}_BLOCKED.json", {**payload, "case_id": case.case_id})
        return
    write_json(generated / "STOP_REASON.json", payload)
    write_json(generated / "FLOW2_STATUS.json", {"status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES", "passed": False})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--k", default=",".join(str(value) for value in K_PILOT))
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated pilot case ids (filters generate_all_cases / cases_for_split).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process pool size. Default: ROADS_WORKERS, else nproc. "
        "When --case-ids selects a subset, this is the unit-level pool size.",
    )
    parser.add_argument(
        "--per-case-only",
        action="store_true",
        help="Write case_<ID>.json only; skip the 8-case L0 gate and shared Flow-2 status files.",
    )
    parser.add_argument(
        "--log-name",
        default="",
        help="Log file name under flow2_oracle_gap/. Default flow2.log, or flow2_<ids>.log with --case-ids.",
    )
    args = parser.parse_args()
    k_values = tuple(int(item) for item in args.k.split(",") if item.strip())
    requested_ids = parse_case_ids(args.case_ids)

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    cache_dir = generated / "cache"
    flow2 = generated / "flow2_oracle_gap"
    flow2.mkdir(parents=True, exist_ok=True)

    cases = select_pilot_cases(requested_ids)
    for case in cases:
        assert_split_readable(case.split, stage="flow2")
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no cases selected")
    _assert_case_hashes(cases, generated / "case_registry.csv")

    all_pilot = list(cases_for_split("pilot_gap", generate_all_cases()))
    subset = {case.case_id for case in cases} != {case.case_id for case in all_pilot}
    per_case_only = bool(args.per_case_only or (requested_ids and subset))

    detected_cpus = os.cpu_count() or 4
    try:
        detected_cpus = max(detected_cpus, int(subprocess.check_output(["nproc", "--all"], text=True).strip()))
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    if args.workers > 0:
        requested_workers = args.workers
    elif "ROADS_WORKERS" in os.environ:
        requested_workers = int(os.environ["ROADS_WORKERS"])
        # Some containers report nproc=1 while os.cpu_count()/nproc --all is 4.
        # Subset runs must saturate the VM unless --workers is explicit.
        if requested_workers < detected_cpus:
            requested_workers = detected_cpus
    else:
        requested_workers = detected_cpus
    requested_workers = max(1, requested_workers)
    if per_case_only:
        unit_workers = requested_workers
        case_workers = max(1, min(requested_workers, len(cases)))
        workers = case_workers
    else:
        workers = max(1, min(int(os.environ.get("ROADS_WORKERS", "4")), len(cases)))
        unit_workers = workers
        case_workers = workers

    if args.log_name:
        log_name = args.log_name
    elif requested_ids:
        log_name = f"flow2_{'_'.join(requested_ids)}.log"
    else:
        log_name = "flow2.log"

    def _log(message: str) -> None:
        print(message, flush=True)
        with (flow2 / log_name).open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    _log(
        f"[flow2] {len(cases)} cases {[case.case_id for case in cases]} "
        f"K={k_values} workers={workers} unit_workers={unit_workers} "
        f"per_case_only={per_case_only} log={log_name}"
    )

    def _run_cases(*, include_disks: bool, label: str) -> list:
        _log(f"[flow2] stage={label} include_disks={include_disks}")
        stage_results = []
        if case_workers == 1:
            for case in cases:
                _log(f"[flow2] {label} {case.case_id}")
                stage_results.append(
                    evaluate_pilot_case(
                        case,
                        waveform_ids=PILOT_WAVEFORMS,
                        cache_dir=cache_dir,
                        k_values=k_values,
                        include_disks=include_disks,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=case_workers) as pool:
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
                    _log(f"[flow2] {label} finished {case_id}")
                    stage_results.append(result)
            stage_results.sort(key=lambda item: item["case_id"])
        return stage_results

    try:
        _ensure_cache_identity(cache_dir)
        if per_case_only:
            started = datetime.now(timezone.utc)
            cache_stats = {"hit": 0, "miss": 0, "n": 0}
            fits_by_id: dict[str, list] = {}
            if case_workers == 1:
                for case in cases:
                    _log(f"[flow2] fitting {case.case_id}")
                    case_id, fits = _fit_one_case(case, k_values)
                    fits_by_id[case_id] = fits
            else:
                with ProcessPoolExecutor(max_workers=case_workers) as pool:
                    futures = {pool.submit(_fit_one_case, case, k_values): case.case_id for case in cases}
                    for future in as_completed(futures):
                        case_id, fits = future.result()
                        fits_by_id[case_id] = fits
                        _log(f"[flow2] fitted {case_id} n={len(fits)}")

            stage_a_point = []
            stage_a_disk = []
            for case in cases:
                stage_a_point.extend(_point_units(case, fits_by_id[case.case_id], PILOT_WAVEFORMS, cache_dir))
                stage_a_disk.extend(
                    _disk_units(
                        case,
                        fits_by_id[case.case_id],
                        PILOT_WAVEFORMS,
                        cache_dir,
                        candidate_ids=set(),
                        include_exact_noip=True,
                    )
                )
            # Longest units first so disk exact/noip occupy workers immediately.
            stage_a = stage_a_disk + stage_a_point
            stats_a = _run_pool(stage_a, unit_workers, _log, label="warm-points+exact-disks")
            for key in cache_stats:
                cache_stats[key] += stats_a[key]

            point_results = _run_cases(include_disks=False, label="points")
            shortlists_by_case = {
                item["case_id"]: disk_shortlists_from_point_result(item) for item in point_results
            }

            stage_c = []
            for case in cases:
                ids = {cid for values in shortlists_by_case[case.case_id].values() for cid in values}
                stage_c.extend(
                    _disk_units(
                        case,
                        fits_by_id[case.case_id],
                        PILOT_WAVEFORMS,
                        cache_dir,
                        candidate_ids=ids,
                        include_exact_noip=False,
                    )
                )
            stats_c = _run_pool(stage_c, unit_workers, _log, label="warm-disk-shortlists")
            for key in cache_stats:
                cache_stats[key] += stats_c[key]

            results = _run_cases(include_disks=True, label="disks")
            finished = datetime.now(timezone.utc)
            transform = evaluation_transform()
            by_case = {case.case_id: case for case in cases}
            for result in results:
                case = by_case[result["case_id"]]
                provenance = {
                    "transform": {"label": transform.label, **transform.hash_payload()},
                    "empymod_version": _module_version("empymod"),
                    "scipy_version": _module_version("scipy"),
                    "numpy_version": _module_version("numpy"),
                    "git_commit": _git_field("rev-parse", "HEAD"),
                    "branch": _git_field("rev-parse", "--abbrev-ref", "HEAD"),
                    "case_registry_sha256": _file_sha256(generated / "case_registry.csv"),
                    "candidate_registry_sha256": _file_sha256(generated / "candidate_registry.csv"),
                    "protocol_md_sha256": _file_sha256(
                        REPO_ROOT / "paper_receiver_adaptive_debye_mvp" / "protocol.md"
                    ),
                    "workers": unit_workers,
                    "n_forward_units": {
                        "point": len(stage_a_point),
                        "disk": len(stage_a_disk) + stats_c["n"],
                    },
                    "cache_hits": cache_stats["hit"],
                    "cache_misses": cache_stats["miss"],
                    "started_at": started.isoformat(),
                    "finished_at": finished.isoformat(),
                    "wall_seconds": (finished - started).total_seconds(),
                }
                payload = build_case_result_payload(
                    result,
                    case=case,
                    k_values=k_values,
                    disk_shortlists=shortlists_by_case[case.case_id],
                    provenance=provenance,
                )
                destination = flow2 / f"case_{case.case_id}.json"
                write_json(destination, payload)
                _log(
                    f"[flow2] wrote {destination.relative_to(REPO_ROOT)} "
                    f"choices={len(payload['choices'])} tasks={len(payload['tasks'])} "
                    f"l0_gate_evaluated=false"
                )
            _log("[flow2] per-case subset finished; 8-case L0 gate not evaluated")
            return 0

        point_results = _run_cases(include_disks=False, label="points")
        point_l0 = evaluate_l0(point_results)
        write_json(flow2 / "L0_point_only_preview.json", {"note": "point receivers only; not the official L0", **point_l0})
        _log(f"[flow2] point-only preview {point_l0['status']} median_ratio={point_l0['best_same_k_median_ratio']}")
        results = _run_cases(include_disks=True, label="disks")
    except BlockedBySoftwareOrResourcesError as exc:
        _write_blocked(flow2, cases, str(exc), per_case=per_case_only)
        print("BLOCKED_BY_SOFTWARE_OR_RESOURCES", flush=True)
        return 2
    for result in results:
        write_json(
            flow2 / f"{result['case_id']}.json",
            {"case_id": result["case_id"], "choices": [item.__dict__ for item in result["choices"]]},
        )

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
