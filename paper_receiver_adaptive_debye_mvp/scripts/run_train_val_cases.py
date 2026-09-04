#!/usr/bin/env python3
"""Flow 3 train/validation layered responses. Does not freeze the selector."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.guards import IndependentTestLeakageError
from atem3d.adaptive_debye_mvp.io import read_json, write_json
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError
from atem3d.adaptive_debye_mvp.oracle_gap import (
    assemble_train_val_metrics,
    evaluate_pilot_case,
    fit_case_candidates,
    resolve_train_val_cases,
    write_split_case_result,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, TRAIN_VAL_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import instantiate_candidates


def _thread_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
    }


def _git_info() -> tuple[str | None, str | None]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return branch, commit


def _empymod_version() -> str | None:
    try:
        import empymod

        return str(empymod.__version__)
    except Exception:
        return None


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    if not isinstance(payload, dict):
        return False
    return (not bool(payload.get("point_only", True))) and str(payload.get("status", "")) == "COMPLETE"


def evaluate_and_write_case(
    case,
    *,
    waveform_ids: tuple[str, ...],
    cache_dir: str,
    k_values: tuple[int, ...],
    include_disks: bool,
    official_variant: str,
    output_dir: str,
    provenance_base: dict,
) -> dict:
    started = time.time()
    result = evaluate_pilot_case(
        case,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        k_values=k_values,
        official_variant=official_variant,
        include_disks=include_disks,
    )
    provenance = {
        **provenance_base,
        "wall_seconds": time.time() - started,
        "stage": "points+disks" if include_disks else "points",
        "shared_survey_hash": result.get("shared_survey_hash", {}),
        "three_d_run": False,
        "independent_test_read": False,
    }
    path = Path(output_dir) / f"case_{case.case_id}.json"
    write_split_case_result(path, result, case=case, split=case.split, provenance=provenance)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "point_only": bool(result.get("point_only", not include_disks)),
        "n_tasks": len(result["tasks"]),
        "n_choices": len(result["choices"]),
        "path": str(path),
    }


def _worker(payload: dict) -> dict:
    return evaluate_and_write_case(
        payload["case"],
        waveform_ids=payload["waveform_ids"],
        cache_dir=payload["cache_dir"],
        k_values=payload["k_values"],
        include_disks=payload["include_disks"],
        official_variant=payload["official_variant"],
        output_dir=payload["output_dir"],
        provenance_base=payload["provenance_base"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="both", help="train, validation, or both")
    parser.add_argument("--case-ids", default="", help="comma-separated case ids")
    parser.add_argument("--k", default=",".join(str(value) for value in K_PILOT))
    parser.add_argument("--points-only", action="store_true")
    parser.add_argument("--fits-only", action="store_true", help="hard-DC fits only; no empymod")
    parser.add_argument("--skip-complete", action="store_true", default=True)
    parser.add_argument("--no-skip-complete", action="store_false", dest="skip_complete")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    flow2 = generated / "flow2_oracle_gap"
    flow3 = generated / "flow3_selector"
    cache_dir = generated / "cache"
    flow3.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "train_val_runner.log"

    def _log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    l0_path = flow2 / "L0_summary.json"
    if not l0_path.is_file() or not bool(read_json(l0_path).get("passed", False)):
        write_json(
            generated / "FLOW3_STATUS.json",
            {"status": "REFUSED", "reason": "L0 did not pass", "l1_frozen": False, "three_d_run": False},
        )
        raise SystemExit("Flow 3 train/val refused: L0 did not pass")

    variant_path = flow2 / "official_spectral_variant.json"
    official_variant = str(read_json(variant_path)["variant"]) if variant_path.is_file() else "S0"
    k_values = tuple(int(item) for item in args.k.split(",") if item.strip())
    case_ids = _parse_case_ids(args.case_ids)
    try:
        cases = resolve_train_val_cases(split=args.split, case_ids=case_ids)
    except IndependentTestLeakageError:
        raise

    cpu_count = os.cpu_count() or 4
    workers = args.workers or int(os.environ.get("ROADS_WORKERS", str(min(4, cpu_count))))
    workers = max(1, min(int(workers), max(len(cases), 1), cpu_count))
    branch, commit = _git_info()
    provenance_base = {
        "empymod_version": _empymod_version(),
        "transform_label": "smoke_fast_lagged_dlf",
        "ft_pts_per_dec": -1,
        "git_branch": branch,
        "git_commit": commit,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "roads_workers": workers,
        "thread_env": _thread_env(),
        "three_d_run": False,
        "independent_test_read": False,
    }
    _log(
        f"[flow3] {len(cases)} cases split={args.split} K={k_values} "
        f"workers={workers} variant={official_variant} "
        f"ids={[case.case_id for case in cases]}"
    )

    if args.fits_only:
        for case in cases:
            candidates = tuple(spec for spec in instantiate_candidates(case) if spec.K in k_values)
            fits = fit_case_candidates(case, candidates)
            write_split_case_result(
                flow3 / f"case_{case.case_id}.json",
                {
                    "case_id": case.case_id,
                    "official_variant": official_variant,
                    "fits": fits,
                    "choices": [],
                    "tasks": [],
                    "point_only": True,
                    "status": "FITS_ONLY",
                    "k_values": list(k_values),
                    "waveform_ids": list(TRAIN_VAL_WAVEFORMS),
                    "disk_shortlist": {},
                },
                case=case,
                split=case.split,
                provenance={**provenance_base, "stage": "fits_only", "wall_seconds": 0.0},
            )
            _log(f"[flow3] fits-only {case.case_id} n_fits={len(fits)}")
        return 0

    def _run_stage(*, include_disks: bool, label: str) -> list[dict]:
        pending = []
        for case in cases:
            destination = flow3 / f"case_{case.case_id}.json"
            if args.skip_complete and _is_complete(destination):
                _log(f"[flow3] skip complete {case.case_id}")
                continue
            pending.append(case)
        _log(f"[flow3] stage={label} include_disks={include_disks} n={len(pending)}")
        results = []
        if not pending:
            return results
        if workers == 1 or len(pending) == 1:
            for case in pending:
                _log(f"[flow3] {label} {case.case_id}")
                results.append(
                    evaluate_and_write_case(
                        case,
                        waveform_ids=TRAIN_VAL_WAVEFORMS,
                        cache_dir=str(cache_dir),
                        k_values=k_values,
                        include_disks=include_disks,
                        official_variant=official_variant,
                        output_dir=str(flow3),
                        provenance_base=provenance_base,
                    )
                )
            return results
        payloads = [
            {
                "case": case,
                "waveform_ids": TRAIN_VAL_WAVEFORMS,
                "cache_dir": str(cache_dir),
                "k_values": k_values,
                "include_disks": include_disks,
                "official_variant": official_variant,
                "output_dir": str(flow3),
                "provenance_base": provenance_base,
            }
            for case in pending
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
            futures = {pool.submit(_worker, payload): payload["case"].case_id for payload in payloads}
            for future in as_completed(futures):
                case_id = futures[future]
                result = future.result()
                _log(f"[flow3] {label} finished {case_id} n_tasks={result['n_tasks']}")
                results.append(result)
        results.sort(key=lambda item: item["case_id"])
        return results

    try:
        _run_stage(include_disks=False, label="points")
        assemble_train_val_metrics(flow3, generated_dir=generated)
        if not args.points_only:
            _run_stage(include_disks=True, label="disks")
            assemble_train_val_metrics(flow3, generated_dir=generated)
    except BlockedBySoftwareOrResourcesError as exc:
        write_json(
            generated / "FLOW3_STATUS.json",
            {
                "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
                "l1_frozen": False,
                "selected": None,
                "reason": str(exc),
                "three_d_run": False,
            },
        )
        _log("BLOCKED_BY_SOFTWARE_OR_RESOURCES")
        print("BLOCKED_BY_SOFTWARE_OR_RESOURCES", flush=True)
        return 2

    _log("[flow3] train/val responses written; selector not frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
