#!/usr/bin/env python3
"""Precompute independent_test layered responses for Flow 4 acceleration.

Does not train or select templates. Does not evaluate the L2 gate. Does not
start 3-D. Writes case_TE*.json for PR 10 to consume after L1 freeze.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from atem3d.adaptive_debye_mvp.guards import ThreeDNotAuthorizedError  # noqa: E402
from atem3d.adaptive_debye_mvp.independent_test import (  # noqa: E402
    OFFICIAL_DIR_NAME,
    SCHEMA,
    STAGE,
    build_provenance,
    case_is_complete,
    empymod_version_text,
    evaluate_independent_case,
    extend_disk_tasks,
    frozen_official_variant,
    hydrate_independent_choices,
    hydrate_independent_tasks,
    independent_test_cases,
    load_independent_case_result,
    refuse_3d_always,
    registry_hashes,
    survey_hashes,
    tilted_projection_record,
    write_independent_case_result,
)
from atem3d.adaptive_debye_mvp.io import read_json, sha256_hex, write_json  # noqa: E402
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError  # noqa: E402
from atem3d.adaptive_debye_mvp.protocol_constants import (  # noqa: E402
    K_PILOT,
    TEST_RECEIVERS,
    TEST_WAVEFORMS,
)
from atem3d.adaptive_debye_mvp.registry import LayeredCase  # noqa: E402


FORBIDDEN_OUTPUT_NAMES = (
    "selected_template",
    "FLOW4_STATUS",
    "LAYERED_GATE_PASSED",
)


def _parse_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_waveforms(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in values if item not in TEST_WAVEFORMS]
    if unknown:
        raise SystemExit(f"waveforms not in the independent_test set: {unknown}")
    return values


def _output_is_official(path: Path, generated: Path) -> bool:
    try:
        return path.resolve() == (generated / OFFICIAL_DIR_NAME).resolve()
    except OSError:
        return path.name == OFFICIAL_DIR_NAME


def _assert_safe_output_name(path: Path) -> None:
    name = path.name
    for forbidden in FORBIDDEN_OUTPUT_NAMES:
        if name.startswith(forbidden):
            raise SystemExit(f"refusing to write forbidden artifact {path}")


def _log(output_dir: Path, message: str) -> None:
    print(message, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "precompute.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _write_manifest(output_dir: Path, cases: list[LayeredCase], complete_ids: list[str]) -> None:
    rows = []
    for case in cases:
        path = output_dir / f"case_{case.case_id}.json"
        row = {
            "case_id": case.case_id,
            "case_hash": case.case_hash(),
            "complete": case.case_id in complete_ids,
        }
        if path.is_file():
            row["file_sha256"] = sha256_hex(path.read_text(encoding="utf-8"))
        rows.append(row)
    write_json(
        output_dir / "PRECOMPUTE_MANIFEST.json",
        {
            "schema": SCHEMA,
            "case_ids": [case.case_id for case in cases],
            "cases": rows,
            "n_cases_complete": len(complete_ids),
            "l2_evaluated": False,
            "three_d_run": False,
            "selector_read": False,
            "selected_template_written": False,
            "stage": STAGE,
        },
    )


def _run_one_case(
    case: LayeredCase,
    *,
    cache_dir: str,
    output_dir: str,
    k_values: tuple[int, ...],
    waveform_ids: tuple[str, ...],
    official_variant: str,
    official_variant_source: str,
    disk_shortlist: int,
    roads_workers: int,
    registry: dict[str, str],
) -> dict[str, str]:
    destination = Path(output_dir) / f"case_{case.case_id}.json"
    _assert_safe_output_name(destination)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    result = evaluate_independent_case(
        case,
        cache_dir=cache_dir,
        k_values=k_values,
        waveform_ids=waveform_ids,
        official_variant=official_variant,
        disk_shortlist=disk_shortlist,
        include_disks=True,
    )
    hashes = survey_hashes(case.case_id, cache_dir, waveform_ids[0])
    provenance = build_provenance(
        started_at=started,
        wall_seconds=time.perf_counter() - t0,
        roads_workers=roads_workers,
        point_hash=hashes["point"],
        disk_hash=hashes["disk"],
        official_variant_source=official_variant_source,
        empymod_version=empymod_version_text(),
    )
    write_independent_case_result(
        destination,
        result,
        case=case,
        k_values=k_values,
        waveform_ids=waveform_ids,
        disk_shortlist=result["disk_shortlist"],
        provenance=provenance,
        tilted_projection=tilted_projection_record(case),
        registry=registry,
    )
    return {"case_id": case.case_id, "path": str(destination)}


def _extend_one_case(
    case: LayeredCase,
    *,
    cache_dir: str,
    output_dir: str,
    extra_ids: tuple[str, ...],
    waveform_ids: tuple[str, ...],
    k_values: tuple[int, ...],
    registry: dict[str, str],
) -> dict[str, str]:
    path = Path(output_dir) / f"case_{case.case_id}.json"
    loaded = load_independent_case_result(path)
    loaded["tasks"] = hydrate_independent_tasks(loaded["tasks"])
    loaded["choices"] = hydrate_independent_choices(loaded["choices"])
    extra = extend_disk_tasks(
        case,
        loaded,
        candidate_ids=extra_ids,
        waveform_ids=tuple(str(item) for item in loaded.get("waveform_ids", waveform_ids)),
        cache_dir=cache_dir,
    )
    loaded["tasks"] = list(loaded["tasks"]) + extra
    write_independent_case_result(
        path,
        loaded,
        case=case,
        k_values=tuple(int(item) for item in loaded.get("k_values", k_values)),
        waveform_ids=tuple(str(item) for item in loaded.get("waveform_ids", waveform_ids)),
        disk_shortlist=loaded.get("disk_shortlist", {}),
        provenance=dict(loaded.get("provenance", {})),
        tilted_projection=loaded.get("tilted_projection"),
        registry=registry or loaded.get("registry_hashes", {}),
    )
    return {"case_id": case.case_id, "added_disk_tasks": str(len(extra))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", default="", help="comma-separated TE ids; default all 10")
    parser.add_argument("--k", default=",".join(str(value) for value in K_PILOT))
    parser.add_argument("--waveforms", default=",".join(TEST_WAVEFORMS))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp" / OFFICIAL_DIR_NAME),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp" / "cache"),
    )
    parser.add_argument("--disk-shortlist", type=int, default=2)
    parser.add_argument("--extend-disks", default="", help="JSON mapping case_id -> extra candidate ids")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="refuse unless --output-dir is outside the official flow4_independent_test dir",
    )
    args = parser.parse_args()

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    if args.smoke and _output_is_official(output_dir, generated):
        raise SystemExit("--smoke refuses to write into the official flow4_independent_test directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        refuse_3d_always()
    except ThreeDNotAuthorizedError:
        pass

    k_values = _parse_ints(args.k)
    waveform_ids = _parse_waveforms(args.waveforms)
    wanted = _parse_ids(args.case_ids)
    all_cases = list(independent_test_cases())
    by_id = {case.case_id: case for case in all_cases}
    if wanted:
        missing = [case_id for case_id in wanted if case_id not in by_id]
        if missing:
            raise SystemExit(f"unknown or non-independent_test case ids: {missing}")
        cases = [by_id[case_id] for case_id in wanted]
    else:
        cases = all_cases

    official_variant_path = generated / "flow2_oracle_gap" / "official_spectral_variant.json"
    official_variant = frozen_official_variant(official_variant_path.parent)
    registry = registry_hashes(generated)
    workers = max(1, min(int(os.environ.get("ROADS_WORKERS", "4")), len(cases)))

    if args.extend_disks:
        mapping = read_json(args.extend_disks)
        if not isinstance(mapping, dict):
            raise SystemExit("--extend-disks JSON must map case_id to candidate id lists")
        for case_id, extra_ids in mapping.items():
            if case_id not in by_id:
                raise SystemExit(f"--extend-disks unknown case {case_id}")
            extra = tuple(str(item) for item in extra_ids)
            _log(output_dir, f"[flow4-precompute] extend-disks {case_id} {extra}")
            _extend_one_case(
                by_id[str(case_id)],
                cache_dir=str(cache_dir),
                output_dir=str(output_dir),
                extra_ids=extra,
                waveform_ids=waveform_ids,
                k_values=k_values,
                registry=registry,
            )
        complete = [case.case_id for case in all_cases if case_is_complete(output_dir / f"case_{case.case_id}.json")]
        _write_manifest(output_dir, all_cases, complete)
        return 0

    pending = []
    for case in cases:
        path = output_dir / f"case_{case.case_id}.json"
        if not args.force and case_is_complete(path, waveform_ids=waveform_ids):
            _log(output_dir, f"[flow4-precompute] skip complete {case.case_id}")
            continue
        pending.append(case)

    _log(
        output_dir,
        f"[flow4-precompute] {len(pending)} pending / {len(cases)} selected, "
        f"K={k_values}, waveforms={waveform_ids}, workers={workers}, variant={official_variant}",
    )

    try:
        if workers == 1 or len(pending) <= 1:
            for case in pending:
                _log(output_dir, f"[flow4-precompute] start {case.case_id}")
                _run_one_case(
                    case,
                    cache_dir=str(cache_dir),
                    output_dir=str(output_dir),
                    k_values=k_values,
                    waveform_ids=waveform_ids,
                    official_variant=official_variant,
                    official_variant_source=str(official_variant_path.relative_to(REPO_ROOT)),
                    disk_shortlist=int(args.disk_shortlist),
                    roads_workers=workers,
                    registry=registry,
                )
                _log(output_dir, f"[flow4-precompute] finished {case.case_id}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _run_one_case,
                        case,
                        cache_dir=str(cache_dir),
                        output_dir=str(output_dir),
                        k_values=k_values,
                        waveform_ids=waveform_ids,
                        official_variant=official_variant,
                        official_variant_source=str(official_variant_path.relative_to(REPO_ROOT)),
                        disk_shortlist=int(args.disk_shortlist),
                        roads_workers=workers,
                        registry=registry,
                    ): case.case_id
                    for case in pending
                }
                for future in as_completed(futures):
                    case_id = futures[future]
                    future.result()
                    _log(output_dir, f"[flow4-precompute] finished {case_id}")
    except BlockedBySoftwareOrResourcesError as exc:
        write_json(
            output_dir / "BLOCKED.json",
            {
                "status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
                "reason": str(exc),
                "three_d_run": False,
                "l2_evaluated": False,
            },
        )
        _log(output_dir, "BLOCKED_BY_SOFTWARE_OR_RESOURCES")
        return 2

    complete = [case.case_id for case in all_cases if case_is_complete(output_dir / f"case_{case.case_id}.json")]
    _write_manifest(output_dir, all_cases, complete)
    _log(output_dir, f"[flow4-precompute] complete={complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
