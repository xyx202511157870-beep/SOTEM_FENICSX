#!/usr/bin/env python3
"""Flow 3: deployable selector. Refuses to run unless L0 passed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.guards import (
    IndependentTestLeakageError,
    assert_case_ids_split_safe,
    assert_records_split_safe,
    assert_split_readable,
    snapshot_cache_keys,
)
from atem3d.adaptive_debye_mvp.io import read_json, sha256_file, write_json
from atem3d.adaptive_debye_mvp.oracle_gap import choose_official_spectral_variant, load_pilot_case_result
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT
from atem3d.adaptive_debye_mvp.selector import records_from_case_results, select_templates


def _load_split_cases(flow3: Path, prefix: str) -> list[dict]:
    paths = sorted(flow3.glob(f"case_{prefix}*.json"))
    results = [load_pilot_case_result(path) for path in paths]
    ids = [item["case_id"] for item in results]
    assert_case_ids_split_safe(ids, stage="flow3")
    return results


def _forced_ids(selected: dict[str, str], spectral: dict[str, str], audit: dict) -> dict[str, list[str]]:
    forced: dict[str, list[str]] = {}
    for poles in selected:
        ids = set(audit.get("T_K", {}).get(poles, []))
        ids.add(selected[poles])
        ids.add(spectral[poles])
        forced[poles] = sorted(ids)
    return forced


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path): sha256_file(path) for path in paths if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rank-points", "freeze"), default="freeze")
    args = parser.parse_args()

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    l0_path = generated / "flow2_oracle_gap" / "L0_summary.json"
    if not l0_path.is_file():
        raise SystemExit("L0_summary.json is missing; Flow 3 is unauthorized")
    l0 = read_json(l0_path)
    if not bool(l0.get("passed", False)):
        write_json(
            generated / "FLOW3_STATUS.json",
            {"status": "REFUSED", "reason": "L0 did not pass", "three_d_run": False},
        )
        raise SystemExit("Flow 3 refused: L0 did not pass")

    flow3 = generated / "flow3_selector"
    flow3.mkdir(parents=True, exist_ok=True)
    if any(flow3.glob("case_TE*.json")) or any(flow3.glob("case_LP*.json")):
        raise IndependentTestLeakageError("flow3_selector contains independent-test JSON")

    train_results = _load_split_cases(flow3, "TR")
    val_results = _load_split_cases(flow3, "VA")
    if args.stage == "freeze":
        if len(train_results) != 12 or len(val_results) != 6:
            raise SystemExit(f"need 12 train + 6 val case JSON, got {len(train_results)}/{len(val_results)}")
        if any(item.get("point_only") for item in [*train_results, *val_results]):
            raise SystemExit("official L1 freeze refuses point-only case JSON")

    variant_path = flow3 / "official_spectral_variant.json"
    if variant_path.is_file():
        official_variant = str(read_json(variant_path)["variant"])
    else:
        summaries = []
        for result in train_results:
            for row in result.get("fit_summary") or []:
                summaries.append(
                    type("Fit", (), row)
                )
        # choose_official_spectral_variant expects FitRecord-like attributes
        from atem3d.adaptive_debye_mvp.oracle_gap import FitRecord

        records = [
            FitRecord(
                candidate_id=row["candidate_id"],
                K=int(row["K"]),
                valid=bool(row["valid"]),
                spectral_error_s0=float(row["spectral_error_s0"]),
                spectral_error_s1=float(row["spectral_error_s1"]),
                condition_number=float(row["condition_number"]),
                relative_dc_error=float(row.get("relative_dc_error", 0.0)),
                optimizer_success=bool(row.get("optimizer_success", True)),
                fit=None,
            )
            for result in train_results
            for row in result.get("fit_summary") or []
        ]
        official_variant = choose_official_spectral_variant(records) if records else "S1"
        write_json(variant_path, {"variant": official_variant})

    scope = "point" if args.stage == "rank-points" else "point_disk"
    train = records_from_case_results(train_results, split="train", official_variant=official_variant, scope=scope)
    validation = records_from_case_results(val_results, split="validation", official_variant=official_variant, scope=scope)
    if args.stage == "freeze":
        train_point = records_from_case_results(train_results, split="train", official_variant=official_variant, scope="point")
        val_point = records_from_case_results(val_results, split="validation", official_variant=official_variant, scope="point")
        train = train_point + train
        validation = val_point + validation
    for row in [*train, *validation]:
        assert_split_readable(str(row["split"]), stage="selector")
        if str(row["split"]) == "independent_test":
            raise IndependentTestLeakageError("train_selector.py read independent_test")
    assert_records_split_safe(train, stage="selector")
    assert_records_split_safe(validation, stage="selector")

    payload = select_templates(
        train_records=train,
        validation_records=validation,
        l0_path=l0_path,
        output_dir=flow3,
        k_values=K_PILOT,
    )
    forced = _forced_ids(payload["selected"], payload["spectral_selected"], payload["audit"])
    write_json(flow3 / "forced_disk_ids.json", forced)

    if args.stage == "rank-points":
        write_json(
            generated / "FLOW3_STATUS.json",
            {
                "status": "L1_POINTS_RANKED",
                "selected_preview": payload["selected"],
                "spectral_selected": payload["spectral_selected"],
                "forced_disk_ids": forced,
                "three_d_run": False,
            },
        )
        return 0

    hashed = _hash_files(
        [
            flow3 / "selected_template_by_K.json",
            flow3 / "spectral_selected_template_by_K.json",
            flow3 / "train_candidate_metrics.csv",
            flow3 / "validation_candidate_metrics.csv",
            flow3 / "selector_audit.json",
            flow3 / "official_spectral_variant.json",
            *sorted(flow3.glob("case_TR*.json")),
            *sorted(flow3.glob("case_VA*.json")),
        ]
    )
    write_json(flow3 / "hashes.sha256.json", hashed)
    write_json(
        generated / "FLOW3_STATUS.json",
        {
            "status": "L1_FROZEN",
            "selected": payload["selected"],
            "spectral_selected": payload["spectral_selected"],
            "official_spectral_variant": official_variant,
            "n_task_families": 1,
            "hashes": hashed,
            "n_train": 12,
            "n_validation": 6,
            "independent_test_unread": True,
            "three_d_run": False,
            "roads_workers": os.environ.get("ROADS_WORKERS"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
