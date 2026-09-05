#!/usr/bin/env python3
"""Flow 4: independent layered test. Refuses unless L1 passed. Never starts 3-D."""

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

from atem3d.adaptive_debye_mvp.guards import ThreeDNotAuthorizedError, refuse_3d_before_l2
from atem3d.adaptive_debye_mvp.io import read_json, sha256_file, to_record, write_json, write_records_csv
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError
from atem3d.adaptive_debye_mvp.oracle_gap import (
    evaluate_l2,
    evaluate_pilot_case,
    load_pilot_case_result,
    write_case_result,
    write_oracle_gap_artifacts,
)
from atem3d.adaptive_debye_mvp.protocol_constants import (
    B1_LOG_UNIFORM_TEMPLATE,
    FINAL_STATUSES,
    FORBIDDEN_STATUSES,
    K_PILOT,
    TEST_WAVEFORMS,
)
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _case_json_satisfies(path: Path, stage: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not payload.get("case_id") or not payload.get("tasks"):
        return False
    provenance = payload.get("provenance") or {}
    # PR13-style precompute has disks but never read the frozen selector.
    if provenance.get("selector_read") is False or provenance.get("l2_evaluated") is False:
        return stage == "points"
    if not payload.get("point_only", True):
        return True
    return stage == "points"


def _workers(n_cases: int) -> int:
    requested = max(4, int(os.cpu_count() or 4))
    env = os.environ.get("ROADS_WORKERS")
    if env:
        requested = max(4, int(env))
    return max(1, min(requested, n_cases))


def _decision_markdown(l2: dict, n_cases: int, l0: dict | None = None) -> str:
    passed = bool(l2.get("passed"))
    status = "3D_AUTHORIZED_PENDING_PREFLIGHT" if passed else "STOP_LAYERED_SELECTOR_FAILED"
    if passed:
        status_line = (
            "L0 passed. L1 is frozen. L2 passed (`L2_PASS`). "
            "FINAL STATUS=`3D_AUTHORIZED_PENDING_PREFLIGHT`. 3-D was not started."
        )
    else:
        status_line = (
            "L0 passed. L1 is frozen. L2 failed (`L2_FAIL`). "
            "FINAL STATUS=`STOP_LAYERED_SELECTOR_FAILED`. 3-D was not started."
        )
    l0 = l0 or {}
    return "\n".join(
        [
            "# LAYERED_DECISION",
            "",
            "## 1. FINAL STATUS",
            "",
            status_line,
            "",
            "## 2. 3-D continue?",
            "",
            "YES (authorized pending preflight). 3-D was not run."
            if passed
            else "NO. L2 failed. 3-D was not run.",
            "",
            "## 3. Numbers",
            "",
            f"- L0 pass/fail: `{l0.get('status', 'L0_PASS')}` (A=`{l0.get('passed_A')}`, B=`{l0.get('passed_B')}`)",
            f"- L0 best same-K OR/B2 median ratio: `{l0.get('best_same_k_median_ratio')}` at K=`{l0.get('best_same_k')}`",
            f"- L0 bootstrap 95% CI: `[{l0.get('bootstrap_ci_low')}, {l0.get('bootstrap_ci_high')}]`",
            f"- L0 win rate: `{l0.get('win_rate')}`",
            f"- L1 frozen: `yes`",
            f"- L2 pass/fail: `{l2['status']}` (A=`{l2['passed_A']}`, B=`{l2['passed_B']}`)",
            f"- same-K P-R/B2 median ratio: `{l2['best_same_k_median_ratio']}` at K=`{l2['best_same_k']}`",
            f"- bootstrap 95% CI: `[{l2['bootstrap_ci_low']}, {l2['bootstrap_ci_high']}]`",
            f"- win rate: `{l2['win_rate']}`",
            f"- qualifying-K difference: median `{l2['median_k_qual_diff']}`, nonnegative rate `{l2['nonnegative_k_qual_rate']}`",
            f"- group outcomes: `{l2.get('group_ok')}`",
            f"- 3-D authorized: `{'yes_pending_preflight' if passed else 'no'}`",
            f"- n independent-test cases: `{n_cases}`",
            "",
            "## 4. Git / tests",
            "",
            "Official L0/L2 call `compute_layered_response`. PR 10 remains draft. No 3-D forwards.",
            "",
            "## 5. If stopped",
            "",
            f"`{status}`",
            "",
            "Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.",
            "",
        ]
    )


def _report(l2: dict, n_cases: int) -> str:
    lines = [
        "# LAYERED_INDEPENDENT_TEST_REPORT",
        "",
        "Official 10-case L2 with W0-W3, point, disk_1.0, disk_4.0, and tilted-coil projection.",
        "B0 is not defined in protocol.md and was not invented. B1 is the frozen log-uniform "
        "time-window template `K{K}_tw_span8.0_shift+0.0_dens1.00` when present in the shortlist.",
        "",
        f"- L2 status: `{l2['status']}`",
        f"- passed A: `{l2['passed_A']}`",
        f"- passed B: `{l2['passed_B']}`",
        f"- n cases: `{n_cases}`",
        f"- best same-K P-R/B2 median ratio: `{l2['best_same_k_median_ratio']}` at K=`{l2['best_same_k']}`",
        f"- bootstrap 95% CI: `[{l2['bootstrap_ci_low']}, {l2['bootstrap_ci_high']}]`",
        f"- win rate: `{l2['win_rate']}`",
        f"- median K_qual_B2 - K_qual_PR: `{l2['median_k_qual_diff']}`",
        f"- nonnegative K_qual rate: `{l2['nonnegative_k_qual_rate']}`",
        f"- group_ok: `{l2.get('group_ok')}`",
        f"- pr_by_k: `{l2.get('pr_by_k')}`",
        f"- b2_by_k: `{l2.get('b2_by_k')}`",
        f"- reselection: `{l2.get('reselection')}`",
        f"- bootstrap: 2000 case-level paired resamples, seed 202609116",
        "",
        "## L2 A/B checklist (P-R vs frozen B2; OR is upper bound only)",
        "",
    ]
    best = (l2.get("same_k") or {}).get(str(l2.get("best_same_k") or ""), {})
    group = l2.get("group_ok") or {}
    lines.extend(
        [
            f"- A1 median P-R/B2 ratio <= 0.80: `{best.get('median_ratio')}` -> "
            f"`{bool(best.get('median_ratio', 9) <= 0.80)}`",
            f"- A2 bootstrap 95% CI upper < 1.00: `{best.get('bootstrap_ci_high')}` -> "
            f"`{bool((best.get('bootstrap_ci_high') or 9) < 1.00)}`",
            f"- A3 P-R better than B2 in >= 70% of cases: `{best.get('win_rate')}` -> "
            f"`{bool((best.get('win_rate') or 0) >= 0.70)}`",
            f"- A4 improvement on at least two waveforms: `{group.get('waveforms')}`",
            f"- A5 improvement on point AND at least one disk: `{group.get('receivers')}`",
            f"- A6 H or dB/dt clearly improved, other not worse >10%: `{group.get('components')}`",
            f"- A7 IP-increment p95 not worse by >5%: `{group.get('ip')}`",
            f"- A overall: `{l2['passed_A']}`",
            f"- B1 median(K_qual_B2 - K_qual_P-R) >= 2: `{l2['median_k_qual_diff']}` -> "
            f"`{bool((l2.get('median_k_qual_diff') or -9) >= 2.0)}`",
            f"- B2 nonnegative qualifying-K difference in >= 70% of cases: "
            f"`{l2['nonnegative_k_qual_rate']}` -> "
            f"`{bool((l2.get('nonnegative_k_qual_rate') or 0) >= 0.70)}`",
            f"- B overall: `{l2['passed_B']}`",
            "",
            "## Same-K A rows",
            "",
        ]
    )
    for key in sorted(l2.get("same_k", {}), key=int):
        row = l2["same_k"][key]
        lines.append(
            f"- K={key}: median_ratio={row['median_ratio']}, "
            f"CI=[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}], "
            f"win_rate={row['win_rate']}, n={row['n_cases']}"
        )
    lines.extend(["", "3-D was not run.", ""])
    return "\n".join(lines)


def _evaluate(case, *, cache_dir, include_disks, forced_disk_ids, official_variant):
    started = time.time()
    allowed = sorted({item for ids in (forced_disk_ids or {}).values() for item in ids})
    result = evaluate_pilot_case(
        case,
        waveform_ids=TEST_WAVEFORMS,
        cache_dir=cache_dir,
        k_values=K_PILOT,
        official_variant=official_variant,
        include_disks=include_disks,
        forced_disk_ids=forced_disk_ids,
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
        "stage": "disks" if include_disks else "points",
        "reselection": False,
        "selector_read": True,
        "l2_evaluated": True,
        "allowed_candidate_ids": allowed,
    }
    return result


def _assemble(generated: Path, flow4: Path, l1: dict) -> int:
    paths = sorted(flow4.glob("case_TE*.json"))
    results = [load_pilot_case_result(path) for path in paths]
    if len(results) != 10:
        print(f"[flow4] assemble found {len(results)} cases, need 10", flush=True)
        return 3
    if any(item.get("point_only") for item in results):
        print("[flow4] official L2 refused: at least one case is point-only", flush=True)
        return 4
    pr_by_k = {str(key): str(value) for key, value in dict(l1["selected"]).items()}
    b2_by_k = {str(key): str(value) for key, value in dict(l1.get("spectral_selected") or {}).items()}
    official_variant = str(l1.get("official_spectral_variant") or "S1")
    l2 = evaluate_l2(
        results,
        pr_by_k=pr_by_k,
        b2_by_k=b2_by_k or None,
        waveform_ids=TEST_WAVEFORMS,
    )
    write_oracle_gap_artifacts(flow4, results, l2)
    write_json(flow4 / "L2_summary.json", l2)
    write_records_csv(flow4 / "independent_test_choices.csv", [to_record(c) for r in results for c in r["choices"]])
    write_json(generated / "FLOW4_STATUS.json", {"status": l2["status"], "passed": l2["passed"], "l2": l2, "reselection": False, "three_d_run": False})
    l0_path = generated / "flow2_oracle_gap" / "L0_summary.json"
    l0 = read_json(l0_path) if l0_path.is_file() else {}
    report = _report(l2, len(results))
    decision = _decision_markdown(l2, len(results), l0)
    (flow4 / "LAYERED_INDEPENDENT_TEST_REPORT.md").write_text(report, encoding="utf-8")
    (REPO_ROOT / "LAYERED_INDEPENDENT_TEST_REPORT.md").write_text(report, encoding="utf-8")
    (flow4 / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
    (REPO_ROOT / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
    if l2["passed"]:
        write_json(
            generated / "LAYERED_GATE_PASSED.json",
            {
                "status": "3D_AUTHORIZED_PENDING_PREFLIGHT",
                "l2_passed": True,
                "l2": l2,
                "official_spectral_variant": official_variant,
            },
        )
        write_json(generated / "selected_method.json", {"method": "P-R", "selected": pr_by_k})
        write_json(generated / "selected_K_policy.json", {"k_values": list(K_PILOT), "practical": [6, 8, 10, 12]})
    else:
        stop = {
            "status": "STOP_LAYERED_SELECTOR_FAILED",
            "gate": "L2",
            "reason": "P-R vs B2 A and B both failed on independent_test",
            "l2": l2,
            "three_d_run": False,
        }
        write_json(generated / "STOP_REASON.json", stop)
        write_json(generated / "STOP_LAYERED_SELECTOR_FAILED.json", stop)
        write_json(REPO_ROOT / "STOP_LAYERED_SELECTOR_FAILED.json", stop)
        (flow4 / "SELECTOR_FAILURE_ANALYSIS.md").write_text(
            "# SELECTOR_FAILURE_ANALYSIS\n\n"
            "L2 A and B both failed. P-R was frozen on train/val and not reselected.\n"
            f"Best same-K median P-R/B2 ratio={l2['best_same_k_median_ratio']} "
            f"CI=[{l2['bootstrap_ci_low']}, {l2['bootstrap_ci_high']}] "
            f"win_rate={l2['win_rate']}.\n",
            encoding="utf-8",
        )
        (REPO_ROOT / "SELECTOR_FAILURE_ANALYSIS.md").write_text(
            (flow4 / "SELECTOR_FAILURE_ANALYSIS.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    print(l2["status"], flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--points-only", action="store_true")
    parser.add_argument("--assemble-l2", action="store_true")
    parser.add_argument("--skip-l2", action="store_true")
    args = parser.parse_args()

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    l1_path = generated / "FLOW3_STATUS.json"
    if not l1_path.is_file() or str(read_json(l1_path).get("status")) != "L1_FROZEN":
        write_json(
            generated / "FLOW4_STATUS.json",
            {"status": "REFUSED", "reason": "L1 not frozen", "three_d_run": False},
        )
        raise SystemExit("Flow 4 refused: L1 not frozen")
    l1 = read_json(l1_path)
    selected_path = generated / "flow3_selector" / "selected_template_by_K.json"
    expected = dict(l1.get("hashes") or {}).get(
        str((generated / "flow3_selector" / "selected_template_by_K.json").relative_to(REPO_ROOT))
    )
    if expected and selected_path.is_file() and sha256_file(selected_path) != expected:
        write_json(
            generated / "FLOW4_STATUS.json",
            {"status": "REFUSED", "reason": "selected_template_by_K.json hash mismatch", "three_d_run": False},
        )
        raise SystemExit("Flow 4 refused: selector hash mismatch")
    try:
        refuse_3d_before_l2(generated)
    except ThreeDNotAuthorizedError:
        pass

    flow4 = generated / "flow4_independent_test"
    flow4.mkdir(parents=True, exist_ok=True)
    if args.assemble_l2:
        return _assemble(generated, flow4, l1)

    cache_dir = generated / "cache"
    cases = list(cases_for_split("independent_test", generate_all_cases()))
    wanted = _parse_case_ids(args.case_ids)
    if wanted:
        cases = [case for case in cases if case.case_id in set(wanted)]
    pr_by_k = {int(key): str(value) for key, value in dict(l1["selected"]).items()}
    b2_by_k = {int(key): str(value) for key, value in dict(l1.get("spectral_selected") or {}).items()}
    forced: dict[int, list[str]] = {}
    for poles in K_PILOT:
        ids = [pr_by_k[int(poles)]]
        if int(poles) in b2_by_k:
            ids.append(b2_by_k[int(poles)])
        ids.append(B1_LOG_UNIFORM_TEMPLATE.format(K=int(poles)))
        forced[int(poles)] = list(dict.fromkeys(ids))
    official_variant = str(l1.get("official_spectral_variant") or "S1")
    workers = _workers(len(cases))

    def _log(message: str) -> None:
        print(message, flush=True)
        with (flow4 / "flow4.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    _log(f"[flow4] n={len(cases)} workers={workers} variant={official_variant} pr={pr_by_k}")
    try:
        stages = (("points", False),)
        if not args.points_only:
            stages = (("points", False), ("disks", True))
        for label, include_disks in stages:
            pending = [
                case
                for case in cases
                if not _case_json_satisfies(flow4 / f"case_{case.case_id}.json", label)
            ]
            skipped = [case.case_id for case in cases if case not in pending]
            if skipped:
                _log(f"[flow4] {label} skip existing {skipped}")
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
                    write_case_result(flow4 / f"case_{result['case_id']}.json", result)
                    _log(f"[flow4] {label} finished {result['case_id']}")
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
                        write_case_result(flow4 / f"case_{result['case_id']}.json", result)
                        _log(f"[flow4] {label} finished {result['case_id']}")
    except BlockedBySoftwareOrResourcesError as exc:
        write_json(
            generated / "FLOW4_STATUS.json",
            {"status": "BLOCKED_BY_SOFTWARE_OR_RESOURCES", "reason": str(exc), "three_d_run": False},
        )
        return 2
    if args.skip_l2 or args.points_only or len(list(flow4.glob("case_TE*.json"))) != 10:
        _log("[flow4] case JSON written; L2 not finalized")
        return 0
    return _assemble(generated, flow4, l1)


if __name__ == "__main__":
    raise SystemExit(main())
