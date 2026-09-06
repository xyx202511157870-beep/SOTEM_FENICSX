#!/usr/bin/env python3
"""Flow 2: cheapest oracle-gap falsification on the 8 pilot cases."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.guards import assert_split_readable
from atem3d.adaptive_debye_mvp.io import write_json
from atem3d.adaptive_debye_mvp.layered_forward import BlockedBySoftwareOrResourcesError
from atem3d.adaptive_debye_mvp.oracle_gap import (
    evaluate_l0,
    evaluate_pilot_case,
    load_official_case_results,
    write_case_result,
    write_oracle_gap_artifacts,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, PILOT_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


def _decision_markdown(l0: dict, n_cases: int) -> str:
    status = (
        "STOP_LAYERED_NO_ACTIONABLE_GAP"
        if not l0["passed"]
        else "L0_PASS_SELECTOR_NOT_STARTED"
    )
    if l0["passed"]:
        status_line = "L0 passed. Selector / independent test were not started in this turn."
    else:
        status_line = "STOP_LAYERED_NO_ACTIONABLE_GAP. Official 8-case L0 A and B both failed."
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
            "NO. L2 has not passed. 3-D was not run.",
            "",
            "## 3. Numbers",
            "",
            f"- L0 pass/fail: `{l0['status']}` (A=`{l0['passed_A']}`, B=`{l0['passed_B']}`)",
            f"- best same-K OR/B2 median ratio: `{l0['best_same_k_median_ratio']}` at K=`{l0['best_same_k']}`",
            f"- bootstrap 95% CI: `[{l0['bootstrap_ci_low']}, {l0['bootstrap_ci_high']}]`",
            f"- win rate: `{l0['win_rate']}`",
            f"- qualifying-K oracle difference: median `{l0['median_k_qual_diff']}`, nonnegative rate `{l0['nonnegative_k_qual_rate']}`",
            f"- n official pilot cases with disks: `{n_cases}`",
            "",
            "## 4. Git / tests",
            "",
            "Official L0 calls `compute_layered_response`. PR 10 remains draft.",
            "",
            "## 5. If stopped",
            "",
            f"`{status}`" if not l0["passed"] else "Not stopped at L0. 3-D still unauthorized.",
            "",
            "Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.",
            "",
        ]
    )


def _markdown_report(l0: dict, n_cases: int) -> str:
    lines = [
        "# LAYERED_ORACLE_GAP_REPORT",
        "",
        "Official 8-case L0 with point and disk-average receivers.",
        "The earlier 4-case point-only preview is not official L0.",
        "",
        f"- L0 status: `{l0['status']}`",
        f"- passed A: `{l0['passed_A']}`",
        f"- passed B: `{l0['passed_B']}`",
        f"- n official pilot cases with disks: `{n_cases}`",
        f"- best same-K OR/B2 median ratio: `{l0['best_same_k_median_ratio']}` at K=`{l0['best_same_k']}`",
        f"- bootstrap 95% CI: `[{l0['bootstrap_ci_low']}, {l0['bootstrap_ci_high']}]`",
        f"- win rate: `{l0['win_rate']}`",
        f"- median K_qual_B2 - K_qual_OR: `{l0['median_k_qual_diff']}`",
        f"- nonnegative K_qual rate: `{l0['nonnegative_k_qual_rate']}`",
        f"- group_ok: `{l0.get('group_ok')}`",
        f"- focus_k: `{l0.get('focus_k')}`",
        "",
        "## Same-K A rows",
        "",
    ]
    for key in sorted(l0.get("same_k", {}), key=int):
        row = l0["same_k"][key]
        lines.append(
            f"- K={key}: median_ratio={row['median_ratio']}, "
            f"CI=[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}], "
            f"win_rate={row['win_rate']}, n={row['n_cases']}"
        )
    lines.extend(["", "3-D was not run.", ""])
    return "\n".join(lines)


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-ids", default="", help="comma-separated case ids, e.g. PG05,PG06,PG07,PG08")
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
        results = load_official_case_results(flow2)
        if wanted:
            results = [item for item in results if item["case_id"] in set(wanted)]
        if len(results) != 8:
            _log(f"[flow2] assemble-l0 found {len(results)} cases, need 8")
            return 3
        if any(item.get("point_only") for item in results):
            _log("[flow2] official L0 refused: at least one case is point-only")
            return 4
        l0 = evaluate_l0(results)
        write_oracle_gap_artifacts(flow2, results, l0)
        report = _markdown_report(l0, len(results))
        decision = _decision_markdown(l0, len(results))
        (flow2 / "LAYERED_ORACLE_GAP_REPORT.md").write_text(report, encoding="utf-8")
        (flow2 / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
        (REPO_ROOT / "LAYERED_ORACLE_GAP_REPORT.md").write_text(report, encoding="utf-8")
        (REPO_ROOT / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
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
        missing = [case_id for case_id in wanted if all(case.case_id != case_id for case in cases)]
        if missing:
            raise SystemExit(f"unknown case ids: {missing}")
        cases = [case for case in cases if case.case_id in set(wanted)]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    workers = max(1, min(int(os.environ.get("ROADS_WORKERS", "4")), len(cases)))
    _log(f"[flow2] {len(cases)} cases, K={k_values}, workers={workers}, case_ids={[c.case_id for c in cases]}")

    def _run_cases(*, include_disks: bool, label: str) -> list:
        _log(f"[flow2] stage={label} include_disks={include_disks}")
        stage_results = []
        if workers == 1:
            for case in cases:
                _log(f"[flow2] {label} {case.case_id}")
                result = evaluate_pilot_case(
                    case,
                    waveform_ids=PILOT_WAVEFORMS,
                    cache_dir=cache_dir,
                    k_values=k_values,
                    include_disks=include_disks,
                )
                write_case_result(flow2 / f"{result['case_id']}.json", result)
                stage_results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
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
                    write_case_result(flow2 / f"{result['case_id']}.json", result)
                    _log(f"[flow2] {label} finished {case_id}")
                    stage_results.append(result)
            stage_results.sort(key=lambda item: item["case_id"])
        return stage_results

    try:
        point_results = _run_cases(include_disks=False, label="points")
        if not args.skip_l0 and len(point_results) >= 2:
            point_l0 = evaluate_l0(point_results)
            write_json(
                flow2 / "L0_point_only_preview.json",
                {"note": "point receivers only; not the official L0", **point_l0},
            )
            _log(
                f"[flow2] point-only preview {point_l0['status']} "
                f"median_ratio={point_l0['best_same_k_median_ratio']}"
            )
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
    report = _markdown_report(l0, len(results))
    decision = _decision_markdown(l0, len(results))
    (flow2 / "LAYERED_ORACLE_GAP_REPORT.md").write_text(report, encoding="utf-8")
    (flow2 / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
    (REPO_ROOT / "LAYERED_ORACLE_GAP_REPORT.md").write_text(report, encoding="utf-8")
    (REPO_ROOT / "LAYERED_DECISION.md").write_text(decision, encoding="utf-8")
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
