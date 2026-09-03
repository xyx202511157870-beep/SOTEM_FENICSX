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
from atem3d.adaptive_debye_mvp.oracle_gap import evaluate_l0, evaluate_pilot_case, write_oracle_gap_artifacts
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--k", default=",".join(str(value) for value in K_PILOT))
    args = parser.parse_args()
    k_values = tuple(int(item) for item in args.k.split(",") if item.strip())

    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    cache_dir = generated / "cache"
    flow2 = generated / "flow2_oracle_gap"
    flow2.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        print(message, flush=True)
        with (flow2 / "flow2.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    cases = list(cases_for_split("pilot_gap", generate_all_cases()))
    for case in cases:
        assert_split_readable(case.split, stage="flow2")
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    workers = max(1, min(int(os.environ.get("ROADS_WORKERS", "4")), len(cases)))
    _log(f"[flow2] {len(cases)} cases, K={k_values}, workers={workers}")

    def _run_cases(*, include_disks: bool, label: str) -> list:
        _log(f"[flow2] stage={label} include_disks={include_disks}")
        stage_results = []
        if workers == 1:
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
                    _log(f"[flow2] {label} finished {case_id}")
                    stage_results.append(result)
            stage_results.sort(key=lambda item: item["case_id"])
        return stage_results

    try:
        point_results = _run_cases(include_disks=False, label="points")
        point_l0 = evaluate_l0(point_results)
        write_json(flow2 / "L0_point_only_preview.json", {"note": "point receivers only; not the official L0", **point_l0})
        _log(f"[flow2] point-only preview {point_l0['status']} median_ratio={point_l0['best_same_k_median_ratio']}")
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
