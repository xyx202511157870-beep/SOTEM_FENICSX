"""Final no-IP/IP acceptance report helpers."""

from __future__ import annotations

from pathlib import Path
import json

REQUIRED_CASES = ("noip", "ip")


def summarize_final_acceptance(case_summaries: dict[str, dict]) -> dict:
    """Summarize task-book final acceptance across no-IP and IP cases."""

    cases = {str(key): dict(value) for key, value in dict(case_summaries).items()}
    missing_cases = [case for case in REQUIRED_CASES if case not in cases]
    passed_cases: list[str] = []
    failed_cases: list[str] = []
    blocking_reasons_by_case: dict[str, list[str]] = {}
    case_status: dict[str, dict] = {}

    for case in REQUIRED_CASES:
        if case in missing_cases:
            continue
        summary = cases[case]
        actual_case_type = str(summary.get("case_type", ""))
        if actual_case_type != case:
            passed = False
            reasons = [f"case_type_mismatch:{actual_case_type or 'missing'}"]
        else:
            passed = bool(summary.get("final_acceptance_passed", False))
            reasons = _blocking_reasons(summary)
        if passed:
            passed_cases.append(case)
        else:
            failed_cases.append(case)
        blocking_reasons_by_case[case] = reasons
        case_status[case] = {
            "case_type": actual_case_type,
            "reference_type": str(summary.get("reference_type", "")),
            "magnetic_quantity": str(summary.get("magnetic_quantity", "")),
            "final_acceptance_passed": passed,
            "blocking_reasons": reasons,
        }

    for case in missing_cases:
        blocking_reasons_by_case[case] = ["missing_error_summary"]
        case_status[case] = {
            "case_type": "",
            "reference_type": "",
            "magnetic_quantity": "",
            "final_acceptance_passed": False,
            "blocking_reasons": ["missing_error_summary"],
        }

    final_passed = bool(not missing_cases and not failed_cases)
    return {
        "final_acceptance_passed": final_passed,
        "required_cases": list(REQUIRED_CASES),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "missing_cases": missing_cases,
        "blocking_reasons_by_case": blocking_reasons_by_case,
        "cases": case_status,
    }


def write_final_acceptance_report(
    *,
    noip_summary_json: str | Path,
    ip_summary_json: str | Path,
    output_dir: str | Path,
) -> dict:
    """Read no-IP/IP validation summaries and write final acceptance artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        "noip": _read_summary_json(noip_summary_json),
        "ip": _read_summary_json(ip_summary_json),
    }
    summary = summarize_final_acceptance(summaries)
    (output / "final_acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "final_acceptance_report.txt").write_text(
        _format_final_acceptance_report(summary),
        encoding="utf-8",
    )
    return summary


def _read_summary_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _blocking_reasons(summary: dict) -> list[str]:
    status = summary.get("acceptance_status", {})
    if isinstance(status, dict):
        reasons = status.get("blocking_reasons", [])
        if isinstance(reasons, list):
            return [str(reason) for reason in reasons]
    if bool(summary.get("final_acceptance_passed", False)):
        return []
    return ["final_acceptance_passed_false"]


def _format_final_acceptance_report(summary: dict) -> str:
    lines = [
        f"FINAL_ACCEPTANCE_PASSED={str(bool(summary['final_acceptance_passed'])).lower()}",
        "required_cases=" + ",".join(summary["required_cases"]),
        "passed_cases=" + ",".join(summary["passed_cases"]),
        "failed_cases=" + ",".join(summary["failed_cases"]),
        "missing_cases=" + ",".join(summary["missing_cases"]),
        "",
        "blocking_reasons:",
    ]
    reasons_by_case = dict(summary["blocking_reasons_by_case"])
    for case in summary["required_cases"]:
        reasons = list(reasons_by_case.get(case, []))
        if reasons:
            lines.append(f"{case}: " + ",".join(reasons))
        else:
            lines.append(f"{case}: none")
    lines.append("")
    return "\n".join(lines)
