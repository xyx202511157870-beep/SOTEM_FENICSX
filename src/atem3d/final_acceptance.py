"""Final no-IP/IP acceptance report helpers."""

from __future__ import annotations

from pathlib import Path
import json

REQUIRED_CASES = ("noip", "ip")


def summarize_final_acceptance(
    case_summaries: dict[str, dict],
    case_diagnostics: dict[str, dict] | None = None,
) -> dict:
    """Summarize task-book final acceptance across no-IP and IP cases."""

    cases = {str(key): dict(value) for key, value in dict(case_summaries).items()}
    diagnostics_by_case = {
        str(key): dict(value) for key, value in dict(case_diagnostics or {}).items()
    }
    missing_cases = [case for case in REQUIRED_CASES if case not in cases]
    passed_cases: list[str] = []
    failed_cases: list[str] = []
    blocking_reasons_by_case: dict[str, list[str]] = {}
    case_status: dict[str, dict] = {}
    failure_diagnostics_by_case: dict[str, dict] = {}

    for case in REQUIRED_CASES:
        if case in missing_cases:
            continue
        summary = cases[case]
        actual_case_type = str(summary.get("case_type", ""))
        if actual_case_type != case:
            passed = False
            reasons = [f"case_type_mismatch:{actual_case_type or 'missing'}"]
        else:
            summary_passed = bool(summary.get("final_acceptance_passed", False))
            reasons = _blocking_reasons(summary)
            internal_grid_verified = _case_internal_time_grid_verified(
                summary,
                diagnostics_by_case.get(case, {}),
            )
            if summary_passed and not internal_grid_verified:
                passed = False
                if "internal_time_grid_not_verified" not in reasons:
                    reasons.append("internal_time_grid_not_verified")
            else:
                passed = summary_passed
        if passed:
            passed_cases.append(case)
        else:
            failed_cases.append(case)
        failure_diagnostics = _failure_diagnostics(diagnostics_by_case.get(case, {}))
        failure_diagnostics_by_case[case] = failure_diagnostics
        blocking_reasons_by_case[case] = reasons
        case_status[case] = {
            "case_type": actual_case_type,
            "reference_type": str(summary.get("reference_type", "")),
            "magnetic_quantity": str(summary.get("magnetic_quantity", "")),
            "final_acceptance_passed": passed,
            "internal_time_grid_verified": _case_internal_time_grid_verified(
                summary,
                diagnostics_by_case.get(case, {}),
            ),
            "blocking_reasons": reasons,
            "failure_diagnostics": failure_diagnostics,
        }

    for case in missing_cases:
        failure_diagnostics_by_case[case] = {}
        blocking_reasons_by_case[case] = ["missing_error_summary"]
        case_status[case] = {
            "case_type": "",
            "reference_type": "",
            "magnetic_quantity": "",
            "final_acceptance_passed": False,
            "blocking_reasons": ["missing_error_summary"],
            "failure_diagnostics": {},
        }

    final_passed = bool(not missing_cases and not failed_cases)
    return {
        "final_acceptance_passed": final_passed,
        "required_cases": list(REQUIRED_CASES),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "missing_cases": missing_cases,
        "blocking_reasons_by_case": blocking_reasons_by_case,
        "failure_diagnostics_by_case": failure_diagnostics_by_case,
        "cases": case_status,
    }


def write_final_acceptance_report(
    *,
    noip_summary_json: str | Path,
    ip_summary_json: str | Path,
    noip_diagnostics_json: str | Path | None = None,
    ip_diagnostics_json: str | Path | None = None,
    output_dir: str | Path,
) -> dict:
    """Read no-IP/IP validation summaries and write final acceptance artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        "noip": _read_summary_json(noip_summary_json),
        "ip": _read_summary_json(ip_summary_json),
    }
    diagnostics = {
        "noip": _read_optional_json(noip_diagnostics_json),
        "ip": _read_optional_json(ip_diagnostics_json),
    }
    summary = summarize_final_acceptance(summaries, case_diagnostics=diagnostics)
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


def _read_optional_json(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _blocking_reasons(summary: dict) -> list[str]:
    status = summary.get("acceptance_status", {})
    if isinstance(status, dict):
        reasons = status.get("blocking_reasons", [])
        if isinstance(reasons, list):
            return [str(reason) for reason in reasons]
    if bool(summary.get("final_acceptance_passed", False)):
        return []
    return ["final_acceptance_passed_false"]


def _case_internal_time_grid_verified(summary: dict, diagnostics: dict) -> bool:
    for source in (summary, diagnostics):
        status = source.get("acceptance_status", {})
        if isinstance(status, dict) and bool(status.get("internal_time_grid_verified", False)):
            return True
    return _diagnostic_internal_time_grid_verified(diagnostics)


def _diagnostic_internal_time_grid_verified(diagnostics: dict) -> bool:
    candidates = []
    direct = diagnostics.get("primary_secondary_internal_time_grid")
    if isinstance(direct, dict):
        candidates.append(direct)
    equation = diagnostics.get("primary_secondary_step_equation")
    if isinstance(equation, dict) and isinstance(equation.get("internal_time_grid"), dict):
        candidates.append(equation["internal_time_grid"])
    for candidate in candidates:
        if (
            bool(candidate.get("contains_turnoff_start", False))
            and bool(candidate.get("contains_turnoff_end", False))
            and bool(candidate.get("contains_all_observation_outputs", False))
            and float(candidate.get("last_output_internal_time_s", 0.0)) >= 1.0
        ):
            return True
    return False


def _failure_diagnostics(diagnostics: dict) -> dict:
    failure = diagnostics.get("validation_failure", {})
    if not isinstance(failure, dict):
        return {}
    return {
        "failed": bool(failure.get("failed", False)),
        "reason_codes": [str(reason) for reason in failure.get("reason_codes", [])],
        "checks": dict(failure.get("checks", {})),
    }


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
    lines.append("diagnostic_reason_codes:")
    diagnostics_by_case = dict(summary.get("failure_diagnostics_by_case", {}))
    for case in summary["required_cases"]:
        diagnostics = dict(diagnostics_by_case.get(case, {}))
        reason_codes = list(diagnostics.get("reason_codes", []))
        if reason_codes:
            lines.append(f"{case}: " + ",".join(reason_codes))
        else:
            lines.append(f"{case}: none")
    lines.append("")
    lines.append("diagnostic_checks:")
    for case in summary["required_cases"]:
        diagnostics = dict(diagnostics_by_case.get(case, {}))
        checks = dict(diagnostics.get("checks", {}))
        check_status = [
            f"{name}={dict(check).get('status', '')}"
            for name, check in sorted(checks.items())
        ]
        if check_status:
            lines.append(f"{case}: " + ",".join(check_status))
        else:
            lines.append(f"{case}: none")
    lines.append("")
    return "\n".join(lines)
