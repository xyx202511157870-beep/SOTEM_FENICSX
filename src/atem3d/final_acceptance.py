"""Final no-IP/IP acceptance report helpers."""

from __future__ import annotations

from pathlib import Path
import json

REQUIRED_CASES = ("noip", "ip")
REQUIRED_CASE_ARTIFACTS = (
    "predictions.csv",
    "reference_empymod_or_1d.csv",
    "errors.csv",
    "error_summary.json",
    "comparison_3comp.png",
    "error_curves_3comp.png",
    "diagnostics.json",
    "model_schematic.png",
    "run_config_resolved.yaml",
)
REQUIRED_POLARIZATION_EFFECT_ARTIFACTS = (
    "polarization_effect_predictions.csv",
    "polarization_effect_reference.csv",
    "polarization_effect_errors.csv",
    "polarization_effect_summary.json",
    "polarization_effect_comparison.png",
    "polarization_effect_error_curves.png",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def summarize_final_acceptance(
    case_summaries: dict[str, dict],
    case_diagnostics: dict[str, dict] | None = None,
    case_artifacts: dict[str, dict] | None = None,
    polarization_effect_status: dict | None = None,
) -> dict:
    """Summarize task-book final acceptance across no-IP and IP cases."""

    cases = {str(key): dict(value) for key, value in dict(case_summaries).items()}
    diagnostics_by_case = {
        str(key): dict(value) for key, value in dict(case_diagnostics or {}).items()
    }
    artifacts_by_case = {
        str(key): dict(value) for key, value in dict(case_artifacts or {}).items()
    }
    missing_cases = [case for case in REQUIRED_CASES if case not in cases]
    passed_cases: list[str] = []
    failed_cases: list[str] = []
    blocking_reasons_by_case: dict[str, list[str]] = {}
    case_status: dict[str, dict] = {}
    failure_diagnostics_by_case: dict[str, dict] = {}
    effect_status = (
        {"enabled": False, "complete": True, "missing": [], "required": []}
        if polarization_effect_status is None
        else dict(polarization_effect_status)
    )
    global_blocking_reasons = []
    if not bool(effect_status.get("complete", True)):
        global_blocking_reasons.append(
            _artifact_blocking_reason(
                effect_status,
                missing_reason="polarization_effect_artifacts_missing",
                invalid_reason="polarization_effect_artifacts_invalid",
            )
        )

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
            artifact_status = artifacts_by_case.get(case, {"complete": True, "missing": []})
            if passed and not bool(artifact_status.get("complete", False)):
                passed = False
                artifact_reason = _artifact_blocking_reason(
                    artifact_status,
                    missing_reason="validation_artifacts_missing",
                    invalid_reason="validation_artifacts_invalid",
                )
                if artifact_reason not in reasons:
                    reasons.append(artifact_reason)
        if passed:
            passed_cases.append(case)
        else:
            failed_cases.append(case)
        failure_diagnostics = _failure_diagnostics(diagnostics_by_case.get(case, {}))
        artifact_status = artifacts_by_case.get(case, {"complete": True, "missing": []})
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
            "artifact_status": artifact_status,
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

    final_passed = bool(not missing_cases and not failed_cases and not global_blocking_reasons)
    return {
        "final_acceptance_passed": final_passed,
        "required_cases": list(REQUIRED_CASES),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "missing_cases": missing_cases,
        "global_blocking_reasons": global_blocking_reasons,
        "polarization_effect_status": effect_status,
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
    polarization_effect_dir: str | Path | None = None,
    output_dir: str | Path,
) -> dict:
    """Read no-IP/IP validation summaries and write final acceptance artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_paths = {
        "noip": Path(noip_summary_json),
        "ip": Path(ip_summary_json),
    }
    diagnostics_paths = {
        "noip": Path(noip_diagnostics_json) if noip_diagnostics_json is not None else None,
        "ip": Path(ip_diagnostics_json) if ip_diagnostics_json is not None else None,
    }
    summaries = {case: _read_summary_json(path) for case, path in summary_paths.items()}
    diagnostics = {
        case: _read_optional_json(path)
        for case, path in diagnostics_paths.items()
    }
    artifacts = {
        case: _case_artifact_status(summary_paths[case], diagnostics_paths[case])
        for case in REQUIRED_CASES
    }
    effect_status = (
        None
        if polarization_effect_dir is None
        else _polarization_effect_artifact_status(Path(polarization_effect_dir))
    )
    summary = summarize_final_acceptance(
        summaries,
        case_diagnostics=diagnostics,
        case_artifacts=artifacts,
        polarization_effect_status=effect_status,
    )
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


def _case_artifact_status(
    summary_json: Path,
    diagnostics_json: Path | None,
) -> dict:
    directory = Path(summary_json).parent
    missing = []
    invalid = []
    for name in REQUIRED_CASE_ARTIFACTS:
        path = Path(diagnostics_json) if name == "diagnostics.json" and diagnostics_json is not None else directory / name
        if not path.exists():
            missing.append(name)
        elif _requires_png_signature(name) and not _has_png_signature(path):
            invalid.append(name)
    return {
        "complete": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "required": list(REQUIRED_CASE_ARTIFACTS),
        "directory": str(directory),
    }


def _polarization_effect_artifact_status(directory: Path) -> dict:
    directory = Path(directory)
    missing = []
    invalid = []
    for name in REQUIRED_POLARIZATION_EFFECT_ARTIFACTS:
        path = directory / name
        if not path.exists():
            missing.append(name)
        elif _requires_png_signature(name) and not _has_png_signature(path):
            invalid.append(name)
    return {
        "enabled": True,
        "complete": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "required": list(REQUIRED_POLARIZATION_EFFECT_ARTIFACTS),
        "directory": str(directory),
    }


def _artifact_blocking_reason(
    status: dict,
    *,
    missing_reason: str,
    invalid_reason: str,
) -> str:
    if status.get("missing"):
        return missing_reason
    if status.get("invalid"):
        return invalid_reason
    return missing_reason


def _requires_png_signature(name: str) -> bool:
    return str(name).lower().endswith(".png")


def _has_png_signature(path: Path) -> bool:
    with Path(path).open("rb") as handle:
        return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


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
        "global_blocking_reasons=" + ",".join(summary.get("global_blocking_reasons", [])),
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
