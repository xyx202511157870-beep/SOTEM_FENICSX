#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


FLOOR_FRACTION = 1.0e-6
RTOL = 1.0e-12
ATOL = 1.0e-12


def _load_response(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(run_dir) / "verification_data.npz"
    with np.load(path, allow_pickle=False) as payload:
        components = [str(value) for value in payload["components"].tolist()]
        component_index = components.index("dBzdt")
        times = np.asarray(payload["times"], dtype=float)[:25]
        response = np.asarray(payload["fem"], dtype=float)[:25, component_index]
        reference = np.asarray(payload["empymod"], dtype=float)[:25, component_index]
    if times.size != 25:
        raise ValueError(f"{path} must contain at least 25 samples")
    return times, response, reference


def _pairwise_metrics(coarse_dir: Path, fine_dir: Path) -> dict:
    coarse_times, coarse, _ = _load_response(coarse_dir)
    fine_times, fine, reference = _load_response(fine_dir)
    if not np.allclose(coarse_times, fine_times, rtol=RTOL, atol=1.0e-30):
        raise ValueError("independent audit found mismatched observation grids")
    amplitude_floor = float(np.max(np.abs(reference))) * FLOOR_FRACTION
    mask = np.abs(reference) >= amplitude_floor
    denominator = np.abs(fine[mask])
    if np.count_nonzero(mask) < 3 or np.any(denominator == 0.0):
        raise ValueError("independent audit found an invalid effective window")
    relative = np.abs(coarse[mask] - fine[mask]) / denominator
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "excluded_below_floor_count": int(mask.size - np.count_nonzero(mask)),
        "amplitude_floor": amplitude_floor,
        "median_percent": 100.0 * float(statistics.median(relative.tolist())),
        "rms_percent": 100.0 * float(math.sqrt(np.mean(relative * relative))),
        "max_percent": 100.0 * float(np.max(relative)),
    }


def _external_metrics(errors_path: Path) -> dict:
    records: list[tuple[float, float, float]] = []
    with Path(errors_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("component") != "dBzdt":
                continue
            values = (
                float(row["time_obs"]),
                float(row["ref"]),
                float(row["ordinary_relative_error"]),
            )
            if all(math.isfinite(value) for value in values):
                records.append(values)
    if not records:
        raise ValueError(f"no finite dBzdt records in {errors_path}")
    peak = max(abs(reference) for _, reference, _ in records)
    amplitude_floor = peak * FLOOR_FRACTION
    effective = [
        error
        for _, reference, error in records
        if abs(reference) >= amplitude_floor
    ]
    median = statistics.median(effective)
    rms = math.sqrt(sum(value * value for value in effective) / len(effective))
    maximum = max(effective)
    return {
        "sample_count": len(effective),
        "excluded_below_floor_count": len(records) - len(effective),
        "amplitude_floor": amplitude_floor,
        "median_percent": 100.0 * median,
        "rms_percent": 100.0 * rms,
        "max_percent": 100.0 * maximum,
        "publication_gate_passed": (
            median <= 0.05 and rms <= 0.05 and maximum <= 0.10
        ),
    }


def _assert_metrics(label: str, actual: dict, expected: dict) -> None:
    for key in ("sample_count", "excluded_below_floor_count"):
        if int(actual[key]) != int(expected[key]):
            raise AssertionError(
                f"{label} {key} mismatch: {actual[key]} != {expected[key]}"
            )
    for key in ("amplitude_floor", "median_percent", "rms_percent", "max_percent"):
        if not math.isclose(
            float(actual[key]),
            float(expected[key]),
            rel_tol=RTOL,
            abs_tol=ATOL,
        ):
            raise AssertionError(
                f"{label} {key} mismatch: {actual[key]} != {expected[key]}"
            )
    if "publication_gate_passed" in expected:
        if bool(actual["publication_gate_passed"]) != bool(
            expected["publication_gate_passed"]
        ):
            raise AssertionError(f"{label} publication gate mismatch")


def _audit_resource_contract(summary: dict, axes: dict[str, dict]) -> dict:
    contract = summary.get("resource_contract")
    if not isinstance(contract, dict):
        raise AssertionError("summary resource contract missing")
    keys = (
        "total_memory_gb",
        "reserve_memory_gb",
        "solver_memory_limit_gb",
    )
    try:
        expected = {key: float(contract[key]) for key in keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError("summary resource contract invalid") from exc
    if not all(math.isfinite(value) for value in expected.values()):
        raise AssertionError("summary resource contract is nonfinite")

    baseline_dir = Path(summary["candidate_baseline"]["run_dir"])
    large_dir = next(
        Path(level["run_dir"])
        for level in axes["domain"]["levels"]
        if level["level_id"] == "large"
    )
    checked: list[str] = []
    for run_dir in (baseline_dir, large_dir):
        path = run_dir / "preflight.json"
        if not path.is_file():
            raise AssertionError(f"resource preflight missing: {path}")
        preflight = json.loads(path.read_text(encoding="utf-8"))
        if preflight.get("passed") is not True:
            raise AssertionError(f"resource preflight did not pass: {path}")
        for key, expected_value in expected.items():
            try:
                actual_value = float(preflight[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise AssertionError(
                    f"resource contract missing or invalid: {path} {key}"
                ) from exc
            if actual_value != expected_value:
                raise AssertionError(f"resource contract mismatch: {path} {key}")
        estimate = float(preflight.get("estimated_memory_gb", math.nan))
        solver_limit = expected["solver_memory_limit_gb"]
        if (
            not math.isfinite(estimate)
            or estimate > solver_limit
            or float(preflight.get("memory_limit_gb", math.nan)) != solver_limit
        ):
            raise AssertionError(f"resource estimate exceeds contract: {path}")
        checked.append(str(path))
    return {
        "resource_contract_verified": True,
        "resource_preflight_count": len(checked),
        "resource_preflights": checked,
    }


def audit(summary_path: Path) -> dict:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    comparison_pairs = {
        "time": (
            ("coarse_to_standard", "coarse", "standard"),
            ("standard_to_fine", "standard", "fine"),
        ),
        "mesh": (
            ("coarse_to_standard", "coarse", "standard"),
            ("standard_to_fine", "standard", "fine"),
        ),
        "domain": (
            ("small_to_standard", "small", "standard"),
            ("standard_to_large", "standard", "large"),
        ),
    }
    comparisons: list[dict] = []
    axes = {axis["axis"]: axis for axis in summary["axes"]}
    resource_audit = _audit_resource_contract(summary, axes)
    for axis_name, pairs in comparison_pairs.items():
        axis = axes[axis_name]
        run_dirs = {
            level["level_id"]: Path(level["run_dir"])
            for level in axis["levels"]
        }
        expected = {
            item["comparison_id"]: item for item in axis["comparisons"]
        }
        for comparison_id, coarse_id, fine_id in pairs:
            actual = _pairwise_metrics(run_dirs[coarse_id], run_dirs[fine_id])
            _assert_metrics(
                f"{axis_name}/{comparison_id}",
                actual,
                expected[comparison_id],
            )
            comparisons.append(
                {"axis": axis_name, "comparison_id": comparison_id, **actual}
            )

    baseline = summary["candidate_baseline"]
    baseline_actual = _external_metrics(Path(baseline["run_dir"]) / "errors.csv")
    _assert_metrics(
        "candidate_baseline/external_reference_gate",
        baseline_actual,
        baseline["external_reference_gate"],
    )
    domain = axes["domain"]
    domain_large_dir = next(
        Path(level["run_dir"])
        for level in domain["levels"]
        if level["level_id"] == "large"
    )
    large_actual = _external_metrics(domain_large_dir / "errors.csv")
    _assert_metrics(
        "domain/large_external_reference_gate",
        large_actual,
        domain["large_external_reference_gate"],
    )
    return {
        "verified": True,
        "study_id": summary["study_id"],
        "comparison_count": len(comparisons),
        "external_gate_count": 2,
        "rtol": RTOL,
        "atol": ATOL,
        "comparisons": comparisons,
        "candidate_baseline_external_gate": baseline_actual,
        "domain_large_external_gate": large_actual,
        **resource_audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently recompute layered convergence evidence."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = audit(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("INDEPENDENT_RECOMPUTE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
