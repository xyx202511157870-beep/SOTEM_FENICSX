"""Aggregate the approved SimPEG/FEniCSx seepage verification matrix."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .seepage_case_matrix import VerificationCase, build_case_matrix
from .seepage_channel_model import model_for_variant
from .seepage_verification import (
    anomaly_energy_trend,
    build_verification_summary,
    cross_solver_agreement,
    discrete_volume_metrics,
    parity_metrics,
    model_fingerprint,
    odd_parity_metrics,
    three_level_convergence,
    zero_contrast_metrics,
)


OPEN3D_REQUIRED_GATES = tuple(
    [
        *(f"zero_contrast_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"conductivity_trend_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"volume_trend_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"spatial_convergence_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"temporal_convergence_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"parity_{solver}" for solver in ("simpeg", "fenicsx")),
        "background_empymod",
        "fenicsx_magnetic_operator",
        "discrete_volume",
        "cross_solver",
    ]
)


def _unavailable(error: str) -> dict[str, Any]:
    return {"available": False, "pass": False, "error": str(error)}


class _MatrixResults:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.cases = {case.case_id: case for case in build_case_matrix()}

    def values(self, case_id: str) -> np.ndarray:
        case = self.cases[case_id]
        path = self.output_root / case.expected_output
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as stored:
            fingerprint = str(np.asarray(stored["case_fingerprint"]).item())
            base_fingerprint = str(np.asarray(stored["base_model_fingerprint"]).item())
            values = np.asarray(stored["values"], dtype=float)
        if fingerprint != case.case_fingerprint:
            raise ValueError(f"stale case fingerprint: {case_id}")
        if base_fingerprint != case.model_fingerprint:
            raise ValueError(f"mixed model fingerprint: {case_id}")
        if values.shape != (5, 31, 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid normalized values: {case_id}")
        return values

    def delta(self, channel_id: str, background_id: str) -> np.ndarray:
        return self.values(channel_id) - self.values(background_id)

    def material_error(self, case_id: str) -> float:
        case = self.cases[case_id]
        path = self.output_root / case.expected_output
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as stored:
            fingerprint = str(np.asarray(stored["case_fingerprint"]).item())
            value = float(np.asarray(stored["material_relative_volume_error"]).item())
        if fingerprint != case.case_fingerprint:
            raise ValueError(f"stale case fingerprint: {case_id}")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid material volume audit: {case_id}")
        return value


def _capture(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        return _unavailable(str(exc))


def _load_magnetic_diagnostics(path: Path) -> dict[str, np.ndarray]:
    fields = (
        "dBzdt_curl",
        "dBzdt_biot_rate",
        "dBzdt_faraday_loop",
        "Hz_biot_center",
        "Hz_biot_tetra4",
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    model_times = np.geomspace(1.0e-5, 1.0e-2, 31)
    arrays = {field: np.full((5, 31), np.nan, dtype=float) for field in fields}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 155:
        raise ValueError(f"{path}: expected 155 magnetic diagnostic rows")
    for row in rows:
        receiver = int(row["receiver_id"].removeprefix("Rx")) - 1
        time = float(row["time_obs"])
        time_index = int(np.argmin(np.abs(model_times - time)))
        if not 0 <= receiver < 5 or not np.isclose(
            time, model_times[time_index], rtol=1.0e-12, atol=0.0
        ):
            raise ValueError(f"{path}: diagnostic receiver/time contract mismatch")
        if row.get("provenance") != "explicit_full_domain":
            raise ValueError(f"{path}: magnetic diagnostic is not full-domain data")
        for field in fields:
            arrays[field][receiver, time_index] = float(row[field])
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"{path}: incomplete or nonfinite magnetic diagnostics")
    return arrays


def _magnetic_operator_gate(results: _MatrixResults) -> dict[str, Any]:
    base = results.output_root / "verification_runs" / "fenicsx"
    background_id = "fenicsx-conductivity-background-reference"
    channel_id = "fenicsx-conductivity-channel-sigma-1"
    background = _load_magnetic_diagnostics(
        base / background_id / "magnetic_receiver_diagnostics.csv"
    )
    channel = _load_magnetic_diagnostics(
        base / channel_id / "magnetic_receiver_diagnostics.csv"
    )
    methods: dict[str, Any] = {}
    for field in background:
        methods[field] = odd_parity_metrics(
            channel[field] - background[field],
            pair_threshold=0.05,
            center_threshold=0.02,
        )

    selected_background = results.values(background_id)
    selected_channel = results.values(channel_id)
    selected_dbdt_error = float(
        np.max(np.abs(background["dBzdt_biot_rate"] - selected_background[:, :, 1]))
        + np.max(np.abs(channel["dBzdt_biot_rate"] - selected_channel[:, :, 1]))
    )
    selected_hz_error = float(
        np.max(np.abs(background["Hz_biot_tetra4"] - selected_background[:, :, 2]))
        + np.max(np.abs(channel["Hz_biot_tetra4"] - selected_channel[:, :, 2]))
    )
    scale = max(
        float(np.max(np.abs(selected_background[:, :, 1:]))),
        float(np.max(np.abs(selected_channel[:, :, 1:]))),
        np.finfo(float).tiny,
    )
    consistency = max(selected_dbdt_error, selected_hz_error) / scale
    selected_pass = (
        methods["dBzdt_biot_rate"]["pass"]
        and methods["Hz_biot_tetra4"]["pass"]
        and consistency <= 1.0e-12
    )
    return {
        "available": True,
        "pass": bool(selected_pass),
        "selected": {"dBzdt": "biot_rate", "Hz": "biot_tetra4"},
        "selected_consistency_relative_error": float(consistency),
        "selected_consistency_threshold": 1.0e-12,
        "methods": methods,
    }


def _background_empymod_gate(results: _MatrixResults) -> dict[str, Any]:
    path = results.output_root / "verification_empymod_background.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        reference = np.asarray(stored["values"], dtype=float)
        fingerprint = str(np.asarray(stored["model_fingerprint"]).item())
    expected = next(iter(results.cases.values())).model_fingerprint
    if fingerprint != expected or reference.shape != (5, 31, 3):
        raise ValueError("empymod background model contract mismatch")
    comparisons = {
        solver: cross_solver_agreement(
            results.values(f"{solver}-conductivity-background-reference"),
            reference,
            median_threshold=0.20,
            p95_threshold=0.35,
        )
        for solver in ("simpeg", "fenicsx")
    }
    return {
        "available": True,
        "pass": bool(all(item["pass"] for item in comparisons.values())),
        "reference": "empymod_uniform_halfspace_only",
        "comparisons": comparisons,
    }


def _solver_gates(results: _MatrixResults, solver: str) -> dict[str, dict[str, Any]]:
    prefix = f"{solver}-conductivity"
    background_id = f"{prefix}-background-reference"

    def conductivity_delta(sigma_slug: str) -> np.ndarray:
        return results.delta(
            f"{prefix}-channel-sigma-{sigma_slug}", background_id
        )

    gates: dict[str, dict[str, Any]] = {}
    gates[f"zero_contrast_{solver}"] = _capture(
        lambda: zero_contrast_metrics(
            conductivity_delta("0p01"),
            results.values(background_id),
            threshold=1.0e-6,
        )
    )
    gates[f"conductivity_trend_{solver}"] = _capture(
        lambda: anomaly_energy_trend(
            [0.01, 0.02, 0.1, 1.0],
            [conductivity_delta(slug) for slug in ("0p01", "0p02", "0p1", "1")],
            results.values(background_id),
        )
    )

    def volume_delta(cross_slug: str) -> np.ndarray:
        return results.delta(
            f"{solver}-volume-channel-cross-{cross_slug}",
            f"{solver}-volume-background-cross-{cross_slug}",
        )

    gates[f"volume_trend_{solver}"] = _capture(
        lambda: anomaly_energy_trend(
            [60.0, 240.0, 6000.0],
            [volume_delta(slug) for slug in ("1", "2", "10")],
            results.values(f"{solver}-volume-background-cross-1"),
        )
    )

    def controlled_delta(study: str, label: str) -> np.ndarray:
        return results.delta(
            f"{solver}-{study}-channel-{label}",
            f"{solver}-{study}-background-{label}",
        )

    gates[f"spatial_convergence_{solver}"] = _capture(
        lambda: three_level_convergence(
            controlled_delta("spatial", "h-0p5"),
            controlled_delta("spatial", "h-0p25"),
            controlled_delta("spatial", "h-0p125"),
            median_threshold=0.10,
            p95_threshold=0.20,
        )
    )
    gates[f"temporal_convergence_{solver}"] = _capture(
        lambda: three_level_convergence(
            controlled_delta("temporal", "dt-1"),
            controlled_delta("temporal", "dt-0p5"),
            controlled_delta("temporal", "dt-0p25"),
            median_threshold=0.05,
            p95_threshold=0.10,
        )
    )
    gates[f"parity_{solver}"] = _capture(
        lambda: parity_metrics(
            conductivity_delta("1"), pair_threshold=0.05, center_threshold=0.02
        )
    )
    return gates


def aggregate_matrix_gates(output_root: str | Path) -> dict[str, dict[str, Any]]:
    """Return fail-closed gates from all normalized open-3D solver cases."""

    results = _MatrixResults(output_root)
    gates: dict[str, dict[str, Any]] = {}
    for solver in ("simpeg", "fenicsx"):
        gates.update(_solver_gates(results, solver))

    def base_delta(solver: str) -> np.ndarray:
        return results.delta(
            f"{solver}-conductivity-channel-sigma-1",
            f"{solver}-conductivity-background-reference",
        )

    gates["cross_solver"] = _capture(
        lambda: cross_solver_agreement(
            base_delta("simpeg"),
            base_delta("fenicsx"),
            median_threshold=0.20,
            p95_threshold=0.35,
        )
    )
    gates["discrete_volume"] = _capture(
        lambda: discrete_volume_metrics(
            {
                solver: [
                    results.material_error(case.case_id)
                    for case in results.cases.values()
                    if case.solver == solver and case.role == "channel"
                ]
                for solver in ("simpeg", "fenicsx")
            },
            threshold=0.02,
        )
    )
    gates["fenicsx_magnetic_operator"] = _capture(
        lambda: _magnetic_operator_gate(results)
    )
    gates["background_empymod"] = _capture(
        lambda: _background_empymod_gate(results)
    )
    return gates


def build_open3d_summary(
    output_root: str | Path, *, require_pass: bool = False
) -> dict[str, Any]:
    """Build the formal SimPEG/FEniCSx verification summary."""

    model = model_for_variant("thin_60x1x1")
    return build_verification_summary(
        model_fingerprint_value=model_fingerprint(model),
        required_gates=OPEN3D_REQUIRED_GATES,
        gates=aggregate_matrix_gates(output_root),
        require_pass=require_pass,
    )


__all__ = [
    "OPEN3D_REQUIRED_GATES",
    "aggregate_matrix_gates",
    "build_open3d_summary",
]
