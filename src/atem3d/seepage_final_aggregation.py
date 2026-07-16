"""Final fail-closed aggregation including the independent COMSOL model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .seepage_channel_model import model_for_variant
from .seepage_matrix_aggregation import OPEN3D_REQUIRED_GATES, build_open3d_summary
from .seepage_verification import (
    build_verification_summary,
    comsol_multi_solver_agreement,
    cross_solver_agreement,
    model_fingerprint,
    zero_contrast_metrics,
)


COMSOL_REQUIRED_GATES = (
    "comsol_zero_contrast",
    "comsol_anomaly_agreement",
    "comsol_uniform_background",
)
FINAL_REQUIRED_GATES = (*OPEN3D_REQUIRED_GATES, *COMSOL_REQUIRED_GATES)


def _unavailable(error: str) -> dict[str, Any]:
    return {"available": False, "pass": False, "error": str(error)}


def _capture(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        return _unavailable(str(exc))


def _load_values(path: Path, expected_fingerprint: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        values = np.asarray(stored["values"], dtype=float)
        key = (
            "base_model_fingerprint"
            if "base_model_fingerprint" in stored.files
            else "model_fingerprint"
        )
        fingerprint = str(np.asarray(stored[key]).item())
    if fingerprint != expected_fingerprint:
        raise ValueError(f"{path}: mixed model fingerprint")
    if values.shape != (5, 31, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: invalid normalized response array")
    return values


def build_final_summary(
    output_root: str | Path, *, require_pass: bool = False
) -> dict[str, Any]:
    """Combine all open-solver and independent COMSOL gates."""

    root = Path(output_root)
    fingerprint = model_fingerprint(model_for_variant("thin_60x1x1"))
    open_summary = build_open3d_summary(root)
    gates = dict(open_summary["gates"])

    def comsol_values(case: str) -> np.ndarray:
        return _load_values(root / "comsol_3d" / case / "normalized.npz", fingerprint)

    gates["comsol_zero_contrast"] = _capture(
        lambda: zero_contrast_metrics(
            comsol_values("zero_contrast") - comsol_values("background"),
            comsol_values("background"),
            threshold=1.0e-6,
        )
    )

    def anomaly_agreement() -> dict[str, Any]:
        comsol_delta = comsol_values("channel") - comsol_values("background")
        open_deltas = {
            solver: _load_values(
                root
                / "verification_runs"
                / solver.lower()
                / f"{solver.lower()}-conductivity-channel-sigma-1"
                / "normalized.npz",
                fingerprint,
            )
            - _load_values(
                root
                / "verification_runs"
                / solver.lower()
                / f"{solver.lower()}-conductivity-background-reference"
                / "normalized.npz",
                fingerprint,
            )
            for solver in ("SimPEG", "FEniCSx")
        }
        return comsol_multi_solver_agreement(
            comsol_delta,
            open_deltas,
            median_threshold=0.25,
            p95_threshold=0.40,
        )

    gates["comsol_anomaly_agreement"] = _capture(anomaly_agreement)
    gates["comsol_uniform_background"] = _capture(
        lambda: cross_solver_agreement(
            comsol_values("uniform_background_reference"),
            _load_values(root / "verification_empymod_background.npz", fingerprint),
            median_threshold=0.30,
            p95_threshold=0.40,
        )
    )
    return build_verification_summary(
        model_fingerprint_value=fingerprint,
        required_gates=FINAL_REQUIRED_GATES,
        gates=gates,
        require_pass=require_pass,
    )


__all__ = [
    "COMSOL_REQUIRED_GATES",
    "FINAL_REQUIRED_GATES",
    "build_final_summary",
]
