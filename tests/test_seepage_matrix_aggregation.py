from __future__ import annotations

from pathlib import Path

import numpy as np

from atem3d.seepage_case_matrix import build_case_matrix
from atem3d.seepage_matrix_aggregation import (
    OPEN3D_REQUIRED_GATES,
    aggregate_matrix_gates,
    build_open3d_summary,
)


def _write_case(root: Path, case_id: str, values: np.ndarray) -> None:
    case = next(item for item in build_case_matrix() if item.case_id == case_id)
    path = root / case.expected_output
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=values,
        times=np.geomspace(1e-5, 1e-2, 31),
        case_fingerprint=np.asarray(case.case_fingerprint),
        execution_fingerprint=np.asarray(case.execution_fingerprint),
        base_model_fingerprint=np.asarray(case.model_fingerprint),
        material_relative_volume_error=np.asarray(0.005),
    )


def _symmetric(scale: float) -> np.ndarray:
    values = np.zeros((5, 31, 3), dtype=float)
    time = np.geomspace(1.0, 0.1, 31)
    values[:, :, 0] = scale * np.asarray([2, 1, 3, 1, 2])[:, None] * time
    for component in (1, 2):
        values[:, :, component] = (
            scale * np.asarray([2, 1, 0, -1, -2])[:, None] * time
        )
    return values


def test_matrix_aggregation_builds_all_open_solver_gates(tmp_path: Path) -> None:
    background = np.ones((5, 31, 3), dtype=float)
    for solver in ("simpeg", "fenicsx"):
        _write_case(tmp_path, f"{solver}-conductivity-background-reference", background)
        for sigma, scale in ((0.01, 0.0), (0.02, 0.01), (0.1, 0.1), (1.0, 1.0)):
            slug = str(sigma).replace(".", "p").rstrip("0").rstrip("p")
            _write_case(
                tmp_path,
                f"{solver}-conductivity-channel-sigma-{slug}",
                background + _symmetric(scale),
            )
        for cross, scale in ((1, 1.0), (2, 2.0), (10, 3.0)):
            for role in ("background", "channel"):
                values = background if role == "background" else background + _symmetric(scale)
                _write_case(tmp_path, f"{solver}-volume-{role}-cross-{cross}", values)
        spatial_error = {0.5: 0.20, 0.25: 0.04, 0.125: 0.0}
        for mesh, error in spatial_error.items():
            slug = str(mesh).replace(".", "p").rstrip("0").rstrip("p")
            for role in ("background", "channel"):
                values = background if role == "background" else background + _symmetric(1.0 + error)
                _write_case(tmp_path, f"{solver}-spatial-{role}-h-{slug}", values)
        temporal_error = {1.0: 0.10, 0.5: 0.02, 0.25: 0.0}
        for factor, error in temporal_error.items():
            slug = str(factor).replace(".", "p").rstrip("0").rstrip("p")
            for role in ("background", "channel"):
                values = background if role == "background" else background + _symmetric(1.0 + error)
                _write_case(tmp_path, f"{solver}-temporal-{role}-dt-{slug}", values)

    gates = aggregate_matrix_gates(tmp_path)

    expected = {
        *(f"zero_contrast_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"conductivity_trend_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"volume_trend_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"spatial_convergence_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"temporal_convergence_{solver}" for solver in ("simpeg", "fenicsx")),
        *(f"parity_{solver}" for solver in ("simpeg", "fenicsx")),
        "cross_solver",
        "discrete_volume",
    }
    assert expected <= gates.keys()
    assert all(gates[name]["available"] and gates[name]["pass"] for name in expected)
    summary = build_open3d_summary(tmp_path)
    assert tuple(summary["required_gates"]) == OPEN3D_REQUIRED_GATES
    assert summary["pass"] is True


def test_matrix_aggregation_marks_missing_cases_unavailable(tmp_path: Path) -> None:
    gates = aggregate_matrix_gates(tmp_path)

    assert gates["zero_contrast_simpeg"]["available"] is False
    assert gates["cross_solver"]["pass"] is False
