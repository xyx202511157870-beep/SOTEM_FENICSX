from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_h_guard_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeKSP:
    def __init__(self, *, reason: int, residual: float, iterations: int = 1):
        self._reason = reason
        self._residual = residual
        self._iterations = iterations

    def getConvergedReason(self):
        return self._reason

    def getResidualNorm(self):
        return self._residual

    def getIterationNumber(self):
        return self._iterations


def test_h_solver_guard_rejects_positive_reason_with_huge_relative_residual():
    sp = _load_pipeline_module()

    with pytest.raises(RuntimeError, match="relative residual"):
        sp._validate_h_solver_state(
            "time:h",
            _FakeKSP(reason=2, residual=1.0e83),
            np.array([1.0, 2.0, 3.0]),
            rhs_norm=1.0,
            config=sp.PipelineConfig(),
        )


def test_h_solver_guard_rejects_nonfinite_solution_after_direct_solve():
    sp = _load_pipeline_module()

    with pytest.raises(RuntimeError, match="non-finite"):
        sp._validate_h_solver_state(
            "initial:h",
            _FakeKSP(reason=4, residual=0.0),
            np.array([1.0, np.nan, 3.0]),
            rhs_norm=1.0,
            config=sp.PipelineConfig(),
        )


def test_h_static_initial_mass_scale_has_nonzero_gauge_floor():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        x_extent=500.0,
        y_extent=500.0,
        air_height=300.0,
        earth_depth=500.0,
        t_max=1.0e-4,
        ramp_off_time=1.0e-5,
        rho_earth=100.0,
        rho_air=1.0e8,
    )

    scale = sp._h_static_initial_mass_scale(config)

    assert scale >= 2.0e-3
    assert scale < 1.0e-2


def test_h_solver_configuration_uses_ams_not_lu(monkeypatch):
    sp = _load_pipeline_module()
    calls = []

    def fake_ams(A, spaces, config):
        calls.append(("ams", A, spaces, config))
        return {"ksp": "ams"}

    def fake_lu(A, *, comm=None):
        raise AssertionError("H-form production solver should not use LU")

    monkeypatch.setattr(sp, "configure_ams_solver", fake_ams)
    monkeypatch.setattr(sp, "configure_lu_solver", fake_lu)
    config = sp.PipelineConfig(formulation="h")
    spaces = {"V": object(), "S": object()}

    context = sp._configure_h_solver("matrix", spaces, config)

    assert context == {"ksp": "ams"}
    assert calls == [("ams", "matrix", spaces, config)]


def test_h_step_solver_reconfigures_ams_for_new_matrix(monkeypatch):
    sp = _load_pipeline_module()
    calls = []

    def fake_ams(A, spaces, config):
        calls.append((A, spaces, config))
        return {"ksp": f"ams-{A}"}

    monkeypatch.setattr(sp, "configure_ams_solver", fake_ams)
    config = sp.PipelineConfig(formulation="h")
    spaces = {"V": object(), "S": object()}

    context = sp._configure_h_step_solver({"ksp": "old"}, "new-matrix", spaces, config)

    assert context == {"ksp": "ams-new-matrix"}
    assert calls == [("new-matrix", spaces, config)]
