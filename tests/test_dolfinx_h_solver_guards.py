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
