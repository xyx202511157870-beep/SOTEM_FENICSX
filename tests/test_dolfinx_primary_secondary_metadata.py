from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atem3d.materials.prony import DebyeTerm, PronyConductivity


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_primary_secondary_metadata",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_record_primary_secondary_step_equation_writes_noip_metadata():
    sp = _load_pipeline_module()
    diagnostics = {}

    sp._record_primary_secondary_step_equation(
        diagnostics,
        material=PronyConductivity.no_ip(0.02),
        sigma_background=0.01,
        dt=0.25,
    )

    metadata = diagnostics["primary_secondary_step_equation"]
    assert metadata["case_type"] == "noip"
    assert metadata["sigma"] == 0.02
    assert metadata["sigma_background"] == 0.01
    assert metadata["dt"] == 0.25
    assert metadata["lhs_operator"] == "K + R + M(sigma)/dt"


def test_record_primary_secondary_step_equation_writes_ip_metadata():
    sp = _load_pipeline_module()
    diagnostics = {}

    sp._record_primary_secondary_step_equation(
        diagnostics,
        material=PronyConductivity(
            sigma_inf=0.03,
            terms=[DebyeTerm(delta_sigma=0.006, tau=0.5)],
        ),
        sigma_background=0.01,
        dt=0.5,
    )

    metadata = diagnostics["primary_secondary_step_equation"]
    assert metadata["case_type"] == "ip"
    assert metadata["sigma0"] == pytest.approx(0.024)
    assert metadata["sigma_eff"] == pytest.approx(0.027)
    assert metadata["alpha"] == [0.5]
    assert metadata["beta"] == [0.5]
    assert metadata["adapter_backend"] == "dolfinx_primary_secondary"


def test_dolfinx_primary_secondary_bridge_passes_adapter_diagnostics(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}
    diagnostics = {}

    def fake_interpolation(_msh, _spaces):
        return {"points": np.array([[0.0, 0.0, 0.0]])}

    def fake_adapters(_msh, _spaces, _materials, _operators, _config, _fem_points, **_kwargs):
        return {
            "secondary_state_initializer": object(),
            "secondary_step_solver": object(),
            "secondary_receiver_projector": object(),
            "secondary_state_stepper": object(),
            "diagnostics": diagnostics,
        }

    class FakePrimarySecondaryForwardOperator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import atem3d.solvers as solvers

    monkeypatch.setattr(sp, "_nedelec_interpolation_points", fake_interpolation)
    monkeypatch.setattr(sp, "_make_dolfinx_primary_secondary_forward_adapters", fake_adapters)
    monkeypatch.setattr(solvers, "PrimarySecondaryForwardOperator", FakePrimarySecondaryForwardOperator)

    result = sp._make_dolfinx_primary_secondary_forward_operator(
        msh=object(),
        spaces={},
        materials={},
        operators={},
        config=SimpleNamespace(),
        primary=object(),
        receiver_locations=np.array([[0.0, 0.0, 0.0]]),
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
        turnoff_time=1.0e-5,
        turnoff_steps=10,
    )

    assert captured["diagnostics"] is diagnostics
    assert result["diagnostics"] is diagnostics
