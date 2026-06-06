from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_secondary_projector_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_secondary_receiver_projector_bridge_uses_evaluate_receivers(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(0.0, -300.0, -0.1))
    seen = {}

    def electric_getter(state, Ep_new, time_value, dt):
        seen["electric_getter"] = (state, Ep_new.copy(), time_value, dt)
        return "E_secondary"

    def dbdt_getter(state, Ep_new, time_value, dt):
        seen["dbdt_getter"] = (state, Ep_new.copy(), time_value, dt)
        return "dBdt_secondary"

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        seen["evaluate"] = (E, dbdt, msh, eval_config)
        return {"Ex": 1.0, "Ey": 2.0, "Hz": 3.0, "dBzdt": 4.0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        electric_getter,
        dbdt_getter,
        msh="mesh",
        config=config,
    )

    values = projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0e-5, 2.0e-6, ("Ex", "Ey", "dBzdt"))

    np.testing.assert_allclose(values, [[1.0, 2.0, 4.0]])
    assert seen["electric_getter"][0] == "state"
    np.testing.assert_allclose(seen["electric_getter"][1], [[0.5, 0.0, 0.0]])
    assert seen["electric_getter"][2] == 1.0e-5
    assert seen["electric_getter"][3] == 2.0e-6
    assert seen["evaluate"] == ("E_secondary", "dBdt_secondary", "mesh", config)
