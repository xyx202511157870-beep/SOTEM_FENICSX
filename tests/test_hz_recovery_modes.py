from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_hz_modes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_biot_recovery_mode_preserves_dbdt_when_adding_hz():
    sp = _load_pipeline_module()
    receiver = {"Ex": 1.0, "Ey": 2.0, "dBzdt": 3.0}

    sp._assign_biot_receiver_hz(receiver, np.asarray([0.0, 0.0, -4.0]))

    assert receiver["Hz"] == -4.0
    assert receiver["dBzdt"] == 3.0


def test_faraday_integrated_mode_advertises_hz_and_dbdt_outputs():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(magnetic_receiver_mode="faraday_integrated")

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["magnetic_receiver_mode"] == "faraday_integrated"
    assert sp._forward_components(config) == ["Ex", "Ey", "Hz", "dBzdt"]
