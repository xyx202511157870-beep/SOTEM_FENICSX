from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_geometry_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_geometry_is_100m_wire_with_50m_parallel_offset():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig()
    diagnostics = sp.validate_geometry_consistency(config)

    assert diagnostics["source_length"] == pytest.approx(100.0)
    assert diagnostics["parallel_offset"] == pytest.approx(50.0)


def test_geometry_consistency_rejects_wrong_parallel_offset():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(receiver=(500.0, 500.0, -0.1))

    with pytest.raises(ValueError, match="parallel survey offset"):
        sp.validate_geometry_consistency(config)
