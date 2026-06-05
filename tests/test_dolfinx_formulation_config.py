from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_formulation_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formulation_defaults_to_e_and_accepts_h():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().formulation == "e"
    assert sp.PipelineConfig(formulation="h").formulation == "h"


def test_formulation_validation_rejects_unknown_value():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="formulation"):
        sp.validate_formulation(sp.PipelineConfig(formulation="bad"))


def test_source_term_mode_defaults_to_impressed_current_and_accepts_primary_dc():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().source_term_mode == "impressed_current"
    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_term_mode="primary_dc"))
    assert diagnostics["source_term_mode"] == "primary_dc"


def test_source_term_mode_validation_rejects_unknown_value():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="source_term_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(source_term_mode="bad"))
