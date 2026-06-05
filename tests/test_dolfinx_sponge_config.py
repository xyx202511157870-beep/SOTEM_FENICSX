from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_sponge_config_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dolfinx_sponge_defaults_to_disabled_and_is_reported():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["sponge"]["enabled"] is False
    assert diagnostics["sponge"]["strength"] == pytest.approx(0.0)
    assert diagnostics["sponge"]["thickness"] == pytest.approx(0.0)
    assert diagnostics["sponge"]["sides"] == sp.SPONGE_ALL_SIDES


def test_dolfinx_sponge_weights_only_increase_outer_shell():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        x_extent=100.0,
        y_extent=100.0,
        air_height=80.0,
        earth_depth=120.0,
        sponge_strength=0.25,
        sponge_thickness=20.0,
        sponge_power=2.0,
    )
    centers = np.asarray(
        [
            [0.0, 0.0, -10.0],
            [95.0, 0.0, -10.0],
            [-95.0, 0.0, -10.0],
            [0.0, 0.0, -115.0],
            [0.0, 0.0, 75.0],
        ],
        dtype=float,
    )

    addition = sp._sponge_sigma_addition_for_centers(centers, config)

    assert addition[0] == pytest.approx(0.0)
    assert np.all(addition[1:] > 0.0)
    assert float(np.max(addition)) <= config.sponge_strength


def test_dolfinx_sponge_can_be_limited_to_one_side():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        x_extent=100.0,
        y_extent=100.0,
        air_height=80.0,
        earth_depth=120.0,
        sponge_strength=1.0,
        sponge_thickness=20.0,
        sponge_sides=("x_max",),
    )
    centers = np.asarray(
        [
            [95.0, 0.0, -10.0],
            [-95.0, 0.0, -10.0],
            [0.0, 95.0, -10.0],
        ],
        dtype=float,
    )

    addition = sp._sponge_sigma_addition_for_centers(centers, config)

    assert addition[0] > 0.0
    assert addition[1] == pytest.approx(0.0)
    assert addition[2] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sponge_strength": -1.0},
        {"sponge_strength": 1.0, "sponge_thickness": 0.0},
        {"sponge_power": 0.0},
        {"sponge_strength": 1.0, "sponge_thickness": 10.0, "sponge_sides": ("bad",)},
    ],
)
def test_dolfinx_sponge_rejects_invalid_config(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="sponge"):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))
