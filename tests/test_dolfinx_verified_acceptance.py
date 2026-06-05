from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_verified_acceptance_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "run_dir, expected_weak_component",
    [
        ("noip_offset50_afterramp_recv5_mean_src65_tmin25e7_t1e4", "Ey"),
        ("cole_offset50_afterramp_recv5_median_src65_tmin25e7_t1e4_debye_be", "Ey"),
    ],
)
def test_verified_offset50_acceptance_metrics_pass_below_five_percent(run_dir, expected_weak_component):
    sp = _load_pipeline_module()
    root = Path(__file__).resolve().parents[1]
    data_path = root / "dolfinx" / "runs" / run_dir / "verification_data.npz"
    if not data_path.exists():
        pytest.skip(f"verified run data not available: {data_path}")

    data = np.load(data_path)
    components = [str(item) for item in data["components"]]

    result = sp.check_physical_error_window(
        data["times"],
        data["fem"],
        data["empymod"],
        components,
        error_min_time=2.5e-6,
        tolerance=0.05,
    )

    assert result["passed"], result["maxima"]

    weak_result = sp.check_weak_component_error_window(
        data["times"],
        data["fem"],
        data["empymod"],
        components,
        error_min_time=2.5e-6,
        tolerance=0.05,
        weak_reference_fraction=0.1,
    )

    assert weak_result["passed"], weak_result["maxima"]
    assert expected_weak_component in weak_result["weak_components"]


def test_weak_horizontal_component_uses_primary_scale_absolute_gate():
    sp = _load_pipeline_module()
    times = np.array([1.0e-6, 2.0e-6])
    components = ["Ex", "Ey", "dBzdt"]
    ref = np.array(
        [
            [1.0e-5, 1.0e-9, 2.0e-9],
            [8.0e-6, -1.0e-9, 1.0e-9],
        ]
    )
    fem = np.array(
        [
            [1.02e-5, 2.0e-7, 2.02e-9],
            [7.9e-6, -1.5e-7, 1.02e-9],
        ]
    )

    result = sp.check_weak_component_error_window(
        times,
        fem,
        ref,
        components,
        error_min_time=1.0e-6,
        tolerance=0.05,
        weak_reference_fraction=0.05,
    )

    assert result["passed"], result["maxima"]
    assert result["maxima"]["Ey"] == pytest.approx(0.0199)
    assert result["weak_components"] == ["Ey"]
