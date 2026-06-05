from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_validation_artifacts_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_validation_artifacts_generates_required_p2_outputs(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.array([1.0e-5, 1.0e-4])
    ref = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]])
    pred = np.array([[1.01, 1.0e-16, 2.02], [2.02, 2.0e-16, 1.01]])
    components = ["Ex", "Ey", "dBzdt"]

    summary = sp.write_validation_artifacts(
        times,
        pred,
        ref,
        components,
        config,
        case_type="noip",
        reference_type="empymod",
        source_info={
            "mode": "manual_line",
            "projection_diagnostics": {
                "applied": True,
                "before_residual": 2.0,
                "after_residual": 1.0e-9,
                "endpoint_norm": 1.4,
            },
        },
        receiver_diagnostic_rows=[
            {"time_obs": 1.0e-5, "receiver_type": "point", "radius": 0.0, "Ex": 1.0, "Ey": 0.0, "Hz": np.nan, "dBzdt": 2.0},
            {
                "time_obs": 1.0e-5,
                "receiver_type": "disk_average",
                "radius": 2.0,
                "Ex": 1.2,
                "Ey": 0.0,
                "Hz": np.nan,
                "dBzdt": 3.0,
            },
        ],
    )

    required = [
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "error_summary.json",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
    ]
    for name in required:
        assert (tmp_path / name).is_file(), name

    with (tmp_path / "errors.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) >= {
        "time_obs",
        "component",
        "pred",
        "ref",
        "abs_error",
        "ordinary_relative_error",
        "relative_error_with_floor",
        "peak_normalized_error",
        "pass_5pct",
    }

    report = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert report["case_type"] == "noip"
    assert report["reference_type"] == "empymod"
    assert report["magnetic_quantity"] == "dBzdt"
    assert report["pass_all_components"] is True
    assert summary["pass_all_components"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["source_consistency"]["source_endpoint_balance_residual"] == 1.0e-9
    assert diagnostics["source_projection"]["before_residual"] == 2.0
    assert diagnostics["source_projection"]["after_residual"] == 1.0e-9
    assert diagnostics["receiver_sampling"]["enabled"] is True
    assert diagnostics["receiver_sampling"]["comparisons"]["disk_average"]["dBzdt"]["max_relative_difference"] == 0.5
