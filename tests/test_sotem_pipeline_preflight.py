import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_pipeline():
    path = Path(__file__).resolve().parents[1] / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_preflight_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_receiver_location_preflight_requires_every_sampling_point(monkeypatch):
    pipeline = _load_pipeline()
    sample_points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    monkeypatch.setattr(
        pipeline,
        "_receiver_sampling_points",
        lambda config: sample_points,
    )
    monkeypatch.setattr(
        pipeline,
        "_find_cells_for_point",
        lambda msh, point: [4] if float(point[0]) == 0.0 else [],
    )

    result = pipeline._receiver_location_preflight(
        object(),
        pipeline.PipelineConfig(),
    )

    assert result == {
        "found": False,
        "sample_count": 2,
        "found_sample_count": 1,
        "missing_sample_indices": [1],
    }
