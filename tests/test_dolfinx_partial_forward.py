from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_partial_forward_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_save_forward_partial_writes_completed_output_rows(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5, 2.0e-5], dtype=float)
    rows = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    solver_log = [
        {"step": 1, "time": 1.0e-5, "dt": 1.0e-5, "its": 10, "residual": 1.0e-8, "reason": 2},
        {"step": 2, "time": 2.0e-5, "dt": 1.0e-5, "its": 11, "residual": 2.0e-8, "reason": 2},
    ]

    sp._save_forward_partial(config, times, rows, ["Ex", "Ey", "dBzdt"], solver_log)

    data = np.load(config.forward_partial_npz(), allow_pickle=False)
    assert data["times"].tolist() == [1.0e-5, 2.0e-5]
    assert data["fem"].tolist() == rows
    assert [str(item) for item in data["components"]] == ["Ex", "Ey", "dBzdt"]
    assert data["solver_steps"].tolist() == [1, 2]
    assert data["solver_iterations"].tolist() == [10, 11]


def test_parse_receiver_diagnostic_types_accepts_comma_string():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver_diagnostic_types="point,disk_average")

    assert sp._parse_receiver_diagnostic_types(config) == ("point", "disk_average")


def test_save_forward_partial_writes_receiver_diagnostics(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5], dtype=float)
    rows = [[1.0, 2.0, 3.0]]
    diagnostics = [
        {
            "time_obs": 1.0e-5,
            "receiver_type": "point",
            "radius": 0.0,
            "Ex": 1.0,
            "Ey": 2.0,
            "Hz": np.nan,
            "dBzdt": 3.0,
        },
        {
            "time_obs": 1.0e-5,
            "receiver_type": "disk_average",
            "radius": 2.0,
            "Ex": 1.1,
            "Ey": 2.1,
            "Hz": np.nan,
            "dBzdt": 3.1,
        },
    ]

    sp._save_forward_partial(
        config,
        times,
        rows,
        ["Ex", "Ey", "dBzdt"],
        [],
        receiver_diagnostic_rows=diagnostics,
    )

    data = np.load(config.forward_partial_npz(), allow_pickle=False)
    assert [str(item) for item in data["receiver_diagnostic_types"]] == ["point", "disk_average"]
    np.testing.assert_allclose(data["receiver_diagnostic_times"], np.asarray([1.0e-5, 1.0e-5]))
    np.testing.assert_allclose(data["receiver_diagnostic_values"][:, [0, 1, 3]], np.asarray([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]))

    csv_text = config.receiver_diagnostics_csv().read_text(encoding="utf-8")
    assert "time_obs,receiver_type,radius,Ex,Ey,Hz,dBzdt" in csv_text
    assert "disk_average" in csv_text


def test_completed_return_times_follow_completed_rows_only():
    sp = _load_pipeline_module()
    return_times = np.asarray([1.0e-5, 2.0e-5, 4.0e-5], dtype=float)
    rows = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    completed = sp._completed_return_times(return_times, rows)

    assert completed.tolist() == [1.0e-5, 2.0e-5]


class _FakeVector:
    def __init__(self, values):
        self.array = np.asarray(values, dtype=float)


class _FakeFunction:
    def __init__(self, values):
        self.x = _FakeVector(values)


def test_forward_checkpoint_round_trips_state_without_pickle(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
    memories = [_FakeFunction([4.0, 5.0, 6.0])]
    rows = [[7.0, 8.0, 9.0]]
    solver_log = [
        {
            "step": 4,
            "time": 2.0e-5,
            "observation_time": 1.0e-5,
            "dt": 1.0e-5,
            "its": 12,
            "residual": 3.0e-9,
            "reason": 2,
            "is_output": True,
            "time_theta": 1.0,
        }
    ]
    receiver_diagnostics = [
        {
            "time_obs": 1.0e-5,
            "receiver_type": "volume_average",
            "radius": 2.0,
            "Ex": 1.0,
            "Ey": 2.0,
            "Hz": np.nan,
            "dBzdt": 3.0,
        }
    ]

    sp._save_forward_checkpoint(
        config,
        completed_step=4,
        previous_time=2.0e-5,
        E_old=e_old,
        memories=memories,
        rows=rows,
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
        h_old_receiver=np.asarray([10.0, 11.0, 12.0]),
        receiver_diagnostic_rows=receiver_diagnostics,
    )

    loaded = sp._load_forward_checkpoint(config)

    assert loaded["completed_step"] == 4
    assert loaded["previous_time"] == 2.0e-5
    np.testing.assert_allclose(loaded["e_old"], np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(loaded["memories"], np.asarray([[4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(loaded["rows"], np.asarray(rows))
    assert loaded["components"] == ["Ex", "Ey", "dBzdt"]
    assert loaded["solver_log"][0]["step"] == 4
    assert loaded["solver_log"][0]["is_output"] is True
    np.testing.assert_allclose(loaded["h_old_receiver"], np.asarray([10.0, 11.0, 12.0]))
    assert loaded["receiver_diagnostic_rows"][0]["receiver_type"] == "volume_average"
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt"] == 3.0
