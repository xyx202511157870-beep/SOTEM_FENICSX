from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


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


def test_save_forward_partial_keeps_full_internal_time_grid_separate_from_output_rows(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5], dtype=float)
    rows = [[1.0, 2.0, 3.0]]
    solver_log = [
        {"step": 0, "time": 1.0e-6, "dt": 1.0e-6, "its": 8, "residual": 4.0e-8, "reason": 2, "is_output": False},
        {"step": 1, "time": 2.0e-6, "dt": 1.0e-6, "its": 9, "residual": 3.0e-8, "reason": 2, "is_output": False},
        {
            "step": 2,
            "time": 2.0e-5,
            "observation_time": 1.0e-5,
            "dt": 1.8e-5,
            "its": 10,
            "residual": 2.0e-8,
            "reason": 2,
            "is_output": True,
        },
    ]

    sp._save_forward_partial(config, times, rows, ["Ex", "Ey", "dBzdt"], solver_log)

    data = np.load(config.forward_partial_npz(), allow_pickle=False)
    assert data["solver_steps"].tolist() == [2]
    assert data["internal_solver_steps"].tolist() == [0, 1, 2]
    assert data["internal_solver_times"].tolist() == [1.0e-6, 2.0e-6, 2.0e-5]
    assert data["internal_solver_dt"].tolist() == [1.0e-6, 1.0e-6, 1.8e-5]
    assert data["internal_solver_is_output"].tolist() == [False, False, True]


def test_load_forward_partial_restores_full_internal_solver_log(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5], dtype=float)
    rows = [[1.0, 2.0, 3.0]]
    solver_log = [
        {"step": 0, "time": 1.0e-6, "dt": 1.0e-6, "its": 8, "residual": 4.0e-8, "reason": 2, "is_output": False},
        {
            "step": 1,
            "time": 2.0e-5,
            "observation_time": 1.0e-5,
            "dt": 1.9e-5,
            "its": 10,
            "residual": 2.0e-8,
            "reason": 2,
            "is_output": True,
        },
    ]

    sp._save_forward_partial(config, times, rows, ["Ex", "Ey", "dBzdt"], solver_log)
    loaded = sp._load_forward_partial(config)

    assert [item["step"] for item in loaded["internal_solver_log"]] == [0, 1]
    assert [item["time"] for item in loaded["internal_solver_log"]] == [1.0e-6, 2.0e-5]
    assert [item["is_output"] for item in loaded["internal_solver_log"]] == [False, True]


def test_solver_internal_time_grid_summary_verifies_after_ramp_outputs():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        formulation="h",
        time_origin="after_ramp",
        ramp_off_time=1.0e-5,
    )
    observation_times = np.asarray([1.0e-5, 2.0e-5], dtype=float)
    solver_log = [
        {"step": 0, "time": 1.0e-6, "dt": 1.0e-6, "is_output": False},
        {"step": 1, "time": 1.0e-5, "dt": 9.0e-6, "is_output": False},
        {"step": 2, "time": 2.0e-5, "dt": 1.0e-5, "is_output": True},
        {"step": 3, "time": 3.0e-5, "dt": 1.0e-5, "is_output": True},
    ]

    summary = sp._solver_internal_time_grid_summary(solver_log, observation_times, config)

    assert summary["contains_turnoff_start"] is True
    assert summary["turnoff_start_source"] == "initial_static_or_dc_state"
    assert summary["contains_turnoff_end"] is True
    assert summary["contains_all_observation_outputs"] is True
    assert summary["last_output_internal_time_s"] == pytest.approx(3.0e-5)


def test_save_forward_partial_round_trips_divergence_cleaning_output_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5], dtype=float)
    rows = [[1.0, 2.0, 3.0]]
    solver_log = [
        {
            "step": 4,
            "time": 2.0e-5,
            "observation_time": 1.0e-5,
            "dt": 5.0e-6,
            "its": 12,
            "residual": 3.0e-9,
            "reason": 2,
            "is_output": True,
            "divergence_clean_before": 8.0,
            "divergence_clean_after": 1.0e-9,
            "divergence_clean_correction_norm": 0.25,
            "divergence_clean_applied_correction_norm": 0.125,
            "divergence_clean_strength": 0.5,
        }
    ]

    sp._save_forward_partial(config, times, rows, ["Ex", "Ey", "dBzdt"], solver_log)
    loaded = sp._load_forward_partial(config)

    item = loaded["solver_log"][0]
    assert item["divergence_clean_before"] == 8.0
    assert item["divergence_clean_after"] == 1.0e-9
    assert item["divergence_clean_correction_norm"] == 0.25
    assert item["divergence_clean_applied_correction_norm"] == 0.125
    assert item["divergence_clean_strength"] == 0.5


def test_save_forward_partial_round_trips_divergence_control_output_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5], dtype=float)
    rows = [[1.0, 2.0, 3.0]]
    solver_log = [
        {
            "step": 4,
            "time": 2.0e-5,
            "observation_time": 1.0e-5,
            "dt": 5.0e-6,
            "its": 12,
            "residual": 3.0e-9,
            "reason": 2,
            "is_output": True,
            "divergence_control_applied": True,
            "divergence_control_scale": "lhs",
            "divergence_control_weight": 0.25,
            "divergence_control_applied_weight": 1.075,
            "divergence_control_reference_norm": 43.0,
            "divergence_control_matrix_norm": 10.0,
            "divergence_control_relative_weight": 0.25,
        }
    ]

    sp._save_forward_partial(config, times, rows, ["Ex", "Ey", "dBzdt"], solver_log)
    loaded = sp._load_forward_partial(config)

    item = loaded["solver_log"][0]
    assert item["divergence_control_applied"] is True
    assert item["divergence_control_scale"] == "lhs"
    assert item["divergence_control_weight"] == 0.25
    assert item["divergence_control_applied_weight"] == 1.075
    assert item["divergence_control_reference_norm"] == 43.0
    assert item["divergence_control_matrix_norm"] == 10.0
    assert item["divergence_control_relative_weight"] == 0.25


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
            "dBzdt_curl": 3.0,
            "dBzdt_biot_rate": 2.8,
            "dBzdt_ampere_rate": 2.7,
            "local_lsq_earth_sample_count_mean": 12.0,
            "local_lsq_air_sample_count_mean": 8.0,
            "local_lsq_earth_value_z_mean": 3.2,
            "local_lsq_air_value_z_mean": 2.9,
        },
        {
            "time_obs": 1.0e-5,
            "receiver_type": "disk_average",
            "radius": 2.0,
            "Ex": 1.1,
            "Ey": 2.1,
            "Hz": np.nan,
            "dBzdt": 3.1,
            "dBzdt_curl": 3.1,
            "dBzdt_biot_rate": np.nan,
            "dBzdt_ampere_rate": np.nan,
            "local_lsq_earth_sample_count_mean": 10.0,
            "local_lsq_air_sample_count_mean": 6.0,
            "local_lsq_earth_value_z_mean": 3.3,
            "local_lsq_air_value_z_mean": 3.0,
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
    np.testing.assert_allclose(data["receiver_diagnostic_dbdt_curl"], np.asarray([3.0, 3.1]))
    np.testing.assert_allclose(data["receiver_diagnostic_dbdt_biot_rate"], np.asarray([2.8, np.nan]), equal_nan=True)
    np.testing.assert_allclose(data["receiver_diagnostic_dbdt_ampere_rate"], np.asarray([2.7, np.nan]), equal_nan=True)
    np.testing.assert_allclose(data["receiver_diagnostic_local_lsq_earth_sample_count_mean"], np.asarray([12.0, 10.0]))
    np.testing.assert_allclose(data["receiver_diagnostic_local_lsq_air_sample_count_mean"], np.asarray([8.0, 6.0]))
    np.testing.assert_allclose(data["receiver_diagnostic_local_lsq_earth_value_z_mean"], np.asarray([3.2, 3.3]))
    np.testing.assert_allclose(data["receiver_diagnostic_local_lsq_air_value_z_mean"], np.asarray([2.9, 3.0]))

    csv_text = config.receiver_diagnostics_csv().read_text(encoding="utf-8")
    assert "time_obs,receiver_type,radius,Ex,Ey,Hz,dBzdt,dBzdt_curl,dBzdt_biot_rate,dBzdt_ampere_rate" in csv_text
    assert "disk_average" in csv_text
    assert config.receiver_diagnostics_png().is_file()
    assert config.receiver_diagnostics_png().stat().st_size > 0

    loaded = sp._load_forward_partial(config)
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt_curl"] == pytest.approx(3.0)
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt_biot_rate"] == pytest.approx(2.8)
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt_ampere_rate"] == pytest.approx(2.7)
    assert loaded["receiver_diagnostic_rows"][0]["local_lsq_earth_sample_count_mean"] == pytest.approx(12.0)
    assert loaded["receiver_diagnostic_rows"][0]["local_lsq_air_sample_count_mean"] == pytest.approx(8.0)
    assert loaded["receiver_diagnostic_rows"][0]["local_lsq_earth_value_z_mean"] == pytest.approx(3.2)
    assert loaded["receiver_diagnostic_rows"][0]["local_lsq_air_value_z_mean"] == pytest.approx(2.9)


def test_h_receiver_diagnostic_row_preserves_candidate_metadata():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver_type="point", receiver_average_radius=3.0)
    rec = {
        "Ex": 1.0,
        "Ey": 2.0,
        "Hz": 3.0,
        "dBzdt": 4.0,
        "candidate_count_min": 2,
        "candidate_count_max": 5,
        "candidate_count_mean": 3.5,
        "multi_candidate_sample_count": 1,
        "candidate_center_z_min": -4.0,
        "candidate_center_z_max": 1.0,
        "local_lsq_earth_sample_count_mean": 12.0,
        "local_lsq_air_sample_count_mean": 8.0,
        "local_lsq_earth_value_z_mean": 3.2,
        "local_lsq_air_value_z_mean": 2.9,
    }

    row = sp._h_receiver_diagnostic_row(config, time_obs=1.0e-5, rec=rec)

    assert row["time_obs"] == pytest.approx(1.0e-5)
    assert row["receiver_type"] == "point"
    assert row["radius"] == pytest.approx(0.0)
    assert row["Ex"] == pytest.approx(1.0)
    assert row["Ey"] == pytest.approx(2.0)
    assert row["Hz"] == pytest.approx(3.0)
    assert row["dBzdt"] == pytest.approx(4.0)
    assert row["dBzdt_curl"] == pytest.approx(4.0)
    assert row["candidate_count_min"] == 2
    assert row["candidate_count_max"] == 5
    assert row["candidate_center_z_min"] == pytest.approx(-4.0)
    assert row["candidate_center_z_max"] == pytest.approx(1.0)
    assert row["local_lsq_earth_sample_count_mean"] == pytest.approx(12.0)
    assert row["local_lsq_air_sample_count_mean"] == pytest.approx(8.0)
    assert row["local_lsq_earth_value_z_mean"] == pytest.approx(3.2)
    assert row["local_lsq_air_value_z_mean"] == pytest.approx(2.9)


def test_receiver_diagnostic_summary_quantifies_point_average_difference():
    sp = _load_pipeline_module()
    rows = [
        {"time_obs": 1.0, "receiver_type": "point", "radius": 0.0, "Ex": 10.0, "Ey": 1.0, "Hz": np.nan, "dBzdt": 4.0},
        {"time_obs": 1.0, "receiver_type": "disk_average", "radius": 2.0, "Ex": 11.0, "Ey": 1.2, "Hz": np.nan, "dBzdt": 5.0},
        {"time_obs": 2.0, "receiver_type": "point", "radius": 0.0, "Ex": 20.0, "Ey": 2.0, "Hz": np.nan, "dBzdt": 8.0},
        {"time_obs": 2.0, "receiver_type": "disk_average", "radius": 2.0, "Ex": 22.0, "Ey": 2.4, "Hz": np.nan, "dBzdt": 10.0},
    ]

    summary = sp._receiver_diagnostic_summary(rows)

    assert summary["enabled"] is True
    assert summary["baseline_receiver_type"] == "point"
    assert summary["receiver_types"] == ["point", "disk_average"]
    assert summary["comparisons"]["disk_average"]["Ex"]["max_relative_difference"] == 0.1
    assert summary["comparisons"]["disk_average"]["Ey"]["max_relative_difference"] == 0.2
    assert summary["comparisons"]["disk_average"]["dBzdt"]["max_relative_difference"] == 0.25
    assert summary["receiver_sampling_issue_suspected"] is True


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
    magnetic_receiver_diagnostics = [
        {
            "receiver_id": "Rx3",
            "receiver_x_m": 0.0,
            "receiver_y_m": 0.0,
            "receiver_z_m": 0.1,
            "time_obs": 1.0e-5,
            "dBzdt_curl": 1.0,
            "dBzdt_biot_rate": 2.0,
            "dBzdt_faraday_loop": 3.0,
            "Hz_biot_center": 4.0,
            "Hz_biot_tetra4": 5.0,
            "faraday_audit": {"point_count": 32},
            "biot_tetra4_audit": {"cell_count": 7},
            "provenance": "explicit_full_domain",
        }
    ]

    sp._save_forward_checkpoint(
        config,
        completed_step=4,
        previous_time=2.0e-5,
        E_old=e_old,
        memories=memories,
        rows=rows,
        row_times=[1.0e-5],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
        h_old_receiver=np.asarray([10.0, 11.0, 12.0]),
        receiver_diagnostic_rows=receiver_diagnostics,
        magnetic_receiver_diagnostic_rows=magnetic_receiver_diagnostics,
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
    assert loaded["magnetic_receiver_diagnostic_rows"] == magnetic_receiver_diagnostics


def test_forward_checkpoint_round_trips_divergence_cleaning_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
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
            "divergence_clean_before": 8.0,
            "divergence_clean_after": 1.0e-9,
            "divergence_clean_correction_norm": 0.25,
            "divergence_clean_applied_correction_norm": 0.125,
            "divergence_clean_strength": 0.5,
        }
    ]

    sp._save_forward_checkpoint(
        config,
        completed_step=4,
        previous_time=2.0e-5,
        E_old=e_old,
        memories=[],
        rows=[[7.0, 8.0, 9.0]],
        row_times=[1.0e-5],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
    )

    loaded = sp._load_forward_checkpoint(config)
    item = loaded["solver_log"][0]
    assert item["divergence_clean_before"] == 8.0
    assert item["divergence_clean_after"] == 1.0e-9
    assert item["divergence_clean_correction_norm"] == 0.25
    assert item["divergence_clean_applied_correction_norm"] == 0.125
    assert item["divergence_clean_strength"] == 0.5


def test_forward_checkpoint_round_trips_divergence_control_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
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
            "divergence_control_applied": True,
            "divergence_control_scale": "lhs",
            "divergence_control_weight": 0.25,
            "divergence_control_applied_weight": 1.075,
            "divergence_control_reference_norm": 43.0,
            "divergence_control_matrix_norm": 10.0,
            "divergence_control_relative_weight": 0.25,
        }
    ]

    sp._save_forward_checkpoint(
        config,
        completed_step=4,
        previous_time=2.0e-5,
        E_old=e_old,
        memories=[],
        rows=[[7.0, 8.0, 9.0]],
        row_times=[1.0e-5],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
    )

    loaded = sp._load_forward_checkpoint(config)
    item = loaded["solver_log"][0]
    assert item["divergence_control_applied"] is True
    assert item["divergence_control_scale"] == "lhs"
    assert item["divergence_control_weight"] == 0.25
    assert item["divergence_control_applied_weight"] == 1.075
    assert item["divergence_control_reference_norm"] == 43.0
    assert item["divergence_control_matrix_norm"] == 10.0
    assert item["divergence_control_relative_weight"] == 0.25
