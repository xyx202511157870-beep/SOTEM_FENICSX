from __future__ import annotations

import importlib.util
import json
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
    assert int(data["producer_schema_version"].item()) == sp._FORWARD_ARTIFACT_SCHEMA_VERSION
    assert int(data["producer_nedelec_order"].item()) == 1
    assert int(data["producer_curl_degree"].item()) == 0
    assert str(data["producer_dbdt_observation_schema"].item()) == sp._dbdt_observation_schema(config)
    observation_schema = json.loads(str(data["producer_dbdt_observation_schema"].item()))
    assert observation_schema["quantity"] == "dBdt"
    assert observation_schema["operator"] == "minus_curl_E"
    assert observation_schema["observation_space"] == {
        "degree": 0,
        "family": "DG",
        "shape": [3],
    }


def _rewrite_npz(path, transform):
    with np.load(path, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]) for key in payload.files}
    transform(values)
    np.savez(path, **values)


def test_legacy_forward_partial_without_producer_identity_fails_closed(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    sp._save_forward_partial(config, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])
    _rewrite_npz(
        config.forward_partial_npz(),
        lambda values: [values.pop(key) for key in sp._FORWARD_ARTIFACT_IDENTITY_KEYS],
    )

    with pytest.raises(ValueError, match="missing producer identity"):
        sp._load_forward_partial(config)


def test_forward_partial_rejects_nedelec_order_mismatch(tmp_path):
    sp = _load_pipeline_module()
    saved = sp.PipelineConfig(workdir=tmp_path, nedelec_order=2)
    sp._save_forward_partial(saved, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])

    with pytest.raises(ValueError, match="producer identity mismatch"):
        sp._load_forward_partial(sp.PipelineConfig(workdir=tmp_path, nedelec_order=1))


def test_forward_partial_rejects_old_producer_schema_version(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    sp._save_forward_partial(config, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])
    _rewrite_npz(
        config.forward_partial_npz(),
        lambda values: values.__setitem__("producer_schema_version", np.asarray(0)),
    )

    with pytest.raises(ValueError, match="producer identity mismatch"):
        sp._load_forward_partial(config)


def test_forward_partial_rejects_curl_degree_mismatch(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, nedelec_order=2)
    sp._save_forward_partial(config, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])
    _rewrite_npz(
        config.forward_partial_npz(),
        lambda values: values.__setitem__("producer_curl_degree", np.asarray(0)),
    )

    with pytest.raises(ValueError, match="producer identity mismatch"):
        sp._load_forward_partial(config)


@pytest.mark.parametrize(
    "changed_config",
    [
        {"rho_earth": 250.0},
        {"source_current": 3.0},
        {"receiver": (125.0, -300.0, -0.1)},
        {"ramp_off_time": 2.0e-5},
        {"outer_boundary_mode": "natural"},
        {"polarization": "cole-cole"},
        {"empymod_srcpts": 17},
    ],
)
def test_forward_partial_rejects_any_resolved_forward_identity_change(
    tmp_path, changed_config
):
    sp = _load_pipeline_module()
    saved = sp.PipelineConfig(workdir=tmp_path)
    sp._save_forward_partial(saved, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])

    resumed = sp.PipelineConfig(workdir=tmp_path, **changed_config)
    with pytest.raises(ValueError, match="producer_forward_config_fingerprint"):
        sp._load_forward_partial(resumed)


def test_forward_partial_rejects_mesh_content_change(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    config.mesh_path().write_bytes(b"mesh-v1")
    config.dolfinx_mesh_path().write_bytes(b"companion-v1")
    sp._save_forward_partial(config, [1.0e-5], [[1.0, 2.0]], ["Ex", "dBzdt"], [])

    config.dolfinx_mesh_path().write_bytes(b"companion-v2")
    with pytest.raises(ValueError, match="producer_forward_config_fingerprint"):
        sp._load_forward_partial(config)


def test_forward_fingerprint_ignores_runtime_and_output_control_fields(tmp_path):
    sp = _load_pipeline_module()
    baseline = sp.PipelineConfig(workdir=tmp_path)
    runtime_variant = sp.PipelineConfig(
        workdir=tmp_path / "other-output",
        force_mesh=True,
        checkpoint_forward=True,
        resume_forward=True,
        stop_after_outputs=3,
        source_only=True,
        memory_limit_gb=12.0,
        memory_safety_fraction=0.8,
    )

    assert sp._forward_config_fingerprint(baseline) == sp._forward_config_fingerprint(
        runtime_variant
    )


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

    csv_text = config.receiver_diagnostics_csv().read_text(encoding="utf-8")
    assert "time_obs,receiver_type,radius,Ex,Ey,Hz,dBzdt,dBzdt_curl,dBzdt_biot_rate" in csv_text
    assert "disk_average" in csv_text
    assert config.receiver_diagnostics_png().is_file()
    assert config.receiver_diagnostics_png().stat().st_size > 0

    loaded = sp._load_forward_partial(config)
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt_curl"] == pytest.approx(3.0)
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt_biot_rate"] == pytest.approx(2.8)


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


def _checkpoint_schedule(sp, config):
    observation_times = np.asarray([1.0e-5, 2.0e-5, 4.0e-5], dtype=float)
    return sp._forward_observation_schedule(observation_times, config)


def test_forward_checkpoint_round_trips_state_without_pickle(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, config)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
    memories = [_FakeFunction([4.0, 5.0, 6.0])]
    rows = [[7.0, 8.0, 9.0]]
    solver_log = [
        {
            "step": 0,
            "time": 1.0e-5,
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
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=e_old,
        memories=memories,
        rows=rows,
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
        h_old_receiver=np.asarray([10.0, 11.0, 12.0]),
        receiver_diagnostic_rows=receiver_diagnostics,
    )

    loaded = sp._load_forward_checkpoint(config, schedule=schedule)

    assert loaded["completed_step"] == 0
    assert loaded["previous_time"] == 1.0e-5
    np.testing.assert_allclose(loaded["e_old"], np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(loaded["memories"], np.asarray([[4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(loaded["rows"], np.asarray(rows))
    assert loaded["components"] == ["Ex", "Ey", "dBzdt"]
    assert loaded["solver_log"][0]["step"] == 0
    assert loaded["solver_log"][0]["is_output"] is True
    np.testing.assert_allclose(loaded["h_old_receiver"], np.asarray([10.0, 11.0, 12.0]))
    assert loaded["receiver_diagnostic_rows"][0]["receiver_type"] == "volume_average"
    assert loaded["receiver_diagnostic_rows"][0]["dBzdt"] == 3.0
    with np.load(config.forward_checkpoint_npz(), allow_pickle=False) as payload:
        assert str(payload["schedule_time_origin"].item()) == "after_ramp"
        assert float(payload["schedule_ramp_off_time"].item()) == 0.0
        assert int(payload["schedule_output_interval_substeps"].item()) == 1
        np.testing.assert_array_equal(
            payload["schedule_observation_times"], schedule["observation_times"]
        )
        np.testing.assert_array_equal(payload["schedule_return_times"], schedule["return_times"])
        np.testing.assert_array_equal(payload["schedule_step_times"], schedule["step_times"])
        np.testing.assert_array_equal(
            payload["schedule_output_step_indices"], schedule["output_step_indices"]
        )
        assert int(payload["producer_schema_version"].item()) == sp._FORWARD_ARTIFACT_SCHEMA_VERSION
        assert int(payload["producer_nedelec_order"].item()) == 1
        assert int(payload["producer_curl_degree"].item()) == 0
        assert str(payload["producer_dbdt_observation_schema"].item()) == sp._dbdt_observation_schema(config)


def test_forward_checkpoint_without_producer_identity_fails_closed(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, config)
    sp._save_forward_checkpoint(
        config,
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=_FakeFunction([1.0, 2.0, 3.0]),
        memories=[],
        rows=[[7.0, 8.0]],
        components=["Ex", "dBzdt"],
        solver_log=[],
    )
    _rewrite_npz(
        config.forward_checkpoint_npz(),
        lambda values: [values.pop(key) for key in sp._FORWARD_ARTIFACT_IDENTITY_KEYS],
    )

    with pytest.raises(ValueError, match="missing producer identity"):
        sp._load_forward_checkpoint(config, schedule=schedule)


def test_forward_checkpoint_rejects_dbdt_observation_schema_mismatch(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, config)
    sp._save_forward_checkpoint(
        config,
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=_FakeFunction([1.0, 2.0, 3.0]),
        memories=[],
        rows=[[7.0, 8.0]],
        components=["Ex", "dBzdt"],
        solver_log=[],
    )
    _rewrite_npz(
        config.forward_checkpoint_npz(),
        lambda values: values.__setitem__(
            "producer_dbdt_observation_schema",
            np.asarray("legacy-minus-curl-E-vector-DG0"),
        ),
    )

    with pytest.raises(ValueError, match="producer identity mismatch"):
        sp._load_forward_checkpoint(config, schedule=schedule)


@pytest.mark.parametrize(
    "changed_config",
    [
        {"rho_earth": 250.0},
        {"source_current": 3.0},
        {"receiver": (125.0, -300.0, -0.1)},
        {"time_method": "bdf2"},
    ],
)
def test_forward_checkpoint_rejects_resolved_forward_fingerprint_change(
    tmp_path, changed_config
):
    sp = _load_pipeline_module()
    saved = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, saved)
    sp._save_forward_checkpoint(
        saved,
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=_FakeFunction([1.0, 2.0, 3.0]),
        memories=[],
        rows=[[7.0, 8.0]],
        components=["Ex", "dBzdt"],
        solver_log=[],
    )
    resumed = sp.PipelineConfig(
        workdir=tmp_path,
        ramp_off_time=0.0,
        **changed_config,
    )

    with pytest.raises(ValueError, match="producer_forward_config_fingerprint"):
        sp._load_forward_checkpoint(resumed, schedule=_checkpoint_schedule(sp, resumed))


def test_forward_checkpoint_rejects_resume_with_different_time_level(tmp_path):
    sp = _load_pipeline_module()
    saved_config = sp.PipelineConfig(
        workdir=tmp_path,
        ramp_off_time=0.0,
        output_interval_substeps=1,
    )
    saved_schedule = _checkpoint_schedule(sp, saved_config)
    sp._save_forward_checkpoint(
        saved_config,
        schedule=saved_schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=_FakeFunction([1.0, 2.0, 3.0]),
        memories=[],
        rows=[[7.0, 8.0, 9.0]],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=[],
    )
    resumed_config = sp.PipelineConfig(
        workdir=tmp_path,
        ramp_off_time=0.0,
        output_interval_substeps=4,
    )
    resumed_schedule = _checkpoint_schedule(sp, resumed_config)

    with pytest.raises(ValueError, match="producer_forward_config_fingerprint"):
        sp._load_forward_checkpoint(resumed_config, schedule=resumed_schedule)


def test_legacy_forward_checkpoint_without_producer_identity_fails_closed(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    np.savez(
        config.forward_checkpoint_npz(),
        completed_step=np.asarray(0),
        previous_time=np.asarray(1.0e-5),
        e_old=np.asarray([1.0, 2.0, 3.0]),
        memories=np.empty((0, 3)),
        rows=np.asarray([[7.0, 8.0, 9.0]]),
        components=np.asarray(["Ex", "Ey", "dBzdt"]),
    )

    with pytest.raises(ValueError, match="missing producer identity"):
        sp._load_forward_checkpoint(config, schedule=_checkpoint_schedule(sp, config))


def test_forward_checkpoint_round_trips_divergence_cleaning_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, config)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
    solver_log = [
        {
            "step": 0,
            "time": 1.0e-5,
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
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=e_old,
        memories=[],
        rows=[[7.0, 8.0, 9.0]],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
    )

    loaded = sp._load_forward_checkpoint(config, schedule=schedule)
    item = loaded["solver_log"][0]
    assert item["divergence_clean_before"] == 8.0
    assert item["divergence_clean_after"] == 1.0e-9
    assert item["divergence_clean_correction_norm"] == 0.25
    assert item["divergence_clean_applied_correction_norm"] == 0.125
    assert item["divergence_clean_strength"] == 0.5


def test_forward_checkpoint_round_trips_divergence_control_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, ramp_off_time=0.0)
    schedule = _checkpoint_schedule(sp, config)
    e_old = _FakeFunction([1.0, 2.0, 3.0])
    solver_log = [
        {
            "step": 0,
            "time": 1.0e-5,
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
        schedule=schedule,
        completed_step=0,
        previous_time=1.0e-5,
        E_old=e_old,
        memories=[],
        rows=[[7.0, 8.0, 9.0]],
        components=["Ex", "Ey", "dBzdt"],
        solver_log=solver_log,
    )

    loaded = sp._load_forward_checkpoint(config, schedule=schedule)
    item = loaded["solver_log"][0]
    assert item["divergence_control_applied"] is True
    assert item["divergence_control_scale"] == "lhs"
    assert item["divergence_control_weight"] == 0.25
    assert item["divergence_control_applied_weight"] == 1.075
    assert item["divergence_control_reference_norm"] == 43.0
    assert item["divergence_control_matrix_norm"] == 10.0
    assert item["divergence_control_relative_weight"] == 0.25
