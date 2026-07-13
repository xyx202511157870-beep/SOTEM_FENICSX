import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_pipeline():
    path = Path(__file__).resolve().parents[1] / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_forward_schedule_inserts_internal_substeps_without_extra_outputs():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        min_steps_during_turnoff=2,
        min_steps_before_first_observation=1,
        max_internal_dt=0.25,
    )
    times = np.asarray([0.1, 1.0])

    schedule = sp._forward_observation_schedule(times, config)

    assert np.allclose(schedule["return_times"], times)
    assert np.allclose(schedule["output_internal_times"], times + config.ramp_off_time)
    assert len(schedule["output_step_indices"]) == len(times)
    assert np.max(np.diff(schedule["step_times"])) <= 0.25 * (1.0 + 1.0e-12)
    assert len(schedule["step_times"]) > len(times) + config.min_steps_during_turnoff


def test_forward_schedule_can_limit_internal_substeps_by_relative_observation_time():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        min_steps_during_turnoff=2,
        min_steps_before_first_observation=1,
        max_internal_dt=0.0,
        max_internal_dt_fraction=0.1,
    )
    times = np.asarray([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(times, config)
    step_times = schedule["step_times"]

    assert len(schedule["output_step_indices"]) == len(times)
    assert np.allclose(schedule["output_internal_times"], times + config.ramp_off_time)

    second_window = step_times[(step_times >= 2.0e-5) & (step_times <= 1.1e-4)]
    third_window = step_times[(step_times >= 1.1e-4) & (step_times <= 1.01e-3)]
    assert np.max(np.diff(second_window)) <= 1.0e-5 * (1.0 + 1.0e-12)
    assert np.max(np.diff(third_window)) <= 1.0e-4 * (1.0 + 1.0e-12)
    assert len(step_times) > len(times) + config.min_steps_during_turnoff


def test_relative_internal_substep_limit_starts_after_first_observation():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        min_steps_during_turnoff=2,
        min_steps_before_first_observation=3,
        max_internal_dt=0.0,
        max_internal_dt_fraction=0.01,
    )
    times = np.asarray([1.0e-5, 1.0e-4])

    schedule = sp._forward_observation_schedule(times, config)
    step_times = schedule["step_times"]
    first_output = config.ramp_off_time + times[0]
    pre_first_observation = step_times[
        (step_times > config.ramp_off_time) & (step_times < first_output)
    ]

    assert pre_first_observation.size == config.min_steps_before_first_observation - 1
    np.testing.assert_allclose(
        pre_first_observation,
        np.linspace(config.ramp_off_time, first_output, config.min_steps_before_first_observation + 1)[
            1:-1
        ],
    )


def test_relative_internal_substep_limit_can_stop_after_configured_observation_time():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        min_steps_during_turnoff=2,
        min_steps_before_first_observation=1,
        max_internal_dt=5.0e-5,
        max_internal_dt_fraction=0.01,
        max_internal_dt_fraction_until=1.0e-4,
    )
    times = np.asarray([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(times, config)
    step_times = schedule["step_times"]

    second_window = step_times[(step_times >= 2.0e-5) & (step_times <= 1.1e-4)]
    third_window = step_times[(step_times >= 1.1e-4) & (step_times <= 1.01e-3)]

    assert np.max(np.diff(second_window)) <= 1.0e-6 * (1.0 + 1.0e-12)
    assert np.max(np.diff(third_window)) <= 5.0e-5 * (1.0 + 1.0e-12)
    assert np.max(np.diff(third_window)) > 1.0e-5


def test_h_form_after_ramp_schedule_auto_substeps_each_observation_interval():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        formulation="h",
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        min_steps_during_turnoff=10,
        min_steps_before_first_observation=1,
        max_internal_dt=0.0,
    )
    times = np.asarray([1.0e-5, 1.0e-4])

    schedule = sp._forward_observation_schedule(times, config)
    step_times = schedule["step_times"]

    first_window = step_times[(step_times > 1.0e-5) & (step_times <= 2.0e-5)]
    second_window = step_times[(step_times > 2.0e-5) & (step_times <= 1.1e-4)]
    assert first_window.size >= 10
    assert second_window.size >= 10
    assert np.max(np.diff(first_window)) <= 1.0e-6 * (1.0 + 1.0e-12)
    assert np.max(np.diff(second_window)) <= 9.0e-6 * (1.0 + 1.0e-12)
    assert schedule["output_step_indices"] == [
        int(np.flatnonzero(np.isclose(step_times, 2.0e-5))[0]),
        int(np.flatnonzero(np.isclose(step_times, 1.1e-4))[0]),
    ]


def test_generate_time_array_uses_explicit_observation_times_when_configured():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0,
        explicit_observation_times=(1.0e-5, 1.2e-2, 4.7e-1, 1.0),
    )

    np.testing.assert_allclose(
        sp.generate_time_array(config),
        np.asarray([1.0e-5, 1.2e-2, 4.7e-1, 1.0]),
    )


def test_generate_time_array_deduplicates_floating_point_tmax_hit():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0e-2,
        time_growth=1.258925411794167,
    )

    times = sp.generate_time_array(config)

    assert times[-1] == pytest.approx(config.t_max)
    assert np.all(np.diff(times) > 0.0)
    assert not np.isclose(times[-2], config.t_max, rtol=1.0e-12, atol=1.0e-30)


def test_generate_time_array_clamps_truncated_growth_near_tmax():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0e-2,
        time_growth=1.25892541179,
    )

    times = sp.generate_time_array(config)

    assert times.size == 31
    assert times[-1] == config.t_max
    assert np.diff(times)[-1] > 0.5 * np.diff(times)[-2]


def test_explicit_observation_times_must_cover_requested_window():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0,
        explicit_observation_times=(1.0e-5, 1.2e-2, 4.7e-1),
    )

    with pytest.raises(ValueError, match="cover t_min and t_max"):
        sp.validate_model_consistency(config)


def test_generate_verification_mesh_can_reuse_locked_mesh_file(tmp_path, monkeypatch):
    sp = load_pipeline()
    source_mesh = tmp_path / "locked_source.msh"
    source_mesh.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    preflight_paths = []

    def fake_preflight(config, path):
        preflight_paths.append((config, Path(path)))

    monkeypatch.setattr(sp, "_mesh_memory_preflight_for_path", fake_preflight)
    config = sp.PipelineConfig(workdir=run_dir, mesh_source_path=source_mesh)

    mesh_path = sp.generate_verification_mesh(config)

    assert mesh_path == run_dir / "verification_mesh.msh"
    assert mesh_path.read_text(encoding="utf-8") == source_mesh.read_text(encoding="utf-8")
    assert preflight_paths == [(config, mesh_path)]


def test_record_timing_event_appends_jsonl(tmp_path):
    sp = load_pipeline()
    config = sp.PipelineConfig(workdir=tmp_path)

    sp._record_timing_event(config, "mesh_start", 10.0, now=12.5, detail="locked")
    sp._record_timing_event(config, "mesh_done", 10.0, now=13.0, seconds=0.5)

    lines = (tmp_path / "timing_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    import json

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event"] == "mesh_start"
    assert first["elapsed_seconds"] == 2.5
    assert first["detail"] == "locked"
    assert second["event"] == "mesh_done"
    assert second["seconds"] == 0.5


def test_timed_stage_records_start_and_done_events(tmp_path, monkeypatch):
    sp = load_pipeline()
    config = sp.PipelineConfig(workdir=tmp_path)
    ticks = iter([20.0, 23.5])
    monkeypatch.setattr(sp.time, "perf_counter", lambda: next(ticks))

    with sp._timed_stage(config, "forward_assemble_operators", 10.0, cells=7):
        pass

    import json

    events = [
        json.loads(line)
        for line in (tmp_path / "timing_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "forward_assemble_operators_start",
        "forward_assemble_operators_done",
    ]
    assert events[0]["elapsed_seconds"] == 10.0
    assert events[0]["cells"] == 7
    assert events[1]["elapsed_seconds"] == 13.5
    assert events[1]["seconds"] == 3.5


def test_resume_start_step_uses_previous_time_when_schedule_changes():
    sp = load_pipeline()
    step_times = np.asarray([0.1, 0.2, 0.25, 0.3, 0.4])

    start = sp._resume_start_step_from_time(step_times, previous_time=0.2, completed_step=10)

    assert start == 2


def test_empymod_reference_averages_disk_receiver_samples(monkeypatch):
    sp = load_pipeline()
    calls = []

    class FakeEmpymod:
        @staticmethod
        def bipole(*, rec, freqtime, **_kwargs):
            calls.append(tuple(rec))
            return np.full_like(np.asarray(freqtime, dtype=float), float(rec[0]) ** 2)

    monkeypatch.setitem(sys.modules, "empymod", FakeEmpymod)
    config = sp.PipelineConfig(
        receiver=(10.0, 20.0, -0.1),
        receiver_type="disk_average",
        receiver_average_radius=2.0,
        ramp_off_time=0.0,
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )

    reference = sp.get_empymod_reference(np.asarray([1.0, 2.0]), config, mode="noip")

    assert len(calls) == 5 * len(reference["components"])
    assert np.allclose(reference["data"][:, 0], (10.0**2 + 12.0**2 + 8.0**2 + 10.0**2 + 10.0**2) / 5.0)


def test_primary_secondary_empymod_primary_can_use_analytic_dc_runner_when_requested(monkeypatch):
    sp = load_pipeline()
    import atem3d.primary as primary_module

    captured = {}

    class FakeEmpymodPrimaryProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(primary_module, "EmpymodPrimaryProvider", FakeEmpymodPrimaryProvider)
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered",
        primary_secondary_dc_mode="analytic_halfspace",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_rho0=33.33333333333334,
        cole_layer_top=100.0,
        empymod_srcpts=11,
    )

    sp._make_pipeline_empymod_primary_provider(config)

    assert captured["dc_runner"] is None
    assert captured["dc_kwargs"] is None


def test_primary_secondary_empymod_primary_can_use_empymod_dc_runner(monkeypatch):
    sp = load_pipeline()
    import atem3d.primary as primary_module

    captured = {}

    class FakeEmpymodPrimaryProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(primary_module, "EmpymodPrimaryProvider", FakeEmpymodPrimaryProvider)
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered",
        primary_secondary_dc_mode="empymod_quasistatic",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_rho0=33.33333333333334,
        cole_layer_top=100.0,
        empymod_srcpts=11,
    )

    sp._make_pipeline_empymod_primary_provider(config)

    assert captured["dc_runner"].__name__ == "empymod_quasistatic_dc_runner"
    assert captured["dc_kwargs"]["frequency"] == pytest.approx(1.0e-9)
    assert captured["dc_kwargs"]["empymod_kwargs"]["srcpts"] == 11


def test_primary_secondary_layered_background_defaults_to_analytic_dc_runner(monkeypatch):
    sp = load_pipeline()
    import atem3d.primary as primary_module

    captured = {}

    class FakeEmpymodPrimaryProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(primary_module, "EmpymodPrimaryProvider", FakeEmpymodPrimaryProvider)
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_rho0=33.33333333333334,
        cole_layer_top=100.0,
        empymod_srcpts=11,
    )

    sp._make_pipeline_empymod_primary_provider(config)

    assert captured["dc_runner"] is None
    assert captured["dc_kwargs"] is None


def test_masked_primary_provider_prefetches_receiver_reference_for_all_times(monkeypatch):
    sp = load_pipeline()
    calls = []

    def fake_get_empymod_reference(times, config, mode):
        calls.append((np.asarray(times, dtype=float).copy(), tuple(config.receiver), mode))
        return {
            "times": np.asarray(times, dtype=float),
            "components": ["Ex", "Ey", "dBzdt"],
            "data": np.column_stack(
                (
                    10.0 * np.asarray(times, dtype=float),
                    20.0 * np.asarray(times, dtype=float),
                    -30.0 * np.asarray(times, dtype=float),
                )
            ),
        }

    monkeypatch.setattr(sp, "get_empymod_reference", fake_get_empymod_reference)
    provider = sp._MaskedPrimaryProvider(base=object(), config=sp.PipelineConfig())
    times = np.asarray([1.0e-5, 2.0e-5, 3.0e-5])
    receivers = np.asarray([[0.0, -300.0, -0.1]])

    provider.prepare_receiver_reference_cache(times, receivers)
    np.testing.assert_allclose(provider.get_receiver_E(2.0e-5, receivers), [[2.0e-4, 4.0e-4, 0.0]])
    np.testing.assert_allclose(provider.get_receiver_dBdt(3.0e-5, receivers), [[0.0, 0.0, -9.0e-4]])

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], times)
    assert calls[0][2] == "noip"


def test_masked_primary_provider_prefetches_cole_reference_for_layered_ip(monkeypatch):
    sp = load_pipeline()
    calls = []

    def fake_get_empymod_reference(times, config, mode):
        calls.append((np.asarray(times, dtype=float).copy(), tuple(config.receiver), mode))
        return {
            "times": np.asarray(times, dtype=float),
            "components": ["Ex", "Ey", "dBzdt"],
            "data": np.column_stack(
                (
                    10.0 * np.asarray(times, dtype=float),
                    20.0 * np.asarray(times, dtype=float),
                    -30.0 * np.asarray(times, dtype=float),
                )
            ),
        }

    monkeypatch.setattr(sp, "get_empymod_reference", fake_get_empymod_reference)
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_ip",
        polarization="cole-cole",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )
    provider = sp._MaskedPrimaryProvider(base=object(), config=config)
    times = np.asarray([1.0e-5, 2.0e-5, 3.0e-5])
    receivers = np.asarray([[0.0, -300.0, -0.1]])

    provider.prepare_receiver_reference_cache(times, receivers)

    assert len(calls) == 1
    assert calls[0][2] == "cole-cole"


def test_empymod_primary_provider_ramp_averages_fem_samples():
    from atem3d.primary import EmpymodPrimaryProvider

    def linear_reference_runner(survey, **_kwargs):
        times = np.asarray(survey.times, dtype=float)
        n_columns = len(survey.receiver_components)
        return np.tile(times[:, None], (1, n_columns))

    provider = EmpymodPrimaryProvider(
        config={
            "source_start": (-500.0, 200.0, -0.1),
            "source_end": (500.0, 200.0, -0.1),
            "depths": [0.0, 100.0],
            "resistivities": [1.0e8, 100.0, 33.333333333333336],
            "strength": 10.0,
            "signal": -1,
            "coordinate_system": "z_up",
            "ramp_off_time": 0.2,
            "time_origin": "after_ramp",
        },
        reference_runner=linear_reference_runner,
    )

    values = provider.get_Ep_on_V(1.0, np.asarray([[0.0, -300.0, -0.1]]))

    np.testing.assert_allclose(values, [[1.1, 1.1, 1.1]], rtol=1.0e-12)


def test_validation_gate_uses_pointwise_relative_error_not_floor_error():
    sp = load_pipeline()

    rows, summary = sp._robust_error_rows(
        np.asarray([1.0]),
        np.asarray([[2.0e-20]]),
        np.asarray([[1.0e-20]]),
        ["Ex"],
        threshold=0.05,
    )

    assert rows[0]["ordinary_relative_error"] == 1.0
    assert rows[0]["relative_error_with_floor"] < 0.05
    assert rows[0]["pass_5pct"] is False
    assert summary["max_error_Ex"] == 1.0


def test_legacy_validation_gate_uses_pointwise_relative_error_not_peak_normalized_error():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "dolfinx" / "legacy_total_field_baseline.py"
    spec = importlib.util.spec_from_file_location("legacy_total_field_baseline_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows, summary = module._robust_error_rows(
        np.asarray([1.0]),
        np.asarray([[2.0e-20]]),
        np.asarray([[1.0e-20]]),
        ["Ex"],
        threshold=0.05,
    )

    assert rows[0]["ordinary_relative_error"] == 1.0
    assert rows[0]["relative_error_with_floor"] < 0.05
    assert rows[0]["peak_normalized_error"] < 0.05
    assert rows[0]["pass_5pct"] is False
    assert summary["max_error_Ex"] == 1.0


def test_source_term_mode_accepts_primary_secondary():
    sp = load_pipeline()

    config = sp.PipelineConfig(source_term_mode="primary_secondary")

    sp.validate_model_consistency(config)


def test_cli_accepts_layered_ip_primary_secondary_background(tmp_path, monkeypatch):
    sp = load_pipeline()
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {"python": "test-python"})

    exit_code = sp.main(
        [
            "--check-env-only",
            "--workdir",
            str(tmp_path),
            "--source-term-mode",
            "primary_secondary",
            "--primary-secondary-background-mode",
            "layered_ip",
            "--polarization",
            "cole-cole",
            "--layer-depths",
            "100",
            "--layer-resistivities",
            "100,33.333333333333336",
            "--cole-rho0",
            "33.333333333333336",
            "--cole-layer-top",
            "100",
        ]
    )

    assert exit_code == 0


def test_primary_secondary_primary_config_uses_full_layered_background():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        rho_air=1.0e8,
        source_current=10.0,
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )

    primary = sp._primary_secondary_empymod_primary_config(config)

    assert primary["depths"] == [0.0, 100.0]
    assert primary["resistivities"] == [1.0e8, 100.0, 33.333333333333336]
    assert primary["strength"] == 10.0
    assert primary["coordinate_system"] == "z_up"


def test_primary_secondary_primary_config_can_use_top_halfspace_background():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        rho_air=1.0e8,
        source_current=10.0,
        primary_secondary_background_mode="top_halfspace",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )

    primary = sp._primary_secondary_empymod_primary_config(config)

    assert primary["depths"] == [0.0]
    assert primary["resistivities"] == [1.0e8, 100.0]


def test_primary_secondary_layered_ip_background_uses_cole_cole_primary():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_ip",
        rho_air=1.0e8,
        source_current=10.0,
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )

    primary = sp._primary_secondary_empymod_primary_config(config)

    assert primary["depths"] == [0.0, 100.0]
    assert isinstance(primary["resistivities"], dict)
    assert primary["resistivities"]["res"][0] == 1.0e8
    assert primary["resistivities"]["res"][1] == 100.0
    assert primary["resistivities"]["res"][2] < 33.333333333333336
    assert callable(primary["resistivities"]["func_eta"])
    assert sp._primary_secondary_has_active_contrast(config) is False


def test_primary_secondary_layered_ip_secondary_primary_keeps_noip_physical_resistivity():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered",
        rho_air=1.0e8,
        source_current=10.0,
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )

    primary = sp._primary_secondary_empymod_primary_config(config)

    assert primary["depths"] == [0.0, 100.0]
    assert primary["resistivities"] == [1.0e8, 100.0, 33.333333333333336]


def test_primary_secondary_layered_background_uses_noip_rho0_conductivity_for_ip_secondary():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )

    sigma = sp._primary_secondary_layered_background_sigma_values(
        np.asarray([50.0, 150.0]),
        config,
    )

    np.testing.assert_allclose(sigma, [0.01, 0.03])


def test_primary_secondary_dc_conductivity_mode_selects_initial_or_infinity():
    sp = load_pipeline()

    materials = {
        "sigma": object(),
        "sigma_initial": object(),
        "sigma_infinity": object(),
    }

    assert (
        sp._primary_secondary_dc_conductivity(
            materials,
            sp.PipelineConfig(primary_secondary_dc_conductivity_mode="sigma_initial"),
        )
        is materials["sigma_initial"]
    )
    assert (
        sp._primary_secondary_dc_conductivity(
            materials,
            sp.PipelineConfig(primary_secondary_dc_conductivity_mode="sigma_infinity"),
        )
        is materials["sigma_infinity"]
    )


def test_primary_secondary_active_primary_mask_ignores_1d_background_layers():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -50.0],
            [0.0, 0.0, -150.0],
        ]
    )

    mask = sp._primary_secondary_active_primary_sample_mask(points, config)

    assert mask.tolist() == [False, False, False]


def test_primary_secondary_layered_ip_mask_ignores_1d_polarizable_background():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_ip",
        polarization="cole-cole",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -50.0],
            [0.0, 0.0, -150.0],
        ]
    )

    mask = sp._primary_secondary_active_primary_sample_mask(points, config)

    assert mask.tolist() == [False, False, False]


def test_primary_secondary_top_halfspace_marks_lower_1d_layers_as_contrast():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="top_halfspace",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -50.0],
            [0.0, 0.0, -150.0],
        ]
    )

    mask = sp._primary_secondary_active_primary_sample_mask(points, config)

    assert mask.tolist() == [False, False, True]


def test_primary_secondary_layered_primary_top_current_marks_lower_layers_as_contrast():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_primary_top_current",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -50.0],
            [0.0, 0.0, -150.0],
        ]
    )

    mask = sp._primary_secondary_active_primary_sample_mask(points, config)

    assert mask.tolist() == [False, False, True]


def test_primary_secondary_layered_primary_top_current_primary_config_keeps_full_layers():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_primary_top_current",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
    )

    primary = sp._primary_secondary_empymod_primary_config(config)

    assert primary["depths"] == [0.0, 100.0]
    assert primary["resistivities"] == [100000000.0, 100.0, 33.333333333333336]


def test_primary_secondary_contrast_detection():
    sp = load_pipeline()

    assert sp._primary_secondary_has_active_contrast(sp.PipelineConfig()) is False
    assert sp._primary_secondary_has_active_contrast(
        sp.PipelineConfig(layer_depths=(100.0,), layer_resistivities=(100.0, 33.333333333333336))
    ) is False
    assert sp._primary_secondary_has_active_contrast(sp.PipelineConfig(polarization="cole-cole")) is True
    assert sp._primary_secondary_has_active_contrast(
        sp.PipelineConfig(
            primary_secondary_background_mode="layered_ip",
            layer_depths=(100.0,),
            layer_resistivities=(100.0, 33.333333333333336),
            polarization="cole-cole",
            cole_rho0=33.333333333333336,
            cole_layer_top=100.0,
        )
    ) is False
    assert sp._primary_secondary_has_active_contrast(
        sp.PipelineConfig(
            primary_secondary_background_mode="top_halfspace",
            layer_depths=(100.0,),
            layer_resistivities=(100.0, 33.333333333333336),
        )
    ) is True
    assert sp._primary_secondary_has_active_contrast(
        sp.PipelineConfig(
            primary_secondary_background_mode="layered_primary_top_current",
            layer_depths=(100.0,),
            layer_resistivities=(100.0, 33.333333333333336),
        )
    ) is True


def test_primary_secondary_zero_contrast_reference_mode_matches_polarization():
    sp = load_pipeline()

    assert sp._primary_secondary_zero_contrast_reference_mode(sp.PipelineConfig()) == "noip"
    assert sp._primary_secondary_zero_contrast_reference_mode(sp.PipelineConfig(polarization="cole-cole")) == "cole-cole"


def test_primary_secondary_layered_ip_zero_contrast_diagnostics_mark_primary_only():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_background_mode="layered_ip",
        polarization="cole-cole",
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        cole_rho0=33.333333333333336,
        cole_layer_top=100.0,
    )

    diagnostics = sp._primary_secondary_zero_contrast_diagnostics(np.asarray([1.0e-5, 1.0]), config)

    assert diagnostics["contrast_is_zero"] is True
    assert diagnostics["secondary_solver_skipped"] is True
    assert diagnostics["zero_contrast_reason"] == "full_1d_cole_cole_layered_primary_background"
    assert diagnostics["primary_reference_mode"] == "cole-cole"


def test_masked_primary_provider_caps_active_fem_samples():
    sp = load_pipeline()

    class Base:
        def __init__(self):
            self.sampled = []

        def get_Ep_on_V(self, _t, points):
            self.sampled.append(len(points))
            return np.asarray(points, dtype=float)

        def get_Ep_dc_on_V(self, points):
            self.sampled.append(len(points))
            return np.asarray(points, dtype=float)

        def get_receiver_E(self, _t, receivers):
            return np.zeros_like(np.asarray(receivers, dtype=float))

        def get_receiver_dBdt(self, _t, receivers):
            return np.zeros_like(np.asarray(receivers, dtype=float))

    config = sp.PipelineConfig(
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=300.0,
        cole_rho0=33.333333333333336,
        primary_secondary_max_primary_samples=2,
    )
    provider = sp._MaskedPrimaryProvider(Base(), config)
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -150.0],
            [10.0, 0.0, -150.0],
            [20.0, 0.0, -150.0],
            [30.0, 0.0, -150.0],
        ]
    )

    values = provider.get_Ep_on_V(1.0e-5, points)

    assert provider.base.sampled == [2]
    assert values.shape == points.shape
    assert np.all(values[0] == 0.0)


def test_primary_secondary_spatial_sample_indices_are_distributed():
    sp = load_pipeline()
    points = np.asarray(
        [
            [0.0, 0.0, -100.0],
            [0.0, 0.0, -200.0],
            [100.0, 0.0, -100.0],
            [100.0, 0.0, -200.0],
            [0.0, 100.0, -100.0],
            [0.0, 100.0, -200.0],
            [100.0, 100.0, -100.0],
            [100.0, 100.0, -200.0],
        ]
    )

    indices = sp._primary_secondary_spatial_sample_indices(points, 4)

    assert len(indices) == 4
    assert len(set(np.sign(points[indices, 2]).tolist())) == 1
    assert len(set(points[indices, 2].tolist())) > 1


def test_primary_secondary_configured_sampling_keeps_high_priority_and_far_points():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        source_start=(0.0, 0.0, -0.1),
        source_end=(100.0, 0.0, -0.1),
        receiver=(50.0, 0.0, -0.1),
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=float("inf"),
    )
    clustered = np.asarray(
        [[50.0 + 0.1 * i, 0.0, -100.0] for i in range(20)],
        dtype=float,
    )
    far_points = np.asarray(
        [
            [-100.0, -100.0, -100.0],
            [200.0, -100.0, -100.0],
            [-100.0, 100.0, -200.0],
            [200.0, 100.0, -200.0],
        ],
        dtype=float,
    )
    points = np.vstack([clustered, far_points])

    indices = sp._primary_secondary_spatial_sample_indices(points, 8, config)
    selected = points[indices]

    assert len(indices) == 8
    assert any(np.linalg.norm(row[:2] - np.asarray([50.0, 0.0])) < 1.0 for row in selected)
    assert np.count_nonzero(selected[:, 0] < 0.0) >= 1
    assert np.count_nonzero(selected[:, 0] > 100.0) >= 1


def test_primary_secondary_weighted_farthest_sampling_covers_priority_and_domain():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        source_start=(0.0, 0.0, -0.1),
        source_end=(100.0, 0.0, -0.1),
        receiver=(50.0, 0.0, -0.1),
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=float("inf"),
    )
    clustered = np.asarray(
        [[50.0 + 0.01 * i, 0.0, -100.0] for i in range(30)],
        dtype=float,
    )
    far_points = np.asarray(
        [
            [-500.0, -500.0, -100.0],
            [500.0, -500.0, -100.0],
            [-500.0, 500.0, -300.0],
            [500.0, 500.0, -300.0],
        ],
        dtype=float,
    )
    points = np.vstack([clustered, far_points])

    indices = sp._primary_secondary_weighted_farthest_sample_indices(
        points,
        8,
        config,
        score_weight=0.2,
    )
    selected = points[indices]

    assert len(indices) == 8
    assert len(set(indices.tolist())) == 8
    assert any(np.linalg.norm(row[:2] - np.asarray([50.0, 0.0])) < 1.0 for row in selected)
    assert np.count_nonzero(selected[:, 0] < -100.0) >= 1
    assert np.count_nonzero(selected[:, 0] > 100.0) >= 1


def test_primary_secondary_sampling_strategy_priority_hull_uses_priority_indices():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_sampling_strategy="priority_hull",
        source_start=(0.0, 0.0, -0.1),
        source_end=(100.0, 0.0, -0.1),
        receiver=(50.0, 0.0, -0.1),
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=float("inf"),
    )
    points = np.asarray(
        [
            [-500.0, -500.0, -100.0],
            [0.0, 0.0, -100.0],
            [50.0, 0.0, -100.0],
            [100.0, 0.0, -100.0],
            [500.0, 500.0, -300.0],
            [600.0, 500.0, -300.0],
        ],
        dtype=float,
    )

    selected = sp._primary_secondary_spatial_sample_indices(points, 4, config)
    expected = sp._primary_secondary_priority_sample_indices(points, 4, config)

    np.testing.assert_array_equal(selected, expected)


def test_primary_secondary_priority_hull_sampling_fills_requested_budget():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        primary_secondary_sampling_strategy="priority_hull",
        source_start=(0.0, 0.0, -0.1),
        source_end=(100.0, 0.0, -0.1),
        receiver=(50.0, 0.0, -0.1),
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=float("inf"),
    )
    points = np.asarray(
        [[float(i % 17), float((i * 7) % 19), -100.0 - float(i % 11)] for i in range(200)],
        dtype=float,
    )

    selected = sp._primary_secondary_spatial_sample_indices(points, 64, config)

    assert selected.shape == (64,)
    assert len(set(selected.tolist())) == 64


def test_fast_tabulated_vector_field_uses_point_lookup():
    sp = load_pipeline()
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    values = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    field = sp._FastTabulatedVectorField(points=points, values=values)

    result = field(points.T)

    np.testing.assert_allclose(result, values.T)


def test_polarizable_depth_mask_respects_infinite_bottom():
    sp = load_pipeline()
    depths = np.asarray([25.0, 99.9, 100.0, 150.0, 500.0])

    mask = sp._polarizable_depth_mask(depths, top=100.0, bottom=float("inf"))

    assert mask.tolist() == [False, False, True, True, True]


def test_resolved_config_yaml_persists_layer_model():
    sp = load_pipeline()
    config = sp.PipelineConfig(
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.333333333333336),
        primary_secondary_sampling_strategy="priority_hull",
        receiver_type="disk_average",
        receiver_average_radius=20.0,
        max_internal_dt=0.02,
    )

    text = sp._resolved_config_yaml(config)

    assert "layer_depths: [100" in text
    assert "layer_resistivities: [100, 33.33333333333334]" in text
    assert "primary_secondary_sampling_strategy: priority_hull" in text
    assert "receiver_type: disk_average" in text
    assert "receiver_average_radius: 20" in text
    assert "max_internal_dt: 0.02" in text


def test_far_field_mesh_size_is_configurable_and_recorded():
    sp = load_pipeline()
    config = sp.PipelineConfig(far_field_mesh_size=6000.0)

    assert sp._far_field_mesh_size(config) == pytest.approx(6000.0)
    assert "far_field_mesh_size: 6000" in sp._resolved_config_yaml(config)


def test_far_field_mesh_size_does_not_expand_local_refinement_transition():
    sp = load_pipeline()
    config = sp.PipelineConfig(far_field_mesh_size=6000.0)

    assert sp._source_refinement_transition_distance(config) == pytest.approx(2500.0)
    assert sp._receiver_refinement_transition_distance(config) == pytest.approx(3000.0)


def test_diffusion_refinement_top_is_configurable_and_recorded():
    sp = load_pipeline()
    config = sp.PipelineConfig(diffusion_refinement_top=-100.0)

    assert sp._diffusion_refinement_box(config)["top"] == pytest.approx(-100.0)
    assert "diffusion_refinement_top: -100" in sp._resolved_config_yaml(config)


def test_primary_secondary_current_correction_uses_new_secondary_solution():
    sp = load_pipeline()

    old_solution = object()
    new_solution = object()
    seen = {}

    def apply_current_continuity_correction(Es_trial):
        seen["Es_trial"] = Es_trial
        return Es_trial

    corrected = sp._primary_secondary_corrected_trial_solution(
        old_solution,
        new_solution,
        apply_current_continuity_correction,
    )

    assert seen["Es_trial"] is new_solution
    assert corrected is new_solution
