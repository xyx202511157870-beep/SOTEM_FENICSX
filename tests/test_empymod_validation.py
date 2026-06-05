import json

import numpy as np
import pytest

from atem3d.empymod_validation import run_empymod_validation, run_empymod_validation_sweep


class FakeResult:
    def __init__(self, times, data):
        self.times = np.asarray(times, dtype=float)
        self.data = np.asarray(data, dtype=float)


def _config():
    return {
        "coordinate_system": "z_up",
        "source": {
            "start": [-25.0, 0.0, -0.5],
            "end": [25.0, 0.0, -0.5],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, -0.5], "component": "Hz"}],
    }


def test_run_empymod_validation_compares_runner_data_to_reference():
    result = FakeResult(times=[0.0, 1.0e-4, 2.0e-4], data=[[99.0], [2.0], [4.0]])

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=lambda survey: np.array([[2.0], [4.0]]),
    )

    assert validation.component_names == ["Hz@0"]
    np.testing.assert_allclose(validation.times, [1.0e-4, 2.0e-4])
    np.testing.assert_allclose(validation.numerical, [[2.0], [4.0]])
    np.testing.assert_allclose(validation.reference, [[2.0], [4.0]])
    assert validation.components["Hz@0"]["relative_l2"] == 0.0
    assert validation.metadata["empymod"]["coordinate_system"] == "z_up"


def test_run_empymod_validation_can_skip_positive_times_and_write_report(tmp_path):
    result = FakeResult(times=[0.0, 1.0e-4, 2.0e-4], data=[[99.0], [999.0], [4.0]])
    report_path = tmp_path / "validation.json"

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=lambda survey: np.array([[4.0]]),
        skip_positive_times=1,
        output_path=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["n_times"] == 1
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert validation.times.tolist() == [2.0e-4]


def test_run_empymod_validation_can_override_empymod_strength():
    result = FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]])
    seen_strengths = []

    def reference_runner(survey):
        seen_strengths.append(survey.strength)
        return np.array([[2.0]])

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=reference_runner,
        empymod_strength=2.5,
    )

    assert seen_strengths == [2.5]
    assert validation.metadata["empymod"]["strength"] == 2.5


def test_run_empymod_validation_can_limit_comparison_to_time_window(tmp_path):
    result = FakeResult(
        times=[0.0, 1.0e-4, 2.0e-4, 3.0e-4],
        data=[[99.0], [999.0], [4.0], [8.0]],
    )
    seen_reference_times = []
    report_path = tmp_path / "windowed_validation.json"

    def reference_runner(survey):
        seen_reference_times.append(survey.times.copy())
        return np.array([[4.0], [8.0]])

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=reference_runner,
        time_min=2.0e-4,
        time_max=3.0e-4,
        output_path=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(seen_reference_times[0], [2.0e-4, 3.0e-4])
    np.testing.assert_allclose(validation.times, [2.0e-4, 3.0e-4])
    np.testing.assert_allclose(validation.numerical, [[4.0], [8.0]])
    assert payload["metadata"]["empymod"]["time_window"] == {
        "min": 2.0e-4,
        "max": 3.0e-4,
    }
    assert payload["n_times"] == 2


def test_run_empymod_validation_time_window_includes_roundoff_endpoint():
    result = FakeResult(
        times=[0.0, 0.001, 0.012000000000000004],
        data=[[99.0], [999.0], [8.0]],
    )

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=lambda survey: np.array([[8.0]]),
        time_min=0.012,
        time_max=0.012,
    )

    np.testing.assert_allclose(validation.times, [0.012])
    np.testing.assert_allclose(validation.numerical, [[8.0]])


def test_run_empymod_validation_rejects_invalid_time_window():
    result = FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]])

    with pytest.raises(ValueError, match="time_min must be <= time_max"):
        run_empymod_validation(
            _config(),
            depths=[0.0],
            resistivities=[1.0e8, 100.0],
            runner=lambda config: result,
            reference_runner=lambda survey: np.array([[2.0]]),
            time_min=2.0e-4,
            time_max=1.0e-4,
        )

    with pytest.raises(ValueError, match="time selection produced no samples"):
        run_empymod_validation(
            _config(),
            depths=[0.0],
            resistivities=[1.0e8, 100.0],
            runner=lambda config: result,
            reference_runner=lambda survey: np.empty((0, 1)),
            time_min=2.0e-4,
        )


def test_run_empymod_validation_reports_relative_and_absolute_pass_fail():
    config = {
        **_config(),
        "receivers": [
            {"location": [0.0, 10.0, -0.5], "component": "Ey"},
            {"location": [0.0, 10.0, -0.5], "component": "Hz"},
            {"location": [0.0, 10.0, -0.5], "component": "Ex"},
        ],
    }
    result = FakeResult(
        times=[0.0, 1.0e-4],
        data=[[99.0, 99.0, 99.0], [1.0e-8, 1.05, 1.5]],
    )

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: result,
        reference_runner=lambda survey: np.array([[1.0e-15, 1.0, 1.0]]),
        tolerance=0.1,
        absolute_tolerance=1.0e-7,
    )

    report = validation.to_report()
    assert report["tolerance"] == 0.1
    assert report["absolute_tolerance"] == 1.0e-7
    assert report["passed"] is False
    assert report["absolute_linf_max"] == 0.5
    assert report["components"]["Ey@0"]["passed"] is True
    assert report["components"]["Ey@0"]["passed_by"] == "absolute"
    assert report["components"]["Hz@1"]["passed"] is True
    assert report["components"]["Hz@1"]["passed_by"] == "relative"
    assert report["components"]["Ex@2"]["passed"] is False
    assert report["components"]["Ex@2"]["passed_by"] == "none"
    assert report["component_groups"]["electric"]["component_count"] == 2
    assert report["component_groups"]["electric"]["components"] == ["Ey@0", "Ex@2"]
    assert report["component_groups"]["magnetic"]["component_count"] == 1
    assert report["component_groups"]["magnetic"]["components"] == ["Hz@1"]
    assert report["component_groups"]["magnetic"]["relative_linf_max"] == (
        report["components"]["Hz@1"]["relative_linf"]
    )


def test_run_empymod_validation_rejects_negative_tolerances():
    result = FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]])

    with pytest.raises(ValueError, match="tolerance must be nonnegative"):
        run_empymod_validation(
            _config(),
            depths=[0.0],
            resistivities=[1.0e8, 100.0],
            runner=lambda config: result,
            reference_runner=lambda survey: np.array([[2.0]]),
            tolerance=-0.1,
        )

    with pytest.raises(ValueError, match="absolute_tolerance must be nonnegative"):
        run_empymod_validation(
            _config(),
            depths=[0.0],
            resistivities=[1.0e8, 100.0],
            runner=lambda config: result,
            reference_runner=lambda survey: np.array([[2.0]]),
            absolute_tolerance=-1.0e-9,
        )


def test_run_empymod_validation_can_use_data_only_simulation(monkeypatch):
    calls = []

    class FakeSimulation:
        def run(self):
            calls.append("run")
            return FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [1.0]])

        def run_data_only(self):
            calls.append("run_data_only")
            return FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]])

    monkeypatch.setattr(
        "atem3d.empymod_validation.build_simulation",
        lambda config: FakeSimulation(),
    )

    validation = run_empymod_validation(
        _config(),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        reference_runner=lambda survey: np.array([[2.0]]),
        data_only=True,
    )

    assert calls == ["run_data_only"]
    assert validation.components["Hz@0"]["relative_linf"] == 0.0


def test_run_empymod_validation_records_magnetic_recovery_metadata():
    config = {
        **_config(),
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_subdivisions": 5,
        "magnetic_recovery_polarization_scale": 1.25,
        "magnetic_recovery_initial_polarization_scale": 0.5,
        "magnetic_recovery_source_primary_delta6": True,
        "magnetic_recovery_source_primary_delta6_basis": "edge_current",
        "magnetic_recovery_source_history": {
            "kind": "prescribed_source_moments",
            "tau": 0.1,
            "max_order": 1,
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03, 0.04, 0.02],
            "receiver_matrix": "current_biot",
        },
    }

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]]),
        reference_runner=lambda survey: np.array([[2.0]]),
    )

    assert validation.metadata["atem3d"]["magnetic_receiver_mode"] == "current_biot"
    assert validation.metadata["atem3d"]["magnetic_recovery_subdivisions"] == 5
    assert validation.metadata["atem3d"]["magnetic_recovery_polarization_scale"] == 1.25
    assert validation.metadata["atem3d"]["magnetic_recovery_initial_polarization_scale"] == 0.5
    assert validation.metadata["atem3d"]["magnetic_recovery_source_primary_delta6"] is True
    assert (
        validation.metadata["atem3d"]["magnetic_recovery_source_primary_delta6_basis"]
        == "edge_current"
    )
    assert validation.metadata["atem3d"]["magnetic_recovery_source_history"] == {
        "diagnostic_only": True,
        "kind": "prescribed_source_moments",
        "requires_ip": True,
        "tau": 0.1,
        "max_order": 1,
        "source_moment_degrees": [0, 2],
        "coefficient_count": 4,
        "receiver_matrix": "current_biot",
    }


def test_run_empymod_validation_reports_hj_default_magnetic_receiver_mode():
    config = {**_config(), "formulation": "hj"}

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]]),
        reference_runner=lambda survey: np.array([[2.0]]),
    )

    assert validation.metadata["atem3d"]["magnetic_receiver_mode"] == "stored_h"


def test_run_empymod_validation_records_multiple_source_history_terms():
    config = {
        **_config(),
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "terms": [
                {
                    "tau": 0.1,
                    "max_order": 1,
                    "source_moment_degrees": [0],
                    "coefficients": [0.1, 0.04],
                },
                {
                    "tau": 0.5,
                    "max_order": 0,
                    "source_moment_degrees": [2],
                    "coefficients": [-0.03],
                    "receiver_matrix": "edge_current",
                },
            ],
        },
    }

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]]),
        reference_runner=lambda survey: np.array([[2.0]]),
    )

    assert validation.metadata["atem3d"]["magnetic_recovery_source_history"] == {
        "term_count": 2,
        "terms": [
            {
                "diagnostic_only": True,
                "kind": "prescribed_source_moments",
                "requires_ip": True,
                "tau": 0.1,
                "max_order": 1,
                "source_moment_degrees": [0],
                "coefficient_count": 2,
                "receiver_matrix": "auto",
            },
            {
                "diagnostic_only": True,
                "kind": "prescribed_source_moments",
                "requires_ip": True,
                "tau": 0.5,
                "max_order": 0,
                "source_moment_degrees": [2],
                "coefficient_count": 1,
                "receiver_matrix": "edge_current",
            },
        ],
    }


def test_run_empymod_validation_records_multimode_driven_recovery_metadata():
    config = {
        **_config(),
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "driven_recovery_source_moments",
            "driver_tau": 0.1,
            "response_taus": [0.02, 0.04],
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03, 0.04, 0.02],
            "receiver_matrix": "current_biot",
        },
    }

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]]),
        reference_runner=lambda survey: np.array([[2.0]]),
    )

    assert validation.metadata["atem3d"]["magnetic_recovery_source_history"] == {
        "diagnostic_only": True,
        "kind": "driven_recovery_source_moments",
        "requires_ip": True,
        "driver_tau": 0.1,
        "response_taus": [0.02, 0.04],
        "source_moment_degrees": [0, 2],
        "coefficient_count": 4,
        "receiver_matrix": "current_biot",
    }


def test_run_empymod_validation_records_source_diffusion_normalized_metadata():
    config = {
        **_config(),
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "source_diffusion_kernel_source_moments",
            "normalized_amplitude": -3.5,
            "tau_multiplier": 1.5,
            "amplitude_time": 1.0e-4,
            "source_moment_degrees": [0],
            "receiver_matrix": "current_biot",
        },
    }

    validation = run_empymod_validation(
        config,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=lambda config: FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [2.0]]),
        reference_runner=lambda survey: np.array([[2.0]]),
    )

    assert validation.metadata["atem3d"]["magnetic_recovery_source_history"] == {
        "diagnostic_only": True,
        "kind": "source_diffusion_kernel_source_moments",
        "requires_ip": False,
        "tau_multiplier": 1.5,
        "amplitude_time": 1.0e-4,
        "basis_kind": "continuous",
        "source_moment_degrees": [0],
        "receiver_matrix": "current_biot",
        "normalized_amplitude": -3.5,
        "coefficient_count": 1,
    }


def test_run_empymod_validation_sweep_applies_overrides_and_writes_report(tmp_path):
    base_config = {
        **_config(),
        "magnetic_receiver_mode": "current_biot",
        "nested": {"keep": 1, "replace": 2},
    }
    seen_configs = []

    def fake_runner(config):
        seen_configs.append(config)
        value = 2.0 if config["magnetic_receiver_mode"] == "current_biot" else 4.0
        return FakeResult(times=[0.0, 1.0e-4], data=[[99.0], [value]])

    report_path = tmp_path / "sweep.json"
    sweep = run_empymod_validation_sweep(
        base_config,
        [
            {"name": "base", "overrides": {}},
            {
                "name": "cell",
                "overrides": {
                    "magnetic_receiver_mode": "edge_basis_cell_biot",
                    "nested": {"replace": 3},
                },
            },
        ],
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        runner=fake_runner,
        reference_runner=lambda survey: np.array([[2.0]]),
        tolerance=0.25,
        absolute_tolerance=1.0e-12,
        output_path=report_path,
    )

    assert list(sweep.cases) == ["base", "cell"]
    assert sweep.cases["base"].components["Hz@0"]["relative_linf"] == 0.0
    assert sweep.cases["cell"].components["Hz@0"]["relative_linf"] == 1.0
    assert seen_configs[1]["nested"] == {"keep": 1, "replace": 3}
    assert base_config["nested"] == {"keep": 1, "replace": 2}

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["cases"]["cell"]["overrides"]["magnetic_receiver_mode"] == (
        "edge_basis_cell_biot"
    )
    assert payload["cases"]["base"]["report"]["tolerance"] == 0.25
    assert payload["cases"]["base"]["report"]["absolute_tolerance"] == 1.0e-12
    assert payload["cases"]["base"]["report"]["passed"] is True
    assert payload["cases"]["cell"]["report"]["passed"] is False
    assert payload["cases"]["cell"]["report"]["components"]["Hz@0"]["relative_linf"] == 1.0
