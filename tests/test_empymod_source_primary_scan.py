import json

import numpy as np

from atem3d.empymod_source_primary_scan import run_empymod_source_primary_tau_scan
from atem3d.empymod_source_primary_scan_cli import main


def _config():
    return {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {"sigma_infinity": 0.2},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Ex", "Hz"],
        },
        "time_steps": [[0.1, 4]],
    }


def test_empymod_source_primary_tau_scan_fits_fake_reference_runner():
    source_h = np.array([2.0, 3.0])
    tau_values = [0.1, 0.2]
    tau_iter = iter(tau_values)
    scales = {0.1: -4.0, 0.2: -8.0}

    def fake_reference_runner(survey, **_kwargs):
        if not isinstance(survey.resistivities, dict):
            return np.zeros((survey.times.size, source_h.size))
        tau = next(tau_iter)
        return scales[tau] * np.exp(-survey.times[:, None] / (2.0 * tau)) * source_h

    report = run_empymod_source_primary_tau_scan(
        _config(),
        depths=[0.0],
        resistivities=[10.0, 5.0],
        delta_sigma=[0.0, 0.02],
        tau_values=tau_values,
        kernel_factors=[1.0, 2.0],
        positive_times_only=True,
        reference_runner=fake_reference_runner,
        source_field_runner=lambda _config, names, _components: (source_h, 1.0),
    )

    assert report["receiver_names"] == ["Hz@x=0", "Hz@x=1"]
    assert report["tau_values"]["0.1"]["best_key"] == "tau_kernel_0.2"
    assert report["tau_values"]["0.2"]["best_key"] == "tau_kernel_0.4"
    np.testing.assert_allclose(
        report["tau_values"]["0.1"]["fits"]["tau_kernel_0.2"]["scale"],
        -4.0,
    )
    np.testing.assert_allclose(
        report["tau_values"]["0.1"]["empirical_kernel"]["kernel"],
        (-4.0 * np.exp(-np.array([0.1, 0.2, 0.3, 0.4]) / 0.2)).tolist(),
    )
    assert report["tau_values"]["0.2"]["best_relative_l2"] < 1.0e-14


def test_empymod_source_primary_scan_cli_writes_report_with_injected_runners(tmp_path):
    import yaml

    config_path = tmp_path / "scan.yaml"
    config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    output_path = tmp_path / "report.json"
    source_h = np.array([1.5, 2.5])
    tau_iter = iter([0.1])

    def fake_reference_runner(survey, **_kwargs):
        if not isinstance(survey.resistivities, dict):
            return np.zeros((survey.times.size, source_h.size))
        tau = next(tau_iter)
        return -3.0 * np.exp(-survey.times[:, None] / (2.0 * tau)) * source_h

    exit_code = main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "10",
            "5",
            "--delta-sigma",
            "0",
            "0.02",
            "--tau-values",
            "0.1",
            "--kernel-factors",
            "1",
            "2",
            "-o",
            str(output_path),
        ],
        reference_runner=fake_reference_runner,
        source_field_runner=lambda _config, names, _components: (source_h, 1.0),
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tau_values"]["0.1"]["best_key"] == "tau_kernel_0.2"
    np.testing.assert_allclose(
        payload["tau_values"]["0.1"]["fits"]["tau_kernel_0.2"]["scale"],
        -3.0,
    )
