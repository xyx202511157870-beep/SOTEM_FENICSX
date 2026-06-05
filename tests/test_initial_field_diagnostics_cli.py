import json
import subprocess
import sys

import h5py
import numpy as np
import yaml

from atem3d import initial_field_diagnostics
from atem3d import initial_field_diagnostics_cli


def test_initial_field_diagnostics_cli_writes_mode_metrics(tmp_path):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "initial_fields.json"
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0]))
        h5.create_dataset("data", data=np.zeros((1, 1)))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    initial_field_diagnostics_cli.main(
        [
            str(result_path),
            "--modes",
            "ampere",
            "zero",
            "-o",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["result_path"] == str(result_path)
    assert [entry["mode"] for entry in payload["modes"]] == ["ampere", "zero"]
    assert payload["modes"][0]["receivers"]["Hz@x=0"] != 0.0
    assert payload["modes"][0]["ampere_relative_residual"] < 1.0e-8
    assert payload["modes"][0]["divergence_norm"] < 1.0e-18
    assert payload["modes"][1]["receivers"]["Hz@x=0"] == 0.0
    assert payload["modes"][1]["ampere_relative_residual"] == 1.0


def test_initial_field_diagnostics_module_help_has_no_runtime_warning():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "atem3d.initial_field_diagnostics_cli",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "RuntimeWarning" not in completed.stderr


def test_initial_field_diagnostics_can_include_empymod_frequency_reference(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "coordinate_system": "z_up",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [
            {"location": [0.0, 0.5, 0.0], "component": "Ex"},
            {"location": [0.0, 0.5, 0.0], "component": "Hz"},
        ],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 0.01]))
        h5.create_dataset("data", data=np.zeros((2, 2)))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append((survey, kwargs))
        return np.array([[1.25, 2.5]])

    monkeypatch.setattr(initial_field_diagnostics, "run_empymod_reference", fake_reference)

    report = initial_field_diagnostics.run_initial_field_diagnostics(
        result_path,
        modes=["ampere"],
        empymod_depths=[0.0],
        empymod_resistivities=[1.0e8, 10.0],
        empymod_frequency=1.0e-4,
        srcpts=7,
        recpts=3,
    )

    survey, kwargs = surveys[0]
    assert survey.signal is None
    np.testing.assert_allclose(survey.times, [1.0e-4])
    assert kwargs == {"srcpts": 7, "recpts": 3}
    assert report["empymod_frequency_reference"]["frequency"] == 1.0e-4
    assert report["empymod_frequency_reference"]["receivers"] == {
        "Ex@0": 1.25,
        "Hz@1": 2.5,
    }


def test_initial_field_diagnostics_cli_accepts_empymod_reference_args(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "initial_fields.json"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0]))
        h5.create_dataset("data", data=np.zeros((1, 1)))
        h5.attrs["config_yaml"] = yaml.safe_dump({"source": {"current": 1.0}})

    calls = []

    def fake_diagnostics(result, modes, **kwargs):
        calls.append((result, modes, kwargs))
        return {"result_path": str(result), "modes": []}

    monkeypatch.setattr(initial_field_diagnostics_cli, "run_initial_field_diagnostics", fake_diagnostics)

    initial_field_diagnostics_cli.main(
        [
            str(result_path),
            "--modes",
            "ampere",
            "--empymod-depths",
            "0",
            "40",
            "--empymod-resistivities",
            "1e8",
            "100",
            "33.3333333333",
            "--empymod-frequency",
            "1e-5",
            "--srcpts",
            "9",
            "--recpts",
            "2",
            "-o",
            str(report_path),
        ]
    )

    assert calls[0][1] == ["ampere"]
    assert calls[0][2]["empymod_depths"] == [0.0, 40.0]
    assert calls[0][2]["empymod_resistivities"] == [1.0e8, 100.0, 33.3333333333]
    assert calls[0][2]["empymod_frequency"] == 1.0e-5
    assert calls[0][2]["srcpts"] == 9
    assert calls[0][2]["recpts"] == 2
