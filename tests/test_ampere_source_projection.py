import json

import numpy as np

from atem3d.ampere_source_projection import (
    ampere_source_projection,
    hj_ampere_source_projection,
)
from atem3d.ampere_source_projection_cli import main
from atem3d.config import build_simulation
from atem3d.io import save_result_hdf5


def _config():
    return {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receivers": [{"location": [0.0, 1.0, 0.0], "component": "Hz"}],
        "time_steps": [[0.1, 2]],
    }


def _hj_config():
    config = _config()
    config["formulation"] = "hj"
    config["magnetic_receiver_mode"] = "current_biot"
    return config


def test_ampere_source_projection_is_zero_for_initial_ampere_balance():
    sim = build_simulation(_config())
    e0 = sim.initial_electric_field()
    b0 = sim.initial_magnetic_flux_density(e0)

    projection = ampere_source_projection(
        sim,
        np.array([0.0]),
        e0.reshape(1, -1),
        b0.reshape(1, -1),
        include_t0=True,
    )

    assert projection.times.tolist() == [0.0]
    assert abs(projection.coefficients[0]) < 1.0e-9
    assert projection.relative_residual_norms[0] < 1.0e-9


def test_hj_ampere_source_projection_is_zero_for_initial_ampere_balance():
    sim = build_simulation(_hj_config())
    result = sim.run()

    projection = hj_ampere_source_projection(
        sim,
        result.times[:1],
        result.e[:1],
        result.h[:1],
        include_t0=True,
    )

    assert projection.times.tolist() == [0.0]
    assert abs(projection.coefficients[0]) < 1.0e-9
    assert projection.relative_residual_norms[0] < 1.0e-9


def test_ampere_source_projection_cli_writes_discrete_basis_report(tmp_path):
    config = _config()
    sim = build_simulation(config)
    result = sim.run()
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "projection.json"
    save_result_hdf5(result_path, result, config)

    exit_code = main(
        [
            str(result_path),
            "--tau",
            "0.2",
            "--kernel-basis-discrete",
            "-o",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["projection"]["n"] == 2
    assert len(payload["projection"]["coefficients"]) == 2
    assert payload["discrete_basis_fit"]["basis_labels"] == [
        "BE relaxation",
        "BE cascade 1",
    ]


def test_ampere_source_projection_cli_accepts_hj_result(tmp_path):
    config = _hj_config()
    sim = build_simulation(config)
    result = sim.run()
    result_path = tmp_path / "hj_result.h5"
    report_path = tmp_path / "hj_projection.json"
    save_result_hdf5(result_path, result, config)

    exit_code = main(
        [
            str(result_path),
            "--include-t0",
            "--tau",
            "0.2",
            "--kernel-basis-discrete",
            "-o",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["formulation"] == "hj"
    assert payload["projection"]["n"] == 3
    assert max(abs(value) for value in payload["projection"]["coefficients"]) < 1.0e-8


def test_ampere_source_projection_cli_subtracts_baseline(tmp_path):
    config = _config()
    sim = build_simulation(config)
    result = sim.run()
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "projection_delta.json"
    save_result_hdf5(result_path, result, config)

    exit_code = main(
        [
            str(result_path),
            "--subtract-baseline",
            str(result_path),
            "--tau",
            "0.2",
            "--kernel-basis-discrete",
            "-o",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["baseline_result_path"] == str(result_path)
    np.testing.assert_allclose(payload["projection_delta"]["coefficients"], 0.0)
    np.testing.assert_allclose(payload["discrete_basis_fit"]["coefficients"], 0.0)
