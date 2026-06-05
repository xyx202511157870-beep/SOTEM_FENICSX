import json

import h5py
import numpy as np
import yaml
from scipy.constants import mu_0

from atem3d.source_primary import (
    discrete_debye_history_basis,
    fit_exponential_source_primary,
    fit_source_history_kernel_discrete_debye_basis,
    fit_source_history_kernel_basis,
    fit_time_dependent_source_primary_kernel,
    normalized_source_primary_scale,
    scan_exponential_source_primary,
)
from atem3d.source_primary_cli import _SampledReport, _target_data, main


def test_fit_exponential_source_primary_recovers_scale_and_normalization():
    times = np.array([0.1, 0.2, 0.3])
    source_amplitudes = np.array([2.0, -0.5])
    target = -4.0 * np.exp(-times[:, None] / 0.2) * source_amplitudes[None, :]

    fit = fit_exponential_source_primary(
        times,
        target,
        source_amplitudes,
        kernel_tau=0.2,
        component_names=["Hz@x=0", "Hz@x=10"],
    )

    np.testing.assert_allclose(fit.scale, -4.0)
    np.testing.assert_allclose(fit.fitted, target)
    assert fit.relative_l2 < 1.0e-14
    assert fit.components["Hz@x=0"] < 1.0e-14
    np.testing.assert_allclose(
        normalized_source_primary_scale(
            fit.scale,
            delta_sigma=0.1,
            source_length=2.0,
            mu=mu_0,
        ),
        -4.0 / (mu_0 * 0.1 * 4.0),
    )


def test_scan_exponential_source_primary_selects_best_kernel_tau():
    times = np.array([0.1, 0.2, 0.3, 0.4])
    source_amplitudes = np.array([1.0])
    target = 3.0 * np.exp(-times[:, None] / 0.3) * source_amplitudes[None, :]

    scan = scan_exponential_source_primary(
        times,
        target,
        source_amplitudes,
        kernel_taus=[0.1, 0.3, 0.6],
        component_names=["Hz@0"],
    )

    assert scan.best.kernel_tau == 0.3
    np.testing.assert_allclose(scan.best.scale, 3.0)
    assert scan.best.relative_l2 < 1.0e-14


def test_target_data_delta_residual_subtracts_noip_residual():
    ip = _SampledReport(
        payload={},
        names=["Hz@x=0"],
        times=np.array([0.1, 0.2]),
        numerical=np.array([[10.0], [12.0]]),
        reference=np.array([[13.0], [15.0]]),
    )
    noip = _SampledReport(
        payload={},
        names=["Hz@x=0"],
        times=np.array([0.1, 0.2]),
        numerical=np.array([[7.0], [8.0]]),
        reference=np.array([[8.5], [10.5]]),
    )

    target = _target_data(ip, noip, "delta_residual")

    expected = (ip.reference - noip.reference) - (ip.numerical - noip.numerical)
    np.testing.assert_allclose(target, expected)


def test_fit_time_dependent_source_primary_kernel_recovers_common_kernel():
    times = np.array([0.1, 0.2, 0.3])
    source_amplitudes = np.array([2.0, 4.0])
    kernel = np.array([-1.0, -0.5, -0.25])
    target = kernel[:, None] * source_amplitudes[None, :]

    fit = fit_time_dependent_source_primary_kernel(
        times,
        target,
        source_amplitudes,
        component_names=["Hz@x=0", "Hz@x=10"],
    )

    np.testing.assert_allclose(fit.kernel, kernel)
    np.testing.assert_allclose(fit.fitted, target)
    assert fit.relative_l2 < 1.0e-14


def test_fit_source_history_kernel_basis_recovers_exp_polynomial_kernel():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    tau = 0.2
    u = times / tau
    kernel = 0.75 - 2.0 * np.exp(-u) + 1.25 * u * np.exp(-u)

    fit = fit_source_history_kernel_basis(
        times,
        kernel,
        tau=tau,
        powers=[0, 1],
        include_constant=True,
    )

    assert fit.basis_labels == [
        "constant",
        "exp(-t/tau)",
        "(t/tau)^1 exp(-t/tau)",
    ]
    np.testing.assert_allclose(fit.coefficients, [0.75, -2.0, 1.25])
    np.testing.assert_allclose(fit.fitted, kernel, atol=1.0e-14)
    assert fit.relative_l2 < 1.0e-14


def test_discrete_debye_history_basis_matches_backward_euler_cascade():
    time_steps = np.full(4, 0.1)
    tau = 0.2
    alpha = tau / (tau + time_steps[0])
    beta = time_steps[0] / (tau + time_steps[0])

    basis = discrete_debye_history_basis(time_steps, tau=tau, max_order=1)

    nodes = np.arange(5, dtype=float)
    np.testing.assert_allclose(basis.times, [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(basis.values[:, 0], alpha**nodes)
    np.testing.assert_allclose(basis.values[:, 1], nodes * beta * alpha**nodes)
    assert basis.basis_labels == ["BE relaxation", "BE cascade 1"]


def test_fit_source_history_kernel_discrete_debye_basis_recovers_coefficients():
    time_steps = np.full(4, 0.1)
    basis = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    sample_times = np.array([0.1, 0.2, 0.3, 0.4])
    selected = basis.values[1:]
    kernel = -2.0 * selected[:, 0] + 1.5 * selected[:, 1]

    fit = fit_source_history_kernel_discrete_debye_basis(
        time_steps,
        sample_times,
        kernel,
        tau=0.2,
        max_order=1,
    )

    np.testing.assert_allclose(fit.coefficients, [-2.0, 1.5])
    np.testing.assert_allclose(fit.fitted, kernel, atol=1.0e-14)
    assert fit.basis_labels == ["BE relaxation", "BE cascade 1"]
    assert fit.relative_l2 < 1.0e-14


def test_source_primary_cli_fits_reference_delta_from_sampled_reports(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.5}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receivers": [{"location": [0.0, 1.0, 0.0], "component": "Hz"}],
        "time_steps": [[0.1, 1]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    times = np.array([0.1, 0.2, 0.3])
    from geoana.em.static import LineCurrentWholeSpace

    line_current = LineCurrentWholeSpace(
        np.array([config["source"]["start"], config["source"]["end"]]),
        current=1.0,
        mu=mu_0,
    )
    source_hz = line_current.magnetic_field(np.array([[0.0, 1.0, 0.0]]))[0, 2]
    scale = -2.5
    ip_reference = scale * np.exp(-times / 0.2) * source_hz

    def write_report(path, reference):
        payload = {
            "result_path": str(result_path),
            "samples": {
                "Hz@0": [
                    {
                        "time": float(time),
                        "numerical": 0.0,
                        "reference": float(value),
                        "difference": float(-value),
                        "ratio_numerical_over_reference": 0.0,
                    }
                    for time, value in zip(times, reference)
                ]
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "fit.json"
    write_report(ip_report, ip_reference)
    write_report(noip_report, np.zeros_like(ip_reference))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--tau-candidates",
            "0.1",
            "0.2",
            "--kernel-basis-tau",
            "0.2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_tau"] == 0.2
    np.testing.assert_allclose(payload["fits"]["tau_0.2"]["scale"], scale)
    assert payload["fits"]["tau_0.2"]["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(
        payload["empirical_kernel"]["kernel"],
        (scale * np.exp(-times / 0.2)).tolist(),
    )
    basis_fit = payload["empirical_kernel"]["basis_fit"]
    assert basis_fit["tau"] == 0.2
    assert basis_fit["basis_labels"] == [
        "exp(-t/tau)",
        "(t/tau)^1 exp(-t/tau)",
    ]
    np.testing.assert_allclose(
        basis_fit["coefficients"],
        [scale, 0.0],
        atol=1.0e-14,
    )
    assert basis_fit["relative_l2"] < 1.0e-14
    assert payload["normalization"]["delta_sigma"] == 0.1


def test_source_primary_cli_reports_empirical_residual_correction(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.5}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receivers": [{"location": [0.0, 1.0, 0.0], "component": "Hz"}],
        "time_steps": [[0.1, 1]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    times = np.array([0.1, 0.2, 0.3])
    from geoana.em.static import LineCurrentWholeSpace

    source_hz = LineCurrentWholeSpace(
        np.array([config["source"]["start"], config["source"]["end"]]),
        current=1.0,
        mu=mu_0,
    ).magnetic_field(np.array([[0.0, 1.0, 0.0]]))[0, 2]
    residual = -2.0 * np.exp(-times / 0.2) * source_hz
    payload = {
        "result_path": str(result_path),
        "samples": {
            "Hz@0": [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, residual)
            ]
        },
    }
    report = tmp_path / "ip.json"
    output = tmp_path / "fit.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            str(report),
            str(report),
            "--target",
            "ip_residual",
            "--tau-candidates",
            "0.2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    correction = result["empirical_residual_correction"]
    assert correction["components"]["Hz@0"]["relative_l2"] < 1.0e-14


def test_source_primary_cli_reports_discrete_debye_basis_fit(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
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
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    from geoana.em.static import LineCurrentWholeSpace

    source_hz = LineCurrentWholeSpace(
        np.array([config["source"]["start"], config["source"]["end"]]),
        current=1.0,
        mu=mu_0,
    ).magnetic_field(np.array([[0.0, 1.0, 0.0]]))[0, 2]
    basis = discrete_debye_history_basis(np.full(3, 0.1), tau=0.2, max_order=1)
    times = basis.times[1:]
    kernel = -2.0 * basis.values[1:, 0] + 1.5 * basis.values[1:, 1]
    payload = {
        "result_path": str(result_path),
        "samples": {
            "Hz@0": [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value * source_hz),
                    "difference": float(-value * source_hz),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, kernel)
            ]
        },
    }
    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "fit.json"
    ip_report.write_text(json.dumps(payload), encoding="utf-8")
    payload["samples"]["Hz@0"] = [
        {**row, "reference": 0.0, "difference": 0.0}
        for row in payload["samples"]["Hz@0"]
    ]
    noip_report.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--tau-candidates",
            "0.2",
            "--kernel-basis-tau",
            "0.2",
            "--kernel-basis-discrete",
            "--kernel-basis-max-order",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    discrete_fit = result["empirical_kernel"]["discrete_basis_fit"]
    assert discrete_fit["basis_labels"] == ["BE relaxation", "BE cascade 1"]
    np.testing.assert_allclose(discrete_fit["coefficients"], [-2.0, 1.5])
    assert discrete_fit["relative_l2"] < 1.0e-14


def test_source_primary_cli_supports_edge_current_source_basis(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
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
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    from atem3d.config import build_simulation
    from atem3d.magnetic_recovery import biot_savart_h_from_edge_current_moments

    sim = build_simulation(config)
    source_hz = biot_savart_h_from_edge_current_moments(
        sim.mesh,
        sim.sources[0].initial_edge_vector(sim.mesh),
        np.array([[0.0, 1.0, 0.0]]),
    )[0, 2]
    times = np.array([0.1, 0.2, 0.3])
    scale = -1.25
    ip_reference = scale * np.exp(-times / 0.2) * source_hz

    def write_report(path, reference):
        payload = {
            "result_path": str(result_path),
            "samples": {
                "Hz@0": [
                    {
                        "time": float(time),
                        "numerical": 0.0,
                        "reference": float(value),
                        "difference": float(-value),
                        "ratio_numerical_over_reference": 0.0,
                    }
                    for time, value in zip(times, reference)
                ]
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "fit.json"
    write_report(ip_report, ip_reference)
    write_report(noip_report, np.zeros_like(ip_reference))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--tau-candidates",
            "0.2",
            "--source-basis",
            "edge_current",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    np.testing.assert_allclose(result["source_amplitudes"]["Hz@0"], source_hz)
    np.testing.assert_allclose(result["fits"]["tau_0.2"]["scale"], scale)
    assert result["source_basis"] == "edge_current"
