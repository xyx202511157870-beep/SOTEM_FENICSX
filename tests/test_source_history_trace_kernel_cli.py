import json

import h5py
import numpy as np
import pytest
import yaml
from scipy.constants import mu_0

from atem3d.source_history_trace_kernel_cli import main
from atem3d.source_primary import (
    discrete_debye_history_basis,
    discrete_driven_relaxation_basis,
    discrete_relaxation_difference_basis,
)


def test_discrete_relaxation_difference_basis_matches_slow_minus_fast():
    time_steps = np.array([0.1, 0.2, 0.3])

    basis = discrete_relaxation_difference_basis(
        time_steps,
        slow_tau=1.0,
        fast_tau=0.2,
    )
    slow = discrete_debye_history_basis(
        time_steps,
        tau=1.0,
        max_order=0,
    ).values[:, 0]
    fast = discrete_debye_history_basis(
        time_steps,
        tau=0.2,
        max_order=0,
    ).values[:, 0]

    np.testing.assert_allclose(basis.times, np.r_[0.0, np.cumsum(time_steps)])
    np.testing.assert_allclose(basis.values, slow - fast)
    assert basis.values[0] == 0.0
    assert basis.basis_labels == ["BE relaxation difference slow-fast"]


def test_discrete_relaxation_difference_basis_rejects_invalid_tau_order():
    with pytest.raises(ValueError, match="fast_tau must be smaller than slow_tau"):
        discrete_relaxation_difference_basis([0.1], slow_tau=0.2, fast_tau=0.2)


def test_discrete_driven_relaxation_basis_matches_backward_euler_recursion():
    time_steps = np.array([0.1, 0.2, 0.1])
    driver_tau = 1.0
    response_tau = 0.2

    basis = discrete_driven_relaxation_basis(
        time_steps,
        driver_tau=driver_tau,
        response_tau=response_tau,
    )
    driver = discrete_debye_history_basis(
        time_steps,
        tau=driver_tau,
        max_order=0,
    ).values[:, 0]
    expected = np.zeros_like(driver)
    for index, dt in enumerate(time_steps):
        alpha = response_tau / (response_tau + dt)
        beta = dt / (response_tau + dt)
        expected[index + 1] = alpha * expected[index] + beta * driver[index + 1]

    np.testing.assert_allclose(basis.driver_values, driver)
    np.testing.assert_allclose(basis.values, expected)
    assert basis.values[0] == 0.0
    assert basis.basis_labels == ["BE driven relaxation response"]


def test_discrete_driven_relaxation_basis_matches_scaled_difference_for_uniform_dt():
    time_steps = np.full(4, 0.1)
    driver_tau = 1.0
    response_tau = 0.2

    driven = discrete_driven_relaxation_basis(
        time_steps,
        driver_tau=driver_tau,
        response_tau=response_tau,
    )
    difference = discrete_relaxation_difference_basis(
        time_steps,
        slow_tau=driver_tau,
        fast_tau=response_tau,
    )

    np.testing.assert_allclose(
        driven.values,
        driver_tau / (driver_tau - response_tau) * difference.values,
    )


def test_discrete_driven_relaxation_basis_rejects_invalid_tau_order():
    with pytest.raises(ValueError, match="response_tau must be smaller than driver_tau"):
        discrete_driven_relaxation_basis([0.1], driver_tau=0.2, response_tau=0.2)


def test_source_history_trace_kernel_cli_scans_rise_decay_basis(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    slow_tau = 0.5
    fast_tau = 0.1
    history = discrete_relaxation_difference_basis(
        np.full(3, 0.1),
        slow_tau=slow_tau,
        fast_tau=fast_tau,
    )
    sample_times = history.times[1:]
    coefficients = np.array([2.0, -3.0])
    coefficient_matrix = history.values[1:, None] * coefficients[None, :]
    matrix_report = {
        "diagnostic_only": True,
        "tau": slow_tau,
        "normalization": {
            "delta_sigma": 0.1,
            "source_length": 2.0,
            "mu": mu_0,
        },
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0", "source_face_moment:2"],
            "coefficient_matrix": {
                "shape": [3, 2],
                "values": coefficient_matrix.tolist(),
            },
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(value) for value in row],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    output = tmp_path / "rise_decay.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--fast-taus",
            "0.2",
            str(fast_tau),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["basis_kind"] == "discrete_relaxation_difference"
    assert payload["best_fast_tau"] == fast_tau
    best = payload["fits"][payload["best_key"]]
    assert best["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(best["coefficient_table"]["values"], [[2.0, -3.0]])
    np.testing.assert_allclose(
        best["coefficient_table_over_mu_delta_l2"]["values"],
        [[2.0 / (mu_0 * 0.1 * 4.0), -3.0 / (mu_0 * 0.1 * 4.0)]],
    )
    multi = payload["multi_basis_fit"]
    coefficient_table = np.asarray(multi["coefficient_table"]["values"], dtype=float)
    normalized_table = np.asarray(
        multi["coefficient_table_over_mu_delta_l2"]["values"],
        dtype=float,
    )
    np.testing.assert_allclose(
        multi["coefficient_sum_table"]["values"],
        [np.sum(coefficient_table, axis=0)],
    )
    np.testing.assert_allclose(
        multi["coefficient_sum_table_over_mu_delta_l2"]["values"],
        [np.sum(normalized_table, axis=0)],
    )


def test_source_history_trace_kernel_cli_scans_driven_relaxation_basis(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    driver_tau = 0.5
    response_tau = 0.1
    history = discrete_driven_relaxation_basis(
        np.full(3, 0.1),
        driver_tau=driver_tau,
        response_tau=response_tau,
    )
    sample_times = history.times[1:]
    coefficients = np.array([2.0, -3.0])
    coefficient_matrix = history.values[1:, None] * coefficients[None, :]
    matrix_report = {
        "diagnostic_only": True,
        "tau": driver_tau,
        "normalization": {
            "delta_sigma": 0.1,
            "source_length": 2.0,
            "mu": mu_0,
        },
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0", "source_face_moment:2"],
            "coefficient_matrix": {
                "shape": [3, 2],
                "values": coefficient_matrix.tolist(),
            },
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(value) for value in row],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    output = tmp_path / "driven_relaxation.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--basis-kind",
            "driven_relaxation",
            "--fast-taus",
            "0.2",
            str(response_tau),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["basis_kind"] == "discrete_driven_relaxation"
    assert payload["best_fast_tau"] == response_tau
    best = payload["fits"][payload["best_key"]]
    assert best["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(best["coefficient_table"]["values"], [[2.0, -3.0]])
    assert best["basis_labels"] == [
        "BE driven relaxation response response_tau=0.1"
    ]


def test_source_history_trace_kernel_cli_uses_recovery_sweep_tau_estimates(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump({"time_steps": [[0.1, 3]]})

    slow_tau = 0.5
    fast_tau = 0.1
    history = discrete_relaxation_difference_basis(
        np.full(3, 0.1),
        slow_tau=slow_tau,
        fast_tau=fast_tau,
    )
    sample_times = history.times[1:]
    coefficient_matrix = history.values[1:, None] * np.array([[2.0]])
    matrix_report = {
        "tau": slow_tau,
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0"],
            "coefficient_matrix": {
                "shape": [3, 1],
                "values": coefficient_matrix.tolist(),
            },
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(row[0])],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    sweep_report = {
        "sweep": {
            "cases": [
                {"diffusion_time_estimate": 0.2},
                {"diffusion_time_estimate": fast_tau},
                {"diffusion_time_estimate": fast_tau},
            ]
        }
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    sweep_report_path = tmp_path / "sweep.json"
    output = tmp_path / "from_sweep.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")
    sweep_report_path.write_text(json.dumps(sweep_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--fast-taus-from-recovery-sweep",
            str(sweep_report_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fast_taus"] == [0.2, fast_tau]
    assert payload["best_fast_tau"] == fast_tau
    assert payload["fast_tau_sources"] == {
        "explicit": [],
        "recovery_sweeps": [str(sweep_report_path)],
    }


def test_source_history_trace_kernel_cli_evaluates_prescribed_normalized_coefficients(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump({"time_steps": [[0.1, 3]]})

    slow_tau = 0.5
    fast_tau = 0.1
    history = discrete_relaxation_difference_basis(
        np.full(3, 0.1),
        slow_tau=slow_tau,
        fast_tau=fast_tau,
    )
    sample_times = history.times[1:]
    coefficients = np.array([2.0, -3.0])
    coefficient_matrix = history.values[1:, None] * coefficients[None, :]
    matrix_report = {
        "tau": slow_tau,
        "normalization": {
            "delta_sigma": 0.1,
            "source_length": 2.0,
            "mu": mu_0,
        },
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0", "source_face_moment:2"],
            "coefficient_matrix": {
                "shape": [3, 2],
                "values": coefficient_matrix.tolist(),
            },
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(value) for value in row],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    output = tmp_path / "prescribed.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--fast-taus",
            str(fast_tau),
            "--prescribed-normalized-coefficients",
            str(2.0 / (mu_0 * 0.1 * 4.0)),
            str(-3.0 / (mu_0 * 0.1 * 4.0)),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    prescribed = payload["prescribed_evaluation"]
    assert prescribed["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(
        prescribed["coefficient_table"]["values"],
        [[2.0, -3.0]],
    )
    np.testing.assert_allclose(
        prescribed["coefficient_table_over_mu_delta_l2"]["values"],
        [[2.0 / (mu_0 * 0.1 * 4.0), -3.0 / (mu_0 * 0.1 * 4.0)]],
    )


def test_source_history_trace_kernel_cli_accepts_negative_scientific_coefficients(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump({"time_steps": [[0.1, 3]]})

    slow_tau = 0.5
    fast_tau = 0.1
    history = discrete_relaxation_difference_basis(
        np.full(3, 0.1),
        slow_tau=slow_tau,
        fast_tau=fast_tau,
    )
    sample_times = history.times[1:]
    normalization_factor = mu_0 * 0.1 * 4.0
    normalized_coefficients = np.array([3.0e-9, -8.01270181928753e-9])
    coefficients = normalized_coefficients * normalization_factor
    coefficient_matrix = history.values[1:, None] * coefficients[None, :]
    matrix_report = {
        "tau": slow_tau,
        "normalization": {
            "delta_sigma": 0.1,
            "source_length": 2.0,
            "mu": mu_0,
        },
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0", "source_face_moment:2"],
            "coefficient_matrix": {
                "shape": [3, 2],
                "values": coefficient_matrix.tolist(),
            },
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(value) for value in row],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    output = tmp_path / "prescribed_scientific.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--fast-taus",
            str(fast_tau),
            "--prescribed-normalized-coefficients",
            "3e-09",
            "-8.01270181928753e-09",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    prescribed = payload["prescribed_evaluation"]
    assert prescribed["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(
        prescribed["coefficient_table_over_mu_delta_l2"]["values"],
        [normalized_coefficients],
    )


def test_source_history_trace_kernel_cli_filters_time_window(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump({"time_steps": [[0.1, 3]]})

    history = discrete_relaxation_difference_basis(
        np.full(3, 0.1),
        slow_tau=0.5,
        fast_tau=0.1,
    )
    sample_times = history.times[1:]
    coefficient_matrix = history.values[1:, None] * np.array([[2.0]])
    coefficient_matrix[0, 0] = 99.0
    matrix_report = {
        "tau": 0.5,
        "spatial_time_series": {
            "coefficient_names": ["source_face_moment:0"],
            "samples": [
                {
                    "time": float(time),
                    "coefficients": [float(row[0])],
                }
                for time, row in zip(sample_times, coefficient_matrix)
            ],
        },
    }
    matrix_report_path = tmp_path / "matrix_report.json"
    output = tmp_path / "windowed.json"
    matrix_report_path.write_text(json.dumps(matrix_report), encoding="utf-8")

    exit_code = main(
        [
            str(matrix_report_path),
            "--result",
            str(result_path),
            "--fast-taus",
            "0.1",
            "--time-min",
            "0.2",
            "--time-max",
            "0.3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["time_window"] == {
        "min": 0.2,
        "max": 0.3,
        "selected_count": 2,
    }
    best = payload["fits"][payload["best_key"]]
    np.testing.assert_allclose(best["coefficient_table"]["values"], [[2.0]])
    assert best["relative_l2"] < 1.0e-14
