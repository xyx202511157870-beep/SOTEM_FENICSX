import json

import numpy as np


def test_recovery_driven_trace_audit_cli_recovers_known_amplitude(tmp_path):
    from atem3d.recovery_driven_trace_audit_cli import main

    factor = 2.0 * 3.0 * 4.0**2
    normalized = np.array([2.0, -1.0])
    physical = normalized * factor
    driven_coefficients = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    target_matrix = np.einsum("tdm,d->tm", driven_coefficients[1:], physical)
    driven_report = {
        "driven_response": {
            "driver_kind": "relaxation_difference",
            "driver_fast_tau": 0.1,
            "driver_tau": 1.0,
            "source_projection": "raw",
            "initial_state_kind": "zero",
            "forcing_kind": "source_edge_rhs",
            "times": [0.0, 0.1, 0.2],
            "source_moment_projection": {
                "coefficients": driven_coefficients.tolist(),
            },
        }
    }
    target_report = {
        "normalization": {
            "mu": 2.0,
            "delta_sigma": 3.0,
            "source_length": 4.0,
        },
        "spatial_time_series": {
            "samples": [
                {"time": 0.1, "coefficients": target_matrix[0].tolist()},
                {"time": 0.2, "coefficients": target_matrix[1].tolist()},
            ]
        },
    }
    driven_path = tmp_path / "driven.json"
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "audit.json"
    driven_path.write_text(json.dumps(driven_report), encoding="utf-8")
    target_path.write_text(json.dumps(target_report), encoding="utf-8")

    status = main(
        [
            str(driven_path),
            "--target-matrix-report",
            str(target_path),
            "--compact-normalized-amplitude",
            "2.0",
            "-1.0",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    item = payload["items"][0]
    np.testing.assert_allclose(item["fit_amplitude_over_mu_delta_l2"], normalized)
    assert item["fit_relative_l2"] < 1.0e-14
    assert item["compact_relative_l2"] < 1.0e-14


def test_recovery_driven_trace_audit_cli_accepts_source_neighborhood_target(tmp_path):
    from atem3d.recovery_driven_trace_audit_cli import main

    tau0 = 2.0e-5
    normalized = np.array([-3.5])
    physical = normalized * tau0
    driven_coefficients = np.array(
        [
            [[0.0]],
            [[1.0]],
            [[2.0]],
        ]
    )
    target_matrix = np.einsum("tdm,d->tm", driven_coefficients[1:], physical)
    driven_report = {
        "driven_response": {
            "driver_kind": "relaxation_difference",
            "driver_fast_tau": tau0,
            "driver_tau": 1.0e-3,
            "source_projection": "raw",
            "initial_state_kind": "zero",
            "forcing_kind": "source_edge_rhs",
            "times": [0.0, 1.0e-4, 2.0e-4],
            "source_moment_projection": {
                "coefficients": driven_coefficients.tolist(),
            },
        }
    }
    source_neighborhood_report = {
        "source_diffusion_time_s": tau0,
        "time_min": 1.0e-4,
        "time_max": 2.0e-4,
        "time_count": 2,
        "candidate_static_fits": {
            "active_source": {
                "per_time_coefficients": target_matrix[:, 0].tolist(),
            }
        },
    }
    driven_path = tmp_path / "driven.json"
    target_path = tmp_path / "source-neighborhood.json"
    output_path = tmp_path / "audit.json"
    driven_path.write_text(json.dumps(driven_report), encoding="utf-8")
    target_path.write_text(json.dumps(source_neighborhood_report), encoding="utf-8")

    status = main(
        [
            str(driven_path),
            "--target-source-neighborhood-report",
            str(target_path),
            "--compact-normalized-amplitude",
            "-3.5",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["target_kind"] == "source_neighborhood"
    assert payload["normalization"]["factor"] == tau0
    item = payload["items"][0]
    np.testing.assert_allclose(item["fit_amplitude_over_normalization"], normalized)
    assert item["fit_relative_l2"] < 1.0e-14
    assert item["compact_relative_l2"] < 1.0e-14


def test_recovery_driven_trace_audit_cli_reports_driver_following_error(tmp_path):
    from atem3d.recovery_driven_trace_audit_cli import main

    factor = 2.0 * 3.0 * 4.0**2
    normalized = np.array([1.5])
    physical = normalized * factor
    driver_values = np.array([1.0, 0.5, 0.25, 0.125])
    driven_coefficients = (2.0 * driver_values).reshape(-1, 1, 1)
    target_matrix = np.einsum("tdm,d->tm", driven_coefficients[1:], physical)
    driven_report = {
        "driven_response": {
            "driver_kind": "debye_decay",
            "driver_tau": 1.0,
            "driver_values": driver_values.tolist(),
            "source_projection": "raw",
            "initial_state_kind": "zero",
            "forcing_kind": "source_edge_rhs",
            "times": [0.0, 0.1, 0.2, 0.3],
            "source_moment_projection": {
                "coefficients": driven_coefficients.tolist(),
            },
        }
    }
    target_report = {
        "normalization": {
            "mu": 2.0,
            "delta_sigma": 3.0,
            "source_length": 4.0,
        },
        "spatial_time_series": {
            "samples": [
                {"time": 0.1, "coefficients": target_matrix[0].tolist()},
                {"time": 0.2, "coefficients": target_matrix[1].tolist()},
                {"time": 0.3, "coefficients": target_matrix[2].tolist()},
            ]
        },
    }
    driven_path = tmp_path / "driven.json"
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "audit.json"
    driven_path.write_text(json.dumps(driven_report), encoding="utf-8")
    target_path.write_text(json.dumps(target_report), encoding="utf-8")

    status = main(
        [
            str(driven_path),
            "--target-matrix-report",
            str(target_path),
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    item = json.loads(output_path.read_text(encoding="utf-8"))["items"][0]
    np.testing.assert_allclose(item["selected_driver_values"], [0.5, 0.25, 0.125])
    assert item["driver_follow_column_count"] == 1
    assert item["driver_follow_relative_l2_mean"] < 1.0e-14
    assert item["driver_follow_relative_l2_max"] < 1.0e-14


def test_recovery_driven_trace_audit_cli_reports_first_gate_compact_scaling(
    tmp_path,
):
    from atem3d.recovery_driven_trace_audit_cli import main

    tau0 = 2.0e-5
    compact_normalized = np.array([-4.0])
    driven_coefficients = np.array(
        [
            [[0.0]],
            [[0.25]],
            [[0.10]],
        ]
    )
    target_matrix = (
        compact_normalized[0]
        * tau0
        * np.array(
            [
                [1.0],
                [0.5],
            ]
        )
    )
    driven_report = {
        "driven_response": {
            "driver_kind": "debye_decay",
            "driver_fast_tau": tau0,
            "driver_tau": 1.0e-3,
            "source_projection": "raw",
            "initial_state_kind": "initial_source",
            "forcing_kind": "homogeneous",
            "times": [0.0, 1.0e-4, 2.0e-4],
            "source_moment_projection": {
                "coefficients": driven_coefficients.tolist(),
            },
        }
    }
    source_neighborhood_report = {
        "source_diffusion_time_s": tau0,
        "time_min": 1.0e-4,
        "time_max": 2.0e-4,
        "time_count": 2,
        "candidate_static_fits": {
            "active_source": {
                "per_time_coefficients": target_matrix[:, 0].tolist(),
            }
        },
    }
    driven_path = tmp_path / "driven.json"
    target_path = tmp_path / "source-neighborhood.json"
    output_path = tmp_path / "audit.json"
    driven_path.write_text(json.dumps(driven_report), encoding="utf-8")
    target_path.write_text(json.dumps(source_neighborhood_report), encoding="utf-8")

    status = main(
        [
            str(driven_path),
            "--target-source-neighborhood-report",
            str(target_path),
            "--compact-normalized-amplitude",
            str(compact_normalized[0]),
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    item = json.loads(output_path.read_text(encoding="utf-8"))["items"][0]
    np.testing.assert_allclose(item["selected_times"], [1.0e-4, 2.0e-4])
    np.testing.assert_allclose(item["first_selected_compact_over_normalization"], [-1.0])
    np.testing.assert_allclose(item["compact_first_gate_scalar"], 4.0)
    np.testing.assert_allclose(
        item["compact_first_gate_amplitude_over_normalization"],
        [-16.0],
    )
    np.testing.assert_allclose(
        item["first_selected_compact_first_gate_scaled_over_normalization"],
        [-4.0],
    )
    np.testing.assert_allclose(
        item["last_selected_compact_first_gate_scaled_over_normalization"],
        [-1.6],
    )
    np.testing.assert_allclose(
        item["compact_first_gate_relative_l2"],
        0.4 / np.linalg.norm([-4.0, -2.0]),
    )
    np.testing.assert_allclose(
        item["compact_time_relative_l2"],
        [0.75, 0.8],
    )
    np.testing.assert_allclose(
        item["compact_first_gate_time_relative_l2"],
        [0.0, 0.2],
    )
    np.testing.assert_allclose(
        item["compact_first_gate_time_error_over_target_norm"],
        [0.0, 0.4 / np.linalg.norm([-4.0, -2.0])],
    )
    np.testing.assert_allclose(
        item["compact_first_gate_time_error_fraction"],
        [0.0, 1.0],
    )
    compact_optimal_scalar = (4.0 + 0.8) / (1.0 + 0.16)
    np.testing.assert_allclose(
        item["compact_optimal_time_relative_l2"],
        [
            abs(compact_optimal_scalar - 4.0) / 4.0,
            abs(0.4 * compact_optimal_scalar - 2.0) / 2.0,
        ],
    )
    np.testing.assert_allclose(
        item["compact_optimal_time_error_over_target_norm"],
        [
            abs(compact_optimal_scalar - 4.0) / np.linalg.norm([-4.0, -2.0]),
            abs(0.4 * compact_optimal_scalar - 2.0)
            / np.linalg.norm([-4.0, -2.0]),
        ],
    )


def test_recovery_driven_trace_audit_cli_combines_reports_with_weights(tmp_path):
    from atem3d.recovery_driven_trace_audit_cli import main

    factor = 2.0 * 3.0 * 4.0**2
    normalized = np.array([2.0, -1.0])
    physical = normalized * factor
    report_a_coefficients = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    report_b_coefficients = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, -1.0]],
            [[1.0, 0.0], [0.0, -2.0]],
        ]
    )
    weights = np.array([1.0, -0.5])
    combined_coefficients = (
        weights[0] * report_a_coefficients + weights[1] * report_b_coefficients
    )
    target_matrix = np.einsum("tdm,d->tm", combined_coefficients[1:], physical)

    def driven_report(coefficients, fast_tau):
        return {
            "driven_response": {
                "driver_kind": "relaxation_difference",
                "driver_fast_tau": fast_tau,
                "driver_tau": 1.0,
                "source_projection": "raw",
                "initial_state_kind": "zero",
                "forcing_kind": "source_edge_rhs",
                "times": [0.0, 0.1, 0.2],
                "source_moment_projection": {
                    "coefficients": coefficients.tolist(),
                },
            }
        }

    target_report = {
        "normalization": {
            "mu": 2.0,
            "delta_sigma": 3.0,
            "source_length": 4.0,
        },
        "spatial_time_series": {
            "samples": [
                {"time": 0.1, "coefficients": target_matrix[0].tolist()},
                {"time": 0.2, "coefficients": target_matrix[1].tolist()},
            ]
        },
    }
    report_a_path = tmp_path / "driven_a.json"
    report_b_path = tmp_path / "driven_b.json"
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "audit.json"
    report_a_path.write_text(json.dumps(driven_report(report_a_coefficients, 0.1)))
    report_b_path.write_text(json.dumps(driven_report(report_b_coefficients, 0.2)))
    target_path.write_text(json.dumps(target_report), encoding="utf-8")

    status = main(
        [
            str(report_a_path),
            str(report_b_path),
            "--target-matrix-report",
            str(target_path),
            "--compact-normalized-amplitude",
            "2.0",
            "-1.0",
            "--combine-report-weights",
            "1.0",
            "-0.5",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    combined = payload["combined"]
    assert combined["report_weights"] == [1.0, -0.5]
    np.testing.assert_allclose(combined["fit_amplitude_over_mu_delta_l2"], normalized)
    assert combined["fit_relative_l2"] < 1.0e-14
    assert combined["compact_relative_l2"] < 1.0e-14


def test_recovery_driven_trace_audit_cli_fits_compact_report_weights(tmp_path):
    from atem3d.recovery_driven_trace_audit_cli import main

    factor = 2.0 * 3.0 * 4.0**2
    normalized = np.array([2.0, -1.0])
    physical = normalized * factor
    report_a_coefficients = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    report_b_coefficients = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [2.0, 0.0]],
            [[0.0, 2.0], [4.0, 0.0]],
        ]
    )
    report_weights = np.array([0.25, 1.5])
    combined_coefficients = (
        report_weights[0] * report_a_coefficients
        + report_weights[1] * report_b_coefficients
    )
    target_matrix = np.einsum("tdm,d->tm", combined_coefficients[1:], physical)

    def driven_report(coefficients):
        return {
            "driven_response": {
                "driver_kind": "relaxation_difference",
                "driver_fast_tau": 0.1,
                "driver_tau": 1.0,
                "source_projection": "raw",
                "initial_state_kind": "zero",
                "forcing_kind": "source_edge_rhs",
                "times": [0.0, 0.1, 0.2],
                "source_moment_projection": {
                    "coefficients": coefficients.tolist(),
                },
            }
        }

    target_report = {
        "normalization": {
            "mu": 2.0,
            "delta_sigma": 3.0,
            "source_length": 4.0,
        },
        "spatial_time_series": {
            "samples": [
                {"time": 0.1, "coefficients": target_matrix[0].tolist()},
                {"time": 0.2, "coefficients": target_matrix[1].tolist()},
            ]
        },
    }
    report_a_path = tmp_path / "driven_a.json"
    report_b_path = tmp_path / "driven_b.json"
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "audit.json"
    report_a_path.write_text(json.dumps(driven_report(report_a_coefficients)))
    report_b_path.write_text(json.dumps(driven_report(report_b_coefficients)))
    target_path.write_text(json.dumps(target_report), encoding="utf-8")

    status = main(
        [
            str(report_a_path),
            str(report_b_path),
            "--target-matrix-report",
            str(target_path),
            "--compact-normalized-amplitude",
            "2.0",
            "-1.0",
            "--fit-compact-report-weights",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    fit = payload["compact_report_weight_fit"]
    np.testing.assert_allclose(fit["fit_report_weights"], report_weights)
    assert fit["fit_relative_l2"] < 1.0e-14
