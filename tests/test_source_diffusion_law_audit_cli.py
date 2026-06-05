import json

import numpy as np


def test_source_diffusion_law_audit_cli_recovers_global_normalized_law(tmp_path):
    from atem3d.source_diffusion_law_audit_cli import main

    normalized_amplitude = -3.5
    multiplier = 1.25
    report_paths = []
    for index, tau0 in enumerate([2.0e-5, 4.0e-5]):
        times = np.linspace(1.0e-4, 1.3e-4, 4)
        coefficients = (
            normalized_amplitude
            * tau0
            * np.exp(-(times - times[0]) / (multiplier * tau0))
        )
        report = {
            "source_diffusion_time_s": tau0,
            "time_min": float(times[0]),
            "time_max": float(times[-1]),
            "time_count": int(times.size),
            "candidate_static_fits": {
                "active_source": {
                    "per_time_coefficients": [
                        float(value) for value in coefficients
                    ],
                    "per_time_coefficient_first": float(coefficients[0]),
                    "coefficient_kernel_fits": [
                        {
                            "multiplier": multiplier,
                            "tau": multiplier * tau0,
                            "amplitude": normalized_amplitude * tau0,
                            "coefficient_relative_l2": 0.0,
                            "corrected_relative_l2": 0.0,
                        }
                    ],
                }
            },
        }
        path = tmp_path / f"audit-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)

    output = tmp_path / "law-audit.json"
    status = main([*(str(path) for path in report_paths), "-o", str(output)])

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["candidate"] == "active_source"
    assert payload["report_count"] == 2
    assert payload["per_report"][0]["best_multiplier"] == multiplier
    np.testing.assert_allclose(
        payload["per_report"][0]["normalized_best_amplitude"],
        normalized_amplitude,
    )
    np.testing.assert_allclose(
        payload["normalization_summary"]["normalized_best_amplitude_mean"],
        normalized_amplitude,
    )
    assert len(payload["global_law_fits"]) == 1
    global_fit = payload["global_law_fits"][0]
    assert global_fit["multiplier"] == multiplier
    np.testing.assert_allclose(
        global_fit["normalized_amplitude"],
        normalized_amplitude,
    )
    assert global_fit["combined_coefficient_relative_l2"] < 1.0e-12
    assert payload["best_global_law"]["multiplier"] == multiplier


def test_source_diffusion_law_audit_cli_recovers_be_normalized_law(tmp_path):
    from atem3d.source_diffusion_law_audit_cli import main

    normalized_amplitude = -2.25
    multiplier = 1.5
    tau0 = 2.0e-5
    dt = 1.0e-5
    times = np.array([3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5])
    tau = multiplier * tau0
    alpha = tau / (tau + dt)
    steps = np.rint(times / dt).astype(int)
    first_step = int(steps[0])
    coefficients = normalized_amplitude * tau0 * alpha ** (steps - first_step)
    report = {
        "source_diffusion_time_s": tau0,
        "time_min": float(times[0]),
        "time_max": float(times[-1]),
        "time_count": int(times.size),
        "candidate_static_fits": {
            "active_source": {
                "per_time_coefficients": [float(value) for value in coefficients],
                "per_time_coefficient_first": float(coefficients[0]),
                "coefficient_kernel_fits": [
                    {
                        "multiplier": 1.0,
                        "tau": tau0,
                        "amplitude": 0.0,
                        "coefficient_relative_l2": 1.0,
                    }
                ],
            }
        },
    }
    report_path = tmp_path / "audit.json"
    output = tmp_path / "law-audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    status = main(
        [
            str(report_path),
            "--basis-kind",
            "be_decay",
            "--multipliers",
            "1.0",
            str(multiplier),
            "-o",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["basis_kind"] == "be_decay"
    assert payload["per_report"][0]["best_multiplier"] == multiplier
    np.testing.assert_allclose(
        payload["per_report"][0]["normalized_best_amplitude"],
        normalized_amplitude,
    )
    global_fit = payload["best_global_law"]
    assert global_fit["multiplier"] == multiplier
    np.testing.assert_allclose(
        global_fit["normalized_amplitude"],
        normalized_amplitude,
    )
    assert global_fit["combined_coefficient_relative_l2"] < 1.0e-12
    effective = payload["per_report"][0]["effective_decay"]
    assert effective["valid_pair_count"] == 3
    np.testing.assert_allclose(
        effective["be_multiplier_weighted_mean"],
        multiplier,
    )
    np.testing.assert_allclose(
        payload["effective_decay_summary"]["be_multiplier_weighted_mean_mean"],
        multiplier,
    )
    suggestion = payload["diagnostic_source_history_suggestion"]
    assert suggestion["diagnostic_only"] is True
    assert suggestion["kind"] == "source_diffusion_kernel_source_moments"
    assert suggestion["basis_kind"] == "be_decay"
    assert suggestion["source_moment_degrees"] == [0]
    assert suggestion["receiver_matrix"] == "auto"
    assert suggestion["amplitude_time_consistent"] is True
    assert suggestion["amplitude_time"] == float(times[0])
    assert suggestion["tau_multiplier"] == multiplier
    np.testing.assert_allclose(
        suggestion["normalized_amplitude"],
        normalized_amplitude,
    )


def test_source_diffusion_law_audit_cli_summarizes_direction_constraints(tmp_path):
    from atem3d.source_diffusion_law_audit_cli import main

    report_paths = []
    for index, projection in enumerate([0.99, 0.95]):
        report = {
            "source_diffusion_time_s": 2.0e-5,
            "time_min": 1.0e-4,
            "time_max": 1.2e-4,
            "time_count": 3,
            "candidate_static_fits": {
                "active_source": {
                    "per_time_coefficients": [-7.0e-5, -4.0e-5, -2.0e-5],
                    "per_time_coefficient_first": -7.0e-5,
                    "all_window_residual_relative_l2": 0.9,
                    "all_window_residual_projection_fraction": 0.1 + index * 0.1,
                    "per_time_residual_relative_l2": float(np.sqrt(1.0 - projection)),
                    "per_time_residual_projection_fraction": projection,
                    "coefficient_kernel_fits": [
                        {
                            "multiplier": 1.0,
                            "tau": 2.0e-5,
                            "amplitude": -7.0e-5,
                            "coefficient_relative_l2": 0.2,
                        }
                    ],
                }
            },
        }
        path = tmp_path / f"audit-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)

    output = tmp_path / "law-audit.json"
    status = main([*(str(path) for path in report_paths), "-o", str(output)])

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = payload["direction_constraint_summary"]
    assert summary["available_report_count"] == 2
    assert summary["missing_report_count"] == 0
    assert summary["per_time_residual_projection_fraction_min"] == 0.95
    assert summary["per_time_residual_projection_fraction_mean"] == 0.97
    assert summary["all_window_residual_projection_fraction_mean"] == 0.15000000000000002
    assert payload["per_report"][0]["direction_metrics"][
        "per_time_residual_projection_fraction"
    ] == 0.99
