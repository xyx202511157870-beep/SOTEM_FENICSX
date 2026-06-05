import json
import math

import numpy as np


def test_source_diffusion_amplitude_scale_audit_cli_reports_pi_half_segment_scale(tmp_path):
    from atem3d.source_diffusion_amplitude_scale_audit_cli import main

    audit = {
        "items": [
            {
                "report": "driven.json",
                "fit_relative_l2": 0.12,
                "fit_amplitude_over_normalization": [-57.0],
                "compact_amplitude_over_normalization": [-3.6],
                "first_selected_compact_over_normalization": [
                    -3.6 / (5.0 * math.pi)
                ],
                "compact_optimal_scalar": 5.0 * math.pi,
                "compact_relative_l2": 0.93,
                "compact_first_gate_scalar": 5.0 * math.pi,
                "compact_first_gate_relative_l2": 0.08,
                "compact_first_gate_amplitude_over_normalization": [-18.0],
                "selected_times": [0.1, 0.2, 0.3],
                "compact_first_gate_time_error_fraction": [0.1, 0.7, 0.2],
                "compact_optimal_time_error_fraction": [0.6, 0.4, 0.0],
            }
        ]
    }
    geometry = {
        "source": {
            "length_m": 50.0,
            "midpoint_cell": {"widths_m": [5.0, 5.0, 1.0]},
        },
        "source_vector": {
            "active_count": 11,
            "active_count_by_orientation": {"x": 11, "y": 0, "z": 0},
        },
    }
    audit_path = tmp_path / "driven-audit.json"
    geometry_path = tmp_path / "geometry.json"
    output_path = tmp_path / "scale-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    status = main(
        [
            str(audit_path),
            "--geometry-report",
            str(geometry_path),
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    item = payload["items"][0]
    assert item["dominant_orientation"] == "x"
    assert item["source_length_over_along_cell_width"] == 10.0
    np.testing.assert_allclose(item["pi_over_2_segment_scale"], 5.0 * math.pi)
    np.testing.assert_allclose(
        item["compact_optimal_scalar_over_pi_half_segments"],
        1.0,
    )
    np.testing.assert_allclose(
        item["first_selected_response_per_compact_amplitude"],
        1.0 / (5.0 * math.pi),
    )
    np.testing.assert_allclose(
        item["compact_optimal_scalar_over_first_response_inverse"],
        1.0,
    )
    np.testing.assert_allclose(item["compact_first_gate_scalar"], 5.0 * math.pi)
    np.testing.assert_allclose(item["compact_first_gate_relative_l2"], 0.08)
    np.testing.assert_allclose(
        item["compact_first_gate_scalar_over_first_response_inverse"],
        1.0,
    )
    np.testing.assert_allclose(
        item["compact_first_gate_scalar_over_pi_half_segments"],
        1.0,
    )
    np.testing.assert_allclose(
        item["compact_first_gate_amplitude_over_normalization"],
        -18.0,
    )
    np.testing.assert_allclose(item["compact_first_gate_peak_error_time_s"], 0.2)
    np.testing.assert_allclose(item["compact_first_gate_peak_error_fraction"], 0.7)
    np.testing.assert_allclose(item["compact_optimal_peak_error_time_s"], 0.1)
    np.testing.assert_allclose(item["compact_optimal_peak_error_fraction"], 0.6)
    np.testing.assert_allclose(
        payload["summary"]["compact_first_gate_scalar_over_first_response_inverse_mean"],
        1.0,
    )
    np.testing.assert_allclose(
        payload["summary"]["compact_first_gate_relative_l2_mean"],
        0.08,
    )
    assert payload["summary"]["case_count"] == 1


def test_source_diffusion_amplitude_scale_audit_cli_reads_sweep_cases(tmp_path):
    from atem3d.source_diffusion_amplitude_scale_audit_cli import main

    audit_a = {
        "items": [
            {
                "fit_relative_l2": 0.1,
                "fit_amplitude_over_normalization": [-20.0],
                "compact_optimal_scalar": math.pi,
            }
        ]
    }
    audit_b = {
        "items": [
            {
                "fit_relative_l2": 0.2,
                "fit_amplitude_over_normalization": [-40.0],
                "compact_optimal_scalar": 2.0 * math.pi,
            }
        ]
    }
    geometry = {
        "cases": {
            "case_a": {
                "source": {
                    "length_m": 20.0,
                    "midpoint_cell": {"widths_m": [10.0, 5.0, 1.0]},
                },
                "source_vector": {
                    "active_count_by_orientation": {"x": 3, "y": 0, "z": 0},
                },
            },
            "case_b": {
                "source": {
                    "length_m": 20.0,
                    "midpoint_cell": {"widths_m": [5.0, 5.0, 1.0]},
                },
                "source_vector": {
                    "active_count_by_orientation": {"x": 5, "y": 0, "z": 0},
                },
            },
        }
    }
    audit_a_path = tmp_path / "audit-a.json"
    audit_b_path = tmp_path / "audit-b.json"
    geometry_path = tmp_path / "geometry.json"
    output_path = tmp_path / "scale-audit.json"
    audit_a_path.write_text(json.dumps(audit_a), encoding="utf-8")
    audit_b_path.write_text(json.dumps(audit_b), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")

    status = main(
        [
            str(audit_a_path),
            str(audit_b_path),
            "--geometry-report",
            str(geometry_path),
            "--case-keys",
            "case_a",
            "case_b",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        [
            item["compact_optimal_scalar_over_pi_half_segments"]
            for item in payload["items"]
        ],
        [1.0, 1.0],
    )
