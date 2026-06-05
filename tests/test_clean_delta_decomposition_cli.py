import json

import numpy as np


def _sample_rows(times, numerical, reference):
    return [
        {
            "time": float(time),
            "numerical": float(num),
            "reference": float(ref),
        }
        for time, num, ref in zip(times, numerical, reference)
    ]


def test_clean_delta_decomposition_cli_reports_clean_delta_mismatch(tmp_path):
    from atem3d.clean_delta_decomposition_cli import main

    times = np.array([0.1, 0.2, 0.3])
    noip_reference = np.array([10.0, 8.0, 6.0])
    noip_error = np.array([1.0, -0.5, 0.25])
    noip_numerical = noip_reference + noip_error

    ip_reference = noip_reference + np.array([2.0, 1.5, 1.0])
    raw_ip_error = np.array([4.0, 2.0, -1.0])
    raw_ip_numerical = ip_reference + raw_ip_error

    ideal_clean_delta = noip_error - raw_ip_error
    actual_correction = ideal_clean_delta + np.array([0.2, -0.1, 0.0])
    corrected_ip_numerical = raw_ip_numerical + actual_correction

    raw_path = tmp_path / "raw_ip.json"
    corrected_path = tmp_path / "corrected_ip.json"
    noip_path = tmp_path / "noip.json"
    output_path = tmp_path / "decomposition.json"
    raw_path.write_text(
        json.dumps(
            {
                "samples": {
                    "Hz@x=0": _sample_rows(times, raw_ip_numerical, ip_reference),
                    "Ex@x=0": _sample_rows(times, raw_ip_numerical * 2.0, ip_reference * 2.0),
                }
            }
        ),
        encoding="utf-8",
    )
    corrected_path.write_text(
        json.dumps(
            {
                "samples": {
                    "Hz@x=0": _sample_rows(
                        times,
                        corrected_ip_numerical,
                        ip_reference,
                    ),
                    "Ex@x=0": _sample_rows(
                        times,
                        corrected_ip_numerical * 2.0,
                        ip_reference * 2.0,
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    noip_path.write_text(
        json.dumps(
            {
                "samples": {
                    "Hz@x=0": _sample_rows(times, noip_numerical, noip_reference),
                    "Ex@x=0": _sample_rows(times, noip_numerical * 2.0, noip_reference * 2.0),
                }
            }
        ),
        encoding="utf-8",
    )

    status = main(
        [
            str(raw_path),
            str(corrected_path),
            str(noip_path),
            "--components",
            "Hz@x=0",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["component_names"] == ["Hz@x=0"]
    assert payload["identity"]["max_abs_residual"] < 1.0e-14

    item = payload["components"]["Hz@x=0"]
    np.testing.assert_allclose(item["first_ideal_clean_delta"], ideal_clean_delta[0])
    np.testing.assert_allclose(item["first_actual_correction"], actual_correction[0])
    np.testing.assert_allclose(item["first_correction_mismatch"], 0.2)
    assert item["actual_correction_relative_l2_to_ideal"] > 0.0
    assert item["corrected_ip_error_relative_l2"] < item["raw_ip_error_relative_l2"]
    assert item["correction_mismatch_relative_l2_against_ip_reference"] < (
        item["raw_ip_error_relative_l2"] - item["noip_baseline_relative_l2_against_ip_reference"]
    )
