import json

import numpy as np


def test_time_warp_audit_cli_recovers_known_affine_time_map(tmp_path):
    from atem3d.time_warp_audit_cli import main

    times = np.linspace(0.1, 1.0, 37)
    reference = np.exp(-2.0 * times) + 0.05 * np.sin(9.0 * times)
    true_scale = 1.05
    true_shift = -0.02
    mapped_times = true_scale * times + true_shift
    numerical = np.interp(mapped_times, times, reference)
    report = {
        "samples": {
            "Hz@x=0": [
                {
                    "time": float(time),
                    "numerical": float(num),
                    "reference": float(ref),
                }
                for time, num, ref in zip(times, numerical, reference)
            ]
        }
    }
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    status = main(
        [
            str(report_path),
            "--components",
            "Hz@x=0",
            "--scale-min",
            "1.0",
            "--scale-max",
            "1.1",
            "--scale-count",
            "51",
            "--shift-min",
            "-0.04",
            "--shift-max",
            "0.0",
            "--shift-count",
            "81",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    item = payload["components"]["Hz@x=0"]
    assert item["base_relative_l2"] > 0.01
    assert item["best_relative_l2"] < 1.0e-12
    assert item["best_scale"] == true_scale
    assert item["best_shift"] == true_shift
