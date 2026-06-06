import csv
import json

import numpy as np

from atem3d.cli import main
from atem3d.polarization_effect import write_polarization_effect_artifacts


def _write_response(path, times, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_obs", "Ex", "Ey", "dBzdt"])
        for time, values in zip(times, rows):
            writer.writerow([time, *values])


def test_write_polarization_effect_artifacts_computes_ip_minus_noip(tmp_path):
    times = np.array([1.0e-5, 1.0])
    noip_dir = tmp_path / "noip"
    ip_dir = tmp_path / "ip"
    _write_response(noip_dir / "predictions.csv", times, [[1.0, 0.0, 4.0], [2.0, 0.0, 8.0]])
    _write_response(ip_dir / "predictions.csv", times, [[1.5, 0.0, 5.0], [3.0, 0.0, 10.0]])
    _write_response(noip_dir / "reference_empymod_or_1d.csv", times, [[1.0, 0.0, 4.0], [2.0, 0.0, 8.0]])
    _write_response(ip_dir / "reference_empymod_or_1d.csv", times, [[1.25, 0.0, 5.5], [2.5, 0.0, 9.0]])

    summary = write_polarization_effect_artifacts(noip_dir, ip_dir, tmp_path / "effect")

    assert summary["component_names"] == ["Ex", "Ey", "dBzdt"]
    assert summary["time_min"] == 1.0e-5
    assert summary["time_max"] == 1.0
    pred_rows = list(csv.DictReader((tmp_path / "effect" / "polarization_effect_predictions.csv").open()))
    assert float(pred_rows[0]["Ex"]) == 0.5
    assert float(pred_rows[1]["dBzdt"]) == 2.0
    for name in (
        "polarization_effect_reference.csv",
        "polarization_effect_errors.csv",
        "polarization_effect_summary.json",
        "polarization_effect_comparison.png",
        "polarization_effect_error_curves.png",
    ):
        assert (tmp_path / "effect" / name).exists()


def test_polarization_effect_cli_writes_artifacts(tmp_path):
    times = [1.0e-5, 1.0]
    noip_dir = tmp_path / "noip"
    ip_dir = tmp_path / "ip"
    _write_response(noip_dir / "predictions.csv", times, [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    _write_response(ip_dir / "predictions.csv", times, [[2.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
    _write_response(noip_dir / "reference_empymod_or_1d.csv", times, [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    _write_response(ip_dir / "reference_empymod_or_1d.csv", times, [[2.0, 0.0, 2.0], [2.0, 0.0, 2.0]])

    exit_code = main(
        [
            "polarization-effect",
            str(noip_dir),
            str(ip_dir),
            "--output-dir",
            str(tmp_path / "effect_cli"),
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "effect_cli" / "polarization_effect_summary.json").read_text())
    assert payload["pass_all_components"] is True
