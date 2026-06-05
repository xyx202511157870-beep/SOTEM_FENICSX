import json

import numpy as np
import yaml

from atem3d.config import build_simulation


def _eb_config(include_ip: bool = True) -> dict:
    model = {"sigma_infinity": 0.1}
    if include_ip:
        model["debye_terms"] = [{"delta_sigma": 0.02, "tau": 0.05}]
    return {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": model,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01, 0.02],
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_subdivisions": 2,
        "magnetic_recovery_initial_polarization_scale": 0.25,
        "receivers": [
            {"location": [0.0, 0.75, 0.0], "component": "Hz"},
            {"location": [0.0, 0.75, 0.0], "component": "Ex"},
        ],
    }


def test_eb_magnetic_recovery_decomposition_sums_to_runtime_receiver_data():
    from atem3d.magnetic_recovery_decomposition import (
        magnetic_recovery_decomposition_at_time,
    )

    sim = build_simulation(_eb_config(include_ip=True))
    result = sim.run()

    report = magnetic_recovery_decomposition_at_time(sim, result, time_index=1)

    assert report["receiver_indices"] == [0]
    assert report["components"] == ["Hz"]
    assert "ohmic_current" in report["terms"]
    assert "polarization_memory" in report["terms"]
    assert "initial_polarization" in report["terms"]
    np.testing.assert_allclose(report["total_data"], result.data[1, [0]], atol=1.0e-14)
    np.testing.assert_allclose(report["data_residual"], [0.0], atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(report["terms"]["polarization_memory"]["data"]),
        np.asarray(report["term_details"]["polarization_memory_0"]["data"]),
    )


def test_eb_magnetic_recovery_decomposition_has_zero_ip_terms_without_ip():
    from atem3d.magnetic_recovery_decomposition import (
        magnetic_recovery_decomposition_at_time,
    )

    sim = build_simulation(_eb_config(include_ip=False))
    result = sim.run()

    report = magnetic_recovery_decomposition_at_time(sim, result, time_index=1)

    np.testing.assert_allclose(report["terms"]["polarization_memory"]["data"], [0.0])
    np.testing.assert_allclose(report["terms"]["initial_polarization"]["data"], [0.0])
    np.testing.assert_allclose(report["total_data"], result.data[1, [0]], atol=1.0e-14)


def test_eb_edge_basis_cell_decomposition_handles_low_frequency_ratio_scale():
    from atem3d.magnetic_recovery_decomposition import (
        magnetic_recovery_decomposition_at_time,
    )

    config = _eb_config(include_ip=True)
    config["magnetic_receiver_mode"] = "edge_basis_cell_biot"
    config["magnetic_recovery_polarization_scale"] = "low_frequency_ratio"
    sim = build_simulation(config)
    result = sim.run()

    report = magnetic_recovery_decomposition_at_time(sim, result, time_index=1)

    np.testing.assert_allclose(report["total_data"], result.data[1, [0]], atol=1.0e-14)
    np.testing.assert_allclose(report["data_residual"], [0.0], atol=1.0e-14)


def test_magnetic_recovery_decomposition_cli_writes_report(tmp_path):
    from atem3d.magnetic_recovery_decomposition_cli import main

    config_path = tmp_path / "case.yaml"
    output_path = tmp_path / "decomposition.json"
    config_path.write_text(yaml.safe_dump(_eb_config(include_ip=True)), encoding="utf-8")

    status = main(
        [
            str(config_path),
            "--time-indices",
            "1",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["formulation"] == "eb"
    assert payload["magnetic_receiver_mode"] == "current_biot"
    assert payload["samples"][0]["time_index"] == 1
    assert payload["summary"]["total_recomputed_relative_l2_to_numerical"] < 1.0e-12


def test_decomposition_alignment_recovers_synthetic_validation_residual():
    from atem3d.magnetic_recovery_decomposition import (
        align_decomposition_samples_to_validation,
        magnetic_recovery_decomposition_at_time,
    )

    sim = build_simulation(_eb_config(include_ip=True))
    result = sim.run()
    samples = [
        magnetic_recovery_decomposition_at_time(sim, result, time_index=index)
        for index in (1, 2)
    ]
    rows = []
    for sample in samples:
        numerical = sample["numerical_data"][0]
        residual = 2.0 * sample["terms"]["polarization_memory"]["data"][0]
        rows.append(
            {
                "time": sample["time"],
                "numerical": numerical,
                "reference": numerical + residual,
                "difference": -residual,
            }
        )
    validation = {
        "samples": {"Hz": rows},
        "component_groups": {"magnetic": {"components": ["Hz"]}},
    }

    alignment = align_decomposition_samples_to_validation(samples, validation)

    polarization = alignment["term_alignment"]["polarization_memory"]
    np.testing.assert_allclose(polarization["best_scalar"], 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(
        polarization["corrected_existing_term_multiplier"],
        3.0,
        rtol=1.0e-12,
    )
    assert polarization["relative_l2_after_best_scalar"] < 1.0e-12
    assert alignment["best_single_term"]["name"] == "polarization_memory"


def test_decomposition_alignment_reports_per_time_scalars():
    from atem3d.magnetic_recovery_decomposition import (
        align_decomposition_samples_to_validation,
        magnetic_recovery_decomposition_at_time,
    )

    sim = build_simulation(_eb_config(include_ip=True))
    result = sim.run()
    samples = [
        magnetic_recovery_decomposition_at_time(sim, result, time_index=index)
        for index in (1, 2)
    ]
    rows = []
    for scale, sample in zip((2.0, 3.0), samples):
        numerical = sample["numerical_data"][0]
        residual = scale * sample["terms"]["polarization_memory"]["data"][0]
        rows.append(
            {
                "time": sample["time"],
                "numerical": numerical,
                "reference": numerical + residual,
                "difference": -residual,
            }
        )
    validation = {
        "samples": {"Hz": rows},
        "component_groups": {"magnetic": {"components": ["Hz"]}},
    }

    alignment = align_decomposition_samples_to_validation(samples, validation)

    per_time = alignment["per_time_alignment"]
    assert [item["time"] for item in per_time] == [samples[0]["time"], samples[1]["time"]]
    np.testing.assert_allclose(
        [
            item["term_alignment"]["polarization_memory"]["best_scalar"]
            for item in per_time
        ],
        [2.0, 3.0],
        rtol=1.0e-12,
    )


def test_magnetic_recovery_decomposition_cli_aligns_validation_report(tmp_path):
    from atem3d.magnetic_recovery_decomposition import (
        magnetic_recovery_decomposition_at_time,
    )
    from atem3d.magnetic_recovery_decomposition_cli import main

    config = _eb_config(include_ip=True)
    sim = build_simulation(config)
    result = sim.run()
    samples = [
        magnetic_recovery_decomposition_at_time(sim, result, time_index=index)
        for index in (1, 2)
    ]
    rows = []
    for sample in samples:
        numerical = sample["numerical_data"][0]
        residual = 1.5 * sample["terms"]["initial_polarization"]["data"][0]
        rows.append(
            {
                "time": sample["time"],
                "numerical": numerical,
                "reference": numerical + residual,
                "difference": -residual,
            }
        )
    validation = {
        "samples": {"Hz": rows},
        "component_groups": {"magnetic": {"components": ["Hz"]}},
    }
    config_path = tmp_path / "case.yaml"
    validation_path = tmp_path / "validation.json"
    output_path = tmp_path / "decomposition.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    status = main(
        [
            str(config_path),
            "--time-indices",
            "1",
            "2",
            "--validation-report",
            str(validation_path),
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    alignment = payload["validation_alignment"]
    np.testing.assert_allclose(
        alignment["term_alignment"]["initial_polarization"]["best_scalar"],
        1.5,
    )
    assert alignment["best_single_term"]["name"] == "initial_polarization"
