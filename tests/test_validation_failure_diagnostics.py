import json

import numpy as np

from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)


def test_failed_validation_writes_structured_task_book_diagnostics(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 1.0e-12, 2.0e-9],
            [0.5, 1.0e-12, 1.0e-9],
            [0.25, 1.0e-12, 5.0e-10],
        ]
    )
    predictions = reference.copy()
    predictions[:, 0] *= 1.20
    predictions[:, 2] *= 0.70

    write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path,
            times=times,
            predictions=predictions,
            reference=reference,
            component_names=["Ex", "Ey", "dBzdt"],
            case_type="noip",
            reference_type="empymod",
            magnetic_quantity="dBzdt",
            validation_scope="corrected_model_full",
            diagnostics={
                "source_consistency": {
                    "source_endpoint_balance_residual": 2.0e-4,
                    "dc_current_conservation_residual": 1.0e-3,
                    "initial_curl_residual": 5.0e-5,
                    "waveform_integral_residual": 3.0e-6,
                },
                "receiver_sampling": {"enabled": True, "receiver_types": ["point", "disk_average"]},
                "magnetic_receiver_mode": "faraday_integrated",
            },
        )
    )

    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    failure = diagnostics["validation_failure"]
    assert failure["failed"] is True
    assert "physical_error_gate_failed" in failure["reason_codes"]

    expected_checks = [
        "time_step_error",
        "mesh_error",
        "boundary_error",
        "source_term_error",
        "receiver_sampling_error",
        "magnetic_recovery_error",
        "ip_memory_error",
    ]
    assert failure["recommended_check_order"] == expected_checks
    assert set(failure["checks"]) == set(expected_checks)
    for check_name in expected_checks:
        check = failure["checks"][check_name]
        assert check["status"] in {"needs_evaluation", "not_applicable"}
        assert isinstance(check["evidence"], dict)
        assert check["recommended_action"]

    assert failure["checks"]["source_term_error"]["evidence"]["source_endpoint_balance_residual"] == 2.0e-4
    assert failure["checks"]["receiver_sampling_error"]["evidence"]["receiver_types"] == ["point", "disk_average"]
    assert failure["checks"]["magnetic_recovery_error"]["evidence"]["magnetic_quantity"] == "dBzdt"
    assert failure["checks"]["ip_memory_error"]["status"] == "not_applicable"
