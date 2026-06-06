import json

import numpy as np
import pytest
import yaml

from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    validation_acceptance_status,
    write_three_component_validation_artifacts,
)


def test_noip_3comp_validation_smoke_writes_required_artifacts(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 0.0, 1.0e-9],
            [0.5, 0.0, 5.0e-10],
            [0.1, 0.0, 1.0e-10],
        ]
    )
    predictions = reference * np.array([[1.01, 1.0, 0.99]])

    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path,
            times=times,
            predictions=predictions,
            reference=reference,
            component_names=["Ex", "Ey", "dBzdt"],
            case_type="noip",
            reference_type="empymod",
            magnetic_quantity="dBzdt",
            diagnostics={"runtime_seconds": 1.25},
        )
    )

    for name in [
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "error_summary.json",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
    ]:
        assert (tmp_path / name).is_file()

    payload = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["case_type"] == "noip"
    assert payload["reference_type"] == "empymod"
    assert payload["magnetic_quantity"] == "dBzdt"
    assert payload["pass_all_components"] is True
    assert summary["pass_all_components"] is True
    errors_header = (tmp_path / "errors.csv").read_text(encoding="utf-8").splitlines()[0]
    assert errors_header == (
        "time_obs,component,pred,ref,abs_error,ordinary_relative_error,"
        "relative_error_with_floor,peak_normalized_error,pass_5pct"
    )
    resolved_text = (tmp_path / "run_config_resolved.yaml").read_text(encoding="utf-8")
    assert not resolved_text.lstrip().startswith("{")
    resolved = yaml.safe_load(resolved_text)
    assert resolved["case_type"] == "noip"
    assert resolved["component_names"] == ["Ex", "Ey", "dBzdt"]
    assert payload["final_acceptance_passed"] is False
    assert payload["acceptance_status"]["validation_scope"] == "smoke"
    assert "validation_scope_not_corrected_model_full" in payload["acceptance_status"]["blocking_reasons"]


def test_corrected_model_full_scope_can_claim_final_acceptance(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 0.0, 1.0e-9],
            [0.5, 0.0, 5.0e-10],
            [0.1, 0.0, 1.0e-10],
        ]
    )
    predictions = reference * np.array([[1.01, 1.0, 0.99]])

    summary = write_three_component_validation_artifacts(
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
        )
    )

    payload = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["final_acceptance_passed"] is True
    assert payload["acceptance_status"]["blocking_reasons"] == []
    assert summary["final_acceptance_passed"] is True


def test_dolfinx_refined_reference_writes_artifacts_but_cannot_claim_final_acceptance(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 0.1, 1.0e-9],
            [0.5, 0.05, 5.0e-10],
            [0.1, 0.01, 1.0e-10],
        ]
    )
    predictions = reference * np.array([[1.01, 0.99, 1.02]])

    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path,
            times=times,
            predictions=predictions,
            reference=reference,
            component_names=["Ex", "Ey", "dBzdt"],
            case_type="noip",
            reference_type="dolfinx_refined",
            magnetic_quantity="dBzdt",
            validation_scope="corrected_model_full",
        )
    )

    payload = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload["reference_type"] == "dolfinx_refined"
    assert payload["physical_pass_all_components"] is True
    assert payload["final_acceptance_passed"] is False
    assert "reference_type_not_final_acceptance" in payload["acceptance_status"]["blocking_reasons"]
    assert "reference_type_not_final_acceptance" in diagnostics["validation_failure"]["reason_codes"]
    assert summary["final_acceptance_passed"] is False


def test_acceptance_status_rejects_partial_time_window_even_when_errors_pass():
    status = validation_acceptance_status(
        np.array([1.0e-5, 1.0e-3, 1.0e-2]),
        ["Ex", "Ey", "dBzdt"],
        {
            "pass_all_components": True,
            "physical_pass_all_components": True,
        },
        case_type="ip",
        reference_type="1d",
        threshold=0.05,
        validation_scope="corrected_model_full",
    )

    assert status["full_window_covered"] is False
    assert status["final_acceptance_passed"] is False
    assert "time_window_not_covered" in status["blocking_reasons"]


def test_validation_rejects_time_table_that_does_not_cover_required_window(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0e-2])
    reference = np.ones((3, 3))

    with pytest.raises(ValueError, match="cover 1e-5 s to 1 s"):
        write_three_component_validation_artifacts(
            ThreeComponentValidationInput(
                output_dir=tmp_path,
                times=times,
                predictions=reference,
                reference=reference,
                component_names=["Ex", "Ey", "dBzdt"],
                case_type="noip",
                reference_type="empymod",
                magnetic_quantity="dBzdt",
            )
        )


def test_validation_failure_writes_automatic_diagnostic_check_order(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.ones((3, 3))
    predictions = 2.0 * reference

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
        )
    )

    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    failure = diagnostics["validation_failure"]
    assert failure["failed"] is True
    assert failure["failed_components"] == ["Ex", "Ey", "dBzdt"]
    assert failure["recommended_check_order"] == [
        "time_step_error",
        "mesh_error",
        "boundary_error",
        "source_term_error",
        "receiver_sampling_error",
        "magnetic_recovery_error",
        "ip_memory_error",
    ]
