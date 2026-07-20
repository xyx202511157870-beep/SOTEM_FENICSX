from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from atem3d.metrics import robust_component_errors
from atem3d.sotem_acceptance import ey_symmetry_diagnostics
from atem3d.sotem_metrics import compare_signed_response
from atem3d.validation_3comp import validation_acceptance_status


COMPONENTS = ("Ex", "Ey", "Hz", "dBzdt")
ACCEPTANCE_COMPONENTS = ("Ex", "Hz", "dBzdt")


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_ey_acceptance_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ey_only_failure():
    times = np.asarray([1.0e-5, 1.0e-3, 1.0])
    reference = np.asarray(
        [
            [1.0, 0.0, 2.0, 4.0],
            [0.5, 0.0, 1.0, 2.0],
            [0.25, 0.0, 0.5, 1.0],
        ]
    )
    prediction = reference.copy()
    prediction[:, 1] = np.asarray([1.0, -1.0, 0.5])
    return times, prediction, reference


def test_generic_error_table_can_exclude_ey_from_acceptance_without_deleting_it():
    times, prediction, reference = _ey_only_failure()

    rows, summary = robust_component_errors(
        times,
        prediction,
        reference,
        COMPONENTS,
        threshold=0.05,
        acceptance_components=ACCEPTANCE_COMPONENTS,
        diagnostic_only_components={"Ey": "transverse_symmetry"},
    )

    assert set(rows["component"]) == set(COMPONENTS)
    assert np.any(~rows[rows["component"] == "Ey"]["pass_5pct"])
    assert summary["pass_all_components"] is True
    assert summary["failed_components"] == []
    assert summary["failed_times"] == []
    assert summary["diagnostic_failed_components"] == ["Ey"]
    assert summary["acceptance_components"] == list(ACCEPTANCE_COMPONENTS)
    assert summary["component_roles"]["Ey"] == {
        "acceptance_status": "excluded_by_design",
        "diagnostic_role": "transverse_symmetry",
    }


def test_dolfinx_formal_summary_uses_three_physical_components_and_keeps_ey_diagnostic():
    sp = _load_pipeline_module()
    times, prediction, reference = _ey_only_failure()

    rows, summary = sp._robust_error_rows(
        times, prediction, reference, COMPONENTS, threshold=0.05
    )

    assert {row["component"] for row in rows} == set(COMPONENTS)
    assert any(
        row["component"] == "Ey" and row["pass_5pct"] is False for row in rows
    )
    assert summary["acceptance_profile"] == "symmetric_sotem_ex_hz_dBzdt/v1"
    assert summary["component_order"] == list(COMPONENTS)
    assert summary["acceptance_components"] == list(ACCEPTANCE_COMPONENTS)
    assert summary["failed_components"] == []
    assert summary["physical_failed_components"] == []
    assert summary["pass_all_components"] is True
    assert summary["diagnostic_failed_components"] == ["Ey"]
    assert summary["symmetry_diagnostics"]["Ey"]["acceptance_status"] == "excluded_by_design"
    assert summary["symmetry_diagnostics"]["Ey"]["diagnostic_role"] == "transverse_symmetry"


def test_sotem_signed_effect_gate_excludes_ey_but_preserves_all_ey_diagnostics():
    times, prediction, reference = _ey_only_failure()
    # The signed-response gate requires non-zero reference peaks for every
    # diagnostic column; keep Ey very small so its deliberately bad response
    # remains a meaningful symmetry diagnostic.
    reference[:, 1] = np.asarray([1.0e-12, -1.0e-12, 0.5e-12])

    result = compare_signed_response(
        times, prediction, reference, COMPONENTS, threshold=0.10
    )
    summary = result["summary"]

    assert set(result["max_acceptance_error_by_component"]) == set(COMPONENTS)
    assert "Ey" in result["zero_crossings"]
    assert summary["acceptance_components"] == list(ACCEPTANCE_COMPONENTS)
    assert summary["pass_all_components"] is True
    assert summary["failed_components"] == []
    assert summary["diagnostic_failed_components"] == ["Ey"]
    assert summary["robust_diagnostic_pass_all_components"] is False
    assert summary["robust_diagnostic_failed_components"] == ["Ey"]
    assert summary["component_roles"]["Ey"]["acceptance_status"] == "excluded_by_design"


def test_final_acceptance_requires_canonical_columns_but_not_ey_error_pass():
    times, prediction, reference = _ey_only_failure()
    _rows, summary = robust_component_errors(
        times,
        prediction,
        reference,
        COMPONENTS,
        threshold=0.05,
        acceptance_components=ACCEPTANCE_COMPONENTS,
        diagnostic_only_components={"Ey": "transverse_symmetry"},
    )
    summary["acceptance_profile"] = "symmetric_sotem_ex_hz_dBzdt/v1"
    summary["component_order"] = list(COMPONENTS)

    status = validation_acceptance_status(
        times,
        COMPONENTS,
        summary,
        case_type="noip",
        reference_type="empymod",
        threshold=0.05,
        validation_scope="corrected_model_full",
        diagnostics={
            "primary_secondary_internal_time_grid": {
                "contains_turnoff_start": True,
                "contains_turnoff_end": True,
                "contains_all_observation_outputs": True,
                "last_output_internal_time_s": 1.0,
            }
        },
    )

    assert status["acceptance_components"] == list(ACCEPTANCE_COMPONENTS)
    assert status["diagnostic_only_components"] == ["Ey"]
    assert status["component_order_preserved"] is True
    assert status["strict_error_gate_passed"] is True
    assert status["final_acceptance_passed"] is True
    assert "strict_error_gate_failed" not in status["blocking_reasons"]


@pytest.mark.parametrize("physical_gate_value", [False, None, 0, 1, "true"])
def test_final_acceptance_fails_closed_when_physical_gate_is_not_strict_true(
    physical_gate_value,
):
    times, prediction, reference = _ey_only_failure()
    _rows, summary = robust_component_errors(
        times,
        prediction,
        reference,
        COMPONENTS,
        threshold=0.05,
        acceptance_components=ACCEPTANCE_COMPONENTS,
        diagnostic_only_components={"Ey": "transverse_symmetry"},
    )
    summary.update(
        {
            "acceptance_profile": "symmetric_sotem_ex_hz_dBzdt/v1",
            "component_order": list(COMPONENTS),
        }
    )
    if physical_gate_value is None:
        summary.pop("physical_pass_all_components", None)
    else:
        summary["physical_pass_all_components"] = physical_gate_value

    status = validation_acceptance_status(
        times,
        COMPONENTS,
        summary,
        case_type="noip",
        reference_type="empymod",
        threshold=0.05,
        validation_scope="corrected_model_full",
        diagnostics={
            "primary_secondary_internal_time_grid": {
                "contains_turnoff_start": True,
                "contains_turnoff_end": True,
                "contains_all_observation_outputs": True,
                "last_output_internal_time_s": 1.0,
            }
        },
    )

    assert status["strict_error_gate_passed"] is True
    assert status["physical_error_gate_passed"] is False
    assert status["final_acceptance_passed"] is False
    assert "physical_error_gate_failed" in status["blocking_reasons"]


@pytest.mark.parametrize(
    "strict_gate_value", [False, None, 0, 1, "true", "false"]
)
def test_final_acceptance_fails_closed_when_strict_gate_is_not_strict_true(
    strict_gate_value,
):
    times, prediction, reference = _ey_only_failure()
    _rows, summary = robust_component_errors(
        times,
        prediction,
        reference,
        COMPONENTS,
        threshold=0.05,
        acceptance_components=ACCEPTANCE_COMPONENTS,
        diagnostic_only_components={"Ey": "transverse_symmetry"},
    )
    summary.update(
        {
            "acceptance_profile": "symmetric_sotem_ex_hz_dBzdt/v1",
            "component_order": list(COMPONENTS),
        }
    )
    if strict_gate_value is None:
        summary.pop("pass_all_components", None)
    else:
        summary["pass_all_components"] = strict_gate_value

    status = validation_acceptance_status(
        times,
        COMPONENTS,
        summary,
        case_type="noip",
        reference_type="empymod",
        threshold=0.05,
        validation_scope="corrected_model_full",
        diagnostics={
            "primary_secondary_internal_time_grid": {
                "contains_turnoff_start": True,
                "contains_turnoff_end": True,
                "contains_all_observation_outputs": True,
                "last_output_internal_time_s": 1.0,
            }
        },
    )

    assert status["physical_error_gate_passed"] is True
    assert status["strict_error_gate_passed"] is False
    assert status["final_acceptance_passed"] is False
    assert status["blocking_reasons"]
    assert "strict_error_gate_failed" in status["blocking_reasons"]


@pytest.mark.parametrize(
    ("summary_update", "expected_reason"),
    [
        ({"component_roles": {}}, "acceptance_component_contract_invalid"),
        (
            {"component_roles": {"Ey": {"acceptance_status": "included"}}},
            "acceptance_component_contract_invalid",
        ),
    ],
)
def test_final_acceptance_rejects_spoofed_component_roles(
    summary_update, expected_reason
):
    times, prediction, reference = _ey_only_failure()
    _rows, summary = robust_component_errors(
        times,
        prediction,
        reference,
        COMPONENTS,
        threshold=0.05,
        acceptance_components=ACCEPTANCE_COMPONENTS,
        diagnostic_only_components={"Ey": "transverse_symmetry"},
    )
    summary.update(
        {
            "acceptance_profile": "symmetric_sotem_ex_hz_dBzdt/v1",
            "component_order": list(COMPONENTS),
            **summary_update,
        }
    )

    status = validation_acceptance_status(
        times,
        COMPONENTS,
        summary,
        case_type="noip",
        reference_type="empymod",
        threshold=0.05,
        validation_scope="corrected_model_full",
        diagnostics={
            "primary_secondary_internal_time_grid": {
                "contains_turnoff_start": True,
                "contains_turnoff_end": True,
                "contains_all_observation_outputs": True,
                "last_output_internal_time_s": 1.0,
            }
        },
    )

    assert status["final_acceptance_passed"] is False
    assert status["acceptance_component_contract_valid"] is False
    assert expected_reason in status["blocking_reasons"]


def test_zero_ex_symmetry_ratio_remains_strict_json_serializable():
    prediction = np.zeros((3, 4), dtype=float)
    reference = np.zeros((3, 4), dtype=float)
    prediction[:, 1] = 1.0

    diagnostics = ey_symmetry_diagnostics(prediction, reference, COMPONENTS)

    assert diagnostics["Ey"]["prediction_to_Ex_peak_ratio"] is None
    assert diagnostics["Ey"]["prediction_to_Ex_peak_ratio_defined"] is False
    assert diagnostics["Ey"]["reference_to_Ex_peak_ratio"] == 0.0
    assert diagnostics["Ey"]["reference_to_Ex_peak_ratio_defined"] is True
    json.dumps(diagnostics, allow_nan=False)


def test_text_report_names_ey_exclusion_and_quantifies_symmetry(tmp_path):
    sp = _load_pipeline_module()
    times, prediction, reference = _ey_only_failure()
    fem_result = {
        "times": times,
        "data": prediction,
        "components": list(COMPONENTS),
        "solver_log": [],
    }
    ref_result = {
        "times": times,
        "data": reference,
        "components": list(COMPONENTS),
    }

    sp.write_report(
        sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-5, t_max=1.0),
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=sp.compute_error(prediction, reference, COMPONENTS),
        source_info={"mode": "manual_line"},
    )

    report = (tmp_path / "verification_report.txt").read_text(encoding="utf-8")
    assert "formal acceptance components: Ex, Hz, dBzdt" in report
    assert "Ey: excluded_by_design" in report
    assert "transverse symmetry diagnostic" in report
    assert "prediction_peak_abs=" in report
    assert "prediction_to_Ex_peak_ratio=" in report


def test_canonical_report_keeps_ey_horizontal_vector_out_of_physical_gate(tmp_path):
    sp = _load_pipeline_module()
    times, prediction, reference = _ey_only_failure()

    physical = sp.check_physical_error_window(
        times,
        prediction,
        reference,
        COMPONENTS,
        error_min_time=times[0],
        tolerance=0.05,
    )

    assert physical["passed"] is True
    assert physical["acceptance_components"] == list(ACCEPTANCE_COMPONENTS)
    assert set(physical["maxima"]) == set(ACCEPTANCE_COMPONENTS)
    assert physical["diagnostic_only_metrics"] == ["Eh_vector"]
    assert physical["diagnostic_maxima"]["Eh_vector"] > 0.05

    fem_result = {
        "times": times,
        "data": prediction,
        "components": list(COMPONENTS),
        "solver_log": [],
    }
    ref_result = {
        "times": times,
        "data": reference,
        "components": list(COMPONENTS),
    }
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-5, t_max=1.0)
    sp.write_report(
        config,
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=sp.compute_error(prediction, reference, COMPONENTS),
        source_info={"mode": "manual_line"},
    )

    report = config.output_report().read_text(encoding="utf-8")
    assert "strict physical-error passing window (Ex, Hz, dBzdt max <=" in report
    assert "Eh_vector [diagnostic_only/excluded_by_design]" in report
    assert "Ex, dBzdt, and Eh_vector" not in report
    assert "The physical gate uses Ex, dBzdt, and Eh_vector" not in report
    assert "configured run exceeds the physical gate" not in report


def test_formal_artifacts_preserve_canonical_ey_column_and_publish_its_role(tmp_path):
    sp = _load_pipeline_module()
    times, prediction, reference = _ey_only_failure()

    summary = sp.write_validation_artifacts(
        times,
        prediction,
        reference,
        COMPONENTS,
        sp.PipelineConfig(workdir=tmp_path),
        case_type="noip",
        reference_type="empymod",
    )

    assert (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "time_obs,Ex,Ey,Hz,dBzdt"
    )
    prediction_rows = list(
        csv.DictReader((tmp_path / "predictions.csv").open(encoding="utf-8"))
    )
    assert [float(row["Ey"]) for row in prediction_rows] == pytest.approx(
        prediction[:, 1]
    )
    error_rows = list(csv.DictReader((tmp_path / "errors.csv").open(encoding="utf-8")))
    assert any(row["component"] == "Ey" for row in error_rows)
    payload = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload == summary
    assert payload["failed_components"] == []
    assert payload["diagnostic_failed_components"] == ["Ey"]
    assert diagnostics["diagnostic_failed_components"] == ["Ey"]
    assert diagnostics["component_roles"]["Ey"]["acceptance_status"] == "excluded_by_design"


@pytest.mark.parametrize("target", ["dolfinx", "signed_effect"])
def test_reordered_canonical_components_fail_closed(target):
    sp = _load_pipeline_module()
    times, prediction, reference = _ey_only_failure()
    reordered = ("Ex", "Hz", "Ey", "dBzdt")
    reordered_indices = [COMPONENTS.index(name) for name in reordered]
    prediction = prediction[:, reordered_indices]
    reference = reference[:, reordered_indices]

    with pytest.raises(ValueError, match="component order must be exactly"):
        if target == "dolfinx":
            sp._robust_error_rows(
                times, prediction, reference, reordered, threshold=0.05
            )
        else:
            # Keep every reference peak non-zero for the signed-effect API.
            reference[:, reordered.index("Ey")] = 1.0e-12
            compare_signed_response(
                times, prediction, reference, reordered, threshold=0.10
            )
