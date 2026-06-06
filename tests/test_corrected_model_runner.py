import json
import builtins
import sys

import numpy as np

from atem3d.cli import main
from atem3d.cli import _load_yaml
from atem3d.corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_model_case_specs,
    build_corrected_leakage_channel_case_specs,
    build_published_paper_model_target_spec,
)
from atem3d.corrected_model_runner import run_corrected_model_validation
from atem3d.corrected_model_runner import run_corrected_model_convergence_validation
from atem3d.corrected_model_runner import _default_reference_runner


def test_run_corrected_model_validation_writes_full_artifact_set(tmp_path):
    config = CorrectedModelValidationConfig(n_observation_times=3)
    spec = build_corrected_model_case_specs(tmp_path, config=config)["noip"]
    seen = {}

    def fake_forward(case_spec):
        seen["forward_case_type"] = case_spec["case_type"]
        return np.column_stack(
            [
                np.ones(len(case_spec["observation_times"])),
                2.0 * np.ones(len(case_spec["observation_times"])),
                3.0 * np.ones(len(case_spec["observation_times"])),
            ]
        )

    def fake_reference(case_spec):
        seen["reference_scope"] = case_spec["validation_scope"]
        return np.column_stack(
            [
                np.ones(len(case_spec["observation_times"])),
                2.0 * np.ones(len(case_spec["observation_times"])),
                3.0 * np.ones(len(case_spec["observation_times"])),
            ]
        )

    summary = run_corrected_model_validation(
        spec,
        forward_runner=fake_forward,
        reference_runner=fake_reference,
    )

    assert seen == {
        "forward_case_type": "noip",
        "reference_scope": "corrected_model_full",
    }
    assert summary["final_acceptance_passed"] is True
    output_dir = tmp_path / "noip_3comp"
    for name in (
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "error_summary.json",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "model_schematic.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
    ):
        assert (output_dir / name).exists()
    payload = json.loads((output_dir / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["validation_scope"] == "corrected_model_full"
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["runtime_seconds"]["forward"] >= 0.0
    assert diagnostics["runtime_seconds"]["reference"] >= 0.0
    assert diagnostics["runtime_seconds"]["artifact_total"] >= 0.0
    assert diagnostics["model_schematic"]["source_length_m"] == 1000.0


def test_run_corrected_model_convergence_validation_uses_refined_reference_case(tmp_path):
    config = CorrectedModelValidationConfig(n_observation_times=3)
    spec = build_corrected_leakage_channel_case_specs(tmp_path, config=config)["noip"]
    spec["dolfinx_forward"]["cells"] = [2, 2, 1]
    spec["convergence_reference"] = {
        "dolfinx_forward": {
            "cells": [4, 4, 2],
            "rtol": 1.0e-9,
        }
    }
    seen = []

    def fake_forward(case_spec):
        cells = tuple(case_spec["dolfinx_forward"]["cells"])
        seen.append(cells)
        scale = 1.0 if cells == (4, 4, 2) else 1.01
        return scale * np.column_stack(
            [
                np.ones(len(case_spec["observation_times"])),
                2.0 * np.ones(len(case_spec["observation_times"])),
                3.0 * np.ones(len(case_spec["observation_times"])),
            ]
        )

    summary = run_corrected_model_convergence_validation(
        spec,
        output_dir=tmp_path / "convergence_noip",
        forward_runner=fake_forward,
    )

    assert seen == [(2, 2, 1), (4, 4, 2)]
    assert summary["reference_type"] == "dolfinx_refined"
    assert summary["physical_pass_all_components"] is True
    assert summary["final_acceptance_passed"] is False
    assert "reference_type_not_final_acceptance" in summary["acceptance_status"]["blocking_reasons"]
    diagnostics = json.loads((tmp_path / "convergence_noip" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["convergence_reference"]["reference_type"] == "dolfinx_refined"
    assert diagnostics["convergence_reference"]["prediction_cells"] == [2, 2, 1]
    assert diagnostics["convergence_reference"]["reference_cells"] == [4, 4, 2]
    assert diagnostics["leakage_marker_preflight"]["prediction"]["leakage_cell_count"] > 0
    assert (
        diagnostics["leakage_marker_preflight"]["reference"]["leakage_cell_count"]
        > diagnostics["leakage_marker_preflight"]["prediction"]["leakage_cell_count"]
    )


def test_default_reference_runner_uses_debye_empymod_resistivity_for_ip(tmp_path, monkeypatch):
    from atem3d import empymod_compare

    config = CorrectedModelValidationConfig(n_observation_times=2)
    spec = build_corrected_model_case_specs(tmp_path, config=config)["ip"]
    seen_resistivities = []

    def fake_reference(survey, **_kwargs):
        seen_resistivities.append(survey.resistivities)
        return np.zeros((1, len(survey.receiver_components)), dtype=float)

    monkeypatch.setattr(empymod_compare, "run_empymod_reference", fake_reference)

    values = _default_reference_runner(spec)

    assert values.shape == (2, 3)
    assert seen_resistivities
    assert all(isinstance(res, dict) for res in seen_resistivities)
    assert all("func_eta" in res for res in seen_resistivities)
    np.testing.assert_allclose(seen_resistivities[0]["res"], [1.0 / 0.012] * 3)


def test_corrected_model_run_cli_dispatches_selected_cases(tmp_path, monkeypatch):
    specs = build_corrected_model_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    calls = []

    def fake_run(case_spec):
        calls.append((case_spec["case_type"], case_spec["output_dir"]))
        return {"final_acceptance_passed": True}

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_validation", fake_run)

    exit_code = main(
        [
            "corrected-model-run",
            str(spec_path),
            "--case",
            "ip",
            "--output-root",
            str(tmp_path / "override"),
        ]
    )

    assert exit_code == 0
    assert calls == [("ip", str(tmp_path / "override" / "ip_3comp"))]


def test_corrected_model_convergence_run_cli_dispatches_selected_cases(tmp_path, monkeypatch):
    specs = build_corrected_leakage_channel_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    calls = []

    def fake_run(case_spec):
        calls.append((case_spec["case_type"], case_spec["output_dir"]))
        return {
            "final_acceptance_passed": False,
            "physical_pass_all_components": True,
        }

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_convergence_validation", fake_run)

    exit_code = main(
        [
            "corrected-model-convergence-run",
            str(spec_path),
            "--case",
            "noip",
            "--output-root",
            str(tmp_path / "convergence"),
        ]
    )

    assert exit_code == 0
    assert calls == [("noip", str(tmp_path / "convergence" / "noip_convergence"))]


def test_corrected_model_spec_cli_allows_observation_time_count_override(tmp_path):
    output = tmp_path / "spec.json"

    exit_code = main(
        [
            "corrected-model-spec",
            str(tmp_path / "run"),
            "--output",
            str(output),
            "--n-observation-times",
            "5",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["noip"]["observation_times"]) == 5
    assert payload["noip"]["observation_times"][0] == 1.0e-5
    assert payload["noip"]["observation_times"][-1] == 1.0


def test_corrected_model_run_cli_writes_acceptance_config_for_both_cases(tmp_path, monkeypatch):
    specs = build_corrected_model_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    output_root = tmp_path / "override"

    def fake_run(case_spec):
        output_dir = tmp_path / case_spec["case_type"]
        output_dir.mkdir()
        offset = 1.0 if case_spec["case_type"] == "ip" else 0.0
        response_csv = (
            "time_obs,Ex,Ey,dBzdt\n"
            f"1e-05,{1.0 + offset},0.0,{2.0 + offset}\n"
            f"1.0,{2.0 + offset},0.0,{3.0 + offset}\n"
        )
        (output_dir / "predictions.csv").write_text(response_csv, encoding="utf-8")
        (output_dir / "reference_empymod_or_1d.csv").write_text(response_csv, encoding="utf-8")
        (output_dir / "error_summary.json").write_text(
            json.dumps(
                {
                    "case_type": case_spec["case_type"],
                    "reference_type": "empymod",
                    "magnetic_quantity": "dBzdt",
                    "final_acceptance_passed": True,
                    "acceptance_status": {"blocking_reasons": []},
                }
            ),
            encoding="utf-8",
        )
        case_spec["output_dir"] = str(output_dir)
        return {"final_acceptance_passed": True}

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_validation", fake_run)

    exit_code = main(
        [
            "corrected-model-run",
            str(spec_path),
            "--case",
            "both",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    acceptance_text = (output_root / "acceptance.yaml").read_text(encoding="utf-8")
    assert not acceptance_text.lstrip().startswith("{")
    acceptance = _load_yaml(output_root / "acceptance.yaml")
    assert acceptance == {
        "acceptance": {
            "noip_summary_json": str(tmp_path / "noip" / "error_summary.json"),
            "ip_summary_json": str(tmp_path / "ip" / "error_summary.json"),
            "noip_diagnostics_json": str(tmp_path / "noip" / "diagnostics.json"),
            "ip_diagnostics_json": str(tmp_path / "ip" / "diagnostics.json"),
            "output_dir": str(output_root / "final_acceptance"),
        }
    }
    assert main(["acceptance-report", str(output_root / "acceptance.yaml")]) == 0
    final_summary = _load_yaml(output_root / "final_acceptance" / "final_acceptance_summary.json")
    assert final_summary["final_acceptance_passed"] is True
    effect_summary = _load_yaml(output_root / "polarization_effect" / "polarization_effect_summary.json")
    assert effect_summary["definition"] == "ip_minus_noip"
    assert effect_summary["pass_all_components"] is True


def test_main_without_argv_uses_process_arguments_for_corrected_model_run(tmp_path, monkeypatch):
    specs = build_corrected_model_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    calls = []

    def fake_run(case_spec):
        calls.append(case_spec["case_type"])
        return {"final_acceptance_passed": True}

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_validation", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tdem-ip-forward", "corrected-model-run", str(spec_path), "--case", "noip"],
    )

    assert main() == 0
    assert calls == ["noip"]


def test_corrected_model_json_config_load_does_not_require_pyyaml(tmp_path, monkeypatch):
    config_path = tmp_path / "spec.json"
    config_path.write_text(json.dumps({"noip": {"case_type": "noip"}}), encoding="utf-8")
    original_import = builtins.__import__
    import atem3d.cli as cli_module

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.delattr(cli_module, "yaml", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _load_yaml(config_path) == {"noip": {"case_type": "noip"}}


def test_corrected_model_yaml_config_load_has_pyyaml_free_fallback(tmp_path, monkeypatch):
    config_path = tmp_path / "acceptance.yaml"
    config_path.write_text(
        """
acceptance:
  noip_summary_json: outputs/noip/error_summary.json
  ip_summary_json: outputs/ip/error_summary.json
  final_acceptance_passed: false
""",
        encoding="utf-8",
    )
    original_import = builtins.__import__
    import atem3d.cli as cli_module

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.delattr(cli_module, "yaml", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _load_yaml(config_path) == {
        "acceptance": {
            "noip_summary_json": "outputs/noip/error_summary.json",
            "ip_summary_json": "outputs/ip/error_summary.json",
            "final_acceptance_passed": False,
        }
    }


def test_published_paper_model_target_spec_records_public_reference_metadata(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path)

    assert spec["published_reference"]["title"] == "Analysis of 3D induced polarization effects of SOTEM"
    assert spec["published_reference"]["journal"] == "Journal of Applied Geophysics"
    assert spec["published_reference"]["volume"] == "233"
    assert spec["published_reference"]["publication_date"] == "February 2025"
    assert spec["published_reference"]["article_id"] == "S092698512400329X"
    assert spec["published_reference"]["doi"] == "10.1016/j.jappgeo.2024.105613"
    assert spec["published_reference"]["reproduction_status"] == (
        "full_text_model_parameters_extracted_response_digitization_pending"
    )
    assert spec["published_reference"]["public_method_summary"] == {
        "frequency_domain_solver": "COMSOL",
        "time_domain_transform": "frequency-time transformation",
        "reported_components": ["Ex", "Hz"],
    }
    assert spec["published_reference"]["public_model_classes"] == [
        "polarized_layer",
        "high_resistivity_polarized_body",
        "low_resistivity_polarized_body",
    ]
    assert spec["model"]["source_start"] == [-500.0, 200.0, -0.1]
    assert spec["model"]["source_end"] == [500.0, 200.0, -0.1]
    assert spec["model"]["receiver"] == [0.0, -300.0, -0.1]
    assert spec["model"]["source_current"] == 10.0
    assert spec["model"]["source_length_m"] == 1000.0
    assert spec["model"]["parallel_offset_m"] == 500.0
    assert spec["model"]["calculation_domain_m"] == [4000.0, 4000.0, 1000.0]
    assert spec["paper_model_parameters"]["accuracy_benchmark_layer"]["polarized_layer_thickness_m"] == 200.0
    assert spec["paper_model_parameters"]["accuracy_benchmark_layer"]["cole_cole"] == {
        "M": 0.3,
        "c": 0.5,
        "tau_s": 1.0,
        "sigma0_s_per_m": 0.01,
    }
    layer = spec["paper_model_parameters"]["layered_polarization_model"]
    assert layer["calculation_domain_m"] == [4000.0, 4000.0, 1000.0]
    assert layer["infinite_element_layer_thickness_m"] == 100.0
    assert layer["frequency_range_hz"] == [0.001, 10000.0]
    assert layer["frequency_count"] == 81
    assert layer["memory_gbytes"] == 15.7
    assert layer["printed_source_position_note"] == "paper prints source along x=200 m; corrected working geometry uses y=200 m"
    anomaly = spec["paper_model_parameters"]["three_dimensional_polarized_body"]
    assert anomaly["body_size_m"] == [400.0, 400.0, 400.0]
    assert anomaly["body_center_m"] == [0.0, -300.0, 400.0]
    assert anomaly["observation_point_m"] == [0.0, -400.0, 0.0]
    assert anomaly["high_resistivity_ohm_m"] == 1000.0
    assert anomaly["low_resistivity_ohm_m"] == 10.0
    assert anomaly["time_window_after_turnoff_s"] == [0.0001, 1.0]
    assert anomaly["time_count"] == 41
    assert spec["paper_response_targets"]["digitized_response_required"] is True
    assert "digitized_or_tabulated_published_response_values" in spec["remaining_reproduction_requirements"]


def test_published_paper_model_spec_cli_writes_json(tmp_path):
    output = tmp_path / "paper_model_target.json"

    exit_code = main(
        [
            "published-paper-model-spec",
            str(tmp_path / "paper_run"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["published_reference"]["article_id"] == "S092698512400329X"
    assert payload["run_contract"]["output_root"] == str(tmp_path / "paper_run")
    assert payload["run_contract"]["validation_scope"] == "published_paper_reproduction_target"


def test_published_paper_digitization_template_cli_writes_manifest_and_csv(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    spec_path = tmp_path / "paper_model_target.json"
    output_dir = tmp_path / "digitization"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    exit_code = main(
        [
            "published-paper-digitization-template",
            str(spec_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "paper_curve_digitization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_article_id"] == "S092698512400329X"
    assert manifest["template_csv"] == "paper_curve_digitization_template.csv"
    assert manifest["targets"][0] == {
        "figure": "Fig. 2",
        "model_key": "accuracy_benchmark_layer",
        "component": "Ex",
        "suggested_curve_labels": ["paper_3d_model", "paper_1d_analytical"],
    }
    csv_lines = (output_dir / "paper_curve_digitization_template.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "figure,model_key,component,curve_label,time_obs,value,notes"
    assert "Fig. 12,three_dimensional_polarized_body,Ex,paper_ip,,,digitize from published plot" in csv_lines
    assert "Fig. 15,three_dimensional_polarized_body,Hz,paper_noip,,,digitize from published plot" in csv_lines


def test_corrected_leakage_channel_case_specs_define_memory_safe_3d_anomaly(tmp_path):
    specs = build_corrected_leakage_channel_case_specs(tmp_path)

    assert set(specs) == {"noip", "ip"}
    noip = specs["noip"]
    ip = specs["ip"]
    assert noip["validation_scope"] == "corrected_model_terrain_leakage_diagnostic"
    assert noip["runner"]["backend"] == "dolfinx_primary_secondary"
    assert noip["source_start"] == [-500.0, 200.0, -0.1]
    assert noip["source_end"] == [500.0, 200.0, -0.1]
    forward = noip["dolfinx_forward"]
    assert forward["domain_min"] == [-2000.0, -2000.0, -1000.0]
    assert forward["domain_max"] == [2000.0, 2000.0, 100.0]
    assert forward["cells"] == [2, 2, 1]
    assert noip["convergence_reference"]["dolfinx_forward"]["cells"] == [4, 4, 2]
    assert noip["convergence_reference"]["dolfinx_forward"]["rtol"] == 1.0e-9
    assert len(forward["leakage_channel"]["points"]) == 4
    assert forward["leakage_channel"]["radius"] == 900.0
    assert forward["leakage_channel"]["min_marked_cells"] == 1
    assert forward["leakage_channel"]["sigma"] == 0.04
    assert ip["dolfinx_forward"]["leakage_channel"]["sigma_inf"] == 0.05
    assert ip["dolfinx_forward"]["leakage_channel"]["delta_sigma_list"] == [0.015]


def test_corrected_leakage_model_spec_cli_writes_json(tmp_path):
    output = tmp_path / "leakage_spec.json"

    exit_code = main(
        [
            "corrected-leakage-model-spec",
            str(tmp_path / "leakage_run"),
            "--output",
            str(output),
            "--n-observation-times",
            "3",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["noip"]["validation_scope"] == "corrected_model_terrain_leakage_diagnostic"
    assert payload["ip"]["observation_times"][0] == 1.0e-5
    assert payload["ip"]["observation_times"][-1] == 1.0
    assert len(payload["ip"]["observation_times"]) == 3


def test_leakage_marker_diagnostics_cli_writes_case_report(tmp_path):
    specs = build_corrected_leakage_channel_case_specs(tmp_path / "run")
    specs["noip"]["dolfinx_forward"]["domain_min"] = [-3000.0, -3000.0, -1500.0]
    specs["noip"]["dolfinx_forward"]["domain_max"] = [3000.0, 3000.0, 200.0]
    spec_path = tmp_path / "spec.json"
    output = tmp_path / "marker_diagnostics.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")

    exit_code = main(
        [
            "leakage-marker-diagnostics",
            str(spec_path),
            "--case",
            "noip",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["case_type"] == "noip"
    assert payload["prediction"]["initial_leakage_cell_count"] == 0
    assert payload["prediction"]["leakage_cell_count"] == 1
    assert payload["prediction"]["fallback_used"] is True
    assert payload["reference"]["leakage_cell_count"] > payload["prediction"]["leakage_cell_count"]
    assert payload["prediction"]["marked"] is True
