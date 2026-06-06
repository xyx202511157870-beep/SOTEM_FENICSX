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


def test_corrected_model_run_cli_writes_acceptance_config_for_both_cases(tmp_path, monkeypatch):
    specs = build_corrected_model_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    output_root = tmp_path / "override"

    def fake_run(case_spec):
        output_dir = tmp_path / case_spec["case_type"]
        output_dir.mkdir()
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
    acceptance = _load_yaml(output_root / "acceptance.yaml")
    assert acceptance == {
        "acceptance": {
            "noip_summary_json": str(tmp_path / "noip" / "error_summary.json"),
            "ip_summary_json": str(tmp_path / "ip" / "error_summary.json"),
            "output_dir": str(output_root / "final_acceptance"),
        }
    }
    assert main(["acceptance-report", str(output_root / "acceptance.yaml")]) == 0
    final_summary = _load_yaml(output_root / "final_acceptance" / "final_acceptance_summary.json")
    assert final_summary["final_acceptance_passed"] is True


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


def test_published_paper_model_target_spec_records_public_reference_metadata(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path)

    assert spec["published_reference"]["title"] == "Analysis of 3D induced polarization effects of SOTEM"
    assert spec["published_reference"]["journal"] == "Journal of Applied Geophysics"
    assert spec["published_reference"]["volume"] == "233"
    assert spec["published_reference"]["publication_date"] == "February 2025"
    assert spec["published_reference"]["article_id"] == "S092698512400329X"
    assert spec["published_reference"]["doi"] == "10.1016/j.jappgeo.2024.105613"
    assert spec["published_reference"]["reproduction_status"] == "target_defined_full_text_parameters_pending"
    assert spec["model"]["source_start"] == [-500.0, 200.0, -0.1]
    assert spec["model"]["source_end"] == [500.0, 200.0, -0.1]
    assert spec["model"]["receiver"] == [0.0, -300.0, -0.1]
    assert spec["model"]["source_current"] == 10.0
    assert spec["model"]["source_length_m"] == 1000.0
    assert spec["model"]["parallel_offset_m"] == 500.0
    assert spec["model"]["calculation_domain_m"] == [4000.0, 4000.0, 1000.0]
    assert "ip_anomaly_geometry" in spec["full_text_parameters_required"]


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
    assert len(forward["leakage_channel"]["points"]) == 4
    assert forward["leakage_channel"]["radius"] == 900.0
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
