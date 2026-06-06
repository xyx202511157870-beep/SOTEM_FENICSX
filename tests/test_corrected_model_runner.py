import json
import builtins
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest

from atem3d.cli import main
from atem3d.cli import _load_yaml
from atem3d.corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_model_case_specs,
    build_corrected_leakage_channel_case_specs,
    build_published_paper_model_target_spec,
    build_corrected_terrain_smoke_case_spec,
    build_corrected_terrain_smoke_case_specs,
)
from atem3d.corrected_model_runner import run_corrected_model_validation
from atem3d.corrected_model_runner import run_corrected_model_convergence_validation
from atem3d.corrected_model_runner import _default_forward_runner
from atem3d.corrected_model_runner import _default_reference_runner
from atem3d.corrected_model_runner import _copy_dolfinx_forward_diagnostics_to_case_spec
from atem3d.corrected_model_runner import _primary_provider_for_case_spec
from atem3d.corrected_model_runner import _terrain_mesh_runtime_config
from atem3d.corrected_model_runner import dolfinx_backend_status
from atem3d.paper_digitization import write_published_paper_figure_page_package


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
    equation = diagnostics["primary_secondary_step_equation"]
    assert equation["case_type"] == "noip"
    assert equation["lhs_operator"] == "K + R + M(sigma)/dt"
    assert equation["rhs_history"] == "M(deltaJ_old - (sigma - sigma_b) Ep_new)/dt"
    assert equation["dt"] == pytest.approx(1.0e-6)
    assert equation["turnoff_time_s"] == pytest.approx(1.0e-5)
    assert equation["turnoff_steps"] == 10
    assert equation["first_observation_time_s"] == pytest.approx(1.0e-5)
    assert equation["first_output_internal_time_s"] == pytest.approx(2.0e-5)
    assert equation["first_internal_step_dt_s"] == pytest.approx(1.0e-6)
    assert equation["dt_source"] == "turnoff_grid_first_step"
    assert equation["internal_time_grid"]["turnoff_grid_points"] == 11
    assert equation["internal_time_grid"]["observation_output_points"] == 3
    assert equation["internal_time_grid"]["total_internal_points"] == 14
    assert equation["internal_time_grid"]["contains_turnoff_end"] is True
    assert equation["internal_time_grid"]["contains_all_observation_outputs"] is True
    assert equation["internal_time_grid"]["last_output_internal_time_s"] == pytest.approx(1.00001)


def test_run_corrected_model_validation_writes_ip_secondary_equation_metadata(tmp_path):
    config = CorrectedModelValidationConfig(n_observation_times=3)
    spec = build_corrected_model_case_specs(tmp_path, config=config)["ip"]

    def fake_response(case_spec):
        return np.column_stack(
            [
                np.ones(len(case_spec["observation_times"])),
                np.zeros(len(case_spec["observation_times"])),
                2.0 * np.ones(len(case_spec["observation_times"])),
            ]
        )

    run_corrected_model_validation(
        spec,
        forward_runner=fake_response,
        reference_runner=fake_response,
    )

    diagnostics = json.loads((tmp_path / "ip_3comp" / "diagnostics.json").read_text(encoding="utf-8"))
    equation = diagnostics["primary_secondary_step_equation"]
    assert equation["case_type"] == "noip"
    assert equation["zero_contrast_condition"] == "sigma == sigma_background"
    assert equation["secondary_material_reason"] == "ip_primary_background_included"
    assert equation["original_ip_material"]["case_type"] == "ip"
    assert equation["original_ip_material"]["delta_sigma_zero_degenerates_to_noip"] is True


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


def test_run_corrected_model_convergence_validation_records_nonzero_secondary_effect(tmp_path):
    config = CorrectedModelValidationConfig(n_observation_times=3)
    spec = build_corrected_leakage_channel_case_specs(tmp_path, config=config)["noip"]
    spec["dolfinx_forward"]["cells"] = [2, 2, 1]
    spec["convergence_reference"] = {
        "dolfinx_forward": {
            "cells": [4, 4, 2],
        }
    }
    times = np.asarray(spec["observation_times"], dtype=float)
    primary = np.column_stack(
        [
            np.ones(times.size),
            2.0 * np.ones(times.size),
            3.0 * np.ones(times.size),
        ]
    )

    def fake_forward(case_spec):
        cells = tuple(case_spec["dolfinx_forward"]["cells"])
        if cells == (4, 4, 2):
            return primary + np.array(
                [
                    [0.20, 0.00, 0.10],
                    [0.10, 0.00, 0.05],
                    [0.05, 0.00, 0.02],
                ]
            )
        return primary + np.array(
            [
                [0.18, 0.00, 0.08],
                [0.09, 0.00, 0.04],
                [0.04, 0.00, 0.015],
            ]
        )

    def fake_primary(case_spec):
        assert case_spec["reference_type"] == "dolfinx_refined"
        return primary

    run_corrected_model_convergence_validation(
        spec,
        output_dir=tmp_path / "convergence_noip",
        forward_runner=fake_forward,
        primary_runner=fake_primary,
    )

    diagnostics = json.loads((tmp_path / "convergence_noip" / "diagnostics.json").read_text(encoding="utf-8"))
    secondary = diagnostics["secondary_effect_diagnostic"]
    assert secondary["reference_type"] == "dolfinx_refined"
    assert secondary["component_names"] == ["Ex", "Ey", "dBzdt"]
    assert secondary["prediction_secondary_effect_nonzero"] is True
    assert secondary["reference_secondary_effect_nonzero"] is True
    assert secondary["secondary_effect_nonzero"] is True
    assert secondary["nonzero_atol"] == pytest.approx(1.0e-15)
    assert secondary["max_abs_prediction_minus_primary_by_component"]["Ex"] == pytest.approx(0.18)
    assert secondary["max_abs_reference_minus_primary_by_component"]["Ex"] == pytest.approx(0.20)
    assert secondary["max_abs_prediction_minus_primary_by_component"]["Ey"] == pytest.approx(0.0)
    assert secondary["max_abs_prediction_minus_reference_by_component"]["dBzdt"] == pytest.approx(0.02)


def test_terrain_mesh_runtime_config_resolves_small_gmsh_mesh_path(tmp_path):
    cfg = _terrain_mesh_runtime_config(
        {
            "terrain_mesh": {
                "mode": "small_gmsh_terrain_leakage",
                "mesh_size": 0.65,
                "msh_name": "terrain_leakage.msh",
            }
        },
        output_dir=tmp_path / "run",
    )

    assert cfg == {
        "mode": "small_gmsh_terrain_leakage",
        "mesh_size": 0.65,
        "msh_name": "terrain_leakage.msh",
        "workdir": str(tmp_path / "run"),
        "mesh_path": str(tmp_path / "run" / "terrain_leakage.msh"),
    }


def test_primary_provider_for_case_spec_supports_constant_diagnostic_mode():
    provider = _primary_provider_for_case_spec(
        {
            "empymod_primary": {
                "source_start": [-500.0, 200.0, -0.1],
                "source_end": [500.0, 200.0, -0.1],
                "current": 10.0,
                "depths": [350.0, 650.0],
                "resistivities": [100.0, 100.0, 100.0],
            },
            "dolfinx_forward": {
                "primary_provider_mode": "constant",
                "constant_primary_E": [1.0, 0.0, 0.0],
                "constant_primary_dBdt": [0.0, 0.0, 0.0],
            },
        }
    )

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.2]])
    np.testing.assert_allclose(provider.get_Ep_on_V(1.0e-5, points), [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(provider.get_Ep_dc_on_V(points), [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(provider.get_receiver_E(1.0e-5, points), [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(provider.get_receiver_dBdt(1.0e-5, points), np.zeros((2, 3)))


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


def test_default_forward_runner_reports_incomplete_dolfinx_backend(tmp_path, monkeypatch):
    config = CorrectedModelValidationConfig(n_observation_times=2)
    spec = build_corrected_model_case_specs(tmp_path, config=config)["noip"]
    monkeypatch.setitem(sys.modules, "dolfinx", types.ModuleType("dolfinx"))
    monkeypatch.delitem(sys.modules, "dolfinx.fem", raising=False)
    monkeypatch.delitem(sys.modules, "dolfinx.mesh", raising=False)

    with pytest.raises(ImportError, match="DOLFINx forward backend is unavailable"):
        _default_forward_runner(spec)


def test_copy_dolfinx_forward_diagnostics_keeps_only_json_safe_acceptance_evidence():
    case_spec = {}
    function_like = object()

    _copy_dolfinx_forward_diagnostics_to_case_spec(
        case_spec,
        {
            "E": function_like,
            "primary_secondary_internal_time_grid": {
                "contains_turnoff_start": True,
                "contains_turnoff_end": True,
                "contains_all_observation_outputs": True,
                "last_output_internal_time_s": 1.00001,
            },
            "primary_secondary_step_equation": {
                "case_type": "noip",
                "dt": 1.0e-6,
            },
            "dc_result": {
                "contrast_is_zero": False,
            },
        },
    )

    diagnostics = case_spec["diagnostics"]
    assert diagnostics["primary_secondary_internal_time_grid"]["last_output_internal_time_s"] == pytest.approx(1.00001)
    assert diagnostics["primary_secondary_step_equation"]["case_type"] == "noip"
    assert diagnostics["dc_result"]["contrast_is_zero"] is False
    assert "E" not in diagnostics


def test_dolfinx_backend_status_reports_runtime_and_test_dependencies(monkeypatch):
    missing = {"pytest"}
    imported = []

    def fake_import_module(name):
        imported.append(name)
        if name in missing:
            raise ImportError(f"No module named {name!r}")
        return SimpleNamespace(__file__=f"/fake/{name.replace('.', '/')}.py")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    status = dolfinx_backend_status()

    assert status["available"] is True
    assert status["runtime_modules"] == [
        "numpy",
        "dolfinx.fem",
        "dolfinx.mesh",
        "mpi4py.MPI",
        "ufl",
        "basix",
        "petsc4py",
    ]
    assert status["test_modules"] == ["pytest"]
    assert status["missing_modules"] == []
    assert status["missing_test_modules"] == ["pytest"]
    assert status["python_executable"]
    assert set(status["checks"]) == set(status["runtime_modules"] + status["test_modules"])
    assert imported == status["runtime_modules"] + status["test_modules"]


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


def test_corrected_model_run_cli_reports_backend_import_error(tmp_path, monkeypatch, capsys):
    specs = build_corrected_model_case_specs(tmp_path / "from_spec")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")

    def fake_run(_case_spec):
        raise ImportError("DOLFINx forward backend is unavailable: missing dolfinx.fem")

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_validation", fake_run)

    exit_code = main(["corrected-model-run", str(spec_path), "--case", "noip"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DOLFINx forward backend is unavailable" in captured.err


def test_dolfinx_backend_check_cli_writes_unavailable_status(tmp_path, monkeypatch, capsys):
    output = tmp_path / "backend_status.json"

    def fake_status():
        return {
            "available": False,
            "required_modules": ["dolfinx.fem", "dolfinx.mesh", "mpi4py.MPI"],
            "missing_modules": ["dolfinx.fem"],
            "missing_test_modules": ["pytest"],
            "checks": {
                "dolfinx.fem": {"available": False, "error": "missing"},
            },
            "message": "DOLFINx forward backend is unavailable: missing dolfinx.fem",
        }

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "dolfinx_backend_status", fake_status)

    exit_code = main(["dolfinx-backend-check", "--output", str(output)])

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["available"] is False
    assert payload["missing_modules"] == ["dolfinx.fem"]
    assert "available: False" in captured.out
    assert "missing_test_modules: pytest" in captured.out


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
    assert payload["noip"]["ramp_off_time"] == 1.0e-5
    assert payload["noip"]["turnoff_steps"] == 10


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
        (output_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "primary_secondary_internal_time_grid": {
                        "contains_turnoff_start": True,
                        "contains_turnoff_end": True,
                        "contains_all_observation_outputs": True,
                        "last_output_internal_time_s": 1.00001,
                    }
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "errors.csv").write_text(
            "time_obs,component,pred,ref,abs_error,ordinary_relative_error,"
            "relative_error_with_floor,peak_normalized_error,pass_5pct\n"
            f"1e-05,Ex,{1.0 + offset},{1.0 + offset},0.0,0.0,0.0,0.0,true\n"
            f"1.0,dBzdt,{3.0 + offset},{3.0 + offset},0.0,0.0,0.0,0.0,true\n",
            encoding="utf-8",
        )
        (output_dir / "comparison_3comp.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
        (output_dir / "error_curves_3comp.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
        (output_dir / "model_schematic.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
        (output_dir / "run_config_resolved.yaml").write_text("case_type: fake\n", encoding="utf-8")
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
            "polarization_effect_dir": str(output_root / "polarization_effect"),
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
    figure_targets = {
        target["figure"]: target
        for target in spec["paper_response_targets"]["digitization_targets"]
    }
    assert figure_targets["Fig. 2"]["pdf_page_number"] == 3
    assert figure_targets["Fig. 2"]["component"] == "Ex"
    assert figure_targets["Fig. 2"]["response_units"] == "V/m"
    assert figure_targets["Fig. 2"]["response_axis"] == {
        "x_scale": "log10",
        "x_min_s": 1.0e-5,
        "x_max_s": 1.0,
        "y_scale": "log10",
        "y_min": 1.0e-8,
        "y_max": 1.0e-3,
    }
    assert figure_targets["Fig. 2"]["figure_crop_fraction"] == [0.15, 0.07, 0.88, 0.335]
    assert figure_targets["Fig. 15"]["pdf_page_number"] == 9
    assert figure_targets["Fig. 15"]["component"] == "Hz"
    assert figure_targets["Fig. 15"]["response_units"] == "nT"
    assert figure_targets["Fig. 15"]["response_axis"]["x_min_s"] == 1.0e-4
    assert figure_targets["Fig. 15"]["response_axis"]["y_max"] == 1.0e-2
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
    assert manifest["targets"][0]["figure"] == "Fig. 2"
    assert manifest["targets"][0]["pdf_page_number"] == 3
    assert manifest["targets"][0]["response_units"] == "V/m"
    csv_lines = (output_dir / "paper_curve_digitization_template.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == (
        "figure,pdf_page_number,model_key,component,response_panel,value_kind,"
        "curve_label,time_obs,value,units,response_axis,axis_notes,figure_crop_fraction,caption,notes"
    )
    assert any(
        line.startswith("Fig. 12,7,three_dimensional_polarized_body,Ex,a,response,paper_ip,,,V/m,\"{\"\"x_max_s\"\": 1.0")
        for line in csv_lines
    )
    assert any(
        line.startswith("Fig. 15,9,three_dimensional_polarized_body,Hz,a,response,paper_noip,,,nT,\"{\"\"x_max_s\"\": 1.0")
        for line in csv_lines
    )


def test_published_paper_figure_pages_cli_writes_page_manifest(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    spec_path = tmp_path / "paper_model_target.json"
    output_dir = tmp_path / "figure_pages"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    exit_code = main(
        [
            "published-paper-figure-pages",
            str(spec_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "paper_figure_page_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rendered"] is False
    assert manifest["pdf_path"] == ""
    assert [page["pdf_page_number"] for page in manifest["pages"]] == [3, 5, 7, 9]
    page3 = manifest["pages"][0]
    assert page3["image"] == ""
    assert [target["figure"] for target in page3["targets"]] == ["Fig. 2", "Fig. 3"]
    assert page3["targets"][0]["component"] == "Ex"
    assert page3["targets"][0]["figure_image"] == ""


def test_published_paper_figure_page_package_renders_unique_pages(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls = []

    def fake_runner(command, check):
        calls.append((list(command), check))
        Path(str(command[-1]) + ".png").write_bytes(b"\x89PNG\r\n\x1a\n")

    manifest = write_published_paper_figure_page_package(
        spec,
        tmp_path / "figure_pages",
        pdf_path=pdf,
        dpi=90,
        render=True,
        renderer="fake-pdftoppm",
        runner=fake_runner,
    )

    assert manifest["rendered"] is True
    assert manifest["dpi"] == 90
    assert [page["pdf_page_number"] for page in manifest["pages"]] == [3, 5, 7, 9]
    assert [page["image"] for page in manifest["pages"]] == [
        "paper_page_003.png",
        "paper_page_005.png",
        "paper_page_007.png",
        "paper_page_009.png",
    ]
    assert len(calls) == 4
    assert calls[0][0][:7] == ["fake-pdftoppm", "-f", "3", "-l", "3", "-r", "90"]
    assert calls[0][1] is True


def test_published_paper_figure_page_package_crops_figures(tmp_path):
    from PIL import Image

    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def fake_runner(command, check):
        image = Image.new("RGB", (1000, 1200), color="white")
        image.save(str(command[-1]) + ".png")

    manifest = write_published_paper_figure_page_package(
        spec,
        tmp_path / "figure_pages",
        pdf_path=pdf,
        render=True,
        crop_figures=True,
        runner=fake_runner,
    )

    assert manifest["crop_figures"] is True
    page3_targets = manifest["pages"][0]["targets"]
    assert page3_targets[0]["figure_image"] == "paper_fig_002.png"
    assert page3_targets[1]["figure_image"] == "paper_fig_003.png"
    assert (tmp_path / "figure_pages" / "paper_fig_002.png").is_file()
    with Image.open(tmp_path / "figure_pages" / "paper_fig_002.png") as cropped:
        assert cropped.size == (730, 318)


def test_published_paper_curve_artifacts_cli_writes_overlay_outputs(tmp_path):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "\n".join(
            [
                "time_obs,Ex,Hz",
                "1e-05,1.02,2.04e-09",
                "0.001,0.51,1.02e-09",
                "1.0,0.102,2.04e-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    digitized = tmp_path / "digitized.csv"
    digitized.write_text(
        "\n".join(
            [
                "figure,model_key,component,curve_label,time_obs,value,notes",
                "Fig. 12,three_dimensional_polarized_body,Ex,paper_ip,1e-05,1.0,",
                "Fig. 12,three_dimensional_polarized_body,Ex,paper_ip,0.001,0.5,",
                "Fig. 12,three_dimensional_polarized_body,Ex,paper_ip,1.0,0.1,",
                "Fig. 15,three_dimensional_polarized_body,Hz,paper_ip,1e-05,2.0e-09,",
                "Fig. 15,three_dimensional_polarized_body,Hz,paper_ip,0.001,1.0e-09,",
                "Fig. 15,three_dimensional_polarized_body,Hz,paper_ip,1.0,2.0e-10,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "paper_overlay"

    exit_code = main(
        [
            "published-paper-curve-artifacts",
            str(predictions),
            str(digitized),
            "--output-dir",
            str(output_dir),
            "--case-type",
            "ip",
            "--curve-label",
            "paper_ip",
            "--component-figure",
            "Ex=Fig. 12",
            "--component-figure",
            "Hz=Fig. 15",
        ]
    )

    assert exit_code == 0
    for name in (
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "error_summary.json",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
        "paper_response_overlay.png",
        "paper_relative_error_curves.png",
        "runtime_diagnostics.json",
    ):
        assert (output_dir / name).is_file()
    payload = json.loads((output_dir / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["reference_type"] == "published_response_curve"
    assert payload["final_acceptance_passed"] is False
    assert "reference_type_not_final_acceptance" in payload["acceptance_status"]["blocking_reasons"]
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["published_response_curve"]["curve_label"] == "paper_ip"
    assert diagnostics["published_response_curve"]["component_figures"] == {"Ex": "Fig. 12", "Hz": "Fig. 15"}


def test_published_paper_curve_artifacts_cli_infers_figures_from_model_key(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    spec_path = tmp_path / "paper_model_target.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "\n".join(
            [
                "time_obs,Ex,Hz",
                "1e-05,1.02,2.04e-09",
                "0.001,0.51,1.02e-09",
                "1.0,0.102,2.04e-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    digitized = tmp_path / "digitized.csv"
    digitized.write_text(
        "\n".join(
            [
                "figure,model_key,component,curve_label,time_obs,value,notes",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,1e-05,1.0,",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,0.001,0.5,",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,1.0,0.1,",
                "Fig. 8,layered_polarization_model,Hz,paper_ip,1e-05,2.0e-09,",
                "Fig. 8,layered_polarization_model,Hz,paper_ip,0.001,1.0e-09,",
                "Fig. 8,layered_polarization_model,Hz,paper_ip,1.0,2.0e-10,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "paper_overlay"

    exit_code = main(
        [
            "published-paper-curve-artifacts",
            str(predictions),
            str(digitized),
            "--output-dir",
            str(output_dir),
            "--case-type",
            "ip",
            "--curve-label",
            "paper_ip",
            "--paper-spec",
            str(spec_path),
            "--model-key",
            "layered_polarization_model",
        ]
    )

    assert exit_code == 0
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["published_response_curve"]["component_figures"] == {"Ex": "Fig. 7", "Hz": "Fig. 8"}
    assert diagnostics["published_response_curve"]["model_key"] == "layered_polarization_model"


def test_published_paper_digitization_audit_cli_reports_missing_records(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    spec_path = tmp_path / "paper_model_target.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "\n".join(
            [
                "time_obs,Ex,Hz",
                "1e-05,1.02,2.04e-09",
                "0.001,0.51,1.02e-09",
                "1.0,0.102,2.04e-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    digitized = tmp_path / "digitized.csv"
    digitized.write_text(
        "\n".join(
            [
                "figure,model_key,component,curve_label,time_obs,value,notes",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,1e-05,1.0,",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,0.001,0.5,",
                "Fig. 7,layered_polarization_model,Ex,paper_ip,1.0,0.1,",
                "Fig. 8,layered_polarization_model,Hz,paper_ip,1e-05,2.0e-09,",
                "Fig. 8,layered_polarization_model,Hz,paper_ip,0.001,1.0e-09,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    exit_code = main(
        [
            "published-paper-digitization-audit",
            str(predictions),
            str(digitized),
            "--output",
            str(output),
            "--curve-label",
            "paper_ip",
            "--paper-spec",
            str(spec_path),
            "--model-key",
            "layered_polarization_model",
        ]
    )

    assert exit_code == 1
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["complete"] is False
    assert audit["schema_valid"] is True
    assert audit["component_figures"] == {"Ex": "Fig. 7", "Hz": "Fig. 8"}
    assert audit["model_key"] == "layered_polarization_model"
    assert audit["missing_record_count"] == 1
    assert audit["missing_records"] == [
        {
            "figure": "Fig. 8",
            "component": "Hz",
            "curve_label": "paper_ip",
            "time_obs": 1.0,
        }
    ]


def test_published_paper_prony_materials_cli_writes_fit_json(tmp_path):
    spec = build_published_paper_model_target_spec(tmp_path / "paper_run")
    spec_path = tmp_path / "paper_model_target.json"
    output = tmp_path / "paper_prony_materials.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    exit_code = main(
        [
            "published-paper-prony-materials",
            str(spec_path),
            "--output",
            str(output),
            "--n-terms",
            "6",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_article_id"] == "S092698512400329X"
    assert payload["frequency_count"] == 81
    assert payload["n_terms"] == 6
    assert set(payload["materials"]) == {
        "accuracy_benchmark_layer",
        "layered_polarization_model",
        "three_dimensional_high_resistivity_body",
        "three_dimensional_low_resistivity_body",
    }
    layered = payload["materials"]["layered_polarization_model"]
    assert layered["rho0_ohm_m"] == 100.0
    assert layered["chargeability"] == 0.3
    assert layered["sigma0"] == 0.01
    assert layered["sigma_inf"] > layered["sigma0"]
    assert len(layered["terms"]) == 6
    low = payload["materials"]["three_dimensional_low_resistivity_body"]
    high = payload["materials"]["three_dimensional_high_resistivity_body"]
    assert low["rho0_ohm_m"] == 10.0
    assert high["rho0_ohm_m"] == 1000.0
    assert low["relative_l2"] >= 0.0
    assert high["relative_l2"] >= 0.0


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


def test_corrected_terrain_smoke_case_spec_uses_constant_primary_and_self_convergence(tmp_path):
    spec = build_corrected_terrain_smoke_case_spec(tmp_path / "terrain_smoke")

    assert spec["case_type"] == "noip"
    assert spec["reference_type"] == "self_convergence"
    assert spec["validation_scope"] == "corrected_model_terrain_leakage_diagnostic"
    assert spec["observation_times"] == [1.0e-5, 1.0]
    assert spec["turnoff_steps"] == 1
    assert spec["receiver"] == [0.25, 0.25, -0.25]
    forward = spec["dolfinx_forward"]
    assert forward["primary_provider_mode"] == "constant"
    assert forward["terrain_mesh"]["mode"] == "small_gmsh_terrain_leakage"
    assert forward["leakage_channel"]["min_marked_cells"] == 1
    assert spec["runner"]["diagnostic"] == "gmsh_terrain_constant_primary_artifact_smoke"


def test_corrected_terrain_smoke_case_specs_include_ip(tmp_path):
    specs = build_corrected_terrain_smoke_case_specs(tmp_path / "terrain_smoke")

    assert set(specs) == {"noip", "ip"}
    assert specs["noip"]["case_type"] == "noip"
    assert specs["ip"]["case_type"] == "ip"
    assert specs["ip"]["reference_type"] == "self_convergence"
    assert specs["ip"]["dolfinx_forward"]["primary_provider_mode"] == "constant"
    leakage = specs["ip"]["dolfinx_forward"]["leakage_channel"]
    assert leakage["sigma_inf"] == 0.05
    assert leakage["delta_sigma_list"] == [0.015]
    assert leakage["tau_list"] == [0.1]
    assert specs["ip"]["output_dir"].endswith("terrain_leakage_ip_smoke")


def test_corrected_terrain_smoke_run_cli_dispatches_self_convergence_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run(case_spec):
        calls.append(case_spec)
        output_dir = Path(case_spec["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "error_summary.json").write_text(
            json.dumps(
                {
                    "case_type": "noip",
                    "reference_type": "self_convergence",
                    "final_acceptance_passed": False,
                }
            ),
            encoding="utf-8",
        )
        return {"final_acceptance_passed": False, "reference_type": "self_convergence"}

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_self_convergence_validation", fake_run)

    exit_code = main(["corrected-terrain-smoke-run", str(tmp_path / "terrain_smoke")])

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["reference_type"] == "self_convergence"
    assert calls[0]["dolfinx_forward"]["primary_provider_mode"] == "constant"


def test_corrected_terrain_smoke_run_cli_dispatches_both_cases(tmp_path, monkeypatch):
    calls = []

    def fake_run(case_spec):
        calls.append(case_spec["case_type"])
        output_dir = Path(case_spec["output_dir"])
        output_dir.mkdir(parents=True)
        return {"final_acceptance_passed": False, "reference_type": "self_convergence"}

    import atem3d.corrected_model_runner as runner

    monkeypatch.setattr(runner, "run_corrected_model_self_convergence_validation", fake_run)

    exit_code = main(["corrected-terrain-smoke-run", str(tmp_path / "terrain_smoke"), "--case", "both"])

    assert exit_code == 0
    assert calls == ["noip", "ip"]


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
