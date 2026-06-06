import json

import yaml

from atem3d import cli


def test_cli_run_subcommand_dispatches_to_main_run(monkeypatch):
    seen = {}

    def fake_main_run(argv):
        seen["argv"] = list(argv)
        return 17

    monkeypatch.setattr(cli, "_main_run", fake_main_run)

    exit_code = cli.main(["run", "config.yaml", "--data-only"])

    assert exit_code == 17
    assert seen["argv"] == ["config.yaml", "--data-only"]


def test_cli_plot_subcommand_regenerates_validation_figures(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.csv").write_text(
        "time_obs,Ex,Ey,dBzdt\n1e-5,1.0,0.1,0.01\n2e-5,0.8,0.08,0.008\n",
        encoding="utf-8",
    )
    (run_dir / "reference_empymod_or_1d.csv").write_text(
        "time_obs,Ex,Ey,dBzdt\n1e-5,1.1,0.11,0.011\n2e-5,0.9,0.09,0.009\n",
        encoding="utf-8",
    )
    (run_dir / "errors.csv").write_text(
        "time_obs,component,pred,ref,abs_error,ordinary_relative_error,relative_error_with_floor,peak_normalized_error,pass_5pct\n"
        "1e-5,Ex,1.0,1.1,0.1,0.0909,0.0909,0.0909,False\n"
        "2e-5,Ex,0.8,0.9,0.1,0.1111,0.1111,0.0909,False\n"
        "1e-5,Ey,0.1,0.11,0.01,0.0909,0.0909,0.0909,False\n"
        "2e-5,Ey,0.08,0.09,0.01,0.1111,0.1111,0.0909,False\n"
        "1e-5,dBzdt,0.01,0.011,0.001,0.0909,0.0909,0.0909,False\n"
        "2e-5,dBzdt,0.008,0.009,0.001,0.1111,0.1111,0.0909,False\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["plot", str(run_dir)])

    assert exit_code == 0
    assert (run_dir / "comparison_3comp.png").exists()
    assert (run_dir / "error_curves_3comp.png").exists()


def test_cli_validate_secondary_writes_zero_contrast_summary(tmp_path):
    output_dir = tmp_path / "secondary"
    config_path = tmp_path / "secondary.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(output_dir),
                "Ep0": [[1.0, 0.0, 0.0], [0.5, 0.25, 0.0]],
                "sigma": 0.01,
                "sigma_background": 0.01,
                "times": [1.0e-5, 2.0e-5],
                "threshold": 1.0e-12,
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-secondary", str(config_path)])

    assert exit_code == 0
    summary = json.loads((output_dir / "secondary_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["case_type"] == "secondary_zero_contrast"
    assert summary["pass_zero_contrast"] is True
    assert summary["max_abs_Es"] == 0.0
    assert summary["max_abs_secondary_dBdt"] == 0.0
    assert summary["total_response_equals_primary"] is True
    trace = (output_dir / "secondary_validation_trace.csv").read_text(encoding="utf-8")
    assert "time_obs,max_abs_Es,max_abs_secondary_dBdt,total_response_equals_primary" in trace
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["validation_type"] == "secondary_zero_contrast"
    resolved = yaml.safe_load((output_dir / "run_config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["sigma"] == 0.01


def test_cli_validate_secondary_writes_forward_core_predictions(tmp_path):
    output_dir = tmp_path / "secondary_forward"
    config_path = tmp_path / "secondary_forward.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(output_dir),
                "Ep0": [[1.0, 0.0, 0.0]],
                "sigma": 0.01,
                "sigma_background": 0.01,
                "times": [1.0e-5, 2.0e-5],
                "threshold": 1.0e-12,
                "receiver_locations": [[0.0, -300.0, -0.1]],
                "components": ["Ex", "Ey", "dBzdt"],
                "receiver_E": [
                    [[10.0, 1.0, 0.0]],
                    [[5.0, 0.5, 0.0]],
                ],
                "receiver_dBdt": [
                    [[0.0, 0.0, -3.0]],
                    [[0.0, 0.0, -1.5]],
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-secondary", str(config_path)])

    assert exit_code == 0
    predictions = (output_dir / "primary_secondary_predictions.csv").read_text(encoding="utf-8")
    assert predictions.splitlines() == [
        "time_obs,Ex,Ey,dBzdt",
        "1.0000000000000001e-05,10,1,-3",
        "2.0000000000000002e-05,5,0.5,-1.5",
    ]
    summary = json.loads((output_dir / "secondary_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["forward_core_used"] is True
    assert summary["max_abs_total_minus_primary"] == 0.0


def test_cli_validate_secondary_supports_ip_zero_delta_material(tmp_path):
    output_dir = tmp_path / "secondary_ip"
    config_path = tmp_path / "secondary_ip.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "secondary": {
                    "output_dir": str(output_dir),
                    "Ep0": [[1.0, 0.0, 0.0]],
                    "sigma": 0.01,
                    "sigma_background": 0.01,
                    "times": [1.0e-5, 2.0e-5],
                    "threshold": 1.0e-12,
                    "receiver_locations": [[0.0, -300.0, -0.1]],
                    "components": ["Ex", "Ey", "dBzdt"],
                    "receiver_E": [
                        [[10.0, 1.0, 0.0]],
                        [[5.0, 0.5, 0.0]],
                    ],
                    "receiver_dBdt": [
                        [[0.0, 0.0, -3.0]],
                        [[0.0, 0.0, -1.5]],
                    ],
                },
                "material": {
                    "sigma_inf": 0.01,
                    "terms": [{"delta_sigma": 0.0, "tau": 0.1}],
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-secondary", str(config_path)])

    assert exit_code == 0
    summary = json.loads((output_dir / "secondary_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["case_type"] == "secondary_zero_contrast"
    assert summary["material_model"] == "prony"
    assert summary["sigma0"] == 0.01
    assert summary["sigma_inf"] == 0.01
    assert summary["delta_sigma_list"] == [0.0]
    assert summary["pass_zero_contrast"] is True
    assert summary["max_abs_total_minus_primary"] == 0.0


def test_cli_acceptance_report_writes_noip_ip_gate_summary(tmp_path):
    noip = tmp_path / "noip" / "error_summary.json"
    ip = tmp_path / "ip" / "error_summary.json"
    noip.parent.mkdir()
    ip.parent.mkdir()
    noip.write_text(json.dumps(_acceptance_summary("noip", True)), encoding="utf-8")
    ip.write_text(
        json.dumps(_acceptance_summary("ip", False, ["physical_error_gate_failed"])),
        encoding="utf-8",
    )
    output_dir = tmp_path / "acceptance"
    config_path = tmp_path / "acceptance.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "acceptance": {
                    "noip_summary_json": str(noip),
                    "ip_summary_json": str(ip),
                    "output_dir": str(output_dir),
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["acceptance-report", str(config_path)])

    assert exit_code == 1
    summary = json.loads((output_dir / "final_acceptance_summary.json").read_text(encoding="utf-8"))
    assert summary["final_acceptance_passed"] is False
    assert summary["failed_cases"] == ["ip"]
    assert summary["blocking_reasons_by_case"]["ip"] == ["physical_error_gate_failed"]


def test_cli_corrected_model_spec_writes_latest_geometry_case_specs(tmp_path):
    spec_path = tmp_path / "corrected_model_spec.json"

    exit_code = cli.main(["corrected-model-spec", str(tmp_path / "runs"), "--output", str(spec_path)])

    assert exit_code == 0
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["noip"]["source_start"] == [-500.0, 200.0, -0.1]
    assert spec["noip"]["source_end"] == [500.0, 200.0, -0.1]
    assert spec["noip"]["receiver"] == [0.0, -300.0, -0.1]
    assert spec["noip"]["validation_scope"] == "corrected_model_full"
    assert spec["ip"]["case_type"] == "ip"
    assert spec["ip"]["output_dir"] == str(tmp_path / "runs" / "ip_3comp")


def _acceptance_summary(case_type: str, passed: bool, reasons=None):
    reasons = [] if reasons is None else list(reasons)
    return {
        "case_type": case_type,
        "reference_type": "empymod",
        "magnetic_quantity": "dBzdt",
        "final_acceptance_passed": passed,
        "acceptance_status": {
            "final_acceptance_passed": passed,
            "blocking_reasons": reasons,
        },
    }
