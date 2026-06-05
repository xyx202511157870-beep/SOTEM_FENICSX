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
