import json

import numpy as np

from atem3d import empymod_validation_cli
from atem3d.empymod_validation import EmpymodValidationResult, EmpymodValidationSweepResult


def _fake_validation(*, passed=True):
    return EmpymodValidationResult(
        times=np.array([1.0e-3]),
        numerical=np.array([[2.0]]),
        reference=np.array([[2.0]]),
        component_names=["Hz@0"],
        components={
            "Hz@0": {
                "relative_l2": 0.0 if passed else 1.0,
                "relative_linf": 0.0 if passed else 1.0,
                "absolute_linf": 0.0 if passed else 1.0,
                "passed": passed,
                "passed_by": "relative" if passed else "none",
            }
        },
        diagnostics={},
        metadata={},
        tolerance=0.2,
        absolute_tolerance=1.0e-9,
    )


def test_empymod_validation_cli_writes_report(tmp_path, monkeypatch):
    config_path = tmp_path / "case.yaml"
    output_path = tmp_path / "report.json"
    config_path.write_text("value: 1\n", encoding="utf-8")

    def fake_validation(config, **kwargs):
        assert config == {"value": 1}
        assert kwargs["depths"] == [0.0]
        assert kwargs["resistivities"] == [1.0e8, 100.0]
        assert kwargs["skip_positive_times"] == 2
        assert kwargs["data_only"] is True
        assert kwargs["time_min"] == 1.0e-4
        assert kwargs["time_max"] == 2.0e-3
        assert kwargs["tolerance"] == 0.2
        assert kwargs["absolute_tolerance"] == 1.0e-9
        assert kwargs["empymod_strength"] == 2.5
        assert kwargs["empymod_kwargs"] == {"srcpts": 11, "recpts": 3}
        assert kwargs["output_path"] == output_path
        return _fake_validation(passed=True)

    monkeypatch.setattr(empymod_validation_cli, "run_empymod_validation", fake_validation)

    status = empymod_validation_cli.main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--skip-positive-times",
            "2",
            "--srcpts",
            "11",
            "--recpts",
            "3",
            "--data-only",
            "--time-min",
            "1e-4",
            "--time-max",
            "2e-3",
            "--tolerance",
            "0.2",
            "--absolute-tolerance",
            "1e-9",
            "--empymod-strength",
            "2.5",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0


def test_empymod_validation_cli_can_write_three_component_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / "case.yaml"
    output_path = tmp_path / "report.json"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text("value: 1\n", encoding="utf-8")

    validation = EmpymodValidationResult(
        times=np.array([1.0e-5, 1.0e-3, 1.0]),
        numerical=np.array(
            [
                [1.0, 0.5, 1.0e-9],
                [0.5, 0.25, 5.0e-10],
                [0.1, 0.05, 1.0e-10],
            ]
        ),
        reference=np.array(
            [
                [1.0, 0.5, 1.0e-9],
                [0.5, 0.25, 5.0e-10],
                [0.1, 0.05, 1.0e-10],
            ]
        ),
        component_names=["Ex", "Ey", "dBzdt"],
        components={
            "Ex": {
                "relative_l2": 0.0,
                "relative_linf": 0.0,
                "absolute_linf": 0.0,
                "passed": True,
                "passed_by": "relative",
            },
            "Ey": {
                "relative_l2": 0.0,
                "relative_linf": 0.0,
                "absolute_linf": 0.0,
                "passed": True,
                "passed_by": "relative",
            },
            "dBzdt": {
                "relative_l2": 0.0,
                "relative_linf": 0.0,
                "absolute_linf": 0.0,
                "passed": True,
                "passed_by": "relative",
            },
        },
        diagnostics={},
        metadata={},
        tolerance=0.05,
    )
    monkeypatch.setattr(
        empymod_validation_cli,
        "run_empymod_validation",
        lambda config, **kwargs: validation,
    )

    status = empymod_validation_cli.main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--artifact-dir",
            str(artifact_dir),
            "--case-type",
            "noip",
            "--magnetic-quantity",
            "dBzdt",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
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
        assert (artifact_dir / name).is_file()


def test_empymod_validation_cli_require_pass_returns_nonzero_for_failed_case(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "case.yaml"
    output_path = tmp_path / "report.json"
    config_path.write_text("value: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        empymod_validation_cli,
        "run_empymod_validation",
        lambda config, **kwargs: _fake_validation(passed=False),
    )

    status = empymod_validation_cli.main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--tolerance",
            "0.2",
            "--require-pass",
            "-o",
            str(output_path),
        ]
    )

    assert status == 1


def test_empymod_validation_cli_runs_sweep_cases(tmp_path, monkeypatch):
    config_path = tmp_path / "case.yaml"
    sweep_path = tmp_path / "sweep.yaml"
    output_path = tmp_path / "sweep_report.json"
    config_path.write_text("value: 1\n", encoding="utf-8")
    sweep_path.write_text(
        """
cases:
  - name: strict
    overrides:
      magnetic_receiver_mode: current_biot
  - name: local
    overrides:
      magnetic_receiver_mode: edge_basis_cell_biot
""",
        encoding="utf-8",
    )

    calls = {}

    class FakeSweep:
        cases = {}

        def to_report(self):
            return {"passed": None}

    def fake_sweep(config, cases, **kwargs):
        calls["config"] = config
        calls["cases"] = cases
        calls["kwargs"] = kwargs
        return FakeSweep()

    monkeypatch.setattr(empymod_validation_cli, "run_empymod_validation_sweep", fake_sweep)

    status = empymod_validation_cli.main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--sweep-cases",
            str(sweep_path),
            "--data-only",
            "--time-min",
            "1e-5",
            "--time-max",
            "1e-3",
            "--tolerance",
            "0.3",
            "--absolute-tolerance",
            "1e-8",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    assert calls["config"] == {"value": 1}
    assert calls["cases"][1]["name"] == "local"
    assert calls["cases"][1]["overrides"]["magnetic_receiver_mode"] == (
        "edge_basis_cell_biot"
    )
    assert calls["kwargs"]["data_only"] is True
    assert calls["kwargs"]["time_min"] == 1.0e-5
    assert calls["kwargs"]["time_max"] == 1.0e-3
    assert calls["kwargs"]["tolerance"] == 0.3
    assert calls["kwargs"]["absolute_tolerance"] == 1.0e-8
    assert calls["kwargs"]["output_path"] == output_path


def test_empymod_validation_cli_require_pass_checks_sweep_report(tmp_path, monkeypatch):
    config_path = tmp_path / "case.yaml"
    sweep_path = tmp_path / "sweep.yaml"
    output_path = tmp_path / "sweep_report.json"
    config_path.write_text("value: 1\n", encoding="utf-8")
    sweep_path.write_text(
        """
cases:
  - name: pass_case
    overrides: {}
  - name: fail_case
    overrides: {}
""",
        encoding="utf-8",
    )

    def fake_sweep(config, cases, **kwargs):
        return EmpymodValidationSweepResult(
            cases={
                "pass_case": _fake_validation(passed=True),
                "fail_case": _fake_validation(passed=False),
            },
            overrides={"pass_case": {}, "fail_case": {}},
        )

    monkeypatch.setattr(empymod_validation_cli, "run_empymod_validation_sweep", fake_sweep)

    status = empymod_validation_cli.main(
        [
            str(config_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--sweep-cases",
            str(sweep_path),
            "--tolerance",
            "0.2",
            "--require-pass",
            "-o",
            str(output_path),
        ]
    )

    assert status == 1
