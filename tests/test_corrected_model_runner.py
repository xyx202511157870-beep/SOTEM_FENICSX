import json

import numpy as np

from atem3d.cli import main
from atem3d.corrected_model import CorrectedModelValidationConfig, build_corrected_model_case_specs
from atem3d.corrected_model_runner import run_corrected_model_validation


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
