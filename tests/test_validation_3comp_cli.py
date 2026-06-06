import numpy as np
import yaml

from atem3d import cli
from atem3d.validation_3comp import ThreeComponentValidationInput
from atem3d.validation_3comp import write_three_component_validation_artifacts


def test_cli_validate_noip_3comp_writes_artifacts_from_csv(tmp_path):
    pred_path = tmp_path / "pred.csv"
    ref_path = tmp_path / "ref.csv"
    _write_response_csv(pred_path, scale=1.01)
    _write_response_csv(ref_path, scale=1.0)
    output_dir = tmp_path / "noip"
    config_path = tmp_path / "validate-noip.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "predictions_csv": str(pred_path),
                    "reference_csv": str(ref_path),
                    "output_dir": str(output_dir),
                    "component_names": ["Ex", "Ey", "dBzdt"],
                    "reference_type": "empymod",
                    "magnetic_quantity": "dBzdt",
                    "validation_scope": "corrected_model_full",
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-noip-3comp", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "reference_empymod_or_1d.csv").is_file()
    assert (output_dir / "errors.csv").is_file()
    assert (output_dir / "error_summary.json").is_file()
    assert (output_dir / "run_config_resolved.yaml").is_file()
    payload = yaml.safe_load((output_dir / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["validation_scope"] == "corrected_model_full"
    assert payload["final_acceptance_passed"] is True


def test_cli_validate_ip_3comp_reads_prony_material_metadata(tmp_path):
    pred_path = tmp_path / "pred.csv"
    ref_path = tmp_path / "ref.csv"
    _write_response_csv(pred_path, scale=1.02, component_names=("Ex", "Ey", "Hz"))
    _write_response_csv(ref_path, scale=1.0, component_names=("Ex", "Ey", "Hz"))
    output_dir = tmp_path / "ip"
    config_path = tmp_path / "validate-ip.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "predictions_csv": str(pred_path),
                    "reference_csv": str(ref_path),
                    "output_dir": str(output_dir),
                    "component_names": ["Ex", "Ey", "Hz"],
                    "reference_type": "1d",
                    "magnetic_quantity": "Hz",
                },
                "material": {
                    "sigma_inf": 0.02,
                    "terms": [{"delta_sigma": 0.003, "tau": 0.1}],
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-ip-3comp", str(config_path)])

    assert exit_code == 0
    payload = yaml.safe_load((output_dir / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["case_type"] == "ip"
    assert payload["sigma_inf"] == 0.02
    assert payload["delta_sigma_list"] == [0.003]


def test_published_response_curve_reference_writes_diagnostic_artifacts(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 1.0e-9],
            [0.5, 5.0e-10],
            [0.1, 1.0e-10],
        ]
    )
    predictions = 1.02 * reference

    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path / "paper_overlay",
            times=times,
            predictions=predictions,
            reference=reference,
            component_names=["Ex", "Hz"],
            case_type="ip",
            reference_type="published_response_curve",
            magnetic_quantity="Hz",
            validation_scope="published_paper_reproduction_target",
            diagnostics={"published_reference": {"figure": "Fig. 12"}},
        )
    )

    assert summary["reference_type"] == "published_response_curve"
    assert summary["final_acceptance_passed"] is False
    assert summary["acceptance_status"]["reference_type_supported"] is True
    assert "reference_type_not_final_acceptance" in summary["acceptance_status"]["blocking_reasons"]
    assert "required_components_missing" in summary["acceptance_status"]["blocking_reasons"]
    assert (tmp_path / "paper_overlay" / "comparison_3comp.png").is_file()
    assert (tmp_path / "paper_overlay" / "error_curves_3comp.png").is_file()


def _write_response_csv(path, *, scale: float, component_names=("Ex", "Ey", "dBzdt")):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    values = np.array(
        [
            [1.0, 0.1, 1.0e-9],
            [0.5, 0.05, 5.0e-10],
            [0.1, 0.01, 1.0e-10],
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time_obs," + ",".join(component_names) + "\n")
        for time, row in zip(times, values * scale):
            handle.write(",".join([f"{time:.12g}", *[f"{value:.12g}" for value in row]]) + "\n")
