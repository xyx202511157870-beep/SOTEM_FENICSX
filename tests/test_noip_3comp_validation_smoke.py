import json

import numpy as np

from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
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
    resolved = json.loads((tmp_path / "run_config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["case_type"] == "noip"
    assert resolved["component_names"] == ["Ex", "Ey", "dBzdt"]
