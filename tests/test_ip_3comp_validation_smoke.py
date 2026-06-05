import json

import numpy as np

from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)


def test_ip_3comp_validation_smoke_records_prony_metadata(tmp_path):
    times = np.array([1.0e-5, 1.0e-2, 1.0])
    reference = np.array(
        [
            [2.0, -1.0, 1.0e-6],
            [1.0, -0.5, 2.0e-7],
            [0.25, -0.1, 1.0e-8],
        ]
    )
    predictions = reference * 1.02
    material = PronyConductivity(
        sigma_inf=0.02,
        terms=[
            DebyeTerm(delta_sigma=0.003, tau=0.1),
            DebyeTerm(delta_sigma=0.002, tau=1.0),
        ],
    )

    write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path,
            times=times,
            predictions=predictions,
            reference=reference,
            component_names=["Ex", "Ey", "Hz"],
            case_type="ip",
            reference_type="1d",
            magnetic_quantity="Hz",
            material=material,
        )
    )

    payload = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert payload["case_type"] == "ip"
    assert payload["reference_type"] == "1d"
    assert payload["sigma0"] == material.sigma0
    assert payload["sigma_inf"] == material.sigma_inf
    assert payload["sum_delta_sigma"] == 0.005
    assert payload["tau_list"] == [0.1, 1.0]
    assert payload["delta_sigma_list"] == [0.003, 0.002]
    assert payload["prony_dc_constraint_error"] == 0.0
    assert payload["pass_all_components"] is True
