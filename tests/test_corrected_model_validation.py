import numpy as np
import pytest

from atem3d.corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_model_case_specs,
)


def test_corrected_model_defaults_match_latest_task_coordinates():
    config = CorrectedModelValidationConfig()

    assert config.source_start == (-500.0, 200.0, -0.1)
    assert config.source_end == (500.0, 200.0, -0.1)
    assert config.receiver == (0.0, -300.0, -0.1)
    assert config.source_current == 10.0
    assert config.source_length == pytest.approx(1000.0)
    assert config.parallel_offset == pytest.approx(500.0)
    assert config.validation_scope == "corrected_model_full"


def test_corrected_model_observation_times_cover_required_full_window():
    config = CorrectedModelValidationConfig(n_observation_times=8)

    times = config.observation_times()

    assert times[0] == pytest.approx(1.0e-5)
    assert times[-1] == pytest.approx(1.0)
    assert np.all(np.diff(times) > 0.0)


def test_corrected_model_empymod_primary_config_uses_correct_source_receiver_and_current():
    config = CorrectedModelValidationConfig()

    provider_config = config.empymod_primary_config()

    assert provider_config["source_start"] == (-500.0, 200.0, -0.1)
    assert provider_config["source_end"] == (500.0, 200.0, -0.1)
    assert provider_config["current"] == 10.0
    assert provider_config["depths"] == (350.0, 650.0)
    assert provider_config["resistivities"] == (100.0, 100.0, 100.0)
    assert provider_config["coordinate_system"] == "depth_down"


def test_corrected_model_case_specs_prepare_noip_and_ip_acceptance_inputs(tmp_path):
    specs = build_corrected_model_case_specs(tmp_path)

    assert sorted(specs) == ["ip", "noip"]
    assert specs["noip"]["case_type"] == "noip"
    assert specs["ip"]["case_type"] == "ip"
    assert specs["noip"]["validation_scope"] == "corrected_model_full"
    assert specs["ip"]["validation_scope"] == "corrected_model_full"
    assert specs["noip"]["output_dir"] == str(tmp_path / "noip_3comp")
    assert specs["ip"]["output_dir"] == str(tmp_path / "ip_3comp")
    assert specs["noip"]["components"] == ["Ex", "Ey", "dBzdt"]
    assert specs["ip"]["magnetic_quantity"] == "dBzdt"
    assert specs["noip"]["runner"] == {
        "backend": "dolfinx_primary_secondary",
        "reference": "empymod",
        "components": ["Ex", "Ey", "dBzdt"],
        "output_root": str(tmp_path),
    }
    assert specs["noip"]["material"]["kind"] == "noip"
    assert specs["noip"]["material"]["sigma"] == pytest.approx(0.01)
    assert specs["ip"]["material"]["kind"] == "ip_prony"
    assert specs["ip"]["material"]["sigma0"] > 0.0
    assert specs["ip"]["material"]["sigma_inf"] >= specs["ip"]["material"]["sigma0"]
    assert specs["ip"]["material"]["prony_dc_constraint_error"] == pytest.approx(0.0)


def test_corrected_model_geometry_check_rejects_old_offset_model():
    with pytest.raises(ValueError, match="source length"):
        CorrectedModelValidationConfig(
            source_start=(-50.0, 0.0, -0.1),
            source_end=(50.0, 0.0, -0.1),
            receiver=(500.0, 50.0, -0.1),
        ).validate_geometry()
