import h5py
import numpy as np
import yaml

from atem3d.config import build_simulation
from atem3d.io import save_result_hdf5
from atem3d import source_history_postprocess_cli


def _hj_config(*, source_history_kind: str | None = None) -> dict:
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.02, "tau": 0.05}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "magnetic_receiver_mode": "current_biot",
        "time_steps": [0.01, 0.02],
        "receivers": [
            {"location": [0.0, 0.5, 0.0], "component": "Ex"},
            {"location": [0.0, 0.5, 0.0], "component": "Hz"},
        ],
    }
    if source_history_kind == "driven":
        config["magnetic_recovery_source_history"] = {
            "kind": "driven_recovery_source_moments",
            "driver_tau": 0.05,
            "response_tau": 0.01,
            "source_moment_degrees": [0],
            "coefficients": [0.2],
            "receiver_matrix": "current_biot",
        }
    elif source_history_kind == "time_series":
        config["magnetic_recovery_source_history"] = {
            "kind": "time_series_source_moments",
            "times": [0.0, 0.01, 0.03],
            "source_moment_degrees": [0],
            "coefficients": [[0.0], [0.1], [0.3]],
            "receiver_matrix": "current_biot",
        }
    elif source_history_kind == "source_diffusion":
        config["model"] = {"sigma_infinity": 0.2}
        config["time_steps"] = [1.0e-7, 1.0e-7]
        config["magnetic_recovery_source_history"] = {
            "kind": "source_diffusion_kernel_source_moments",
            "amplitude": -0.25,
            "tau_multiplier": 2.0,
            "amplitude_time": 0.0,
            "source_moment_degrees": [0],
            "receiver_matrix": "current_biot",
        }
    return config


def test_source_history_postprocess_cli_matches_runtime_receiver_hook(tmp_path):
    base_config = _hj_config()
    corrected_config = _hj_config(source_history_kind="driven")
    base_result = build_simulation(base_config).run_data_only()
    expected_result = build_simulation(corrected_config).run_data_only()

    input_path = tmp_path / "base-data-only.h5"
    output_path = tmp_path / "corrected-data-only.h5"
    config_path = tmp_path / "corrected.yaml"
    save_result_hdf5(input_path, base_result, base_config)
    config_path.write_text(yaml.safe_dump(corrected_config), encoding="utf-8")

    exit_code = source_history_postprocess_cli.main(
        [str(input_path), "--config", str(config_path), "-o", str(output_path)]
    )

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], base_result.times)
        np.testing.assert_allclose(h5["data"][:], expected_result.data)
        np.testing.assert_allclose(h5["data"][:, 0], base_result.data[:, 0])
        assert np.linalg.norm(h5["data"][:, 1] - base_result.data[:, 1]) > 0.0
        assert "e" not in h5
        assert "h" not in h5
        assert bool(h5.attrs["receiver_data_only"]) is True
        assert bool(h5.attrs["source_history_postprocessed"]) is True
        saved_config = yaml.safe_load(h5.attrs["config_yaml"])
        assert saved_config["magnetic_recovery_source_history"]["kind"] == (
            "driven_recovery_source_moments"
        )


def test_source_history_postprocess_cli_matches_time_series_receiver_hook(tmp_path):
    base_config = _hj_config()
    corrected_config = _hj_config(source_history_kind="time_series")
    base_result = build_simulation(base_config).run_data_only()
    expected_result = build_simulation(corrected_config).run_data_only()

    input_path = tmp_path / "base-data-only.h5"
    output_path = tmp_path / "corrected-data-only.h5"
    config_path = tmp_path / "corrected.yaml"
    save_result_hdf5(input_path, base_result, base_config)
    config_path.write_text(yaml.safe_dump(corrected_config), encoding="utf-8")

    exit_code = source_history_postprocess_cli.main(
        [str(input_path), "--config", str(config_path), "-o", str(output_path)]
    )

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], base_result.times)
        np.testing.assert_allclose(h5["data"][:], expected_result.data)
        np.testing.assert_allclose(h5["data"][:, 0], base_result.data[:, 0])
        assert np.linalg.norm(h5["data"][:, 1] - base_result.data[:, 1]) > 0.0
        saved_config = yaml.safe_load(h5.attrs["config_yaml"])
        assert saved_config["magnetic_recovery_source_history"]["kind"] == (
            "time_series_source_moments"
        )


def test_source_history_postprocess_cli_matches_source_diffusion_kernel_receiver_hook(
    tmp_path,
):
    base_config = _hj_config(source_history_kind="source_diffusion")
    corrected_config = _hj_config(source_history_kind="source_diffusion")
    base_config.pop("magnetic_recovery_source_history")
    base_result = build_simulation(base_config).run_data_only()
    expected_result = build_simulation(corrected_config).run_data_only()

    input_path = tmp_path / "base-data-only.h5"
    output_path = tmp_path / "corrected-data-only.h5"
    config_path = tmp_path / "corrected.yaml"
    save_result_hdf5(input_path, base_result, base_config)
    config_path.write_text(yaml.safe_dump(corrected_config), encoding="utf-8")

    exit_code = source_history_postprocess_cli.main(
        [str(input_path), "--config", str(config_path), "-o", str(output_path)]
    )

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], base_result.times)
        np.testing.assert_allclose(h5["data"][:], expected_result.data)
        np.testing.assert_allclose(h5["data"][:, 0], base_result.data[:, 0])
        assert np.linalg.norm(h5["data"][:, 1] - base_result.data[:, 1]) > 0.0
        saved_config = yaml.safe_load(h5.attrs["config_yaml"])
        assert saved_config["magnetic_recovery_source_history"]["kind"] == (
            "source_diffusion_kernel_source_moments"
        )


def test_source_history_postprocess_cli_accepts_full_field_hdf5(tmp_path):
    base_config = _hj_config(source_history_kind="source_diffusion")
    corrected_config = _hj_config(source_history_kind="source_diffusion")
    base_config.pop("magnetic_recovery_source_history")
    base_result = build_simulation(base_config).run()
    expected_result = build_simulation(corrected_config).run()

    input_path = tmp_path / "base-full.h5"
    output_path = tmp_path / "corrected-data-only.h5"
    config_path = tmp_path / "corrected.yaml"
    save_result_hdf5(input_path, base_result, base_config)
    config_path.write_text(yaml.safe_dump(corrected_config), encoding="utf-8")

    exit_code = source_history_postprocess_cli.main(
        [str(input_path), "--config", str(config_path), "-o", str(output_path)]
    )

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], base_result.times)
        np.testing.assert_allclose(h5["data"][:], expected_result.data)
        assert bool(h5.attrs["receiver_data_only"]) is True
        assert bool(h5.attrs["source_history_postprocessed"]) is True
