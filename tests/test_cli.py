import h5py
import yaml

from atem3d import cli


def test_cli_can_run_receiver_data_only_result(tmp_path):
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Ex"}],
    }
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "data-only.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = cli.main([str(config_path), "--data-only", "-o", str(output_path)])

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        assert h5["data"].shape == (2, 1)
        assert "e" not in h5
        assert "b" not in h5
        assert bool(h5.attrs["receiver_data_only"]) is True


def test_cli_can_run_hj_receiver_data_only_result(tmp_path):
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    config_path = tmp_path / "hj-config.yaml"
    output_path = tmp_path / "hj-data-only.h5"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = cli.main([str(config_path), "--data-only", "-o", str(output_path)])

    assert exit_code == 0
    with h5py.File(output_path, "r") as h5:
        assert h5["data"].shape == (2, 1)
        assert "e" not in h5
        assert "h" not in h5
        assert bool(h5.attrs["receiver_data_only"]) is True
        assert h5.attrs["formulation"] == "hj"
