import h5py
import numpy as np
import pytest
import yaml
from pathlib import Path

from atem3d import compare_cli


def test_compare_cli_passes_empymod_integration_points(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "coordinate_system": "depth_down",
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    calls = []

    def fake_reference(survey, **kwargs):
        calls.append(kwargs)
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--srcpts",
            "9",
            "--recpts",
            "3",
        ]
    )

    assert calls == [{"srcpts": 9, "recpts": 3}]


def test_compare_cli_sets_local_numba_cache_when_missing(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    cache_dirs = []

    def fake_reference(survey, **kwargs):
        import os

        cache_dirs.append(os.environ.get("NUMBA_CACHE_DIR"))
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
        ]
    )

    assert Path(cache_dirs[0]).name == ".numba_cache"
    assert Path(cache_dirs[0]).is_absolute()


def test_compare_cli_can_override_empymod_strength(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append(survey)
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--empymod-strength",
            "50",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert surveys[0].strength == 50.0
    assert payload["metadata"]["empymod"]["strength"] == 50.0


def test_compare_cli_preserves_coordinate_system_in_survey_and_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "coordinate_system": "z_up",
        "source": {
            "start": [-25.0, 0.0, -0.5],
            "end": [25.0, 0.0, -0.5],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, -0.5], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append(survey)
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert surveys[0].coordinate_system == "z_up"
    assert payload["metadata"]["empymod"]["coordinate_system"] == "z_up"


def test_compare_cli_writes_empymod_settings_to_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[1.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--srcpts",
            "7",
            "--recpts",
            "1",
            "--signal",
            "-1",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["empymod"]["srcpts"] == 7
    assert payload["metadata"]["empymod"]["recpts"] == 1
    assert payload["metadata"]["empymod"]["signal"] == -1


def test_compare_cli_can_skip_initial_positive_time_samples(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "dBzdt"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3, 2.0e-3, 7.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [100.0], [2.0], [3.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append(survey)
        return np.array([[2.0], [3.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--skip-positive-times",
            "1",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(surveys[0].times, [2.0e-3, 7.0e-3])
    assert payload["n_times"] == 2
    assert payload["metadata"]["empymod"]["skip_positive_times"] == 1


def test_compare_cli_can_limit_receiver_columns(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receiver_line": {
            "x": [0.0],
            "y": 10.0,
            "z": 0.0,
            "components": ["Ex", "Ey", "Hz"],
        },
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append(survey)
        return np.array([[3.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--receiver-indices",
            "2",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert surveys[0].receiver_components == [((0.0, 10.0, 0.0), "Hz")]
    assert payload["n_components"] == 1
    assert set(payload["components"]) == {"Hz@x=0"}
    assert payload["metadata"]["empymod"]["receiver_indices"] == [2]


def test_compare_cli_can_write_plot(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    plot_path = tmp_path / "comparison.png"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3, 2.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0], [0.5]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[1.0], [0.25]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--plot",
            str(plot_path),
        ]
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_compare_cli_can_use_configured_debye_ip_model(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "model": {
            "layers": [
                {"top": -1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {
                    "top": 0.0,
                    "bottom": 1.0e9,
                    "sigma_infinity": 0.02,
                    "debye_terms": [{"delta_sigma": 0.005, "tau": 0.1}],
                },
            ]
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    surveys = []

    def fake_reference(survey, **kwargs):
        surveys.append(survey)
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "50",
            "--use-config-ip",
        ]
    )

    assert isinstance(surveys[0].resistivities, dict)
    assert "func_eta" in surveys[0].resistivities


def test_compare_cli_can_include_per_time_samples_in_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3, 2.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [2.0], [6.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[1.0], [3.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--include-samples",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["samples"]["Ex@0"][0]["time"] == 1.0e-3
    assert payload["samples"]["Ex@0"][1]["ratio_numerical_over_reference"] == 2.0


def test_compare_cli_can_recompute_current_biot_data_from_saved_fields(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, receiver_indices=None):
        assert path == result_path
        assert subdivisions == 3
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-subdivisions",
            "3",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["recomputed_current_biot"] is True
    assert payload["metadata"]["atem3d"]["magnetic_recovery_subdivisions"] == 3


def test_compare_cli_can_recompute_edge_current_biot_data_from_saved_fields(
    tmp_path,
    monkeypatch,
):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(
        path,
        *,
        subdivisions,
        magnetic_receiver_mode,
        receiver_indices=None,
    ):
        assert path == result_path
        assert subdivisions is None
        assert magnetic_receiver_mode == "edge_current_biot"
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-edge-current-biot",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_receiver_mode"] == "edge_current_biot"


def test_compare_cli_can_recompute_edge_basis_biot_data_from_saved_fields(
    tmp_path,
    monkeypatch,
):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(
        path,
        *,
        subdivisions,
        magnetic_receiver_mode,
        receiver_indices=None,
    ):
        assert path == result_path
        assert subdivisions is None
        assert magnetic_receiver_mode == "edge_basis_biot"
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-edge-basis-biot",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_receiver_mode"] == "edge_basis_biot"


def test_compare_cli_can_recompute_edge_basis_cell_biot_data_from_saved_fields(
    tmp_path,
    monkeypatch,
):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(
        path,
        *,
        subdivisions,
        magnetic_receiver_mode,
        receiver_indices=None,
    ):
        assert path == result_path
        assert subdivisions is None
        assert magnetic_receiver_mode == "edge_basis_cell_biot"
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-edge-basis-cell-biot",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_receiver_mode"] == "edge_basis_cell_biot"


def test_compare_cli_can_set_recomputed_polarization_scale(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, polarization_scale, receiver_indices=None):
        assert path == result_path
        assert subdivisions is None
        assert polarization_scale == 1.25
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-polarization-scale",
            "1.25",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["atem3d"]["magnetic_recovery_polarization_scale"] == 1.25


def test_compare_cli_can_set_recomputed_initial_polarization_scale(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(
        path,
        *,
        subdivisions,
        initial_polarization_scale,
        receiver_indices=None,
    ):
        assert path == result_path
        assert subdivisions is None
        assert initial_polarization_scale == 0.01
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-initial-polarization-scale",
            "0.01",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert (
        payload["metadata"]["atem3d"]["magnetic_recovery_initial_polarization_scale"]
        == 0.01
    )


def test_compare_cli_can_apply_source_primary_delta6_correction(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
        "model": {
            "layers": [
                {"top": 1e9, "bottom": 0.0, "sigma_infinity": 1e-8},
                {
                    "top": 0.0,
                    "bottom": -1e9,
                    "sigma_infinity": 0.012,
                    "debye_terms": [{"delta_sigma": 0.002, "tau": 0.001}],
                },
            ]
        },
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, receiver_indices=None):
        assert path == result_path
        assert subdivisions is None
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    def fake_correction(numerical, *, times, config, receiver_components):
        assert np.allclose(times, [0.0, 1.0e-3])
        assert receiver_components == [((0.0, 10.0, 0.0), "Hz")]
        assert config["model"]["layers"][1]["debye_terms"][0]["delta_sigma"] == 0.002
        return numerical + np.array([[0.0], [3.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(
        compare_cli,
        "_apply_source_primary_delta6_correction",
        fake_correction,
    )
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[5.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-source-primary-delta6",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_recovery_source_primary_correction"] == {
        "diagnostic_only": True,
        "kind": "delta6",
        "formula": "-6 * mu0 * delta_sigma * L^2 * H_wire * exp(-t / (2 tau))",
    }


def test_compare_cli_rejects_nonfinite_initial_polarization_scale(tmp_path, capsys):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    with pytest.raises(SystemExit) as excinfo:
        compare_cli.main(
            [
                str(result_path),
                "--depths",
                "0",
                "--resistivities",
                "1e8",
                "100",
                "--recompute-current-biot",
                "--magnetic-recovery-initial-polarization-scale",
                "nan",
            ]
        )

    assert excinfo.value.code == 2
    assert "--magnetic-recovery-initial-polarization-scale must be finite" in (
        capsys.readouterr().err
    )


def test_compare_cli_rejects_initial_polarization_scale_with_fit(tmp_path, capsys):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    with pytest.raises(SystemExit) as excinfo:
        compare_cli.main(
            [
                str(result_path),
                "--depths",
                "0",
                "--resistivities",
                "1e8",
                "100",
                "--recompute-current-biot",
                "--fit-magnetic-recovery-memory-scale",
                "--magnetic-recovery-initial-polarization-scale",
                "0.01",
            ]
        )

    assert excinfo.value.code == 2
    assert (
        "magnetic recovery fit flags cannot be combined with "
        "--magnetic-recovery-initial-polarization-scale"
    ) in capsys.readouterr().err


def test_compare_cli_can_set_low_frequency_ratio_polarization_scale(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, polarization_scale, receiver_indices=None):
        assert path == result_path
        assert subdivisions is None
        assert polarization_scale == "low_frequency_ratio"
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-polarization-scale",
            "low_frequency_ratio",
        ]
    )


def test_compare_cli_can_set_component_polarization_scale(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, polarization_scale, receiver_indices=None):
        assert path == result_path
        assert subdivisions is None
        assert polarization_scale == [1.1, 0.9, 0.5]
        assert receiver_indices is None
        return np.array([[0.0], [2.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--magnetic-recovery-polarization-scale",
            "1.1,0.9,0.5",
        ]
    )


def test_compare_cli_recomputes_only_requested_receiver_columns(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receiver_line": {
            "x": [0.0],
            "y": 10.0,
            "z": 0.0,
            "components": ["Ex", "Ey", "Hz"],
        },
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    def fake_recompute(path, *, subdivisions, receiver_indices):
        assert path == result_path
        assert subdivisions is None
        assert receiver_indices == [2]
        return np.array([[0.0], [3.0]])

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[3.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--receiver-indices",
            "2",
            "--recompute-current-biot",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["n_components"] == 1
    assert payload["metadata"]["empymod"]["receiver_indices"] == [2]


def test_fit_recomputed_component_polarization_scale_uses_component_basis(monkeypatch):
    base = np.array([[10.0], [8.0], [6.0]])
    px = np.array([[1.0], [2.0], [3.0]])
    py = np.array([[0.5], [1.0], [0.0]])
    pz = np.zeros_like(px)
    reference = base - 1.25 * px - 0.5 * py
    calls = []

    def fake_recompute(path, *, subdivisions, polarization_scale, receiver_indices=None):
        calls.append((path, subdivisions, polarization_scale, receiver_indices))
        if polarization_scale == [0.0, 0.0, 0.0]:
            return base
        if polarization_scale == [1.0, 0.0, 0.0]:
            return base - px
        if polarization_scale == [0.0, 1.0, 0.0]:
            return base - py
        if polarization_scale == [0.0, 0.0, 1.0]:
            return base - pz
        raise AssertionError(f"unexpected scale {polarization_scale}")

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)

    fit = compare_cli.fit_recomputed_component_polarization_scale(
        "result.h5",
        reference[1:],
        fit_mask=np.array([False, True, True]),
        subdivisions=3,
        receiver_indices=[2],
    )

    np.testing.assert_allclose(fit.weights[:2], [1.25, 0.5])
    np.testing.assert_allclose(fit.fitted_data, reference)
    assert calls[0] == ("result.h5", 3, [0.0, 0.0, 0.0], [2])


def test_fit_recomputed_memory_polarization_scale_uses_initial_memory_basis(monkeypatch):
    base = np.array([[10.0], [8.0], [6.0]])
    current_memory = np.array([[1.0], [2.0], [3.0]])
    initial_memory = np.array([[0.5], [1.0], [0.0]])
    reference = base - 1.25 * current_memory + 0.5 * initial_memory
    calls = []

    def fake_recompute(
        path,
        *,
        subdivisions,
        polarization_scale=None,
        initial_polarization_scale=None,
        receiver_indices=None,
    ):
        calls.append(
            (
                path,
                subdivisions,
                polarization_scale,
                initial_polarization_scale,
                receiver_indices,
            )
        )
        polarization_scale = 1.0 if polarization_scale is None else polarization_scale
        initial_polarization_scale = (
            0.0 if initial_polarization_scale is None else initial_polarization_scale
        )
        return (
            base
            - float(polarization_scale) * current_memory
            + float(initial_polarization_scale) * initial_memory
        )

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)

    fit = compare_cli.fit_recomputed_memory_polarization_scale(
        "result.h5",
        reference[1:],
        fit_mask=np.array([False, True, True]),
        subdivisions=3,
        receiver_indices=[2],
    )

    np.testing.assert_allclose(fit.weights, [1.25, 0.5])
    np.testing.assert_allclose(fit.fitted_data, reference)
    assert calls[0] == ("result.h5", 3, 0.0, 0.0, [2])
    assert calls[1] == ("result.h5", 3, 1.0, 0.0, [2])
    assert calls[2] == ("result.h5", 3, 0.0, 1.0, [2])


def test_fit_recomputed_memory_polarization_scale_per_receiver_uses_independent_weights(
    monkeypatch,
):
    base = np.array([[10.0, 4.0], [8.0, 3.0], [6.0, 2.0]])
    current_memory = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
    initial_memory = np.array([[0.5, 1.0], [1.0, 0.0], [0.0, 2.0]])
    weights = np.array([[1.25, 0.5], [0.75, -0.25]])
    reference = np.empty_like(base)
    for column in range(base.shape[1]):
        reference[:, column] = (
            base[:, column]
            - weights[column, 0] * current_memory[:, column]
            + weights[column, 1] * initial_memory[:, column]
        )

    def fake_recompute(
        path,
        *,
        subdivisions,
        polarization_scale=None,
        initial_polarization_scale=None,
        receiver_indices=None,
    ):
        assert path == "result.h5"
        assert subdivisions == 3
        assert receiver_indices == [2, 5]
        polarization_scale = 1.0 if polarization_scale is None else polarization_scale
        initial_polarization_scale = (
            0.0 if initial_polarization_scale is None else initial_polarization_scale
        )
        return (
            base
            - float(polarization_scale) * current_memory
            + float(initial_polarization_scale) * initial_memory
        )

    monkeypatch.setattr(compare_cli, "recompute_current_biot_receiver_data", fake_recompute)

    fit = compare_cli.fit_recomputed_memory_polarization_scale_per_receiver(
        "result.h5",
        reference[1:],
        fit_mask=np.array([False, True, True]),
        subdivisions=3,
        receiver_indices=[2, 5],
    )

    np.testing.assert_allclose(fit.weights, weights)
    np.testing.assert_allclose(fit.fitted_data, reference)


def test_compare_cli_can_write_fitted_component_scale_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    class FakeFit:
        weights = np.array([1.1, 0.9, 0.0])
        rank = 2
        singular_values = np.array([3.0, 1.0])
        fitted_data = np.array([[0.0], [2.0]])

    def fake_fit(path, reference, *, fit_mask, subdivisions, receiver_indices=None):
        assert path == result_path
        np.testing.assert_array_equal(fit_mask, [False, True])
        assert subdivisions is None
        assert receiver_indices is None
        np.testing.assert_allclose(reference, [[2.0]])
        return FakeFit()

    monkeypatch.setattr(compare_cli, "fit_recomputed_component_polarization_scale", fake_fit)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--fit-magnetic-recovery-component-scale",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_recovery_component_fit"]["weights"] == [
        1.1,
        0.9,
        0.0,
    ]


def test_compare_cli_can_write_fitted_memory_scale_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Hz"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [99.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    class FakeFit:
        weights = np.array([1.2, 0.3])
        rank = 2
        singular_values = np.array([3.0, 1.0])
        fitted_data = np.array([[0.0], [2.0]])

    def fake_fit(path, reference, *, fit_mask, subdivisions, receiver_indices=None):
        assert path == result_path
        np.testing.assert_array_equal(fit_mask, [False, True])
        assert subdivisions is None
        assert receiver_indices is None
        np.testing.assert_allclose(reference, [[2.0]])
        return FakeFit()

    monkeypatch.setattr(compare_cli, "fit_recomputed_memory_polarization_scale", fake_fit)
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--fit-magnetic-recovery-memory-scale",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Hz@0"]["relative_linf"] == 0.0
    assert payload["metadata"]["atem3d"]["magnetic_recovery_memory_fit"]["weights"] == [
        1.2,
        0.3,
    ]


def test_compare_cli_can_write_per_receiver_memory_scale_report(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    report_path = tmp_path / "report.json"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receiver_line": {
            "x": [-20.0, 0.0],
            "y": 10.0,
            "z": 0.0,
            "components": ["Hz"],
        },
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0, 0.0], [99.0, 88.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    class FakeFit:
        weights = np.array([[1.2, 0.3], [0.8, -0.1]])
        ranks = np.array([2, 1])
        singular_values = [np.array([3.0, 1.0]), np.array([2.0])]
        fitted_data = np.array([[0.0, 0.0], [2.0, 4.0]])

    def fake_fit(path, reference, *, fit_mask, subdivisions, receiver_indices=None):
        assert path == result_path
        np.testing.assert_array_equal(fit_mask, [False, True])
        assert subdivisions is None
        assert receiver_indices is None
        np.testing.assert_allclose(reference, [[2.0, 4.0]])
        return FakeFit()

    monkeypatch.setattr(
        compare_cli,
        "fit_recomputed_memory_polarization_scale_per_receiver",
        fake_fit,
    )
    monkeypatch.setattr(compare_cli, "run_empymod_reference", lambda survey, **kwargs: np.array([[2.0, 4.0]]))

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
            "--recompute-current-biot",
            "--fit-magnetic-recovery-memory-scale-per-receiver",
            "-o",
            str(report_path),
        ]
    )

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    fit_payload = payload["metadata"]["atem3d"]["magnetic_recovery_memory_fit_per_receiver"]
    assert fit_payload["Hz@x=-20"]["weights"] == [1.2, 0.3]
    assert fit_payload["Hz@x=0"]["weights"] == [0.8, -0.1]


def test_recompute_current_biot_receiver_data_uses_saved_field_shapes(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0],
            "origin": [-1.0, -1.0, -0.5],
        },
        "model": {"sigma_infinity": 0.1},
        "time_steps": [0.01],
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [
            {"location": [0.0, 0.0, 0.0], "component": "Ex"},
            {"location": [0.0, 0.0, 0.0], "component": "Hz"},
        ],
    }
    sim = compare_cli.build_simulation(config)
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=sim.times)
        h5.create_dataset("e", data=np.zeros((sim.times.size, sim.mesh.n_edges)))
        h5.create_dataset("b", data=np.zeros((sim.times.size, sim.mesh.n_faces)))
        h5.create_dataset("data", data=np.full((sim.times.size, len(sim.receivers)), 99.0))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    recovered = compare_cli.recompute_current_biot_receiver_data(result_path)

    np.testing.assert_allclose(recovered, 0.0)
    assert recovered.shape == (2, 2)


def test_recompute_current_biot_receiver_data_accepts_low_frequency_ratio_scale(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0],
            "origin": [-1.0, -1.0, -0.5],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "time_steps": [0.01],
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 0.0, 0.0], "component": "Hz"}],
    }
    sim = compare_cli.build_simulation(config)
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=sim.times)
        h5.create_dataset("e", data=np.zeros((sim.times.size, sim.mesh.n_edges)))
        h5.create_dataset("b", data=np.zeros((sim.times.size, sim.mesh.n_faces)))
        h5.create_dataset("data", data=np.full((sim.times.size, len(sim.receivers)), 99.0))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    recovered = compare_cli.recompute_current_biot_receiver_data(
        result_path,
        polarization_scale="low_frequency_ratio",
    )

    np.testing.assert_allclose(recovered, 0.0)
    assert recovered.shape == (2, 1)


def test_compare_cli_defaults_to_finite_source_integration(tmp_path, monkeypatch):
    result_path = tmp_path / "result.h5"
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    calls = []

    def fake_reference(survey, **kwargs):
        calls.append(kwargs)
        return np.array([[1.0]])

    monkeypatch.setattr(compare_cli, "run_empymod_reference", fake_reference)

    compare_cli.main(
        [
            str(result_path),
            "--depths",
            "0",
            "--resistivities",
            "1e8",
            "100",
        ]
    )

    assert calls[0]["srcpts"] >= 51
