import json

import h5py
import numpy as np
import yaml
from discretize import TensorMesh

from atem3d.config import build_simulation
from atem3d.magnetic_recovery import face_current_biot_matrix


def test_source_neighborhood_audit_cli_fits_compact_group_weight(tmp_path):
    from atem3d.source_neighborhood_audit_cli import main

    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {"sigma_infinity": 1.0},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.1, 0.1],
        "receiver_line": {
            "x": [0.0],
            "y": 0.5,
            "z": 0.0,
            "components": ["Hz"],
        },
        "magnetic_receiver_mode": "current_biot",
    }
    sim = build_simulation(config)
    mesh: TensorMesh = sim.mesh
    times = np.array([0.0, 0.1, 0.2])
    h = np.zeros((times.size, mesh.n_edges), dtype=float)
    h[1, :6] = [0.0, 0.3, -0.1, 0.2, -0.05, 0.15]
    h[2, :6] = [0.0, 0.18, -0.08, 0.12, -0.03, 0.09]
    current = (mesh.edge_curl @ h[1:].T).T
    receiver_matrix = face_current_biot_matrix(mesh, [[0.0, 0.5, 0.0]], subdivisions=1)
    numerical = np.einsum("kcf,tf->tkc", receiver_matrix, current)[:, 0, 2]
    reference = 0.5 * numerical

    hdf5_path = tmp_path / "result.h5"
    with h5py.File(hdf5_path, "w") as h5:
        h5.create_dataset("times", data=times)
        h5.create_dataset("h", data=h)
        h5.create_dataset("e", data=np.zeros((times.size, mesh.n_faces)))
        h5.create_dataset("data", data=np.c_[np.zeros(times.size), np.zeros(times.size), np.r_[0.0, numerical]])
        h5.attrs["formulation"] = "hj"
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    report = {
        "samples": {
            "Hz@x=0": [
                {
                    "time": float(time),
                    "numerical": float(num),
                    "reference": float(ref),
                }
                for time, num, ref in zip(times[1:], numerical, reference)
            ]
        }
    }
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    status = main(
        [
            str(hdf5_path),
            str(report_path),
            "--components",
            "Hz@x=0",
            "--radial-edges",
            "0",
            "10",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["max_abs_difference_from_report_numerical"] < 1.0e-14
    assert payload["base_relative_l2"] == 1.0
    assert payload["groups"][0]["face_count"] == mesh.n_faces
    assert payload["neighborhood_fit"]["rank"] == 1
    assert payload["neighborhood_fit"]["relative_l2"] < 1.0e-12
    assert payload["neighborhood_fit"]["weights"] == [0.5]


def test_source_neighborhood_audit_cli_recovers_static_candidate_weight(tmp_path):
    from atem3d.source_neighborhood_audit_cli import main

    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {"sigma_infinity": 1.0},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.1, 0.1],
        "receiver_line": {
            "x": [0.0],
            "y": 0.5,
            "z": 0.0,
            "components": ["Hz"],
        },
        "magnetic_receiver_mode": "current_biot",
    }
    sim = build_simulation(config)
    mesh: TensorMesh = sim.mesh
    times = np.array([0.0, 0.1, 0.2])
    h = np.zeros((times.size, mesh.n_edges), dtype=float)
    candidate = -sim.sources[0].initial_face_vector(mesh)
    receiver_matrix = face_current_biot_matrix(mesh, [[0.0, 0.5, 0.0]], subdivisions=1)
    candidate_hz = float(np.einsum("kcf,f->kc", receiver_matrix, candidate)[0, 2])
    reference = np.full(2, 2.0 * candidate_hz)

    hdf5_path = tmp_path / "result.h5"
    with h5py.File(hdf5_path, "w") as h5:
        h5.create_dataset("times", data=times)
        h5.create_dataset("h", data=h)
        h5.create_dataset("e", data=np.zeros((times.size, mesh.n_faces)))
        h5.create_dataset("data", data=np.zeros((times.size, 3)))
        h5.attrs["formulation"] = "hj"
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    report = {
        "samples": {
            "Hz@x=0": [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(ref),
                }
                for time, ref in zip(times[1:], reference)
            ]
        }
    }
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    status = main(
        [
            str(hdf5_path),
            str(report_path),
            "--components",
            "Hz@x=0",
            "--radial-edges",
            "0",
            "10",
            "--candidates",
            "active_source",
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    candidate_fit = payload["candidate_static_fits"]["active_source"]
    assert candidate_fit["all_window_coefficient"] == 2.0
    assert candidate_fit["all_window_relative_l2"] < 1.0e-12
    assert candidate_fit["per_time_relative_l2"] < 1.0e-12
    assert candidate_fit["per_time_coefficients"] == [2.0, 2.0]


def test_coefficient_kernel_fits_recovers_exponential_amplitude():
    from atem3d.source_neighborhood_audit_cli import _coefficient_kernel_fits

    sample_times = np.array([0.1, 0.2, 0.3, 0.4])
    tau = 0.05
    coefficients = -3.0 * np.exp(-(sample_times - sample_times[0]) / tau)

    fits = _coefficient_kernel_fits(
        coefficients,
        sample_times,
        source_diffusion_time=tau,
        multipliers=[1.0],
    )

    assert fits[0]["tau"] == tau
    assert fits[0]["amplitude"] == -3.0
    assert fits[0]["coefficient_relative_l2"] < 1.0e-14


def test_candidate_static_fit_reports_residual_space_direction_metrics():
    from atem3d.source_neighborhood_audit_cli import _candidate_static_fit_report

    candidate_vector = np.array([1.0])
    receiver_matrix = np.array([[1.0], [0.0]])
    recomputed = np.zeros((2, 2), dtype=float)
    reference = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    sample_times = np.array([0.1, 0.2])

    report = _candidate_static_fit_report(
        candidate_vector,
        receiver_matrix,
        recomputed,
        reference,
        sample_times,
        source_diffusion_time=0.1,
    )

    assert np.isclose(report["all_window_residual_relative_l2"], np.sqrt(0.75))
    assert np.isclose(report["per_time_residual_relative_l2"], 1.0 / np.sqrt(2.0))
    assert np.isclose(report["per_time_residual_projection_fraction"], 0.5)
    assert report["per_time_response_target_cosines"] == [1.0, 0.0]
