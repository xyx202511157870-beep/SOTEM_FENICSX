import json

import numpy as np
import yaml


def _small_hj_config():
    return {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {"sigma_infinity": 2.0},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 3.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.1],
        "receiver_line": {
            "x": [0.0],
            "y": 0.5,
            "z": 0.0,
            "components": ["Hz"],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "source_diffusion_kernel_source_moments",
            "normalized_amplitude": -3.5,
            "tau_multiplier": 1.25,
            "amplitude_time": 1.0e-4,
            "source_moment_degrees": [0],
        },
    }


def _small_eb_config():
    config = _small_hj_config()
    config["formulation"] = "eb"
    config["receiver_line"]["components"] = ["Ex"]
    config["magnetic_receiver_mode"] = "stored_b"
    config.pop("magnetic_recovery_source_history")
    return config


def test_audit_source_geometry_reports_eb_edge_orientation():
    from atem3d.source_geometry_audit_cli import audit_source_geometry

    payload = audit_source_geometry(_small_eb_config(), config_path="small-eb.yaml")

    assert payload["formulation"] == "eb"
    assert payload["source_vector"]["location"] == "edge"
    orientation = payload["source_vector"]["edge_orientation"]
    assert orientation["source_length_m"] == 1.0
    assert orientation["edge_block_sizes"] == [
        payload["mesh"]["n_edges_x"],
        payload["mesh"]["n_edges_y"],
        payload["mesh"]["n_edges_z"],
    ]
    assert orientation["orientation_cosine"] > 0.99
    assert orientation["relative_parallel_length_error"] < 1.0e-12
    assert orientation["transverse_residual_m"] < 1.0e-12
    assert orientation["reversed_orientation"] is False


def test_audit_source_geometry_reports_hj_face_metrics():
    from atem3d.source_geometry_audit_cli import audit_source_geometry

    payload = audit_source_geometry(_small_hj_config(), config_path="small.yaml")

    assert payload["config_path"] == "small.yaml"
    assert payload["formulation"] == "hj"
    assert payload["source"]["length_m"] == 1.0
    assert payload["source"]["current_a"] == 3.0
    assert payload["source_vector"]["location"] == "face"
    assert payload["source_vector"]["active_count"] > 0
    assert payload["source_vector"]["active_count"] == sum(
        payload["source_vector"]["active_count_by_orientation"].values()
    )
    assert payload["source_vector"]["l1_abs"] > 0.0
    assert payload["source_vector"]["l2"] > 0.0
    assert payload["diffusion_scale"]["sigma_midpoint_s_per_m"] == 2.0
    np.testing.assert_allclose(
        payload["diffusion_scale"]["tau0_s"],
        8.0e-7 * np.pi,
    )
    configured = payload["configured_source_diffusion"]
    assert configured["present"] is True
    assert configured["tau_multiplier"] == 1.25
    assert configured["basis_kind"] == "continuous"
    assert configured["normalized_amplitude"] == -3.5
    np.testing.assert_allclose(
        configured["amplitude"],
        -3.5 * payload["diffusion_scale"]["tau0_s"],
    )


def test_audit_source_geometry_reports_hj_face_inner_product_norms():
    from atem3d.config import build_simulation
    from atem3d.source_geometry_audit_cli import audit_source_geometry

    config = _small_hj_config()
    sim = build_simulation(config)
    mesh = sim.mesh
    source_vector = -sim.sources[0].initial_face_vector(mesh)
    unit_face = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    rho_face = mesh.get_face_inner_product(0.5 * np.ones(mesh.n_cells)).tocsr()
    unit_quadratic = float(source_vector @ (unit_face @ source_vector))
    rho_quadratic = float(source_vector @ (rho_face @ source_vector))
    curl_t_rho = mesh.edge_curl.T @ (rho_face @ source_vector)

    payload = audit_source_geometry(config)
    inner = payload["source_vector"]["face_inner_products"]

    np.testing.assert_allclose(inner["unit_quadratic"], unit_quadratic)
    np.testing.assert_allclose(inner["unit_l2"], np.sqrt(unit_quadratic))
    np.testing.assert_allclose(inner["rho_quadratic"], rho_quadratic)
    np.testing.assert_allclose(inner["rho_l2"], np.sqrt(rho_quadratic))
    np.testing.assert_allclose(
        inner["curl_transpose_rho_source_l2"],
        np.linalg.norm(curl_t_rho),
    )


def test_audit_source_geometry_reports_receiver_static_response():
    from atem3d.config import build_simulation
    from atem3d.magnetic_recovery import face_current_biot_matrix
    from atem3d.source_geometry_audit_cli import audit_source_geometry

    config = _small_hj_config()
    sim = build_simulation(config)
    mesh = sim.mesh
    source_vector = -sim.sources[0].initial_face_vector(mesh)
    locations = np.array([[0.0, 0.5, 0.0]])
    matrix = face_current_biot_matrix(
        mesh,
        locations,
        subdivisions=sim.magnetic_recovery_subdivisions,
    )
    expected_h = np.einsum("kcf,f->kc", matrix, source_vector)

    payload = audit_source_geometry(config)
    response = payload["source_vector"]["receiver_static_response"]

    assert response["present"] is True
    assert response["receiver_matrix"] == "face_current_biot"
    np.testing.assert_allclose(response["locations"], locations)
    np.testing.assert_allclose(response["h_vectors"], expected_h)
    np.testing.assert_allclose(response["h_l2"], np.linalg.norm(expected_h))
    assert response["component_values"][0]["component"] == "Hz"
    np.testing.assert_allclose(
        response["component_values"][0]["value"],
        expected_h[0, 2],
    )


def test_source_geometry_audit_cli_writes_json(tmp_path):
    from atem3d.source_geometry_audit_cli import main

    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "geometry.json"
    config_path.write_text(yaml.safe_dump(_small_hj_config()), encoding="utf-8")

    status = main([str(config_path), "-o", str(output_path)])

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["config_path"] == str(config_path)
    assert payload["source_vector"]["location"] == "face"
    assert payload["configured_source_diffusion"]["normalized_amplitude"] == -3.5


def test_source_geometry_audit_cli_writes_sweep_json(tmp_path):
    from atem3d.source_geometry_audit_cli import main

    config_path = tmp_path / "config.yaml"
    sweep_path = tmp_path / "sweep.yaml"
    output_path = tmp_path / "geometry-sweep.json"
    config_path.write_text(yaml.safe_dump(_small_hj_config()), encoding="utf-8")
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "cases": {
                    "current_2a": {"source": {"current": 2.0}},
                    "current_4a": {"source": {"current": 4.0}},
                }
            }
        ),
        encoding="utf-8",
    )

    status = main(
        [
            str(config_path),
            "--sweep-cases",
            str(sweep_path),
            "-o",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["base_config_path"] == str(config_path)
    assert payload["sweep_cases_path"] == str(sweep_path)
    assert payload["case_count"] == 2
    assert set(payload["cases"]) == {"current_2a", "current_4a"}
    assert payload["cases"]["current_2a"]["source"]["current_a"] == 2.0
    assert payload["cases"]["current_4a"]["source"]["current_a"] == 4.0
    assert payload["cases"]["current_4a"]["source_vector"]["l1_abs"] == (
        2.0 * payload["cases"]["current_2a"]["source_vector"]["l1_abs"]
    )
