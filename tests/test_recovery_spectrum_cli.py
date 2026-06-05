import json

import yaml

from atem3d.recovery_spectrum_cli import main


def test_recovery_spectrum_cli_writes_local_spectrum_report(tmp_path):
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "spectrum.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(
        [
            str(config_path),
            "--cell-indices",
            "0",
            "1",
            "--padding",
            "0",
            "--max-modes",
            "2",
            "--conductivity-model",
            "sigma0",
            "-o",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["conductivity_model"] == "sigma0"
    assert report["requested_cell_indices"] == [0, 1]
    assert report["support"]["global_cell_indices"] == [0, 1]
    assert report["support"]["n_cells"] == 2
    assert len(report["support"]["global_face_indices"]) == report["support"]["n_faces"]
    assert len(report["support"]["global_edge_indices"]) == report["support"]["n_edges"]
    assert len(report["spectrum"]["eigenvalues"]) == 2
    assert len(report["spectrum"]["time_constants"]) == 2
    assert report["spectrum"]["eigenvalue_filter"]["discarded_count"] > 0
    assert report["spectrum"]["eigenvalue_filter"]["last_discarded_eigenvalue"] <= (
        report["spectrum"]["eigenvalue_filter"]["eigenvalue_floor"]
    )
    assert report["spectrum"]["eigenvalue_filter"]["first_kept_eigenvalue"] == (
        report["spectrum"]["eigenvalues"][0]
    )
    assert all(value > 0.0 for value in report["spectrum"]["time_constants"])


def test_recovery_spectrum_cli_can_select_source_receiver_support(tmp_path):
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "spectrum.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--source-cell-radius",
            "0",
            "--receiver-cell-radius",
            "0",
            "--max-modes",
            "2",
            "--include-modal-coupling",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driven-source-projection",
            "charge_conserving",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["support_selection"]["kind"] == "source_receiver"
    assert report["support_selection"]["field_location"] == "face"
    assert report["support_selection"]["source_dof_indices"]
    assert report["support_selection"]["receiver_cell_indices"]
    assert report["support"]["n_cells"] >= len(
        report["support_selection"]["source_cell_indices"]
    )
    assert len(report["support"]["global_face_indices"]) == report["support"]["n_faces"]
    assert len(report["support"]["global_edge_indices"]) == report["support"]["n_edges"]
    assert len(report["spectrum"]["time_constants"]) == 2
    assert report["modal_coupling"]["source_moment_degrees"] == [0, 2]
    assert report["modal_coupling"]["receiver_mode"] == "face_current_biot"
    assert len(report["modal_coupling"]["modal_forcing"]) == 2
    assert len(report["modal_coupling"]["source_receiver_response"]) == 2
    projection = report["modal_coupling"]["source_moment_projection"]
    assert projection["design_matrix"]["shape"][1] == 2
    assert len(projection["coefficients"]) == 2
    assert len(projection["coefficients"][0]) == 2
    assert len(projection["coefficients"][0][0]) == 2
    assert projection["aggregate_relative_l2"] >= 0.0
    driven = report["modal_coupling"]["driven_response"]
    assert driven["driver_tau"] == 0.1
    assert driven["source_projection"] == "charge_conserving"
    assert driven["initial_state_kind"] == "zero"
    assert len(driven["times"]) == 2
    assert len(driven["source_moment_projection"]["coefficients"]) == 2


def test_recovery_spectrum_cli_sweeps_source_receiver_supports(tmp_path):
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "sweep.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--sweep-source-cell-radii",
            "0",
            "1",
            "--sweep-receiver-cell-radii",
            "0",
            "--sweep-padding",
            "0",
            "1",
            "--max-modes",
            "2",
            "-o",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["sweep"]["case_count"] == 4
    assert report["sweep"]["cases"][0]["source_cell_radius"] == 0
    assert report["sweep"]["cases"][0]["padding"] == 0
    assert report["sweep"]["cases"][0]["support"]["n_cells"] > 0
    assert len(report["sweep"]["cases"][0]["support"]["global_face_indices"]) == (
        report["sweep"]["cases"][0]["support"]["n_faces"]
    )
    assert len(report["sweep"]["cases"][0]["support"]["global_edge_indices"]) == (
        report["sweep"]["cases"][0]["support"]["n_edges"]
    )
    assert report["sweep"]["cases"][0]["diffusion_time_estimate"] > 0.0
    assert "spectrum" in report["sweep"]["cases"][0]


def test_recovery_spectrum_cli_can_skip_spectrum_for_driven_response(tmp_path):
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "driven.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--skip-spectrum",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driver-kind",
            "debye_build_up",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "spectrum" not in report
    assert report["spectrum_skipped"]["reason"] == "requested"
    assert report["support"]["n_cells"] > 0
    assert len(report["support"]["global_face_indices"]) == report["support"]["n_faces"]
    assert len(report["support"]["global_edge_indices"]) == report["support"]["n_edges"]
    assert report["driven_response"]["driver_tau"] == 0.1
    assert report["driven_response"]["driver_kind"] == "debye_build_up"
    assert report["driven_response"]["initial_state_kind"] == "zero"
    assert report["driven_response"]["forcing_kind"] == "source_edge_rhs"
    assert report["driven_response"]["driver_values"][0] == 0.0
    assert len(report["driven_response"]["source_moment_projection"]["coefficients"]) == 2

    relaxation_output = tmp_path / "driven_relaxation_difference.json"
    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--skip-spectrum",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driver-kind",
            "relaxation_difference",
            "--driver-fast-tau",
            "0.02",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(relaxation_output),
        ]
    )

    assert exit_code == 0
    relaxation_report = json.loads(relaxation_output.read_text(encoding="utf-8"))
    relaxation = relaxation_report["driven_response"]
    assert relaxation["driver_kind"] == "relaxation_difference"
    assert relaxation["driver_fast_tau"] == 0.02
    assert relaxation["driver_values"][0] == 0.0
    assert relaxation["driver_values"][1] > 0.0

    nonzero_initial_output = tmp_path / "driven_charge_conserving_mmr_initial.json"
    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--skip-spectrum",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driven-initial-state",
            "charge_conserving_mmr",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(nonzero_initial_output),
        ]
    )

    assert exit_code == 0
    nonzero_initial_report = json.loads(
        nonzero_initial_output.read_text(encoding="utf-8")
    )
    nonzero_initial = nonzero_initial_report["driven_response"]
    assert nonzero_initial["initial_state_kind"] == "charge_conserving_mmr"
    assert any(
        abs(value) > 0.0
        for source_response in nonzero_initial["receiver_response"][0]
        for receiver_response in source_response
        for value in receiver_response
    )

    global_initial_output = tmp_path / "driven_global_charge_conserving_mmr_initial.json"
    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--skip-spectrum",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driven-initial-state",
            "global_charge_conserving_mmr",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(global_initial_output),
        ]
    )

    assert exit_code == 0
    global_initial_report = json.loads(global_initial_output.read_text(encoding="utf-8"))
    global_initial = global_initial_report["driven_response"]
    assert global_initial["initial_state_kind"] == "global_charge_conserving_mmr"
    assert any(
        abs(value) > 0.0
        for source_response in global_initial["receiver_response"][0]
        for receiver_response in source_response
        for value in receiver_response
    )

    global_forcing_output = tmp_path / "driven_global_mmr_steady_forcing.json"
    exit_code = main(
        [
            str(config_path),
            "--support",
            "source_receiver",
            "--field-location",
            "face",
            "--skip-spectrum",
            "--include-driven-response",
            "--driver-tau",
            "0.1",
            "--driven-forcing",
            "global_mmr_steady",
            "--source-moment-degrees",
            "0",
            "2",
            "--modal-receiver-mode",
            "face_current_biot",
            "-o",
            str(global_forcing_output),
        ]
    )

    assert exit_code == 0
    global_forcing_report = json.loads(global_forcing_output.read_text(encoding="utf-8"))
    global_forcing = global_forcing_report["driven_response"]
    assert global_forcing["forcing_kind"] == "global_mmr_steady"
    assert len(global_forcing["source_moment_projection"]["coefficients"]) == 2
