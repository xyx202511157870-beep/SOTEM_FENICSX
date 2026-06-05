import json

import h5py
import numpy as np
import yaml
from scipy.constants import mu_0

from atem3d.config import build_simulation
from atem3d.hj import hj_dc_initial_current_density, hj_dc_initial_electric_field
from atem3d.local_coupling import local_edge_basis, source_face_moment_basis
from atem3d.magnetic_recovery import (
    biot_savart_h_from_face_basis_currents,
    cell_current_biot_matrix,
    edge_current_biot_matrix,
    face_current_biot_matrix,
)
from atem3d.source_history_matrix_cli import (
    _coefficient_table,
    _format_coefficients,
    main,
)
from atem3d.source_history_matrix_scan_cli import main as scan_main
from atem3d.source_history_operator import (
    project_vector_to_spatial_basis,
    source_history_receiver_basis,
    source_history_receiver_basis_from_spatial_vectors,
    source_history_receiver_basis_from_static_response,
    source_history_receiver_basis_from_vectors,
)
from atem3d.source_history_runtime import charge_conserving_face_current


def test_source_history_matrix_cli_recovers_known_coefficients(tmp_path):
    _assert_source_history_matrix_cli_recovers_known_coefficients(
        tmp_path,
        receiver_matrix_name="edge_current",
        receiver_matrix_builder=edge_current_biot_matrix,
        source_vector_name="wire",
        source_vector_builder=_wire_source_vector,
    )


def test_source_history_matrix_cli_supports_current_biot_receiver_matrix(tmp_path):
    _assert_source_history_matrix_cli_recovers_known_coefficients(
        tmp_path,
        receiver_matrix_name="current_biot",
        receiver_matrix_builder=cell_current_biot_matrix,
        source_vector_name="wire",
        source_vector_builder=_wire_source_vector,
    )


def test_source_history_matrix_cli_supports_dc_total_source_vector(tmp_path):
    _assert_source_history_matrix_cli_recovers_known_coefficients(
        tmp_path,
        receiver_matrix_name="current_biot",
        receiver_matrix_builder=cell_current_biot_matrix,
        source_vector_name="dc_total_current",
        source_vector_builder=_dc_total_source_vector,
    )


def test_source_history_matrix_cli_supports_time_window(tmp_path):
    _assert_source_history_matrix_cli_recovers_known_coefficients(
        tmp_path,
        receiver_matrix_name="current_biot",
        receiver_matrix_builder=cell_current_biot_matrix,
        source_vector_name="wire",
        source_vector_builder=_wire_source_vector,
        extra_args=["--time-min", "0.2", "--time-max", "0.3"],
        expected_selected_times=2,
    )


def test_source_history_matrix_cli_supports_source_vectors_by_order(tmp_path):
    _assert_source_history_matrix_cli_recovers_known_coefficients(
        tmp_path,
        receiver_matrix_name="current_biot",
        receiver_matrix_builder=cell_current_biot_matrix,
        source_vector_name="wire",
        source_vector_builder=_wire_source_vector,
        source_vector_builders_by_order=[
            _wire_source_vector,
            _dc_total_source_vector,
        ],
        extra_args=["--source-vectors", "wire,dc_total_current"],
    )


def test_source_history_matrix_cli_reports_per_column_fits(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = cell_current_biot_matrix(sim.mesh, locations)
    basis = source_history_receiver_basis(
        sim.time_steps,
        tau=0.2,
        source_vector=_wire_source_vector(sim),
        receiver_matrix=receiver_matrix,
        max_order=1,
    )
    column_coefficients = np.array([[-2.0, 0.5], [-3.0, 1.25]])
    target = np.zeros((basis.responses.shape[0], locations.shape[0], 3))
    for receiver_index, coefficients in enumerate(column_coefficients):
        target[:, receiver_index, 2] = np.einsum(
            "p,tp->t",
            coefficients,
            basis.responses[:, :, receiver_index, 2],
        )
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "matrix_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--source-vector",
            "wire",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--per-column",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [item["component_name"] for item in result["per_column_fit"]] == [
        "Hz@x=0",
        "Hz@x=1",
    ]
    for item, coefficients in zip(result["per_column_fit"], column_coefficients):
        np.testing.assert_allclose(item["fit"]["coefficients"], coefficients)
        assert item["fit"]["relative_l2"] < 1.0e-14
    np.testing.assert_allclose(
        [item["receiver_location"] for item in result["per_column_fit"]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )


def test_source_history_matrix_cli_supports_local_edge_spatial_basis(tmp_path):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = cell_current_biot_matrix(sim.mesh, locations)
    local_basis = local_edge_basis(
        sim.mesh,
        _wire_source_vector(sim),
        locations,
        source_cell_radius=0,
        receiver_cell_radius=0,
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=local_basis.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=local_basis.basis_labels,
    )
    coefficients = np.linspace(-0.25, 0.5, basis.responses.shape[1])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "matrix_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--source-cell-radius",
            "0",
            "--receiver-cell-radius",
            "0",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["spatial_basis"] == "local_edges"
    assert result["local_edge_basis"]["source_cell_radius"] == 0
    assert result["local_edge_basis"]["receiver_cell_radius"] == 0
    assert result["local_edge_basis"]["support_edge_count"] == int(
        local_basis.support_edge_indices.size
    )
    np.testing.assert_array_equal(
        result["local_edge_basis"]["support_edge_indices"],
        local_basis.support_edge_indices,
    )
    assert len(result["basis_labels"]) == basis.responses.shape[1]
    assert result["fit"]["relative_l2"] < 1.0e-13
    assert result["fit"]["design_matrix"]["shape"] == [
        3 * locations.shape[0],
        basis.responses.shape[1],
    ]
    assert result["fit"]["design_matrix"]["rank_deficient"] is True
    assert "coefficients_over_mu_delta_l2" not in result["fit"]


def test_source_history_matrix_cli_supports_source_edge_local_basis_scope(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "source-edge-fit.json"

    exit_code = main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--local-basis-scope",
            "source_edges",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["local_edge_basis"]["basis_scope"] == "source_edges"
    assert result["local_edge_basis"]["basis_edge_count"] == int(
        report_paths["source_edge_count"]
    )
    assert result["fit"]["design_matrix"]["shape"] == [
        3 * 2,
        2 * report_paths["source_edge_count"],
    ]


def test_source_history_matrix_cli_supports_source_moment_local_basis_scope(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "source-moment-fit.json"

    exit_code = main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--local-basis-scope",
            "source_moments",
            "--source-moment-degree",
            "2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["local_edge_basis"]["basis_scope"] == "source_moments"
    assert result["local_edge_basis"]["source_moment_degree"] == 2
    assert result["local_edge_basis"]["basis_edge_count"] == int(
        report_paths["source_edge_count"]
    )
    assert result["local_edge_basis"]["basis_vector_count"] == 3
    assert result["fit"]["design_matrix"]["shape"] == [3 * 2, 2 * 3]
    assert result["coefficient_table"]["row_labels"] == [
        "BE relaxation",
        "BE cascade 1",
    ]
    assert result["coefficient_table"]["column_labels"] == [
        "source_moment:0",
        "source_moment:1",
        "source_moment:2",
    ]
    assert np.asarray(result["coefficient_table"]["values"]).shape == (2, 3)


def test_source_history_matrix_cli_supports_explicit_source_moment_degrees(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "source-even-moment-fit.json"

    exit_code = main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--local-basis-scope",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["local_edge_basis"]["source_moment_degrees"] == [0, 2]
    assert result["local_edge_basis"]["basis_vector_count"] == 2
    assert result["fit"]["design_matrix"]["shape"] == [3 * 2, 2 * 2]
    assert result["coefficient_table"]["column_labels"] == [
        "source_moment:0",
        "source_moment:2",
    ]


def test_source_history_matrix_cli_supports_hj_face_source_moments(tmp_path):
    result_path = tmp_path / "hj_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.array([0.2, -0.1, 0.05, 0.3])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "hj_ip.json"
    noip_report = tmp_path / "hj_noip.json"
    output = tmp_path / "hj_face_source_moments_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["field_location"] == "face"
    assert result["source_moment_basis"]["basis_vector_count"] == 2
    assert result["coefficient_table"]["column_labels"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]
    assert result["fit"]["design_matrix"]["shape"] == [3 * 3, coefficients.size]
    assert result["fit"]["design_matrix"]["rank_deficient"] is True
    assert result["fit"]["relative_l2"] < 1.0e-13

    static_response = np.einsum(
        "lcf,sf->slc",
        receiver_matrix,
        moments.basis_vectors,
    )
    static_design = static_response[:, np.arange(3), [2, 2, 2]].T
    static_report = result["static_response_matrix"]
    assert static_report["shape"] == [3, 2]
    assert static_report["rank"] == int(np.linalg.matrix_rank(static_design))
    assert static_report["rank_deficient"] == (static_report["rank"] < 2)
    np.testing.assert_allclose(
        static_report["column_norms"],
        np.linalg.norm(static_design, axis=0),
    )
    time_series = result["spatial_time_series"]
    assert time_series["coefficient_names"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]
    assert time_series["coefficient_matrix"]["shape"] == [3, 2]
    assert time_series["history_basis_fits"][0]["max_order"] == 1
    assert time_series["history_basis_fits"][0]["relative_l2"] < 1.0e-13
    assert time_series["history_basis_fits"][0]["coefficient_table"]["column_labels"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]


def test_source_history_matrix_cli_uses_matrix_free_face_basis_source_moments(
    tmp_path,
    monkeypatch,
):
    result_path = tmp_path / "hj_face_basis_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "face_basis_cell_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    static_response = np.asarray(
        [
            biot_savart_h_from_face_basis_currents(
                sim.mesh,
                vector,
                locations,
                subdivisions=2,
            )
            for vector in moments.basis_vectors
        ],
        dtype=float,
    )
    basis = source_history_receiver_basis_from_static_response(
        sim.time_steps,
        tau=0.2,
        static_response=static_response,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.array([0.2, -0.1, 0.05, 0.3])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "hj_face_basis_ip.json"
    noip_report = tmp_path / "hj_face_basis_noip.json"
    output = tmp_path / "hj_face_basis_source_moments_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    def fail_full_face_basis_matrix(*args, **kwargs):
        raise AssertionError("full face_basis_biot_matrix should not be built")

    monkeypatch.setattr(
        "atem3d.source_history_matrix_cli.face_basis_biot_matrix",
        fail_full_face_basis_matrix,
    )

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "face_basis",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--subdivisions",
            "2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["receiver_matrix"] == "face_basis"
    assert result["fit"]["relative_l2"] < 1.0e-13
    assert result["static_response_matrix"]["shape"] == [3, 2]
    assert result["spatial_time_series"]["history_basis_fits"][0]["relative_l2"] < 1.0e-13


def test_source_history_matrix_cli_loads_prescribed_file_for_spatial_trace(tmp_path):
    result_path = tmp_path / "hj_trace_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    trace_coefficients = np.array([-0.1, 0.3])
    coefficients = np.array([0.0, trace_coefficients[0], 0.0, trace_coefficients[1]])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"discrete_basis_fit": {"coefficients": trace_coefficients.tolist()}}),
        encoding="utf-8",
    )
    ip_report = tmp_path / "hj_trace_ip.json"
    noip_report = tmp_path / "hj_trace_noip.json"
    output = tmp_path / "hj_trace_prescribed.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--prescribed-coefficients-file",
            str(candidate),
            "--prescribed-coefficients-key",
            "discrete_basis_fit.coefficients",
            "--prescribed-spatial-trace-index",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["prescribed"]["relative_l2"] < 1.0e-13
    prescribed = result["spatial_time_series"]["prescribed_history_basis"]
    assert prescribed["coefficient_table"]["values"] == [
        [0.0, trace_coefficients[0]],
        [0.0, trace_coefficients[1]],
    ]


def test_source_history_matrix_cli_generates_initial_polarization_candidate(tmp_path):
    result_path = tmp_path / "hj_initial_pol_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    initial_e = hj_dc_initial_electric_field(sim.mesh, sim.ip_model, sim.sources)
    unit_face = sim.mesh.get_face_inner_product(np.ones(sim.mesh.n_cells)).tocsr()
    delta_face = sim.mesh.get_face_inner_product(sim.ip_model.terms[0].delta_sigma).tocsr()
    polarization = delta_face.diagonal() / unit_face.diagonal() * initial_e
    candidate_coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        moments.basis_vectors,
        -polarization,
        projection="receiver_l2",
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.r_[candidate_coefficients, np.zeros_like(candidate_coefficients)]
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "hj_initial_pol_ip.json"
    noip_report = tmp_path / "hj_initial_pol_noip.json"
    output = tmp_path / "hj_initial_pol_candidate.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--prescribed-candidate",
            "initial_polarization",
            "--prescribed-candidate-projection",
            "receiver_l2",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["prescribed_candidate"]["kind"] == "initial_polarization"
    assert result["prescribed_candidate"]["requires_ip"] is True
    assert result["prescribed"]["relative_l2"] < 1.0e-13
    table = result["prescribed"]["coefficient_table"]["values"]
    np.testing.assert_allclose(table[0], candidate_coefficients)
    np.testing.assert_allclose(table[1], np.zeros_like(candidate_coefficients))


def test_source_history_matrix_cli_generates_charge_conserving_initial_polarization_candidate(
    tmp_path,
):
    result_path = tmp_path / "hj_charge_conserving_initial_pol_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    initial_e = hj_dc_initial_electric_field(sim.mesh, sim.ip_model, sim.sources)
    unit_face = sim.mesh.get_face_inner_product(np.ones(sim.mesh.n_cells)).tocsr()
    delta_face = sim.mesh.get_face_inner_product(sim.ip_model.terms[0].delta_sigma).tocsr()
    polarization = delta_face.diagonal() / unit_face.diagonal() * initial_e
    charge_conserving_current = charge_conserving_face_current(
        sim.mesh,
        sim.ip_model.sigma_infinity,
        -polarization,
    )
    candidate_coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        moments.basis_vectors,
        charge_conserving_current,
        projection="receiver_l2",
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.r_[candidate_coefficients, np.zeros_like(candidate_coefficients)]
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "hj_charge_conserving_initial_pol_ip.json"
    noip_report = tmp_path / "hj_charge_conserving_initial_pol_noip.json"
    output = tmp_path / "hj_charge_conserving_initial_pol_candidate.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--prescribed-candidate",
            "charge_conserving_initial_polarization",
            "--prescribed-candidate-projection",
            "receiver_l2",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["prescribed_candidate"]["kind"] == (
        "charge_conserving_initial_polarization"
    )
    assert result["prescribed_candidate"]["requires_ip"] is True
    assert result["prescribed"]["relative_l2"] < 1.0e-13
    table = result["prescribed"]["coefficient_table"]["values"]
    np.testing.assert_allclose(table[0], candidate_coefficients)
    np.testing.assert_allclose(table[1], np.zeros_like(candidate_coefficients))


def test_source_history_matrix_cli_source_moments_honor_source_vector_choice(tmp_path):
    result_path = tmp_path / "hj_dc_total_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    dc_total = hj_dc_initial_current_density(
        sim.mesh,
        sim.ip_model,
        sim.sources,
    ) + _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        dc_total,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.array([0.4, -0.2, 0.1, 0.15])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        path.write_text(
            json.dumps({"result_path": str(result_path), "samples": samples}),
            encoding="utf-8",
        )

    ip_report = tmp_path / "hj_dc_total_ip.json"
    noip_report = tmp_path / "hj_dc_total_noip.json"
    output = tmp_path / "hj_dc_total_source_moments_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-vector",
            "dc_total_current",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_vector"] == "dc_total_current"
    assert result["source_moment_basis"]["source_vector"] == "dc_total_current"
    assert result["fit"]["relative_l2"] < 1.0e-13
    np.testing.assert_allclose(result["fit"]["coefficients"], coefficients)


def test_source_history_matrix_cli_evaluates_prescribed_coefficients(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "prescribed-fit.json"
    coefficients = [0.1, -0.2, 0.3, -0.4]

    exit_code = main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--local-basis-scope",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--prescribed-coefficients",
            ",".join(str(value) for value in coefficients),
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        result["prescribed"]["coefficients"],
        coefficients,
    )
    assert result["prescribed"]["coefficient_table"]["values"] == [
        [0.1, -0.2],
        [0.3, -0.4],
    ]
    assert result["prescribed"]["relative_l2"] >= 0.0


def test_source_history_matrix_cli_accepts_normalized_prescribed_coefficients(tmp_path):
    report_paths = _write_known_hj_face_source_moment_reports(tmp_path)
    output = tmp_path / "normalized-prescribed-fit.json"
    normalized = [6.0, 2.0, 3.0, 0.5]
    normalizer = mu_0 * 0.1 * 2.0**2
    expected = [value * normalizer for value in normalized]

    exit_code = main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "--prescribed-normalized-coefficients",
            ",".join(str(value) for value in normalized),
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    np.testing.assert_allclose(result["prescribed"]["coefficients"], expected)
    np.testing.assert_allclose(
        result["prescribed"]["coefficients_over_mu_delta_l2"],
        normalized,
    )
    assert result["prescribed"]["coefficient_table_over_mu_delta_l2"]["values"] == [
        [6.0, 2.0],
        [3.0, 0.5],
    ]


def test_format_coefficients_summarizes_long_vectors():
    values = np.arange(8.0)

    text = _format_coefficients(values, max_values=4)

    assert text == "0,1,...,6,7 (n=8)"


def test_coefficient_table_groups_flat_history_spatial_labels():
    table = _coefficient_table(
        np.array([1.0, 2.0, 3.0, 4.0]),
        [
            "g0 * m0",
            "g0 * m1",
            "g1 * m0",
            "g1 * m1",
        ],
    )

    assert table == {
        "row_labels": ["g0", "g1"],
        "column_labels": ["m0", "m1"],
        "values": [[1.0, 2.0], [3.0, 4.0]],
    }


def test_source_history_matrix_scan_cli_scans_orders_and_windows(tmp_path):
    report_paths = _write_known_source_history_reports(tmp_path)
    output = tmp_path / "scan.json"

    exit_code = scan_main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--source-vector",
            "wire",
            "--tau",
            "0.2",
            "--orders",
            "1,2",
            "--windows",
            "all,0.2:0.3",
            "--per-column",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["diagnostic_only"] is True
    assert [case["label"] for case in result["cases"]] == [
        "order_1_all",
        "order_1_0.2_0.3",
        "order_2_all",
        "order_2_0.2_0.3",
    ]
    assert [case["max_order"] for case in result["cases"]] == [1, 1, 2, 2]
    assert result["cases"][0]["time_window"]["selected_count"] == 3
    assert result["cases"][1]["time_window"]["selected_count"] == 2
    for case in result["cases"]:
        assert "design_matrix" in case["fit"]
        matrix = case["fit"]["design_matrix"]
        if matrix["condition_number"] is None:
            assert matrix["rank_deficient"] is True
        else:
            assert np.isfinite(matrix["condition_number"])
        if matrix["column_normalized_condition_number"] is None:
            assert matrix["column_normalized_rank_deficient"] is True
        else:
            assert np.isfinite(matrix["column_normalized_condition_number"])
        assert len(case["per_column_fit"]) == 2
        np.testing.assert_allclose(
            [item["receiver_location"] for item in case["per_column_fit"]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        )
        if case["max_order"] == 1:
            for column_fit in case["per_column_fit"]:
                np.testing.assert_allclose(
                    column_fit["fit"]["coefficients"],
                    report_paths["coefficients"],
                )
    assert result["cases"][3]["fit"]["design_matrix"]["rank_deficient"] is True
    np.testing.assert_allclose(
        result["cases"][0]["fit"]["coefficients"],
        report_paths["coefficients"],
    )
    np.testing.assert_allclose(
        result["cases"][2]["fit"]["coefficients"][:2],
        report_paths["coefficients"],
    )


def test_source_history_matrix_scan_cli_supports_local_edge_spatial_basis(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "local-scan.json"

    exit_code = scan_main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--source-cell-radius",
            "0",
            "--receiver-cell-radius",
            "0",
            "--tau",
            "0.2",
            "--orders",
            "1",
            "--windows",
            "all,0.2:0.3",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["spatial_basis"] == "local_edges"
    assert result["local_edge_basis"]["support_edge_count"] == int(
        report_paths["support_edge_count"]
    )
    assert len(result["cases"]) == 2
    assert result["cases"][0]["fit"]["relative_l2"] < 1.0e-13
    assert result["cases"][0]["fit"]["design_matrix"]["shape"] == [
        3 * 2,
        2 * report_paths["support_edge_count"],
    ]
    assert "coefficients_over_mu_delta_l2" not in result["cases"][0]["fit"]
    assert result["cases"][1]["time_window"]["selected_count"] == 2


def test_source_history_matrix_scan_cli_evaluates_prescribed_coefficients(tmp_path):
    report_paths = _write_known_local_edge_source_history_reports(tmp_path)
    output = tmp_path / "local-prescribed-scan.json"
    coefficients = [0.1, -0.2, 0.3, -0.4]

    exit_code = scan_main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "local_edges",
            "--local-basis-scope",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--orders",
            "1",
            "--windows",
            "all,0.2:0.3",
            "--prescribed-coefficients",
            ",".join(str(value) for value in coefficients),
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["cases"]) == 2
    for case in result["cases"]:
        np.testing.assert_allclose(case["prescribed"]["coefficients"], coefficients)
        assert case["prescribed"]["coefficient_table"]["values"] == [
            [0.1, -0.2],
            [0.3, -0.4],
        ]
        assert case["prescribed"]["relative_l2"] >= 0.0


def test_source_history_matrix_scan_cli_accepts_normalized_prescribed_coefficients(
    tmp_path,
):
    report_paths = _write_known_hj_face_source_moment_reports(tmp_path)
    output = tmp_path / "normalized-prescribed-scan.json"
    normalized = [6.0, 2.0, 3.0, 0.5]
    normalizer = mu_0 * 0.1 * 2.0**2
    expected = [value * normalizer for value in normalized]

    exit_code = scan_main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--orders",
            "1",
            "--windows",
            "all,0.2:0.3",
            "--prescribed-normalized-coefficients",
            ",".join(str(value) for value in normalized),
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    for case in result["cases"]:
        np.testing.assert_allclose(case["prescribed"]["coefficients"], expected)
        np.testing.assert_allclose(
            case["prescribed"]["coefficients_over_mu_delta_l2"],
            normalized,
        )
        assert case["prescribed"]["coefficient_table_over_mu_delta_l2"]["values"] == [
            [6.0, 2.0],
            [3.0, 0.5],
        ]


def test_source_history_matrix_scan_cli_supports_hj_face_source_moments(tmp_path):
    report_paths = _write_known_hj_face_source_moment_reports(tmp_path)
    output = tmp_path / "hj-face-source-moment-scan.json"

    exit_code = scan_main(
        [
            str(report_paths["ip_report"]),
            str(report_paths["noip_report"]),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--field-location",
            "face",
            "--receiver-matrix",
            "current_biot",
            "--spatial-basis",
            "source_moments",
            "--source-moment-degrees",
            "0,2",
            "--tau",
            "0.2",
            "--orders",
            "1",
            "--windows",
            "all,0.2:0.3",
            "--prescribed-coefficients",
            "0.2,-0.1,0.05,0.3",
            "-o",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["field_location"] == "face"
    assert result["source_moment_basis"]["basis_vector_count"] == 2
    assert len(result["cases"]) == 2
    assert result["cases"][0]["fit"]["relative_l2"] < 1.0e-13
    assert result["cases"][0]["coefficient_table"]["column_labels"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]
    assert result["static_response_matrix"]["shape"] == [3, 2]
    assert "rank_deficient" in result["static_response_matrix"]
    time_series = result["spatial_time_series"]
    assert time_series["coefficient_names"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]
    assert time_series["coefficient_matrix"]["shape"] == [3, 2]
    assert len(time_series["samples"]) == 3
    assert time_series["projection_fit"]["design_matrix"]["shape"] == [3, 2]
    assert time_series["history_basis_fits"][0]["max_order"] == 1
    assert time_series["history_basis_fits"][0]["relative_l2"] < 1.0e-13
    assert time_series["history_basis_fits"][0]["coefficient_table"]["column_labels"] == [
        "source_face_moment:0",
        "source_face_moment:2",
    ]
    prescribed_trace = time_series["prescribed_history_basis"]
    assert prescribed_trace["max_order"] == 1
    assert prescribed_trace["relative_l2"] > 0.0
    assert prescribed_trace["coefficient_table"]["values"] == [
        [0.2, -0.1],
        [0.05, 0.3],
    ]


def _assert_source_history_matrix_cli_recovers_known_coefficients(
    tmp_path,
    *,
    receiver_matrix_name,
    receiver_matrix_builder,
    source_vector_name,
    source_vector_builder,
    source_vector_builders_by_order=None,
    extra_args=None,
    expected_selected_times=None,
):
    result_path = tmp_path / "result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Ex", "Hz"],
        },
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    source_vector = source_vector_builder(sim)
    source_vectors_by_order = None
    if source_vector_builders_by_order is not None:
        source_vectors_by_order = np.vstack(
            [builder(sim) for builder in source_vector_builders_by_order]
        )
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = receiver_matrix_builder(sim.mesh, locations)
    if source_vectors_by_order is None:
        basis = source_history_receiver_basis(
            sim.time_steps,
            tau=0.2,
            source_vector=source_vector,
            receiver_matrix=receiver_matrix,
            max_order=1,
        )
    else:
        basis = source_history_receiver_basis_from_vectors(
            sim.time_steps,
            tau=0.2,
            source_vectors=source_vectors_by_order,
            receiver_matrix=receiver_matrix,
        )
    coefficients = np.array([-2.0, 0.5])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "ip.json"
    noip_report = tmp_path / "noip.json"
    output = tmp_path / "matrix_fit.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))

    exit_code = main(
        [
            str(ip_report),
            str(noip_report),
            "--target",
            "reference_delta",
            "--component-prefix",
            "Hz",
            "--receiver-matrix",
            receiver_matrix_name,
            "--source-vector",
            source_vector_name,
            "--tau",
            "0.2",
            "--max-order",
            "1",
            "-o",
            str(output),
        ]
        + list(extra_args or [])
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    np.testing.assert_allclose(result["fit"]["coefficients"], coefficients)
    np.testing.assert_allclose(
        result["fit"]["coefficients_over_mu_delta_l2"],
        (coefficients / (mu_0 * 0.1 * 1.0)).tolist(),
    )
    assert result["fit"]["relative_l2"] < 1.0e-14
    selected_times = expected_selected_times if expected_selected_times is not None else 3
    assert result["fit"]["design_matrix"]["shape"] == [selected_times * 2, 2]
    assert all(value > 0.0 for value in result["fit"]["design_matrix"]["column_norms"])
    assert np.isfinite(result["fit"]["design_matrix"]["condition_number"])
    assert np.isfinite(
        result["fit"]["design_matrix"]["column_normalized_condition_number"]
    )
    if expected_selected_times is not None:
        assert result["time_window"]["selected_count"] == expected_selected_times


def _wire_source_vector(sim):
    return sum(
        (source.initial_edge_vector(sim.mesh) for source in sim.sources),
        np.zeros(sim.mesh.n_edges, dtype=float),
    )


def _hj_face_wire_source_vector(sim):
    return sum(
        (-source.initial_face_vector(sim.mesh) for source in sim.sources),
        np.zeros(sim.mesh.n_faces, dtype=float),
    )


def _dc_total_source_vector(sim):
    source_vector = _wire_source_vector(sim)
    e0 = sim.initial_electric_field()
    sigma0 = sim.mesh.get_edge_inner_product(sim.ip_model.low_frequency_sigma()).tocsr()
    return np.asarray(sigma0 @ e0 + source_vector, dtype=float)


def _write_known_source_history_reports(tmp_path):
    result_path = tmp_path / "scan_result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = cell_current_biot_matrix(sim.mesh, locations)
    basis = source_history_receiver_basis(
        sim.time_steps,
        tau=0.2,
        source_vector=_wire_source_vector(sim),
        receiver_matrix=receiver_matrix,
        max_order=1,
    )
    coefficients = np.array([-2.0, 0.5])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "scan_ip.json"
    noip_report = tmp_path / "scan_noip.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))
    return {
        "ip_report": ip_report,
        "noip_report": noip_report,
        "coefficients": coefficients,
    }


def _write_known_local_edge_source_history_reports(tmp_path):
    result_path = tmp_path / "local_scan_result.h5"
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = cell_current_biot_matrix(sim.mesh, locations)
    local_basis = local_edge_basis(sim.mesh, _wire_source_vector(sim), locations)
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=local_basis.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=local_basis.basis_labels,
    )
    coefficients = np.linspace(-0.2, 0.3, basis.responses.shape[1])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "local_scan_ip.json"
    noip_report = tmp_path / "local_scan_noip.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))
    return {
        "ip_report": ip_report,
        "noip_report": noip_report,
        "support_edge_count": int(local_basis.support_edge_indices.size),
        "source_edge_count": int(local_basis.source_edge_indices.size),
    }


def _write_known_hj_face_source_moment_reports(tmp_path):
    result_path = tmp_path / "hj_face_source_moment_result.h5"
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
            "debye_terms": [{"delta_sigma": 0.1, "tau": 0.2}],
        },
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 1.0,
            "z": 0.0,
            "components": ["Hz"],
        },
        "time_steps": [[0.1, 3]],
        "magnetic_receiver_mode": "current_biot",
    }
    with h5py.File(result_path, "w") as h5:
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)

    sim = build_simulation(config)
    hz_receivers = [receiver for receiver in sim.receivers if receiver.component == "Hz"]
    locations = np.array([receiver.location for receiver in hz_receivers])
    receiver_matrix = face_current_biot_matrix(sim.mesh, locations)
    face_source = _hj_face_wire_source_vector(sim)
    moments = source_face_moment_basis(
        sim.mesh,
        face_source,
        start=sim.sources[0].locations[0],
        end=sim.sources[0].locations[-1],
        degrees=[0, 2],
    )
    basis = source_history_receiver_basis_from_spatial_vectors(
        sim.time_steps,
        tau=0.2,
        spatial_vectors=moments.basis_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=moments.basis_labels,
    )
    coefficients = np.array([0.2, -0.1, 0.05, 0.3])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)
    times = basis.times[1:]

    def write_report(path, reference):
        samples = {}
        for receiver_index, x in enumerate(config["receiver_line"]["x"]):
            samples[f"Hz@x={float(x):g}"] = [
                {
                    "time": float(time),
                    "numerical": 0.0,
                    "reference": float(value),
                    "difference": float(-value),
                    "ratio_numerical_over_reference": 0.0,
                }
                for time, value in zip(times, reference[:, receiver_index, 2])
            ]
        payload = {"result_path": str(result_path), "samples": samples}
        path.write_text(json.dumps(payload), encoding="utf-8")

    ip_report = tmp_path / "hj_face_source_moment_ip.json"
    noip_report = tmp_path / "hj_face_source_moment_noip.json"
    write_report(ip_report, target[1:])
    write_report(noip_report, np.zeros_like(target[1:]))
    return {
        "ip_report": ip_report,
        "noip_report": noip_report,
    }
