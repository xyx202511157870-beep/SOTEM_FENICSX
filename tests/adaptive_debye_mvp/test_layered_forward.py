from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atem3d.adaptive_debye_mvp import layered_forward as lf
from atem3d.adaptive_debye_mvp.layered_forward import (
    APPROVED_PRODUCTION_TRANSFORM,
    CHANNELS,
    SMOKE_FAST_TRANSFORM,
    BlockedBySoftwareOrResourcesError,
    TimeGrid,
    W0_IDEAL_STEP_OFF,
    W1_LINEAR_RAMP_5US,
    W2_LINEAR_RAMP_20US,
    build_empymod_resistivity,
    compute_hashes,
    compute_layered_response,
    default_smoke_case,
    disk_average_channel_pair,
    resolve_waveform,
    w3_tabulated,
)
from atem3d.adaptive_debye_mvp.reference_audit import (
    annotate_reference_type,
    run_reference_audit,
)
from atem3d.empymod_magnetic6 import MAGNETIC6_UNITS


def _zero_data(n_times: int = 31) -> np.ndarray:
    return np.zeros((n_times, 1, 6), dtype="<f8")


@pytest.fixture(scope="module")
def case():
    return default_smoke_case()


@pytest.fixture(scope="module")
def point_rx(case):
    return (case["receivers"][0],)


@pytest.fixture(scope="module")
def resp_exact(case, point_rx):
    return compute_layered_response(
        case["materials"]["exact_m0p2"],
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def resp_m0(case, point_rx):
    return compute_layered_response(
        case["materials"]["exact_m0"],
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def resp_plain(case, point_rx):
    return compute_layered_response(
        case["materials"]["non_polarizable"],
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def resp_debye(case, point_rx):
    return compute_layered_response(
        case["materials"]["debye_dummy"],
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def resp_w1(case, point_rx):
    return compute_layered_response(
        case["materials"]["exact_m0p2"],
        case["geometry"],
        case["waveforms"]["W1"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def resp_w2(case, point_rx):
    return compute_layered_response(
        case["materials"]["exact_m0p2"],
        case["geometry"],
        case["waveforms"]["W2"],
        point_rx,
        case["times"],
        case["transform"],
    )


@pytest.fixture(scope="module")
def disk_pairs(case):
    resistivity = build_empymod_resistivity(case["materials"]["exact_m0p2"])
    backend = lf._require_empymod()
    kwargs = case["transform"].empymod_kwargs(case["geometry"].n_layers)
    times = case["times"].as_array()
    center = case["receivers"][0].location
    pairs = {}
    for radius in (0.25, 1.0):
        data, record = disk_average_channel_pair(
            "z",
            center,
            radius,
            geometry=case["geometry"],
            times=times,
            resistivity=resistivity,
            waveform=None,
            waveform_spec=case["waveforms"]["W0"],
            transform_settings=case["transform"],
            call_kwargs=kwargs,
            backend=backend,
        )
        pairs[radius] = (data, record)
    return pairs


@pytest.fixture(scope="module")
def audit_report(case, point_rx, resp_exact):
    return run_reference_audit(
        case["materials"]["exact_m0p2"],
        case["geometry"],
        case["waveforms"]["W0"],
        case["receivers"][:2],
        case["times"],
        case["transform"],
        baseline=resp_exact,
        disk_receiver_index=1,
        disk_axes=("z",),
    )


def test_empymod_missing_raises_blocked(monkeypatch, case, point_rx):
    def _boom():
        raise ImportError("empymod missing")

    monkeypatch.setattr(lf, "_import_empymod", _boom)
    with pytest.raises(BlockedBySoftwareOrResourcesError, match="BLOCKED_BY_SOFTWARE_OR_RESOURCES"):
        compute_layered_response(
            case["materials"]["exact_m0p2"],
            case["geometry"],
            case["waveforms"]["W0"],
            point_rx,
            case["times"],
            case["transform"],
        )


def test_approved_identity_matches_production_literal():
    assert APPROVED_PRODUCTION_TRANSFORM.identity_dict() == {
        "equation": "quasistatic",
        "electric_permittivity": {"horizontal": 0.0, "vertical": 0.0},
        "magnetic_permeability": {"horizontal": 1.0, "vertical": 1.0},
        "hankel_transform": {
            "method": "dlf",
            "parameters": {"filter": "key_201_2009", "pts_per_dec": 0},
        },
        "fourier_transform": {
            "method": "dlf",
            "parameters": {"filter": "key_201_2012", "pts_per_dec": 0},
        },
    }
    assert APPROVED_PRODUCTION_TRANSFORM.srcpts == 9
    assert APPROVED_PRODUCTION_TRANSFORM.is_approved_production_identity() is True
    assert SMOKE_FAST_TRANSFORM.is_approved_production_identity() is False
    assert SMOKE_FAST_TRANSFORM.ft_pts_per_dec == -1


def test_hash_contract(case, point_rx):
    material_a = case["materials"]["exact_m0p2"]
    material_b = case["materials"]["debye_dummy"]
    data = _zero_data()
    first = compute_hashes(
        material_a,
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
        data,
        empymod_version="2.6.0",
    )
    second = compute_hashes(
        material_a,
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
        data,
        empymod_version="2.6.0",
    )
    assert first == second
    swapped = compute_hashes(
        material_b,
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        case["transform"],
        data,
        empymod_version="2.6.0",
    )
    assert swapped["shared_survey_hash"] == first["shared_survey_hash"]
    assert swapped["material_hash"] != first["material_hash"]
    assert swapped["config_hash"] != first["config_hash"]
    srcpts = compute_hashes(
        material_a,
        case["geometry"],
        case["waveforms"]["W0"],
        point_rx,
        case["times"],
        replace(case["transform"], srcpts=17),
        data,
        empymod_version="2.6.0",
    )
    assert srcpts["geometry_hash"] == first["geometry_hash"]
    assert srcpts["waveform_hash"] == first["waveform_hash"]
    assert srcpts["times_hash"] == first["times_hash"]
    assert srcpts["receiver_hash"] == first["receiver_hash"]
    assert srcpts["transform_hash"] != first["transform_hash"]
    waveform = compute_hashes(
        material_a,
        case["geometry"],
        case["waveforms"]["W1"],
        point_rx,
        case["times"],
        case["transform"],
        data,
        empymod_version="2.6.0",
    )
    assert waveform["geometry_hash"] == first["geometry_hash"]
    assert waveform["times_hash"] == first["times_hash"]
    assert waveform["waveform_hash"] != first["waveform_hash"]


def test_flatten_frequency_context_accepts_2d_dlf_grid(case):
    resistivity = build_empymod_resistivity(case["materials"]["exact_m0p2"])
    assert isinstance(resistivity, dict)
    frequencies = np.linspace(1.0, 4.0, 12).reshape(3, 4)
    eta_h, eta_v = resistivity["func_eta"]({}, {"freq": frequencies})
    assert eta_h.shape == (12, 3)
    assert eta_v.shape == (12, 3)


def test_six_channels_finite_and_nontrivial_off_axis(resp_exact):
    data = np.asarray(resp_exact["data"], dtype=float)
    assert data.shape == (31, 1, 6)
    assert np.all(np.isfinite(data))
    assert resp_exact["channels"] == list(CHANNELS)
    assert resp_exact["units"] == list(MAGNETIC6_UNITS)
    peaks = np.max(np.abs(data[:, 0, :]), axis=0)
    h_max = float(np.max(peaks[:3]))
    dbdt_max = float(np.max(peaks[3:]))
    assert np.all(peaks[:3] >= 1.0e-3 * h_max)
    assert np.all(peaks[3:] >= 1.0e-3 * dbdt_max)
    audit = next(iter(resp_exact["dbdt_route_audit"].values()))
    assert audit["passed"] is True
    assert resp_exact["provenance"]["primary_dbdt_reference"] in {"native_b", "impulse_h"}
    assert resp_exact["provenance"]["source"]["finite_source_quadrature_points"] == 9
    assert resp_exact["time_origin"] == "complete_current_shutoff"


def test_m_zero_ip_increment_is_numerical_zero(resp_m0, resp_plain, resp_exact):
    delta_m0 = np.abs(resp_m0["data"] - resp_plain["data"])
    peak = np.maximum(np.max(np.abs(resp_plain["data"]), axis=(0, 1)), np.finfo(float).tiny)
    assert np.all(np.max(delta_m0, axis=(0, 1)) <= 1.0e-12 * peak)
    ip = np.max(np.abs(resp_exact["data"] - resp_m0["data"]), axis=(0, 1))
    exact_peak = np.maximum(np.max(np.abs(resp_exact["data"]), axis=(0, 1)), np.finfo(float).tiny)
    assert np.any(ip > 1.0e-2 * exact_peak)


def test_exact_vs_debye_candidate_share_survey_hashes(resp_exact, resp_debye):
    shared = (
        "geometry_hash",
        "waveform_hash",
        "times_hash",
        "transform_hash",
        "receiver_hash",
        "shared_survey_hash",
    )
    for key in shared:
        assert resp_exact["hashes"][key] == resp_debye["hashes"][key]
    assert resp_exact["hashes"]["material_hash"] != resp_debye["hashes"]["material_hash"]
    assert resp_exact["hashes"]["config_hash"] != resp_debye["hashes"]["config_hash"]
    assert resp_exact["hashes"]["output_hash"] != resp_debye["hashes"]["output_hash"]
    assert np.all(np.isfinite(resp_debye["data"]))
    assert not np.array_equal(resp_exact["data"], resp_debye["data"])


def test_waveforms_share_time_origin(resp_exact, resp_w1, resp_w2):
    for response in (resp_exact, resp_w1, resp_w2):
        assert response["time_origin"] == "complete_current_shutoff"
        assert response["hashes"]["times_hash"] == resp_exact["hashes"]["times_hash"]
        np.testing.assert_allclose(response["times"], resp_exact["times"])
    assert resp_w1["hashes"]["waveform_hash"] != resp_exact["hashes"]["waveform_hash"]
    assert resp_w2["hashes"]["waveform_hash"] != resp_w1["hashes"]["waveform_hash"]
    for response in (resp_w1, resp_w2):
        assert response["provenance"]["waveform"]["time_origin"] == "ramp_end"
        assert abs(response["provenance"]["convolution"]["weight_sum_error"]) <= 1.0e-12
    late = -1
    ratio_w1 = resp_w1["data"][late] / np.where(
        resp_exact["data"][late] == 0.0, np.nan, resp_exact["data"][late]
    )
    assert np.all(np.abs(ratio_w1 - 1.0) < 1.0e-2)
    early_w0 = np.abs(resp_exact["data"][0, 0, 3:])
    early_w1 = np.abs(resp_w1["data"][0, 0, 3:])
    early_w2 = np.abs(resp_w2["data"][0, 0, 3:])
    assert np.all(early_w2 < early_w1)
    assert np.all(early_w1 < early_w0)


def test_tabulated_waveform_loader_stub(tmp_path):
    path = tmp_path / "w3.csv"
    path.write_text("time_s,current_scale\n0.0,1.0\n1.0e-5,0.5\n2.0e-5,0.0\n", encoding="utf-8")
    waveform = resolve_waveform(w3_tabulated(path))
    assert waveform is not None
    assert float(waveform.times[-1]) == 0.0
    assert waveform.duration == pytest.approx(2.0e-5)
    assert waveform.times.size - 1 == 2
    with pytest.raises(FileNotFoundError):
        resolve_waveform(w3_tabulated(tmp_path / "missing.csv"))


def test_disk_average_converges_to_point_as_radius_shrinks(resp_exact, disk_pairs):
    point_z = resp_exact["data"][:, 0, [CHANNELS.index("Hz"), CHANNELS.index("dBzdt")]]
    small, small_record = disk_pairs[0.25]
    large, large_record = disk_pairs[1.0]
    assert small_record["n_points"] == 36
    assert large_record["n_points"] == 36
    assert small_record["weight_sum"] == pytest.approx(1.0)
    peak = np.maximum(np.max(np.abs(point_z), axis=0), np.finfo(float).tiny)
    err_small = np.max(np.abs(small - point_z), axis=0) / peak
    err_large = np.max(np.abs(large - point_z), axis=0) / peak
    assert np.all(err_large <= 1.0e-3)
    assert np.all(err_small < err_large)


def test_reference_audit_cheap_checks_pass_and_mark_converged(audit_report, resp_exact, case):
    for name in (
        "frequency_range_expansion",
        "frequency_sampling_density",
        "fourier_method_pair",
        "source_quadrature_9_vs_17",
        "hankel_filter_pair",
        "disk_quadrature_order",
        "six_channel_signs_and_near_zero",
    ):
        item = audit_report["checks"][name]
        assert item["performed"] is True
        assert item["passed"] is True
    assert audit_report["near_zero_channels"] == []
    assert audit_report["approved_identity_tied"] is True
    annotated = annotate_reference_type(resp_exact, audit_report)
    assert annotated["reference_type"] == "empymod_converged_cole_cole"
    failed = dict(audit_report)
    failed_checks = dict(audit_report["checks"])
    failed_checks["source_quadrature_9_vs_17"] = {
        **failed_checks["source_quadrature_9_vs_17"],
        "passed": False,
    }
    failed["checks"] = failed_checks
    failed["reference_type"] = "empymod_audit_failed"
    assert annotate_reference_type(resp_exact, failed)["reference_type"] == "empymod_audit_failed"
    debye = dict(audit_report)
    debye["reference_type"] = "empymod_audited_debye_candidate"
    assert (
        annotate_reference_type(resp_debye_stub(case), debye)["reference_type"]
        == "empymod_audited_debye_candidate"
    )


def resp_debye_stub(case):
    return {
        "schema": lf.SCHEMA,
        "reference_type": "empymod_unaudited",
        "hashes": {},
    }


def test_layered_forward_does_not_import_fem_or_dolfinx():
    root = Path(lf.__file__).resolve().parent
    forbidden = ("source_history_operator", "dolfinx", "sotem_pipeline")
    for name in ("layered_forward.py", "reference_audit.py"):
        imports = [
            line
            for line in (root / name).read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith(("import ", "from "))
        ]
        joined = "\n".join(imports)
        for token in forbidden:
            assert token not in joined


def test_time_grid_default_window():
    grid = TimeGrid.default()
    assert len(grid.times_s) == 31
    assert grid.times_s[0] == pytest.approx(1.0e-5)
    assert grid.times_s[-1] == pytest.approx(1.0e-2)
    assert grid.origin == "complete_current_shutoff"
    assert W0_IDEAL_STEP_OFF.kind == "ideal_step_off"
    assert W1_LINEAR_RAMP_5US.ramp_duration_s == pytest.approx(5.0e-6)
    assert W2_LINEAR_RAMP_20US.ramp_duration_s == pytest.approx(20.0e-6)
