from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.guards import IndependentTestLeakageError, assert_split_readable
from atem3d.adaptive_debye_mvp.independent_test import (
    SCHEMA,
    STAGE,
    case_is_complete,
    independent_test_cases,
    load_independent_case_result,
    project_onto_normal,
    projected_channel_column,
    tilted_tasks_from_cache,
    write_independent_case_result,
)
from atem3d.adaptive_debye_mvp.oracle_gap import CaseKChoice, FitRecord, TaskMetrics, load_pilot_case_result
from atem3d.adaptive_debye_mvp.protocol_constants import TEST_RECEIVERS, TEST_WAVEFORMS, normalize_tilted_normal
from atem3d.adaptive_debye_mvp.receiver_metrics import ReceiverCase, evaluate_case
from atem3d.adaptive_debye_mvp.registry import generate_all_cases


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "paper_receiver_adaptive_debye_mvp"
    / "scripts"
    / "precompute_independent_test.py"
)


def _synthetic_response(*, field, n_times=8, n_rx=2):
    data = np.zeros((n_times, n_rx, 6), dtype=float)
    data[..., 0] = field[0]
    data[..., 1] = field[1]
    data[..., 2] = field[2]
    data[..., 3] = field[0]
    data[..., 4] = field[1]
    data[..., 5] = field[2]
    return {
        "data": data,
        "channels": ["Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt"],
        "receiver_labels": ["point:0", "point:1"],
        "times": np.logspace(-5, -2, n_times),
        "hashes": {"shared_survey_hash": "abc123"},
    }


def test_project_onto_normal_aligned_and_orthogonal():
    normal = (0.0, 0.0, 1.0)
    aligned = project_onto_normal(_synthetic_response(field=(0.0, 0.0, 4.0)), normal)
    assert aligned["channels"] == ["Hn", "dBndt"]
    assert aligned["hashes"]["shared_survey_hash"] == "abc123"
    assert np.allclose(aligned["data"][..., 0], 4.0)
    assert np.allclose(aligned["data"][..., 1], 4.0)
    orthogonal = project_onto_normal(_synthetic_response(field=(1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
    assert np.allclose(orthogonal["data"], 0.0)
    mixed = project_onto_normal(_synthetic_response(field=(3.0, 4.0, 0.0)), (3.0, 4.0, 0.0))
    assert np.allclose(mixed["data"][..., 0], 5.0)
    column = projected_channel_column(aligned, 1)
    assert set(column) == {"Hn", "dBndt"}
    assert column["Hn"].shape == (8,)
    unit = np.asarray(normalize_tilted_normal())
    assert abs(float(np.linalg.norm(unit)) - 1.0) < 1.0e-12


def test_evaluate_case_accepts_projected_channels():
    times = np.linspace(1.0e-4, 1.0e-3, 10)
    reference = {"Hn": np.linspace(1.0, 0.1, 10), "dBndt": np.linspace(2.0, 0.2, 10)}
    candidate = {name: value * 1.001 for name, value in reference.items()}
    baseline = {name: value * 0.5 for name, value in reference.items()}
    metrics = evaluate_case(
        ReceiverCase(
            case_id="tilted",
            times=times,
            reference=reference,
            candidate=candidate,
            reference_no_ip=baseline,
        )
    )
    assert set(metrics.channels) == {"Hn", "dBndt"}
    assert np.isfinite(metrics.case_total_p95)


def test_write_independent_case_roundtrip(tmp_path):
    case = [item for item in generate_all_cases() if item.case_id == "TE01"][0]
    choice = CaseKChoice(
        case_id="TE01",
        K=4,
        spectral_id="K04_cc_span4.0_shift+0.0_dens1.00",
        oracle_id="K04_cc_span4.0_shift+0.0_dens1.25",
        ids_differ=True,
        e_b2=0.01,
        e_or=0.008,
        ratio=0.8,
        gap=0.2,
        ip_b2=0.02,
        ip_or=0.015,
        log_hausdorff=0.1,
        qualifies_b2=True,
        qualifies_or=True,
    )
    task = TaskMetrics(
        case_id="TE01",
        candidate_id="K04_cc_span4.0_shift+0.0_dens1.25",
        K=4,
        waveform_id="W3",
        receiver_id="tilted_coil",
        receiver_index=0,
        total_p95=0.002,
        ip_increment_nrmse=0.003,
        passed=True,
        unexplained_sign_flips=0,
        peak_time_error_steps_max=0.0,
        h_p95=0.001,
        dbdt_p95=0.002,
    )
    path = tmp_path / "case_TE01.json"
    write_independent_case_result(
        path,
        {
            "official_variant": "S1",
            "choices": [choice],
            "tasks": [task],
            "fit_summary": [{"K": 4, "candidate_id": task.candidate_id, "valid": True}],
        },
        case=case,
        k_values=(4, 6, 8, 10, 12),
        waveform_ids=TEST_WAVEFORMS,
        disk_shortlist={"4": [task.candidate_id]},
        provenance={
            "three_d_run": False,
            "l2_evaluated": False,
            "transform_label": "smoke_fast_lagged_dlf",
            "ft_pts_per_dec": -1,
            "shared_survey_hash": {"point": "p", "disk": "d"},
        },
        tilted_projection={"normal": list(normalize_tilted_normal()), "derived_from": "point", "disk_projection": False},
        registry={"case_registry_sha256": "x"},
    )
    payload = path.read_text(encoding="utf-8")
    assert SCHEMA in payload
    assert "tilted_coil" in payload
    loaded = load_independent_case_result(path)
    assert loaded["schema"] == SCHEMA
    assert loaded["case_id"] == "TE01"
    assert any(item.receiver_id == "tilted_coil" for item in loaded["tasks"])
    pilot = load_pilot_case_result(path)
    assert pilot["case_id"] == "TE01"
    assert case_is_complete(path)
    assert not list(tmp_path.glob("selected_template*"))


def test_tilted_tasks_from_cache_raises_on_missing_cache(tmp_path):
    case = [item for item in generate_all_cases() if item.case_id == "TE01"][0]
    record = FitRecord(
        candidate_id="K04_cc_span4.0_shift+0.0_dens1.00",
        K=4,
        valid=True,
        spectral_error_s0=0.1,
        spectral_error_s1=0.1,
        condition_number=1.0,
        relative_dc_error=0.0,
        optimizer_success=True,
        fit=None,
    )
    with pytest.raises(FileNotFoundError, match="missing point cache"):
        tilted_tasks_from_cache(case, [record], waveform_ids=("W0",), cache_dir=tmp_path)


def test_precompute_stage_can_read_independent_test():
    assert_split_readable("independent_test", stage=STAGE)
    with pytest.raises(IndependentTestLeakageError):
        assert_split_readable("independent_test", stage="selector")
    cases = independent_test_cases()
    assert [case.case_id for case in cases] == [f"TE{index:02d}" for index in range(1, 11)]
    assert all(case.sensor_frame == "tilted" for case in cases)
    assert all("W3" in case.waveform_ids for case in cases)


def test_runner_source_has_no_selector_or_l2_or_3d():
    source = RUNNER.read_text(encoding="utf-8")
    for needle in (
        "evaluate_l0",
        "evaluate_l2",
        "select_templates",
        "train_selector",
        "dolfinx",
    ):
        assert needle not in source
    assert "FLOW4_STATUS.json" not in source
    assert "LAYERED_GATE_PASSED.json" not in source
    assert "FORBIDDEN_OUTPUT_NAMES" in source
    assert STAGE not in {"train", "validation", "selector", "flow3"}
    assert "refuse_3d_always" in source


def test_runner_rejects_pilot_case_ids(tmp_path, monkeypatch):
    import runpy

    monkeypatch.setattr(
        "sys.argv",
        [
            str(RUNNER),
            "--case-ids",
            "PG01",
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(RUNNER), run_name="__main__")
    assert exc.value.code not in {0, None}


def test_receiver_and_waveform_constants():
    assert TEST_RECEIVERS == ("point", "disk_1.0", "disk_4.0", "tilted_coil")
    assert TEST_WAVEFORMS == ("W0", "W1", "W2", "W3")
    assert inspect.signature(write_independent_case_result)
