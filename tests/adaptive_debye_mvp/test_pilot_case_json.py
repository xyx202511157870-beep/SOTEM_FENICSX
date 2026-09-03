from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from atem3d.adaptive_debye_mvp.io import write_json
from atem3d.adaptive_debye_mvp.oracle_gap import (
    CaseKChoice,
    TaskMetrics,
    evaluate_l0,
    load_pilot_case_result,
    plan_disk_forward_units,
    plan_point_forward_units,
    write_pilot_case_result,
)
from atem3d.adaptive_debye_mvp.protocol_constants import K_PILOT, PILOT_WAVEFORMS
from atem3d.adaptive_debye_mvp.registry import generate_split


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_oracle_gap.py"


def _choice(case_id: str, K: int, ratio: float = 0.7) -> CaseKChoice:
    e_b2 = 0.02
    return CaseKChoice(
        case_id=case_id,
        K=K,
        spectral_id="K08_cc_span4.0_shift+0.0_dens1.00",
        oracle_id="K08_tw_span6.0_shift-0.5_dens1.25",
        ids_differ=True,
        e_b2=e_b2,
        e_or=ratio * e_b2,
        ratio=ratio,
        gap=1.0 - ratio,
        ip_b2=0.02,
        ip_or=0.015,
        log_hausdorff=0.1,
        qualifies_b2=False,
        qualifies_or=True,
    )


def _task(case_id: str, candidate_id: str, K: int, receiver_id: str, waveform_id: str) -> TaskMetrics:
    return TaskMetrics(
        case_id=case_id,
        candidate_id=candidate_id,
        K=K,
        waveform_id=waveform_id,
        receiver_id=receiver_id,
        receiver_index=0,
        total_p95=0.01,
        ip_increment_nrmse=0.01,
        passed=True,
        unexplained_sign_flips=0,
        peak_time_error_steps_max=0.0,
        h_p95=0.01,
        dbdt_p95=0.01,
    )


def test_plan_point_units_match_evaluate_cache_keys():
    case = generate_split("pilot_gap")[0]
    units = plan_point_forward_units(case, k_values=K_PILOT)
    keys = [unit["cache_key"] for unit in units]
    assert f"{case.case_id}:exact:point:W0" in keys
    assert f"{case.case_id}:noip:point:W2" in keys
    assert any(key.endswith(":point:W1") and "K04_" in key for key in keys)
    assert len(keys) == len(set(keys))
    assert all(unit["receiver_kind"] == "point" for unit in units)
    assert set(unit["waveform_id"] for unit in units) == set(PILOT_WAVEFORMS)


def test_plan_disk_units_cover_shortlist_and_exact():
    case = generate_split("pilot_gap")[0]
    shortlist = ["K08_cc_span4.0_shift+0.0_dens1.00", "K08_tw_span6.0_shift-0.5_dens1.25"]
    units = plan_disk_forward_units(case, shortlist)
    keys = {unit["cache_key"] for unit in units}
    assert f"{case.case_id}:exact:disk:W0" in keys
    assert f"{case.case_id}:noip:disk:W2" in keys
    for candidate_id in shortlist:
        for waveform_id in PILOT_WAVEFORMS:
            assert f"{case.case_id}:{candidate_id}:disk:{waveform_id}" in keys
    assert len(keys) == (2 + len(shortlist)) * len(PILOT_WAVEFORMS)


def test_pilot_case_json_round_trip_evaluate_l0(tmp_path):
    cases = {item.case_id: item for item in generate_split("pilot_gap")}
    spectral = "K08_cc_span4.0_shift+0.0_dens1.00"
    oracle = "K08_tw_span6.0_shift-0.5_dens1.25"
    loaded = []
    for case_id in ("PG03", "PG04"):
        result = {
            "case_id": case_id,
            "official_variant": "S0",
            "point_only": False,
            "disk_shortlist": {"8": [spectral, oracle]},
            "fits": [],
            "choices": [_choice(case_id, k) for k in K_PILOT],
            "tasks": [
                _task(case_id, oracle, 8, "point", "W0"),
                _task(case_id, oracle, 8, "disk_1.0", "W0"),
                _task(case_id, spectral, 8, "point", "W0"),
                _task(case_id, spectral, 8, "disk_1.0", "W0"),
            ],
        }
        path = tmp_path / f"case_{case_id}.json"
        write_pilot_case_result(
            path,
            result,
            case=cases[case_id],
            provenance={
                "transform_label": "smoke_fast_lagged_dlf",
                "ft_pts_per_dec": -1,
                "empymod_version": "2.6.0",
                "three_d_run": False,
            },
        )
        item = load_pilot_case_result(path)
        assert item["case_id"] == case_id
        assert item["case_hash"] == cases[case_id].case_hash()
        assert item["official_variant"] == "S0"
        assert item["point_only"] is False
        assert len(item["choices"]) == 5
        loaded.append(item)
    l0 = evaluate_l0(loaded)
    assert "status" in l0
    assert "passed" in l0


def test_cli_case_ids_flag_and_unknown_ids():
    help_run = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_run.returncode == 0
    assert "--case-ids" in help_run.stdout
    for bad in ("PG99", "TR01"):
        failed = subprocess.run(
            [sys.executable, str(SCRIPT), "--case-ids", bad],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        text = failed.stderr + failed.stdout
        assert "unknown pilot case ids" in text
        assert "PG03" in text


def test_write_json_keeps_evaluate_l0_keys(tmp_path):
    loaded = []
    for case_id in ("PG03", "PG04"):
        payload = {
            "case_id": case_id,
            "official_variant": "S1",
            "point_only": False,
            "choices": [_choice(case_id, k).__dict__ for k in K_PILOT],
            "tasks": [_task(case_id, "or", 8, "point", "W0").__dict__],
        }
        path = tmp_path / f"case_{case_id}.json"
        write_json(path, payload)
        loaded.append(load_pilot_case_result(path))
    l0 = evaluate_l0(loaded)
    assert l0["status"] in {"L0_PASS", "L0_FAIL"}
