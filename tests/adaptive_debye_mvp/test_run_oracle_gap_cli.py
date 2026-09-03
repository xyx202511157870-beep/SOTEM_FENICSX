from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from atem3d.adaptive_debye_mvp.oracle_gap import CaseKChoice, TaskMetrics, evaluate_l0
from atem3d.adaptive_debye_mvp.registry import cases_for_split, generate_all_cases


SCRIPT = Path(__file__).resolve().parents[2] / "paper_receiver_adaptive_debye_mvp" / "scripts" / "run_oracle_gap.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_oracle_gap_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_and_filter_case_ids():
    mod = _load_script()
    assert mod.parse_case_ids("PG01,PG02") == ("PG01", "PG02")
    assert mod.parse_case_ids(" PG01, PG01, PG02 ") == ("PG01", "PG02")
    selected = mod.select_pilot_cases(("PG01", "PG02"))
    assert [case.case_id for case in selected] == ["PG01", "PG02"]
    all_pilot = list(cases_for_split("pilot_gap", generate_all_cases()))
    assert [case.case_id for case in selected] == [case.case_id for case in all_pilot if case.case_id in {"PG01", "PG02"}]
    with pytest.raises(SystemExit, match="unknown or non-pilot"):
        mod.select_pilot_cases(("PG99",))


def _choice(case_id: str, k: int) -> CaseKChoice:
    return CaseKChoice(
        case_id=case_id,
        K=k,
        spectral_id="b2",
        oracle_id="or",
        ids_differ=True,
        e_b2=0.02,
        e_or=0.01,
        ratio=0.5,
        gap=0.5,
        ip_b2=0.02,
        ip_or=0.015,
        log_hausdorff=0.1,
        qualifies_b2=False,
        qualifies_or=True,
    )


def _task(case_id: str, candidate_id: str, k: int, waveform_id: str, receiver_id: str) -> TaskMetrics:
    return TaskMetrics(
        case_id=case_id,
        candidate_id=candidate_id,
        K=k,
        waveform_id=waveform_id,
        receiver_id=receiver_id,
        receiver_index=0,
        total_p95=0.01 if candidate_id == "or" else 0.02,
        ip_increment_nrmse=0.01,
        passed=True,
        unexplained_sign_flips=0,
        peak_time_error_steps_max=0.0,
        h_p95=0.01,
        dbdt_p95=0.02,
    )


def test_case_result_payload_is_evaluate_l0_ready():
    mod = _load_script()
    cases = {item.case_id: item for item in generate_all_cases()}
    payloads = []
    rehydrated = []
    for case_id in ("PG01", "PG02"):
        choices = [_choice(case_id, k) for k in (4, 6, 8, 10, 12)]
        tasks = []
        for k in (4, 6, 8, 10, 12):
            for candidate_id in ("b2", "or"):
                for waveform_id in ("W0", "W1", "W2"):
                    for receiver_id in ("point", "disk_1.0", "disk_4.0"):
                        tasks.append(_task(case_id, candidate_id, k, waveform_id, receiver_id))
        result = {
            "case_id": case_id,
            "official_variant": "S0",
            "fits": [],
            "choices": choices,
            "tasks": tasks,
        }
        payload = mod.build_case_result_payload(result, case=cases[case_id], k_values=(4, 6, 8, 10, 12))
        assert payload["schema"] == mod.CASE_RESULT_SCHEMA
        assert payload["l0_gate_evaluated"] is False
        assert payload["three_d_run"] is False
        assert payload["k_qual"]["or"] == 4
        assert set(payload["groups"]["8"]["waveform"]) == {"W0", "W1", "W2"}
        assert "point" in payload["groups"]["8"]["receiver"]
        assert "H" in payload["groups"]["8"]["channel"]
        payloads.append(payload)
        rehydrated.append(
            {
                "choices": [CaseKChoice(**row) for row in payload["choices"]],
                "tasks": [TaskMetrics(**row) for row in payload["tasks"]],
                "official_variant": payload["official_variant"],
            }
        )
    l0 = evaluate_l0(rehydrated)
    assert "status" in l0
    assert "group_ok" in l0
