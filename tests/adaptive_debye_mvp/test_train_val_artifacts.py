from pathlib import Path

import pytest

from atem3d.adaptive_debye_mvp.guards import IndependentTestLeakageError
from atem3d.adaptive_debye_mvp.io import read_csv, read_json
from atem3d.adaptive_debye_mvp.oracle_gap import (
    CANDIDATE_POINT_METRIC_COLUMNS,
    POINT_METRIC_SENTINEL,
    CaseKChoice,
    FitRecord,
    TaskMetrics,
    assemble_train_val_metrics,
    candidate_point_metrics,
    load_split_case_result,
    resolve_train_val_cases,
    write_split_case_result,
)
from atem3d.adaptive_debye_mvp.registry import generate_all_cases
from atem3d.adaptive_debye_mvp.selector import select_templates


def _fit(candidate_id: str, K: int, *, valid: bool = True) -> FitRecord:
    return FitRecord(
        candidate_id=candidate_id,
        K=K,
        valid=valid,
        spectral_error_s0=0.02 if valid else 0.9,
        spectral_error_s1=0.01 if valid else 0.8,
        condition_number=4.0,
        relative_dc_error=0.0,
        optimizer_success=valid,
        fit=None,
    )


def _task(case_id: str, candidate_id: str, K: int, receiver_id: str, total_p95: float) -> TaskMetrics:
    return TaskMetrics(
        case_id=case_id,
        candidate_id=candidate_id,
        K=K,
        waveform_id="W0",
        receiver_id=receiver_id,
        receiver_index=0,
        total_p95=total_p95,
        ip_increment_nrmse=total_p95 * 0.5,
        passed=True,
        unexplained_sign_flips=0,
        peak_time_error_steps_max=0.0,
        h_p95=total_p95,
        dbdt_p95=total_p95,
    )


def _choice(case_id: str, K: int, candidate_id: str) -> CaseKChoice:
    return CaseKChoice(
        case_id=case_id,
        K=K,
        spectral_id=candidate_id,
        oracle_id=candidate_id,
        ids_differ=False,
        e_b2=0.01,
        e_or=0.01,
        ratio=1.0,
        gap=0.0,
        ip_b2=0.01,
        ip_or=0.01,
        log_hausdorff=0.0,
        qualifies_b2=True,
        qualifies_or=True,
    )


def _case(case_id: str):
    return next(case for case in generate_all_cases() if case.case_id == case_id)


def _synthetic_result(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "official_variant": "S1",
        "fits": [
            _fit("K04_good", 4, valid=True),
            _fit("K04_bad", 4, valid=False),
        ],
        "choices": [_choice(case_id, 4, "K04_good")],
        "tasks": [
            _task(case_id, "K04_good", 4, "point", 0.01),
            _task(case_id, "K04_good", 4, "disk_1.0", 0.99),
            _task(case_id, "K04_good", 4, "disk_4.0", 0.98),
        ],
        "point_only": False,
        "disk_shortlist": {"4": ["K04_good"]},
        "k_values": [4],
        "waveform_ids": ["W0"],
        "shared_survey_hash": {"point": "abc", "disk": "def"},
    }


def test_write_and_load_split_case_round_trip(tmp_path):
    case = _case("TR01")
    path = tmp_path / "case_TR01.json"
    write_split_case_result(path, _synthetic_result("TR01"), case=case, split="train")
    loaded = load_split_case_result(path)
    payload = read_json(path)
    assert payload["schema"] == "atem3d.adaptive_debye_mvp.split_case_result.v1"
    assert payload["split"] == "train"
    assert payload["case_hash"] == case.case_hash()
    assert payload["official_variant"] == "S1"
    assert loaded["split"] == "train"
    rows = payload["candidate_point_metrics"]
    assert len(rows) == 2
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["K04_good"]["total_p95"] == 0.01
    assert by_id["K04_good"]["n_tasks"] == 1
    assert by_id["K04_good"]["receiver_scope"] == "point"
    assert by_id["K04_good"]["spectral_error"] == 0.01
    assert by_id["K04_bad"]["total_p95"] == POINT_METRIC_SENTINEL
    assert by_id["K04_bad"]["n_tasks"] == 0
    assert set(rows[0]) >= set(CANDIDATE_POINT_METRIC_COLUMNS)


def test_write_split_case_refuses_independent_test(tmp_path):
    case = _case("TE01")
    with pytest.raises(IndependentTestLeakageError):
        write_split_case_result(
            tmp_path / "case_TE01.json",
            _synthetic_result("TE01"),
            case=case,
            split="independent_test",
        )


def test_candidate_point_metrics_ignore_disks():
    rows = candidate_point_metrics(_synthetic_result("TR01"), split="train", case_hash="h")
    assert [row["candidate_id"] for row in rows] == ["K04_bad", "K04_good"]
    assert rows[1]["total_p95"] == 0.01


def test_resolve_cases_refuses_independent_test_and_pressure():
    with pytest.raises(IndependentTestLeakageError):
        resolve_train_val_cases(split="independent_test")
    with pytest.raises(IndependentTestLeakageError):
        resolve_train_val_cases(split="both", case_ids=("TE01",))
    with pytest.raises(IndependentTestLeakageError):
        resolve_train_val_cases(split="both", case_ids=("LP01",))
    cases = resolve_train_val_cases(split="both")
    assert {case.split for case in cases} == {"train", "validation"}
    assert len(cases) == 18
    assert all(case.case_id.startswith(("TR", "VA")) for case in cases)


def test_assembler_refuses_test_json_and_does_not_freeze(tmp_path):
    generated = tmp_path / "generated"
    flow3 = generated / "flow3_selector"
    flow3.mkdir(parents=True)
    write_split_case_result(
        flow3 / "case_TR01.json",
        _synthetic_result("TR01"),
        case=_case("TR01"),
        split="train",
    )
    (flow3 / "case_TE01.json").write_text('{"schema":"x","case_id":"TE01","split":"independent_test"}', encoding="utf-8")
    with pytest.raises(IndependentTestLeakageError):
        assemble_train_val_metrics(flow3, generated_dir=generated)
    (flow3 / "case_TE01.json").unlink()
    write_split_case_result(
        flow3 / "case_VA01.json",
        _synthetic_result("VA01"),
        case=_case("VA01"),
        split="validation",
    )
    payload = assemble_train_val_metrics(flow3, generated_dir=generated)
    assert payload["status"] == "L1_INPUTS_PARTIAL"
    assert payload["l1_frozen"] is False
    assert not (flow3 / "selected_template_by_K.json").exists()
    status = read_json(generated / "FLOW3_STATUS.json")
    assert status["status"] == "L1_INPUTS_PARTIAL"
    assert status["l1_frozen"] is False
    assert status["selected"] is None
    train = read_csv(flow3 / "train_candidate_metrics.csv")
    validation = read_csv(flow3 / "validation_candidate_metrics.csv")
    assert len(train) == 2
    assert len(validation) == 2
    assert tuple(train[0]) == CANDIDATE_POINT_METRIC_COLUMNS
    select_templates(
        train_records=train,
        validation_records=validation,
        l0_path=_write_l0(tmp_path),
        output_dir=tmp_path / "selector_scratch",
    )
    assert not (flow3 / "selected_template_by_K.json").exists()


def _write_l0(tmp_path: Path) -> Path:
    path = tmp_path / "L0_summary.json"
    path.write_text('{"passed": true}', encoding="utf-8")
    return path
