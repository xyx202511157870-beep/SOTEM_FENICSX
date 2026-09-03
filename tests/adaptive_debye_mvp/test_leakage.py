from pathlib import Path

import pytest

from atem3d.adaptive_debye_mvp.guards import (
    IndependentTestLeakageError,
    ThreeDNotAuthorizedError,
    assert_3d_authorized,
    assert_records_split_safe,
    assert_split_readable,
    refuse_3d_before_l2,
)
from atem3d.adaptive_debye_mvp.io import write_json
from atem3d.adaptive_debye_mvp.selector import load_independent_test_for_selector, select_templates


def test_selector_stage_cannot_read_independent_test():
    with pytest.raises(IndependentTestLeakageError):
        assert_split_readable("independent_test", stage="selector")
    with pytest.raises(IndependentTestLeakageError):
        assert_records_split_safe([{"split": "independent_test"}], stage="train")
    with pytest.raises(IndependentTestLeakageError):
        load_independent_test_for_selector([{"split": "independent_test", "K": 6}])


def test_selector_requires_l0_and_rejects_test_rows(tmp_path):
    l0 = tmp_path / "L0_summary.json"
    write_json(l0, {"passed": True})
    with pytest.raises(IndependentTestLeakageError):
        select_templates(
            train_records=[{"split": "independent_test", "K": 6, "candidate_id": "x", "total_p95": 0.1, "spectral_error": 0.1, "condition_number": 1.0}],
            validation_records=[{"split": "validation", "K": 6, "candidate_id": "x", "total_p95": 0.1, "spectral_error": 0.1, "condition_number": 1.0}],
            l0_path=l0,
            output_dir=tmp_path / "out",
        )


def test_3d_before_l2_errors(tmp_path):
    with pytest.raises(ThreeDNotAuthorizedError):
        assert_3d_authorized(tmp_path)
    with pytest.raises(ThreeDNotAuthorizedError):
        refuse_3d_before_l2(tmp_path)
    write_json(tmp_path / "LAYERED_GATE_PASSED.json", {"status": "STOP_LAYERED_NO_ACTIONABLE_GAP", "l2_passed": False})
    with pytest.raises(ThreeDNotAuthorizedError):
        refuse_3d_before_l2(tmp_path)
