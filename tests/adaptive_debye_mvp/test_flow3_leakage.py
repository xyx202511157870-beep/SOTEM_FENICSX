from pathlib import Path

import pytest

from atem3d.adaptive_debye_mvp.guards import (
    IndependentTestLeakageError,
    assert_cache_untouched,
    assert_case_ids_split_safe,
)
from atem3d.adaptive_debye_mvp.selector import records_from_case_results


def test_case_ids_reject_test_and_pilot_in_flow3():
    with pytest.raises(IndependentTestLeakageError):
        assert_case_ids_split_safe(["TE01"], stage="flow3")
    with pytest.raises(IndependentTestLeakageError):
        assert_case_ids_split_safe(["PG01"], stage="selector")
    assert_case_ids_split_safe(["TR01", "VA02"], stage="flow3")


def test_cache_audit_rejects_test_prefix(tmp_path):
    before = set()
    (tmp_path / "TE01:exact:point:W0.npz").write_bytes(b"x")
    with pytest.raises(IndependentTestLeakageError):
        assert_cache_untouched(tmp_path, before=before)


def test_records_from_case_results_reject_te_case():
    with pytest.raises(IndependentTestLeakageError):
        records_from_case_results(
            [{"case_id": "TE01", "choices": [], "tasks": [], "split": "train"}],
            split="train",
            official_variant="S1",
            scope="point",
        )
