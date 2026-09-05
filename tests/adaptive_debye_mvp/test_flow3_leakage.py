import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "paper_receiver_adaptive_debye_mvp" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from consume_sibling_case_json import _payload_rank
from run_layered_test import _case_json_satisfies as l2_case_json_satisfies
from run_selector_cases import case_json_satisfies

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


def test_case_json_satisfies_does_not_overwrite_disks(tmp_path):
    point = tmp_path / "case_TR01.json"
    point.write_text(
        json.dumps({"case_id": "TR01", "point_only": True, "tasks": [{"receiver_id": "point"}]}),
        encoding="utf-8",
    )
    assert case_json_satisfies(point, "points") is True
    assert case_json_satisfies(point, "disks") is False
    disks = tmp_path / "case_TR02.json"
    disks.write_text(
        json.dumps(
            {
                "case_id": "TR02",
                "point_only": False,
                "receiver_ids": ["point", "disk_1.0", "disk_4.0"],
                "tasks": [{"receiver_id": "disk_1.0"}],
            }
        ),
        encoding="utf-8",
    )
    assert case_json_satisfies(disks, "points") is True
    assert case_json_satisfies(disks, "disks") is True


def test_l2_does_not_skip_selector_unread_precompute(tmp_path):
    path = tmp_path / "case_TE01.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "TE01",
                "point_only": False,
                "schema": "atem3d.adaptive_debye_mvp.independent_test_case_result.v1",
                "provenance": {"selector_read": False, "l2_evaluated": False},
                "tasks": [{"receiver_id": "disk_1.0"}],
            }
        ),
        encoding="utf-8",
    )
    assert l2_case_json_satisfies(path, "points") is True
    assert l2_case_json_satisfies(path, "disks") is False


def test_consume_prefers_official_forced_disk_json_over_partial_pr14():
    pr14 = {
        "case_id": "TR01",
        "point_only": False,
        "schema": "atem3d.adaptive_debye_mvp.split_case_result.v1",
        "tasks": [{"receiver_id": "disk_1.0"}] * 72,
    }
    official = {
        "case_id": "TR01",
        "point_only": False,
        "schema": "atem3d.adaptive_debye_mvp.pilot_case_result.v1",
        "tasks": [{"receiver_id": "disk_1.0"}] * 192,
    }
    assert _payload_rank(official) > _payload_rank(pr14)
    point = {"case_id": "TR01", "point_only": True, "schema": "atem3d.adaptive_debye_mvp.pilot_case_result.v1", "tasks": [{"receiver_id": "point"}]}
    assert _payload_rank(point) > _payload_rank(pr14)
    assert _payload_rank(official) > _payload_rank(point)
