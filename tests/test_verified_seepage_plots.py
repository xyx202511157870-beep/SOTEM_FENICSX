from __future__ import annotations

import json
from pathlib import Path

import pytest

from atem3d.seepage_verification import VerificationGateError
from tools.plot_verified_seepage_report import (
    REPORT_RECEIVER_INDICES,
    require_verified_summary,
)


def test_verified_plots_use_only_four_formal_receivers() -> None:
    assert REPORT_RECEIVER_INDICES == (0, 1, 3, 4)
    assert 2 not in REPORT_RECEIVER_INDICES


def test_verified_plots_fail_closed_without_passing_summary(tmp_path: Path) -> None:
    with pytest.raises(VerificationGateError, match="verification_summary.json"):
        require_verified_summary(tmp_path)

    (tmp_path / "verification_summary.json").write_text(
        json.dumps({"pass": False, "failed_gates": ["cross_solver"]}),
        encoding="utf-8",
    )
    with pytest.raises(VerificationGateError, match="cross_solver"):
        require_verified_summary(tmp_path)


def test_verified_summary_is_returned_when_all_gates_pass(tmp_path: Path) -> None:
    expected = {
        "pass": True,
        "failed_gates": [],
        "model_fingerprint": "a" * 64,
    }
    (tmp_path / "verification_summary.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    assert require_verified_summary(tmp_path) == expected
