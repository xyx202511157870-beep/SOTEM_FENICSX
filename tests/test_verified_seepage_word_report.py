from __future__ import annotations

import json
from pathlib import Path

import pytest

from atem3d.seepage_verification import VerificationGateError
from tools.build_verified_seepage_word_report import (
    _gate_status_rows,
    build_verified_report,
)


def test_verified_report_refuses_missing_or_failed_final_summary(
    tmp_path: Path,
) -> None:
    with pytest.raises(VerificationGateError, match="verification_summary.json"):
        build_verified_report(tmp_path, tmp_path / "report.docx")

    (tmp_path / "verification_summary.json").write_text(
        json.dumps({"pass": False, "failed_gates": ["comsol_zero_contrast"]}),
        encoding="utf-8",
    )
    with pytest.raises(VerificationGateError, match="comsol_zero_contrast"):
        build_verified_report(tmp_path, tmp_path / "report.docx")


def test_gate_rows_are_derived_from_final_json() -> None:
    rows = _gate_status_rows(
        {
            "required_gates": ["zero_contrast_simpeg", "cross_solver"],
            "gates": {
                "zero_contrast_simpeg": {
                    "available": True,
                    "pass": True,
                    "normalized_l2": [1e-10, 2e-10, 3e-10],
                },
                "cross_solver": {
                    "available": True,
                    "pass": True,
                    "median_threshold": 0.2,
                    "p95_threshold": 0.35,
                },
            },
        }
    )

    assert rows[0][0] == "zero_contrast_simpeg"
    assert rows[0][-1] == "通过"
    assert "L2" in rows[0][1]
    assert "20.0%" in rows[1][1]
