from __future__ import annotations

from pathlib import Path

from atem3d.seepage_final_aggregation import (
    COMSOL_REQUIRED_GATES,
    build_final_summary,
)


def test_final_summary_fails_closed_when_comsol_cases_are_missing(
    tmp_path: Path,
) -> None:
    summary = build_final_summary(tmp_path)

    assert summary["pass"] is False
    assert tuple(name for name in summary["required_gates"] if name.startswith("comsol_")) == COMSOL_REQUIRED_GATES
    assert all(summary["gates"][name]["available"] is False for name in COMSOL_REQUIRED_GATES)
