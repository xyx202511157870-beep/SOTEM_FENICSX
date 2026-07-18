from __future__ import annotations

from pathlib import Path

import atem3d.seepage_final_aggregation as final_aggregation
from atem3d.seepage_matrix_aggregation import OPEN3D_REQUIRED_GATES


def test_final_summary_uses_only_open3d_required_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    expected = {
        "pass": True,
        "failed_gates": [],
        "required_gates": list(OPEN3D_REQUIRED_GATES),
        "gates": {name: {"available": True, "pass": True} for name in OPEN3D_REQUIRED_GATES},
        "model_fingerprint": "a" * 64,
    }
    calls: list[tuple[Path, bool]] = []

    def fake_open3d_summary(output_root: str | Path, *, require_pass: bool = False):
        calls.append((Path(output_root), require_pass))
        return expected

    monkeypatch.setattr(final_aggregation, "build_open3d_summary", fake_open3d_summary)

    summary = final_aggregation.build_final_summary(tmp_path, require_pass=True)

    assert summary is expected
    assert final_aggregation.FINAL_REQUIRED_GATES == OPEN3D_REQUIRED_GATES
    assert calls == [(tmp_path, True)]
