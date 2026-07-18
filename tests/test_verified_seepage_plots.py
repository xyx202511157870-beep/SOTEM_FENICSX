from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atem3d.seepage_verification import VerificationGateError
import tools.plot_verified_seepage_report as verified_plots
from tools.plot_verified_seepage_report import (
    FIGURE_NAMES,
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


def test_verified_figure_manifest_is_two_solver_only() -> None:
    assert "verified_two_solver_anomaly.png" in FIGURE_NAMES
    assert "verified_three_solver_anomaly.png" not in FIGURE_NAMES


def test_plot_generation_does_not_read_comsol_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    values = np.ones((5, 31, 3), dtype=float)
    times = np.geomspace(1.0e-5, 1.0e-2, 31)
    loaded_paths: list[Path] = []
    sweep_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        verified_plots,
        "require_verified_summary",
        lambda _root: {"pass": True, "model_fingerprint": "a" * 64, "gates": {}},
    )
    monkeypatch.setattr(verified_plots, "_plot_geometry", lambda *_args: None)
    monkeypatch.setattr(
        verified_plots,
        "_case",
        lambda *_args: {"values": values.copy()},
    )

    def fake_load(path: Path, _fingerprint: str):
        loaded_paths.append(Path(path))
        return {"values": values.copy(), "times": times.copy()}

    monkeypatch.setattr(verified_plots, "_load", fake_load)
    monkeypatch.setattr(verified_plots, "_plot_grid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        verified_plots,
        "_plot_sweep",
        lambda _path, _summary, stem, xlabel: sweep_calls.append((stem, xlabel)),
    )
    monkeypatch.setattr(verified_plots, "_plot_convergence", lambda *_args: None)
    monkeypatch.setattr(verified_plots, "_plot_parity", lambda *_args: None)
    monkeypatch.setattr(verified_plots, "_plot_two_solver", lambda *_args: None, raising=False)
    monkeypatch.setattr(verified_plots, "_plot_three_solver", lambda *_args: None, raising=False)

    paths = verified_plots.generate_verified_plots(tmp_path)

    assert paths == [tmp_path / name for name in FIGURE_NAMES]
    assert loaded_paths == [tmp_path / "verification_empymod_background.npz"]
    assert all("comsol" not in str(path).lower() for path in loaded_paths)
    assert sweep_calls == [
        ("conductivity", "channel conductivity (S/m)"),
        ("volume", "channel volume (m^3)"),
    ]
