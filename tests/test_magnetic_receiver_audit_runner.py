from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_runner():
    path = Path("tools/run_fenicsx_magnetic_receiver_audit.py")
    spec = importlib.util.spec_from_file_location("magnetic_receiver_audit_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact_values(scale=1.0):
    values = np.zeros((5, 4, 3), dtype=float)
    values[:, :, 0] = np.asarray([[1], [2], [3], [2], [1]])
    values[:, :, 1] = scale * np.asarray([[-4], [-2], [0], [2], [4]])
    values[:, :, 2] = scale * np.asarray([[-8], [-3], [0], [3], [8]])
    return values


def test_build_method_summary_aggregates_all_requested_methods():
    runner = load_runner()
    names = (
        "curl",
        "biot_rate",
        "faraday_loop_16",
        "faraday_loop_32",
        "faraday_loop_64",
        "biot_center",
        "biot_tetra4",
    )
    methods = {
        name: {"background": exact_values(), "channel": exact_values(1.1)}
        for name in names
    }

    summary = runner.build_method_summary(
        methods,
        components=("Ex", "dBzdt", "Hz"),
        times=np.geomspace(1.0e-5, 1.0e-3, 4),
        signal_floor=1.0e-20,
    )

    assert set(summary) == set(names)
    assert all(set(item["models"]) == {"background", "channel", "delta"} for item in summary.values())
    assert summary["curl"]["rx3_zero_ratio"] == 0.0


def test_rx3_relative_error_is_none_below_signal_floor():
    runner = load_runner()
    methods = {"curl": {"background": exact_values(0.0), "channel": exact_values(0.0)}}
    summary = runner.build_method_summary(
        methods,
        components=("Ex", "dBzdt", "Hz"),
        times=np.arange(4, dtype=float),
        signal_floor=1.0e-12,
    )
    assert summary["curl"]["rx3_relative_error"] is None


def test_formal_method_selection_requires_every_gate():
    runner = load_runner()
    good = {
        "rx3_zero_ratio": 0.001,
        "pair_24_residual": 0.002,
        "pair_15_residual": 0.003,
        "strong_signal_error_increase_percentage_points": 1.0,
        "ex_median_change": 0.001,
        "strong_signal_median_error": 0.02,
    }
    bad = {**good, "pair_24_residual": 0.02, "strong_signal_median_error": 0.0}
    selected = runner.select_formal_method({"stable": good, "low_rx3_only": bad})
    assert selected["passed"] is True
    assert selected["selected"] == "stable"
    assert "pair_24_residual" in selected["rejected"]["low_rx3_only"]
