from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_runtime_report_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _error_metrics():
    return {
        name: {
            "mean": 0.0,
            "median": 0.0,
            "rms": 0.0,
            "max": 0.0,
            "max_index": 0,
            "floor": 1.0e-12,
            "reference_too_small": False,
        }
        for name in ["Ex", "Ey", "dBzdt"]
    }


def _plot_error_metrics(times):
    return {
        name: {
            "relative": np.linspace(0.01, 0.04, len(times)),
            "absolute": np.zeros(len(times)),
            "mean": 0.0,
            "median": 0.0,
            "rms": 0.0,
            "max": 0.04,
            "max_index": len(times) - 1,
            "floor": 1.0e-12,
            "reference_too_small": False,
        }
        for name in ["Ex", "Ey", "dBzdt"]
    }


def test_plot_verification_handles_log_amplitude_response_axis(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-6, t_max=3.0e-6, time_growth=2.0)
    times = np.asarray([1.0e-6, 2.0e-6, 3.0e-6])
    fem_data = np.asarray(
        [
            [2.0e-5, -1.0e-8, -1.0e-8],
            [1.0e-5, -1.0e-9, -1.0e-9],
            [2.0e-6, 1.0e-10, -1.0e-10],
        ],
        dtype=float,
    )
    ref_data = fem_data * 1.01

    sp.plot_verification(
        times,
        fem_data,
        ref_data,
        _plot_error_metrics(times),
        ["Ex", "Ey", "dBzdt"],
        config,
    )

    assert config.output_png().is_file()


def test_postprocess_saved_forward_uses_forward_partial_without_rerunning_fem(tmp_path, monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-6, t_max=2.0e-6, time_growth=2.0)
    times = np.asarray([1.0e-6, 2.0e-6])
    data = np.asarray(
        [
            [1.0e-5, 1.0e-8, 2.0e-8, -1.0e-8],
            [8.0e-6, 8.0e-9, 1.6e-8, -8.0e-9],
        ],
        dtype=float,
    )
    np.savez(
        config.forward_partial_npz(),
        times=times,
        fem=data,
        components=np.asarray(["Ex", "Ey", "Hz", "dBzdt"]),
        solver_steps=np.asarray([10, 11]),
        solver_iterations=np.asarray([20, 21]),
        solver_residuals=np.asarray([1.0e-9, 2.0e-9]),
        solver_reasons=np.asarray([2, 2]),
    )

    reference_srcpts = []

    def fake_reference(t_array, _config, mode="noip", *, srcpts=None):
        assert mode == "noip"
        reference_srcpts.append(srcpts)
        return {
            "times": np.asarray(t_array),
            "data": data.copy(),
            "components": ["Ex", "Ey", "Hz", "dBzdt"],
        }

    monkeypatch.setattr(sp, "get_empymod_reference", fake_reference)

    result = sp.postprocess_saved_forward(config, env={})

    assert reference_srcpts == [None, 17]
    evidence = json.loads(
        config.reference_source_quadrature_audit_json().read_text(encoding="utf-8")
    )
    summary = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert evidence == result["reference_audit_summary"]
    assert summary["reference_source_quadrature_audit"] == evidence
    assert result["fem_result"]["data"].shape == (2, 4)
    assert config.output_npz().is_file()
    assert config.output_png().is_file()
    assert config.output_report().is_file()
    text = config.output_report().read_text(encoding="utf-8")
    assert "source mode: postprocess_partial/auto" in text
    assert "reference source-quadrature acceptance gate" in text
    assert "distinct from the ordinary 1e-6 floor error metric" in text
    begin = "BEGIN_REFERENCE_SOURCE_QUADRATURE_AUDIT_JSON"
    end = "END_REFERENCE_SOURCE_QUADRATURE_AUDIT_JSON"
    report_audit = json.loads(text.split(begin, 1)[1].split(end, 1)[0].strip())
    assert report_audit == evidence
    assert report_audit == summary["reference_source_quadrature_audit"]
    assert sp._canonical_json_bytes(report_audit) == sp._canonical_json_bytes(evidence)
    assert sp._canonical_json_bytes(report_audit) == sp._canonical_json_bytes(
        summary["reference_source_quadrature_audit"]
    )


def test_write_report_records_model_runtime(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-6, t_max=2.0e-6, time_growth=2.0)
    times = np.asarray([1.0e-6, 2.0e-6])
    data = np.asarray(
        [
            [1.0, 0.0, 2.0],
            [1.5, 0.0, 2.5],
        ],
        dtype=float,
    )
    fem_result = {
        "times": times,
        "data": data,
        "components": ["Ex", "Ey", "dBzdt"],
        "solver_log": [],
    }
    ref_result = {"times": times, "data": data, "components": ["Ex", "Ey", "dBzdt"]}
    runtime = {
        "total_seconds": 12.3456,
        "mesh_seconds": 1.25,
        "forward_seconds": 9.5,
        "reference_seconds": 0.75,
        "postprocess_seconds": 0.5,
    }

    sp.write_report(
        config,
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=_error_metrics(),
        source_info={"mode": "manual_line"},
        runtime=runtime,
    )

    text = config.output_report().read_text(encoding="utf-8")
    assert "model runtime:" in text
    assert "total: 12.346 s" in text
    assert "mesh: 1.250 s" in text
    assert "forward solve: 9.500 s" in text
    assert "empymod reference: 0.750 s" in text
    assert "postprocess/report: 0.500 s" in text


def test_write_report_records_manual_line_mesh_segment_integration(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-6, t_max=2.0e-6, time_growth=2.0)
    times = np.asarray([1.0e-6])
    data = np.asarray([[1.0, 0.0, 2.0]], dtype=float)
    fem_result = {
        "times": times,
        "data": data,
        "components": ["Ex", "Ey", "dBzdt"],
        "solver_log": [],
    }
    ref_result = {"times": times, "data": data, "components": ["Ex", "Ey", "dBzdt"]}
    source_info = {
        "mode": "manual_line",
        "local_projection_diagnostics": {
            "integration_mode": "mesh_segments",
            "segment_count": 200,
            "segment_total_length": 1000.0,
            "quadrature_points_per_segment_min": 26,
            "quadrature_points_per_segment_max": 26,
            "quadrature_points_per_segment_mean": 26.0,
            "quadrature_points": 5200,
            "added_points": 5200,
            "missed_points": 0,
            "unique_hit_cells": 206,
            "cell_contribution_top_fraction": 0.005,
            "dof_contribution_top_fraction": 0.008,
        },
    }

    sp.write_report(
        config,
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=_error_metrics(),
        source_info=source_info,
    )

    text = config.output_report().read_text(encoding="utf-8")
    assert "source line integration: mode=mesh_segments; segments=200; segment_length_total=1000 m" in text
    assert "quadrature_per_segment[min/mean/max]=26/26/26" in text


def test_write_report_records_divergence_cleaning_strength(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        workdir=tmp_path,
        t_min=1.0e-6,
        t_max=2.0e-6,
        time_growth=2.0,
        divergence_cleaning="conductivity",
        divergence_cleaning_strength=0.25,
    )
    times = np.asarray([1.0e-6])
    data = np.asarray([[1.0, 0.0, 2.0]], dtype=float)
    fem_result = {
        "times": times,
        "data": data,
        "components": ["Ex", "Ey", "dBzdt"],
        "solver_log": [],
    }
    ref_result = {"times": times, "data": data, "components": ["Ex", "Ey", "dBzdt"]}

    sp.write_report(
        config,
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=_error_metrics(),
        source_info={"mode": "manual_line"},
    )

    text = config.output_report().read_text(encoding="utf-8")
    assert "divergence cleaning: conductivity; strength=0.25" in text


def test_write_report_records_divergence_cleaning_solver_stats(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, t_min=1.0e-6, t_max=2.0e-6, time_growth=2.0)
    times = np.asarray([1.0e-6])
    data = np.asarray([[1.0, 0.0, 2.0]], dtype=float)
    fem_result = {
        "times": times,
        "data": data,
        "components": ["Ex", "Ey", "dBzdt"],
        "solver_log": [
            {
                "step": 4,
                "time": 2.0e-6,
                "observation_time": 1.0e-6,
                "dt": 1.0e-6,
                "its": 12,
                "residual": 3.0e-9,
                "reason": 2,
                "is_output": True,
                "divergence_clean_before": 8.0,
                "divergence_clean_after": 1.0e-9,
                "divergence_clean_correction_norm": 0.25,
                "divergence_clean_applied_correction_norm": 0.125,
                "divergence_clean_strength": 0.5,
            }
        ],
    }
    ref_result = {"times": times, "data": data, "components": ["Ex", "Ey", "dBzdt"]}

    sp.write_report(
        config,
        env={},
        fem_result=fem_result,
        ref_result=ref_result,
        errors=_error_metrics(),
        source_info={"mode": "manual_line"},
    )

    text = config.output_report().read_text(encoding="utf-8")
    assert "div_clean_before=8.000000e+00" in text
    assert "div_clean_after=1.000000e-09" in text
    assert "div_clean_correction=2.500000e-01" in text
    assert "div_clean_applied=1.250000e-01" in text
    assert "div_clean_strength=0.5" in text
