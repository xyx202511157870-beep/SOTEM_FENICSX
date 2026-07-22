from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_receiver_depth_profile_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_terminal_depth_profile_reuses_fields_and_negative_z(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        workdir=tmp_path,
        receiver=(1.0, 2.0, -0.1),
        receiver_depth_profile_depths=(300.0, 400.0),
    )
    seen = []

    def fake_evaluate(E, dbdt, msh, eval_config):
        seen.append((E, dbdt, msh, eval_config.receiver, eval_config.receiver_type))
        return {
            "Ex": eval_config.receiver[2],
            "Ey": 0.0,
            "dBzdt": -eval_config.receiver[2],
            "sample_count": 1,
            "candidate_count_min": 1,
            "candidate_count_max": 1,
            "candidate_count_mean": 1.0,
            "multi_candidate_sample_count": 0,
        }

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate)
    msh = SimpleNamespace(comm=SimpleNamespace(rank=0))
    E, dbdt = object(), object()

    rows = sp._evaluate_terminal_receiver_depth_profile(
        E,
        dbdt,
        msh,
        config,
        time_obs=7.943282347242813e-4,
    )

    assert [item[3] for item in seen] == [
        (1.0, 2.0, -300.0),
        (1.0, 2.0, -400.0),
    ]
    assert all(
        item[0] is E and item[1] is dbdt and item[2] is msh and item[4] == "point"
        for item in seen
    )
    assert [row["depth_m"] for row in rows] == [300.0, 400.0]
    assert [row["dBzdt"] for row in rows] == pytest.approx([300.0, 400.0])


def test_receiver_depth_profile_csv_is_root_only_atomic_and_ordered(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    rows = [
        {
            "time_obs": 7.943282347242813e-4,
            "depth_m": depth,
            "receiver_x": 0.0,
            "receiver_y": -500.0,
            "receiver_z": -depth,
            "Ex": depth * 1.0e-9,
            "Ey": 0.0,
            "dBzdt": depth * 1.0e-12,
            "sample_count": 1,
            "candidate_count_min": 1,
            "candidate_count_max": 1,
            "candidate_count_mean": 1.0,
            "multi_candidate_sample_count": 0,
        }
        for depth in (300.0, 400.0)
    ]

    sp._write_receiver_depth_profile_csv(
        config,
        rows,
        comm=SimpleNamespace(rank=0),
    )

    path = config.receiver_depth_profile_csv()
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "time_obs",
            "depth_m",
            "receiver_x",
            "receiver_y",
            "receiver_z",
            "Ex",
            "Ey",
            "dBzdt",
            "sample_count",
            "candidate_count_min",
            "candidate_count_max",
            "candidate_count_mean",
            "multi_candidate_sample_count",
            "candidate_center_distance_min",
            "candidate_center_distance_max",
            "candidate_center_distance_mean",
            "selected_center_distance_mean",
            "selected_center_distance_max",
            "candidate_center_z_min",
            "candidate_center_z_max",
            "selected_center_z_mean",
        ]
        loaded = list(reader)
    assert [float(row["depth_m"]) for row in loaded] == [300.0, 400.0]

    other = sp.PipelineConfig(workdir=tmp_path / "nonroot")
    sp._write_receiver_depth_profile_csv(
        other,
        rows,
        comm=SimpleNamespace(rank=1),
    )
    assert not other.receiver_depth_profile_csv().exists()
