from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_mesh_config_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_receiver_refinement_defaults_are_explicit():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig()

    assert config.source_mesh_size == 5.0
    assert config.source_refinement_radius == 100.0
    assert config.receiver_mesh_size == 10.0
    assert config.receiver_anchor_mesh_size == 0.0
    assert config.receiver_refinement_radius == 60.0


def test_receiver_refinement_cloud_keeps_local_points_in_earth():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(500.0, 50.0, -0.1), receiver_mesh_size=10.0)

    points = sp._receiver_refinement_cloud_points(config)

    assert points
    assert all(point != config.receiver for point in points)
    assert all(point[2] <= config.receiver[2] for point in points)
    assert (510.0, 50.0, -0.1) in points
    assert (500.0, 50.0, -10.1) in points


def test_surface_receiver_volume_refinement_cloud_stays_below_interface():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(500.0, 50.0, 0.0), receiver_mesh_size=10.0)

    points = sp._receiver_refinement_cloud_points(config)

    assert points
    assert all(point[2] < 0.0 for point in points)
    assert (510.0, 50.0, 0.0) not in points
    assert (500.0, 50.0, -10.0) in points


def test_receiver_surface_classification_uses_geometry_tolerance():
    sp = _load_pipeline_module()

    assert sp._receiver_is_on_surface(sp.PipelineConfig(receiver=(0.0, -50.0, 0.0)))
    assert not sp._receiver_is_on_surface(sp.PipelineConfig(receiver=(0.0, -50.0, -0.1)))


def test_receiver_anchor_mesh_size_decouples_local_cloud_from_receiver_mesh_size():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(500.0, 50.0, -0.1),
        receiver_mesh_size=200.0,
        receiver_anchor_mesh_size=10.0,
    )

    points = sp._receiver_refinement_cloud_points(config)

    assert (510.0, 50.0, -0.1) in points
    assert (500.0, 50.0, -10.1) in points
    assert (700.0, 50.0, -0.1) not in points


def test_receiver_surface_refinement_cloud_adds_interface_points():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(500.0, 50.0, -0.1), receiver_mesh_size=10.0)

    points = sp._receiver_surface_refinement_points(config)

    assert points
    assert all(point[2] == 0.0 for point in points)
    assert (500.0, 50.0, 0.0) in points
    assert (510.0, 50.0, 0.0) in points
    assert (500.0, 60.0, 0.0) in points


def test_surface_receiver_surface_refinement_cloud_excludes_receiver_anchor():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(500.0, 50.0, 0.0), receiver_mesh_size=10.0)

    points = sp._receiver_surface_refinement_points(config)

    assert (500.0, 50.0, 0.0) not in points
    assert (510.0, 50.0, 0.0) in points


def test_receiver_anchor_mesh_size_decouples_surface_cloud_from_receiver_mesh_size():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(500.0, 50.0, -0.1),
        receiver_mesh_size=200.0,
        receiver_anchor_mesh_size=10.0,
    )

    points = sp._receiver_surface_refinement_points(config)

    assert (510.0, 50.0, 0.0) in points
    assert (500.0, 60.0, 0.0) in points
    assert (700.0, 50.0, 0.0) not in points


def test_source_refinement_cloud_adds_points_below_and_crossline():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-50.0, 0.0, -0.1),
        source_end=(50.0, 0.0, -0.1),
        source_mesh_size=10.0,
    )

    points = sp._source_refinement_cloud_points(config)

    assert points
    assert all(point != config.source_start for point in points)
    assert all(point != config.source_end for point in points)
    assert (-50.0, 10.0, -0.1) in points
    assert (-50.0, 0.0, -10.1) in points
    assert (-60.0, 0.0, -0.1) in points
    assert (60.0, 0.0, -0.1) in points


def test_memory_preflight_accepts_one_million_cells_on_32gb_workstation():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(memory_limit_gb=32.0, memory_safety_fraction=0.95)
    mesh_stats = {"cells_blocks": 1_000_000, "nodes": 250_000}

    audit = sp._mesh_memory_preflight(config, mesh_stats)

    assert audit["limit_gb"] == 32.0
    assert audit["usable_limit_gb"] == 30.4
    assert audit["estimated_gb"] < audit["usable_limit_gb"]
    assert audit["ok"] is True


def test_memory_preflight_is_calibrated_to_observed_one_million_cell_rss():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(memory_limit_gb=32.0, memory_safety_fraction=1.0)
    mesh_stats = {"cells_blocks": 1_044_028, "nodes": 164_973}

    audit = sp._mesh_memory_preflight(config, mesh_stats)

    assert 29.0 <= audit["estimated_gb"] <= 31.0


def test_memory_preflight_rejects_too_large_mesh_for_32gb_workstation():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(memory_limit_gb=32.0)
    mesh_stats = {"cells_blocks": 2_000_000, "nodes": 500_000}

    try:
        sp._mesh_memory_preflight(config, mesh_stats)
    except MemoryError as exc:
        assert "estimated solver memory" in str(exc)
        assert "32" in str(exc)
    else:
        raise AssertionError("expected MemoryError for oversized 32 GB run")


def test_existing_mesh_reuse_runs_memory_preflight(tmp_path, monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, force_mesh=False, memory_limit_gb=32.0)
    config.mesh_path().write_text("$MeshFormat\n", encoding="utf-8")
    calls = []

    def fake_stats(path):
        calls.append(("stats", path))
        return {"cells_blocks": 2_000_000, "nodes": 500_000}

    def fake_preflight(received_config, stats):
        calls.append(("preflight", received_config, stats))
        raise MemoryError("estimated solver memory exceeds configured workstation budget")

    monkeypatch.setattr(sp, "_mesh_size_statistics", fake_stats)
    monkeypatch.setattr(sp, "_mesh_memory_preflight", fake_preflight)

    try:
        sp.generate_verification_mesh(config)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError from reused mesh preflight")

    assert calls[0] == ("stats", config.mesh_path())
    assert calls[1][0] == "preflight"
