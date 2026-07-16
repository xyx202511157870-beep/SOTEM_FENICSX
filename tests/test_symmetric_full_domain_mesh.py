from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load_module():
    path = Path("dolfinx/symmetric_full_domain_mesh.py")
    spec = importlib.util.spec_from_file_location("symmetric_full_domain_mesh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def half_fixture():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    tetrahedra = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    tetra_tags = np.asarray([101, 102], dtype=np.int32)
    triangles = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    triangle_tags = np.asarray([999, 202], dtype=np.int32)
    lines = np.asarray([[0, 1], [1, 4]], dtype=np.int64)
    line_tags = np.asarray([301, 302], dtype=np.int32)
    return points, tetrahedra, tetra_tags, triangles, triangle_tags, lines, line_tags


def signed_volume(points, tetra):
    xyz = points[tetra]
    return np.linalg.det(
        np.column_stack((xyz[1] - xyz[0], xyz[2] - xyz[0], xyz[3] - xyz[0]))
    ) / 6.0


def test_mirror_half_topology_reuses_plane_and_preserves_tags():
    module = load_module()
    result = module.mirror_half_topology(*half_fixture())

    assert result["points"].shape == (7, 3)
    assert np.count_nonzero(result["points"][:, 1] < 0.0) == 2
    assert result["tetrahedra"].shape == (4, 4)
    assert result["tetra_tags"].tolist() == [101, 102, 101, 102]
    assert np.all(
        [signed_volume(result["points"], tetra) != 0.0 for tetra in result["tetrahedra"]]
    )
    assert 999 not in result["triangle_tags"][2:].tolist()
    assert result["triangle_tags"].tolist() == [202, 202]
    assert result["line_tags"].tolist().count(301) == 1
    assert result["line_tags"].tolist().count(302) == 2


def test_mirror_half_topology_rejects_negative_input_nodes():
    module = load_module()
    fixture = list(half_fixture())
    fixture[0] = fixture[0].copy()
    fixture[0][3, 1] = -1.0
    with pytest.raises(ValueError, match="positive-y half"):
        module.mirror_half_topology(*fixture)


def test_audit_y_reflection_fails_missing_cell_or_tag_mismatch():
    module = load_module()
    result = module.mirror_half_topology(*half_fixture())
    passed = module.audit_y_reflection(
        result["points"], result["tetrahedra"], result["tetra_tags"], 1.0e-10
    )
    assert passed["passed"] is True
    assert passed["exact_node_pair_fraction"] == 1.0
    assert passed["exact_centroid_pair_fraction"] == 1.0

    missing = module.audit_y_reflection(
        result["points"], result["tetrahedra"][:-1], result["tetra_tags"][:-1], 1.0e-10
    )
    assert missing["passed"] is False
    mismatched_tags = result["tetra_tags"].copy()
    mismatched_tags[-1] = 999
    mismatch = module.audit_y_reflection(
        result["points"], result["tetrahedra"], mismatched_tags, 1.0e-10
    )
    assert mismatch["passed"] is False
    assert mismatch["tag_mismatch_count"] > 0


def test_pipeline_mesh_y_bounds_support_half_source_only():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location("pipeline_for_mesh_bounds", path)
    assert spec is not None and spec.loader is not None
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    assert pipeline._mesh_y_bounds(pipeline.PipelineConfig(y_extent=20.0)) == (-20.0, 20.0)
    assert pipeline._mesh_y_bounds(
        pipeline.PipelineConfig(y_extent=20.0, mesh_symmetry_mode="positive_half_source")
    ) == (0.0, 20.0)
    assert pipeline._mesh_source_embedding_dimension(pipeline.PipelineConfig()) == 3
    assert pipeline._mesh_source_embedding_dimension(
        pipeline.PipelineConfig(mesh_symmetry_mode="positive_half_source")
    ) == 2
