from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_mesh_quality_gate_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingField:
    def __init__(self):
        self.next_tag = 1
        self.added = []
        self.numbers = []
        self.number_lists = []
        self.background = None

    def add(self, kind):
        tag = self.next_tag
        self.next_tag += 1
        self.added.append((tag, kind))
        return tag

    def setNumber(self, tag, name, value):
        self.numbers.append((tag, name, value))

    def setNumbers(self, tag, name, values):
        self.number_lists.append((tag, name, list(values)))

    def setAsBackgroundMesh(self, tag):
        self.background = tag


class _RecordingOcc:
    def __init__(self):
        self.next_tag = 10
        self.centers = {}

    def _tag(self):
        tag = self.next_tag
        self.next_tag += 1
        return tag

    def addBox(self, x, y, z, dx, dy, dz):
        tag = self._tag()
        self.centers[(3, tag)] = (x + dx / 2.0, y + dy / 2.0, z + dz / 2.0)
        return tag

    def fragment(self, *_args):
        return None

    def addPoint(self, *_args):
        return self._tag()

    def addLine(self, *_args):
        return self._tag()

    def synchronize(self):
        return None

    def getCenterOfMass(self, dim, tag):
        return self.centers[(dim, tag)]


class _RecordingMesh:
    def __init__(self):
        self.field = _RecordingField()
        self.embeds = []

    def embed(self, source_dim, tags, target_dim, target_tag):
        self.embeds.append((source_dim, list(tags), target_dim, target_tag))

    def generate(self, _dim):
        return None

    def getNodes(self):
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float)

    def getElements(self):
        return [], [], []


class _RecordingModel:
    def __init__(self):
        self.occ = _RecordingOcc()
        self.mesh = _RecordingMesh()
        self.physical = []
        self.removed_physical = []

    def add(self, _name):
        return None

    def getEntities(self, dim):
        if dim == 3:
            return [(3, tag) for (entity_dim, tag) in self.occ.centers if entity_dim == 3]
        if dim == 2:
            return [(2, 501), (2, 502)]
        return []

    def getBoundingBox(self, dim, tag):
        assert dim == 2
        if tag == 501:
            return (-25_000.0, -25_000.0, 0.0, 25_000.0, 25_000.0, 0.0)
        return (-25_000.0, -25_000.0, -25_000.0, -25_000.0, 25_000.0, 10_000.0)

    def addPhysicalGroup(self, dim, tags, physical_tag):
        self.physical.append((dim, list(tags), physical_tag))

    def setPhysicalName(self, *_args):
        return None

    def removePhysicalGroups(self, groups):
        self.removed_physical.extend(groups)


class _RecordingGmsh:
    def __init__(self):
        self.model = _RecordingModel()
        self.option = SimpleNamespace(setNumber=lambda *_args: None)
        self.written = None
        self.written_paths = []

    def initialize(self):
        return None

    def finalize(self):
        return None

    def write(self, path):
        self.written = Path(path)
        self.written_paths.append(self.written)
        self.written.write_text("synthetic mesh", encoding="utf-8")


def test_gmsh_refinement_entities_are_not_embedded_in_earth_volume(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    fake = _RecordingGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)
    monkeypatch.setattr(sp, "_mesh_memory_preflight_for_path", lambda *_args: None)
    monkeypatch.setattr(
        sp,
        "_write_dolfinx_companion_mesh",
        lambda config: config.dolfinx_mesh_path().write_text("volume mesh", encoding="utf-8"),
        raising=False,
    )
    config = sp.PipelineConfig(workdir=tmp_path, force_mesh=True)

    sp.generate_verification_mesh(config)

    assert fake.model.mesh.embeds
    assert all(target_dim == 2 for _source_dim, _tags, target_dim, _target in fake.model.mesh.embeds)
    assert all(source_dim == 0 for source_dim, _tags, _target_dim, _target in fake.model.mesh.embeds)
    assert any(dim == 1 and physical_tag == sp.PHYS_SOURCE_LINE for dim, _tags, physical_tag in fake.model.physical)
    assert any(name == "CurvesList" for _tag, name, _values in fake.model.mesh.field.number_lists)
    assert any(name == "PointsList" for _tag, name, _values in fake.model.mesh.field.number_lists)


def test_gmsh_writes_dolfinx_companion_without_orphan_source_physical_group(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    fake = _RecordingGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)
    monkeypatch.setattr(sp, "_mesh_memory_preflight_for_path", lambda *_args: None)
    calls = []

    def write_companion(config):
        calls.append(config.mesh_path())
        config.dolfinx_mesh_path().write_text("volume mesh", encoding="utf-8")

    monkeypatch.setattr(
        sp, "_write_dolfinx_companion_mesh", write_companion, raising=False
    )
    config = sp.PipelineConfig(workdir=tmp_path, force_mesh=True)

    sp.generate_verification_mesh(config)

    assert fake.written_paths == [config.mesh_path()]
    assert calls == [config.mesh_path()]
    assert config.mesh_path().is_file()
    assert config.dolfinx_mesh_path().is_file()


def test_dolfinx_companion_data_drops_orphan_line_and_compacts_nodes():
    sp = _load_pipeline_module()
    points = np.asarray(
        [
            [-1.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cell_blocks = [
        ("line", np.asarray([[0, 1]], dtype=np.int64)),
        ("triangle", np.asarray([[2, 3, 4]], dtype=np.int64)),
        ("tetra", np.asarray([[2, 3, 4, 5]], dtype=np.int64)),
    ]
    cell_data = {
        "gmsh:physical": [
            np.asarray([sp.PHYS_SOURCE_LINE]),
            np.asarray([sp.PHYS_SURFACE]),
            np.asarray([sp.PHYS_EARTH]),
        ],
        "gmsh:geometrical": [np.asarray([1]), np.asarray([2]), np.asarray([3])],
    }
    field_data = {
        "source_wire": np.asarray([sp.PHYS_SOURCE_LINE, 1]),
        "air_earth_interface": np.asarray([sp.PHYS_SURFACE, 2]),
        "earth": np.asarray([sp.PHYS_EARTH, 3]),
    }

    compact = sp._dolfinx_companion_mesh_data(
        points, cell_blocks, cell_data, field_data
    )

    assert [block[0] for block in compact["cells"]] == ["triangle", "tetra"]
    assert compact["points"].shape == (4, 3)
    np.testing.assert_array_equal(compact["cells"][1][1], np.asarray([[0, 1, 2, 3]]))
    assert "source_wire" not in compact["field_data"]
    assert compact["field_data"]["earth"].tolist() == [sp.PHYS_EARTH, 3]


def test_tetra_quality_is_one_for_regular_tetrahedron():
    sp = _load_pipeline_module()
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0, 0.0],
            [0.5, math.sqrt(3.0) / 6.0, math.sqrt(2.0 / 3.0)],
        ]
    )

    metrics = sp._tetra_quality_metrics(vertices)

    assert metrics["quality_3r_over_R"] == pytest.approx(1.0)
    assert metrics["aspect_R_over_3r"] == pytest.approx(1.0)
    assert metrics["volume"] > 0.0


def test_tetra_quality_detects_sliver():
    sp = _load_pipeline_module()
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.4, 1.0e-8]]
    )

    metrics = sp._tetra_quality_metrics(vertices)

    assert metrics["quality_3r_over_R"] < 1.0e-6
    assert metrics["aspect_R_over_3r"] > 1.0e6


@pytest.mark.parametrize(
    "vertices, message",
    [
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "inverted",
        ),
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.25, 0.0]]
            ),
            "zero-volume",
        ),
    ],
)
def test_tetra_quality_rejects_nonpositive_or_inverted_cells(vertices, message):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match=message):
        sp._tetra_quality_metrics(vertices)


def test_local_mesh_quality_gate_uses_fixed_thresholds_and_fails_closed():
    sp = _load_pipeline_module()
    summary = {
        "selections": {
            "source_hit_cells": {"quality_3r_over_R": {"min": 0.009}, "aspect_R_over_3r": {"max": 80.0}, "all_positive_volume": True},
            "receiver_colliding_or_nearest_cells": {"quality_3r_over_R": {"min": 0.5}, "aspect_R_over_3r": {"max": 2.0}, "all_positive_volume": True},
            "interface_source_receiver_patch": {"quality_3r_over_R": {"min": 0.5}, "aspect_R_over_3r": {"max": 2.0}, "all_positive_volume": True},
        }
    }

    gated = sp._apply_local_mesh_quality_gate(summary)

    assert gated["thresholds"] == {"min_quality_3r_over_R": 0.01, "max_aspect_R_over_3r": 100.0}
    assert gated["passed"] is False
    assert gated["selections"]["source_hit_cells"]["gate_passed"] is False
    with pytest.raises(RuntimeError, match="local tetra quality gate failed"):
        sp._require_local_mesh_quality_gate(gated)


def test_physical_tag_counts_require_air_earth_outer_and_interface():
    sp = _load_pipeline_module()

    class Tags:
        def __init__(self, values):
            self.values = values

        def find(self, tag):
            return np.arange(self.values.get(tag, 0), dtype=np.int32)

    counts = sp._physical_tag_counts(
        Tags({sp.PHYS_AIR: 5, sp.PHYS_EARTH: 7}),
        Tags({sp.PHYS_OUTER: 11, sp.PHYS_SURFACE: 13}),
    )

    assert counts == {
        "air_cells": 5,
        "earth_cells": 7,
        "outer_boundary_facets": 11,
        "air_earth_interface_facets": 13,
    }
    with pytest.raises(RuntimeError, match="air_earth_interface_facets"):
        sp._physical_tag_counts(
            Tags({sp.PHYS_AIR: 5, sp.PHYS_EARTH: 7}),
            Tags({sp.PHYS_OUTER: 11, sp.PHYS_SURFACE: 0}),
        )


def test_physical_tag_counts_accept_empty_interior_rank_via_global_sum():
    sp = _load_pipeline_module()

    class Tags:
        def __init__(self, values):
            self.values = values

        def find(self, tag):
            return np.arange(self.values.get(tag, 0), dtype=np.int32)

    class Comm:
        def __init__(self):
            self.global_counts = iter((50, 70, 11, 13))

        def allreduce(self, _local_count):
            return next(self.global_counts)

    counts = sp._physical_tag_counts(
        Tags({sp.PHYS_AIR: 5, sp.PHYS_EARTH: 7}),
        Tags({sp.PHYS_OUTER: 0, sp.PHYS_SURFACE: 0}),
        comm=Comm(),
    )

    assert counts == {
        "air_cells": 50,
        "earth_cells": 70,
        "outer_boundary_facets": 11,
        "air_earth_interface_facets": 13,
    }


def test_source_only_artifacts_include_mesh_quality_and_preflight(tmp_path):
    sp = _load_pipeline_module()
    quality = {
        "passed": True,
        "thresholds": {"min_quality_3r_over_R": 0.01, "max_aspect_R_over_3r": 100.0},
        "selections": {
            "source_hit_cells": {
                "selection_definition": "actual source quadrature hit cells",
                "selection_count": 4,
                "quality_3r_over_R": {"min": 0.2, "p01": 0.21, "median": 0.5, "max": 0.9},
                "aspect_R_over_3r": {"min": 1.1, "p01": 1.2, "median": 2.0, "max": 5.0},
                "all_positive_volume": True,
                "gate_passed": True,
            }
        },
    }
    preflight = {
        "mesh_sha256": "abc",
        "dolfinx_mesh_sha256": "def",
        "physical_tags": {
            "air_cells": 5,
            "earth_cells": 7,
            "outer_boundary_facets": 11,
            "air_earth_interface_facets": 13,
        },
        "global_cells": 10,
        "global_nedelec_dofs": 20,
        "memory": {"estimated_gb": 0.1, "ok": True},
        "receiver": {"colliding_cell_count": 1, "selection_mode": "colliding"},
        "polarization": {"mode": "none", "polarizable_cell_count": None, "debye_fit_relative_l2": None},
    }

    sp.write_source_only_diagnostics(
        sp.PipelineConfig(workdir=tmp_path, source_only=True, ramp_off_time=0.0),
        env={"python": "test-python"},
        source_info={"mode": "manual_line"},
        mesh_quality=quality,
        preflight=preflight,
    )

    payload = json.loads((tmp_path / "source_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["mesh_quality"]["passed"] is True
    assert payload["preflight"]["receiver"]["colliding_cell_count"] == 1
    report = (tmp_path / "source_diagnostics_report.txt").read_text(encoding="utf-8")
    assert "local tetra quality gate: PASS" in report
    assert "global Nedelec DOFs: 20" in report


def test_mesh_contract_identity_changes_reuse_decision(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    config.mesh_path().write_text("old embedded mesh", encoding="utf-8")
    config.dolfinx_mesh_path().write_text("volume mesh", encoding="utf-8")

    assert sp._mesh_contract_matches(config) is False
    sp._write_mesh_contract(config)
    assert sp._mesh_contract_matches(config) is True
