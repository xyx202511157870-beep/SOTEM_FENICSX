from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_mpi_boundary_dof_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeIndexMap:
    size_local = 3
    num_ghosts = 2

    def __init__(self):
        self._global = np.asarray([10, 11, 12, 20, 21], dtype=np.int64)

    def local_to_global(self, local):
        return self._global[np.asarray(local, dtype=np.int32)]


class _FakeComm:
    size = 2

    def allgather(self, local_global):
        np.testing.assert_array_equal(local_global, np.asarray([10], dtype=np.int64))
        return [local_global, np.asarray([21, 21], dtype=np.int64)]


class _FakeCountComm:
    def allreduce(self, local_count):
        assert local_count == 0
        return 17


class _FakeVector:
    def __init__(self):
        self.set_values_calls = 0
        self.assemble_calls = 0

    def setValues(self, *_args):
        self.set_values_calls += 1

    def assemble(self):
        self.assemble_calls += 1


class _FakeQualityComm:
    rank = 0
    size = 2

    def allgather(self, local_rows):
        assert local_rows == []
        return [
            [],
            [
                {
                    "owner_rank": 1,
                    "local_cell_id": 4,
                    "global_cell_id": 99,
                    "metrics": {
                        "volume": 2.0,
                        "inradius": 0.2,
                        "circumradius": 0.5,
                        "quality_3r_over_R": 0.8,
                        "aspect_R_over_3r": 1.25,
                    },
                }
            ],
        ]


class _FakeVectorSumComm:
    size = 2

    def allreduce(self, local):
        return np.asarray(local, dtype=float) + np.asarray([1.0, 2.0, 3.0])


class _FakeCellIndexMap:
    def local_to_global(self, local):
        return np.asarray(local, dtype=np.int64) + 1000


class _FakeTopology:
    dim = 3

    def index_map(self, _dim):
        return _FakeCellIndexMap()


class _FakeQualityMesh:
    comm = _FakeQualityComm()
    topology = _FakeTopology()


class _PermutedDG0DofMap:
    bs = 1

    def cell_dofs(self, cell):
        return np.asarray([1 - int(cell)], dtype=np.int32)


class _FakeDG0Function:
    x = type("X", (), {"array": np.asarray([100.0, 200.0])})()
    function_space = type("Space", (), {"dofmap": _PermutedDG0DofMap()})()


class _MisalignedCellIndexMap:
    size_local = 1


class _MisalignedConnectivity:
    def links(self, _cell):
        return np.asarray([0, 1, 2, 3], dtype=np.int32)


class _MisalignedTopology:
    dim = 3

    def create_connectivity(self, *_args):
        return None

    def connectivity(self, *_args):
        return _MisalignedConnectivity()

    def index_map(self, _dim):
        return _MisalignedCellIndexMap()


class _MisalignedGeometry:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, 0.0, 1.0],
        ]
    )
    dofmap = np.asarray([[4, 5, 6, 7]], dtype=np.int32)


class _MisalignedGeometryMesh:
    topology = _MisalignedTopology()
    geometry = _MisalignedGeometry()


def test_collective_boundary_dof_closure_adds_remote_ghost_copy():
    sp = _load_pipeline_module()

    dofs = sp._collective_boundary_dof_closure(
        _FakeComm(),
        _FakeIndexMap(),
        np.asarray([0], dtype=np.int32),
    )

    np.testing.assert_array_equal(dofs, np.asarray([0, 4], dtype=np.int32))


def test_collective_entity_count_accepts_empty_interior_rank():
    sp = _load_pipeline_module()

    assert sp._collective_entity_count(_FakeCountComm(), 0) == 17


def test_zero_rhs_entries_assembles_on_rank_without_local_boundary_dofs():
    sp = _load_pipeline_module()
    vector = _FakeVector()

    sp._zero_rhs_entries(vector, np.empty(0, dtype=np.int64))

    assert vector.set_values_calls == 0
    assert vector.assemble_calls == 1


def test_tetra_quality_summary_accepts_empty_interior_rank_and_remote_rows():
    sp = _load_pipeline_module()

    summary = sp._summarize_tetra_cell_quality(
        _FakeQualityMesh(),
        [],
        selection_definition="distributed test selection",
    )

    assert summary["selection_count"] == 1
    assert summary["global_cell_ids"] == [99]
    assert summary["quality_3r_over_R"]["min"] == 0.8


def test_collective_vector_sum_adds_remote_biot_contribution():
    sp = _load_pipeline_module()

    total = sp._collective_vector_sum(
        _FakeVectorSumComm(),
        np.asarray([0.5, 1.0, 1.5]),
    )

    np.testing.assert_allclose(total, np.asarray([1.5, 3.0, 4.5]))


def test_dg0_values_at_cells_respect_distributed_dof_permutation():
    sp = _load_pipeline_module()

    values = sp._dg0_values_at_cells(
        _FakeDG0Function(),
        np.asarray([0, 1, 0], dtype=np.int32),
    )

    np.testing.assert_allclose(values, np.asarray([200.0, 100.0, 200.0]))


def test_tetra_quadrature_uses_geometry_dofmap_not_topology_vertex_ids():
    sp = _load_pipeline_module()

    points, cells, weights = sp._tetrahedron_quadrature_geometry(
        _MisalignedGeometryMesh(),
        degree=2,
    )

    assert np.min(points[:, 0]) > 9.0
    np.testing.assert_array_equal(cells, np.zeros(cells.size, dtype=np.int32))
    assert np.isclose(np.sum(weights), 1.0 / 6.0)
