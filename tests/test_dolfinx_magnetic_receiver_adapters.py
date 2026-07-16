from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load_operator_module():
    path = Path("dolfinx/magnetic_receiver_operators.py")
    spec = importlib.util.spec_from_file_location(
        "magnetic_receiver_adapters_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline_module():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_magnetic_adapters",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConstantField:
    def __init__(self, value: tuple[float, float, float]):
        self.value = np.asarray(value, dtype=float)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def eval(self, points, cells):
        point_array = np.asarray(points, dtype=float)
        cell_array = np.asarray(cells, dtype=np.int32)
        self.calls.append((point_array.copy(), cell_array.copy()))
        return np.repeat(self.value.reshape(1, 3), point_array.shape[0], axis=0)


def test_evaluate_faraday_loop_rejects_any_unlocated_point() -> None:
    module = load_operator_module()
    field = ConstantField((0.0, 0.0, 0.0))

    def locator(_mesh, point):
        return [] if point[0] > 0.0 else [0]

    with pytest.raises(RuntimeError, match="Faraday loop point"):
        module.evaluate_faraday_loop_field(
            field,
            object(),
            center=(0.0, 0.0, 0.1),
            radius=2.0,
            point_count=32,
            find_cells=locator,
        )

    assert field.calls == []


def test_evaluate_faraday_loop_uses_smallest_shared_cell_candidate() -> None:
    module = load_operator_module()
    field = ConstantField((1.0, 0.0, 0.0))

    value, audit = module.evaluate_faraday_loop_field(
        field,
        object(),
        center=(0.0, 0.0, 0.1),
        radius=2.0,
        point_count=8,
        find_cells=lambda _mesh, _point: [7, 3, 5],
    )

    assert abs(value) < 1.0e-14
    assert audit == {
        "method": "faraday_loop",
        "radius_m": 2.0,
        "point_count": 8,
        "located_point_count": 8,
        "missing_point_indices": [],
    }
    assert len(field.calls) == 1
    np.testing.assert_array_equal(field.calls[0][1], np.full(8, 3, dtype=np.int32))


def test_evaluate_biot_current_field_uses_supplied_cells_and_weights() -> None:
    module = load_operator_module()
    current = ConstantField((1.0, 0.0, 0.0))

    value, audit = module.evaluate_biot_current_field(
        current,
        receiver=(0.0, 0.0, 0.0),
        points=np.array([[0.0, 1.0, 0.0]]),
        cells=np.array([9], dtype=np.int32),
        weights=np.array([2.0]),
    )

    assert value[2] < 0.0
    assert audit["method"] == "biot_tetra4"
    assert audit["sample_count"] == 1
    np.testing.assert_array_equal(current.calls[0][1], [9])


def test_dolfinx_constant_current_adapter_is_finite() -> None:
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")

    from dolfinx import fem, mesh
    from mpi4py import MPI

    module = load_operator_module()
    sp = load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    spaces = sp.build_function_spaces(msh, sp.PipelineConfig())
    current = fem.Function(spaces["V"])
    current.interpolate(
        lambda x: np.vstack(
            (
                np.ones(x.shape[1]),
                np.zeros(x.shape[1]),
                np.zeros(x.shape[1]),
            )
        )
    )
    points, cells, weights = sp._cell_biot_quadrature_points_weights(msh)

    value, audit = module.evaluate_biot_current_field(
        current,
        receiver=(0.5, 1.5, 0.5),
        points=points,
        cells=cells,
        weights=weights,
    )

    assert value.shape == (3,)
    assert np.all(np.isfinite(value))
    assert value[2] > 0.0
    assert audit["sample_count"] == len(points)
