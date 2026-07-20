from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_curl_space_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSpace:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.dofmap = SimpleNamespace(index_map=SimpleNamespace(size_global=1))


@pytest.mark.parametrize("nedelec_order, expected_curl_degree", [(1, 0), (2, 1)])
def test_build_function_spaces_matches_curl_degree_without_changing_shared_dg0(
    monkeypatch, nedelec_order, expected_curl_degree
):
    sp = _load_pipeline_module()
    calls = []

    def functionspace(_msh, descriptor):
        calls.append(descriptor)
        return _FakeSpace(descriptor)

    fake_fem = SimpleNamespace(functionspace=functionspace)
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(fem=fake_fem))
    msh = SimpleNamespace(comm=SimpleNamespace(rank=0))

    spaces = sp.build_function_spaces(msh, sp.PipelineConfig(nedelec_order=nedelec_order))

    assert spaces["W"].descriptor == ("DG", 0, (3,))
    assert spaces["W_curl"].descriptor == ("DG", expected_curl_degree, (3,))
    assert spaces["W_curl"] is not spaces["W"]
    assert ("DG", expected_curl_degree, (3,)) in calls


def test_compute_dbdt_uses_dedicated_curl_observation_space(monkeypatch):
    sp = _load_pipeline_module()
    shared_dg0 = SimpleNamespace(
        name="shared-current-dg0",
        element=SimpleNamespace(interpolation_points=lambda: np.zeros((1, 3))),
    )
    curl_space = SimpleNamespace(
        name="curl-observation",
        element=SimpleNamespace(interpolation_points=lambda: np.zeros((1, 3))),
    )
    created_on = []

    class _FakeFunction:
        def __init__(self, space, name):
            created_on.append(space)
            self.name = name
            self.x = SimpleNamespace(scatter_forward=lambda: None)

        def interpolate(self, _expr):
            return None

    fake_fem = SimpleNamespace(
        Function=_FakeFunction,
        Expression=lambda expression, points, comm: (expression, points, comm),
    )
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(fem=fake_fem))
    class _ExpressionTerm:
        def __neg__(self):
            return self

    fake_ufl = SimpleNamespace(curl=lambda _field: _ExpressionTerm())
    monkeypatch.setitem(sys.modules, "ufl", fake_ufl)
    E = SimpleNamespace(function_space=SimpleNamespace(mesh=SimpleNamespace(comm="comm")))

    result = sp.compute_dbdt(E, {"W": shared_dg0, "W_curl": curl_space})

    assert created_on == [curl_space]
    assert result.name == "dBdt"


def test_nedelec_order_two_linear_curl_is_recovered_exactly_and_beats_dg0():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    spaces = sp.build_function_spaces(msh, sp.PipelineConfig(nedelec_order=2))
    E = fem.Function(spaces["V"], name="quadratic_nedelec_field")
    E.interpolate(
        lambda x: np.vstack(
            (
                -x[0] * x[1],
                x[0] ** 2,
                np.zeros_like(x[0]),
            )
        )
    )
    E.x.scatter_forward()

    point = np.asarray([0.211, 0.327, 0.463], dtype=float)
    cell = sp._find_cell_for_point(msh, point)
    assert cell is not None
    eval_point = point.reshape(1, 3)
    eval_cell = np.asarray([cell], dtype=np.int32)
    expected = np.asarray([0.0, 0.0, -3.0 * point[0]])

    recovered = sp.compute_dbdt(E, spaces)
    recovered_value = np.asarray(recovered.eval(eval_point, eval_cell), dtype=float).reshape(-1)

    old_dg0 = fem.Function(spaces["W"], name="old_dg0_dBdt")
    old_expr = fem.Expression(
        -ufl.curl(E),
        spaces["W"].element.interpolation_points(),
        comm=msh.comm,
    )
    old_dg0.interpolate(old_expr)
    old_dg0.x.scatter_forward()
    old_value = np.asarray(old_dg0.eval(eval_point, eval_cell), dtype=float).reshape(-1)

    np.testing.assert_allclose(recovered_value, expected, rtol=1.0e-11, atol=1.0e-12)
    assert np.linalg.norm(recovered_value - expected) < 1.0e-6 * np.linalg.norm(old_value - expected)
