from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("dolfinx.fem")
pytest.importorskip("dolfinx.mesh")


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_secondary_step_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dolfinx_secondary_step_solver_zero_rhs_returns_zero_samples():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=200,
    )
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = 0.01
    sigma.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma,
        "sigma_infinity": sigma,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
    solver = sp._make_dolfinx_secondary_step_solver(spaces, operators, config)
    rhs_samples = np.zeros((2, 3))

    solved = solver(rhs_samples, 0.01, 1.0e-5)

    np.testing.assert_allclose(solved, rhs_samples, atol=1.0e-14)


def test_dolfinx_secondary_step_solver_accepts_constant_sample_rhs_hook():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=200,
    )
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = 0.01
    sigma.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma,
        "sigma_infinity": sigma,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
    sample_points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
        ]
    )
    rhs_to_function = sp._make_nedelec_rhs_interpolator_from_samples(spaces)
    solution_to_samples = sp._make_nedelec_solution_sampler_at_points(msh, sample_points)

    solver = sp._make_dolfinx_secondary_step_solver(
        spaces,
        operators,
        config,
        rhs_to_function=rhs_to_function,
        solution_to_samples=solution_to_samples,
    )
    rhs_samples = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    solved = solver(rhs_samples, 0.01, 1.0e-5)

    assert solved.shape == rhs_samples.shape
    assert np.all(np.isfinite(solved))
    assert np.max(np.abs(solved)) > 0.0


def test_nedelec_solution_sampler_returns_values_at_points():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig()
    spaces = sp.build_function_spaces(msh, config)
    field = fem.Function(spaces["V"], name="constant_field")
    field.interpolate(lambda x: np.vstack((2.0 * np.ones(x.shape[1]), -1.0 * np.ones(x.shape[1]), 0.5 * np.ones(x.shape[1]))))
    field.x.scatter_forward()
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
        ]
    )
    sampler = sp._make_nedelec_solution_sampler_at_points(msh, points)

    values = sampler(field, np.zeros((2, 3)))

    np.testing.assert_allclose(values, [[2.0, -1.0, 0.5], [2.0, -1.0, 0.5]], atol=1.0e-12)
