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
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_ps_forward_smoke", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dolfinx_primary_secondary_zero_contrast_forward_returns_primary_response():
    from atem3d.materials.prony import PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        receiver=(0.25, 0.25, 0.25),
        receiver_evaluation_mode="first_cell",
    )
    spaces = sp.build_function_spaces(msh, config)
    times = np.array([1.0e-5, 2.0e-5])
    fem_points = np.array([[0.25, 0.25, 0.25]])
    receiver_locations = np.array([[0.25, 0.25, 0.25]])
    primary = CachedPrimaryProvider(
        times=times,
        points=fem_points,
        receivers=receiver_locations,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]], [[5.0, 0.5, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]], [[0.0, 0.0, -1.5]]]),
        Ep_dc_on_V=np.array([[1.0, 0.0, 0.0]]),
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
        secondary_receiver_projector=sp._make_dolfinx_zero_secondary_receiver_projector(
            msh,
            spaces,
            config,
        ),
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.0, 1.0, -3.0], [5.0, 0.5, -1.5]], atol=1.0e-12)


def test_dolfinx_primary_secondary_nonzero_contrast_forward_runs_secondary_path():
    from atem3d.materials.prony import PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        receiver=(0.25, 0.25, 0.25),
        receiver_evaluation_mode="first_cell",
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=400,
    )
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = 0.02
    sigma.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma,
        "sigma_initial": sigma,
        "sigma_infinity": sigma,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
    times = np.array([1.0e-5, 2.0e-5])
    fem_points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
        ]
    )
    receiver_locations = np.array([[0.25, 0.25, 0.25]])
    primary = CachedPrimaryProvider(
        times=times,
        points=fem_points,
        receivers=receiver_locations,
        Ep_on_V=np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array([[[10.0, 1.0, 0.0]], [[5.0, 0.5, 0.0]]]),
        receiver_dBdt=np.zeros((2, 1, 3)),
        Ep_dc_on_V=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh,
        spaces,
        materials,
        operators,
        config,
        fem_points,
        sigma_background=0.01,
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.02),
        sigma_background=0.01,
        secondary_field_solver=adapters["secondary_field_solver"],
        secondary_step_solver=adapters["secondary_step_solver"],
        secondary_receiver_projector=adapters["secondary_receiver_projector"],
    )

    predicted = operator.forward(times)

    assert predicted.shape == (2, 3)
    assert np.all(np.isfinite(predicted))
    assert np.linalg.norm(predicted - np.array([[10.0, 1.0, 0.0], [5.0, 0.5, 0.0]])) > 0.0
    assert adapters["diagnostics"]["dc_result"] is not None
    assert adapters["diagnostics"]["dc_result"]["contrast_is_zero"] is False


def test_dolfinx_primary_secondary_variable_contrast_forward_runs_state_stepper():
    from atem3d.materials.prony import PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        receiver=(0.25, 0.25, 0.25),
        receiver_evaluation_mode="first_cell",
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=400,
    )
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma_variable")
    sigma.x.array[:] = np.linspace(0.01, 0.02, sigma.x.array.size)
    sigma.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma,
        "sigma_initial": sigma,
        "sigma_infinity": sigma,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
    times = np.array([1.0e-5, 2.0e-5])
    fem_points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
        ]
    )
    receiver_locations = np.array([[0.25, 0.25, 0.25]])
    primary = CachedPrimaryProvider(
        times=times,
        points=fem_points,
        receivers=receiver_locations,
        Ep_on_V=np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array([[[10.0, 1.0, 0.0]], [[5.0, 0.5, 0.0]]]),
        receiver_dBdt=np.zeros((2, 1, 3)),
        Ep_dc_on_V=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh,
        spaces,
        materials,
        operators,
        config,
        fem_points,
        sigma_background=0.01,
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.02),
        sigma_background=0.01,
        secondary_field_solver=adapters["secondary_field_solver"],
        secondary_step_solver=adapters["secondary_step_solver"],
        secondary_receiver_projector=adapters["secondary_receiver_projector"],
        secondary_state_stepper=adapters["secondary_state_stepper"],
    )

    predicted = operator.forward(times)

    assert predicted.shape == (2, 3)
    assert np.all(np.isfinite(predicted))
    assert np.linalg.norm(predicted - np.array([[10.0, 1.0, 0.0], [5.0, 0.5, 0.0]])) > 0.0
    assert adapters["diagnostics"]["dc_result"] is not None
    assert adapters["diagnostics"]["dc_result"]["contrast_is_zero"] is False
