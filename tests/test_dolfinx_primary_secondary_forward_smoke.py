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


def test_dolfinx_primary_secondary_ip_forward_runs_state_stepper():
    from atem3d.materials.prony import DebyeTerm, PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    material = PronyConductivity(
        sigma_inf=0.02,
        terms=[DebyeTerm(delta_sigma=0.003, tau=0.1)],
    )
    times = np.array([1.0e-5, 2.0e-5])
    dt = float(times[0])
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
    sigma_eff = fem.Function(spaces["Q"], name="sigma_eff")
    sigma_eff.x.array[:] = material.sigma_eff(dt)
    sigma_eff.x.scatter_forward()
    sigma_initial = fem.Function(spaces["Q"], name="sigma_initial")
    sigma_initial.x.array[:] = material.sigma0
    sigma_initial.x.scatter_forward()
    sigma_infinity = fem.Function(spaces["Q"], name="sigma_infinity")
    sigma_infinity.x.array[:] = material.sigma_inf
    sigma_infinity.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma_eff,
        "sigma_initial": sigma_initial,
        "sigma_infinity": sigma_infinity,
        "mu_inv": mu_inv,
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config)
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
        material=material,
        sigma_background=0.01,
        secondary_state_initializer=adapters["secondary_state_initializer"],
        secondary_step_solver=adapters["secondary_step_solver"],
        secondary_receiver_projector=adapters["secondary_receiver_projector"],
        secondary_state_stepper=adapters["secondary_state_stepper"],
    )

    predicted = operator.forward(times)

    assert predicted.shape == (2, 3)
    assert np.all(np.isfinite(predicted))
    assert np.linalg.norm(predicted - np.array([[10.0, 1.0, 0.0], [5.0, 0.5, 0.0]])) > 0.0
    assert len(adapters["diagnostics"]["chi"]) == 1
    assert adapters["diagnostics"]["dc_result"] is not None
    assert adapters["diagnostics"]["dc_result"]["contrast_is_zero"] is False


def test_dolfinx_primary_secondary_spatial_ip_forward_uses_state_initializer():
    from atem3d.materials.prony import DebyeTerm, PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    material = PronyConductivity(
        sigma_inf=0.02,
        terms=[DebyeTerm(delta_sigma=0.003, tau=0.1)],
    )
    times = np.array([1.0e-5, 2.0e-5])
    dt = float(times[0])
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
    delta_fn = fem.Function(spaces["Q"], name="delta_sigma_variable")
    delta_fn.x.array[:] = np.linspace(0.0, 0.003, delta_fn.x.array.size)
    delta_fn.x.scatter_forward()
    sigma_initial = fem.Function(spaces["Q"], name="sigma_initial_variable_ip")
    sigma_initial.x.array[:] = material.sigma_inf - delta_fn.x.array
    sigma_initial.x.scatter_forward()
    sigma_eff = fem.Function(spaces["Q"], name="sigma_eff_variable_ip")
    sigma_eff.x.array[:] = material.sigma_inf - float(material.beta(dt)[0]) * delta_fn.x.array
    sigma_eff.x.scatter_forward()
    sigma_infinity = fem.Function(spaces["Q"], name="sigma_infinity")
    sigma_infinity.x.array[:] = material.sigma_inf
    sigma_infinity.x.scatter_forward()
    mu_inv = fem.Function(spaces["Q"], name="mu_inv")
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {
        "sigma": sigma_eff,
        "sigma_initial": sigma_initial,
        "sigma_infinity": sigma_infinity,
        "mu_inv": mu_inv,
    }
    debye = {
        "terms": material.terms,
        "delta_functions": [delta_fn],
    }
    operators = sp.assemble_operators(msh, spaces, materials, facet_tags=None, config=config, debye=debye)
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
        debye=debye,
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=("Ex", "Ey", "dBzdt"),
        material=material,
        sigma_background=0.01,
        secondary_state_initializer=adapters["secondary_state_initializer"],
        secondary_step_solver=adapters["secondary_step_solver"],
        secondary_receiver_projector=adapters["secondary_receiver_projector"],
        secondary_state_stepper=adapters["secondary_state_stepper"],
    )

    predicted = operator.forward(times)

    assert predicted.shape == (2, 3)
    assert np.all(np.isfinite(predicted))
    assert np.linalg.norm(predicted - np.array([[10.0, 1.0, 0.0], [5.0, 0.5, 0.0]])) > 0.0
    assert len(adapters["diagnostics"]["chi"]) == 1
    assert adapters["diagnostics"]["dc_result"] is not None
    assert adapters["diagnostics"]["dc_result"]["contrast_is_zero"] is False


def test_dolfinx_primary_secondary_operator_helper_samples_provider_on_nedelec_points():
    from atem3d.materials.prony import PronyConductivity
    from atem3d.primary import ZeroPrimaryProvider
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
    expected_points = sp._nedelec_interpolation_points(msh, spaces)["points"]
    seen = {}

    class RecordingPrimaryProvider(ZeroPrimaryProvider):
        def get_Ep_on_V(self, t, points):
            seen.setdefault("transient_points", []).append(np.asarray(points, dtype=float).copy())
            return np.column_stack((np.ones(len(points)), np.zeros(len(points)), np.zeros(len(points))))

        def get_Ep_dc_on_V(self, points):
            seen["dc_points"] = np.asarray(points, dtype=float).copy()
            return np.column_stack((np.ones(len(points)), np.zeros(len(points)), np.zeros(len(points))))

        def get_receiver_E(self, t, receivers):
            return np.array([[10.0, 1.0, 0.0]])

        def get_receiver_dBdt(self, t, receivers):
            return np.zeros((1, 3))

    built = sp._make_dolfinx_primary_secondary_forward_operator(
        msh,
        spaces,
        materials,
        operators,
        config,
        primary=RecordingPrimaryProvider(),
        receiver_locations=np.array([[0.25, 0.25, 0.25]]),
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.02),
        sigma_background=0.01,
    )

    predicted = built["operator"].forward(np.array([1.0e-5]))

    assert predicted.shape == (1, 3)
    np.testing.assert_allclose(seen["dc_points"], expected_points)
    np.testing.assert_allclose(seen["transient_points"][0], expected_points)
    np.testing.assert_allclose(built["fem_points"], expected_points)
