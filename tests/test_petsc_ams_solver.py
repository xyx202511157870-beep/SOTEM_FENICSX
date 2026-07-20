import numpy as np
import pytest
import scipy.sparse as sp


def test_petsc_aij_destroys_matrix_when_assembly_fails():
    from atem3d.solvers.petsc_ams import _petsc_aij_from_csr

    class MatrixProbe:
        def __init__(self):
            self.destroyed = False

        def createAIJ(self, **_kwargs):
            return self

        def assemble(self):
            raise RuntimeError("synthetic matrix assembly failure")

        def destroy(self):
            self.destroyed = True

    matrix_probe = MatrixProbe()

    class PetscProbe:
        IntType = np.int64
        ScalarType = np.float64
        COMM_SELF = object()

        @staticmethod
        def Mat():
            return matrix_probe

    with pytest.raises(RuntimeError, match="synthetic matrix assembly failure"):
        _petsc_aij_from_csr(sp.eye(2, format="csr"), PetscProbe)

    assert matrix_probe.destroyed is True


def test_external_true_residual_gate_rejects_backend_success_above_tolerance():
    from atem3d.solvers.petsc_ams import require_true_residual

    matrix = sp.diags([1.0, 4.0], format="csr")
    rhs = np.array([1.0, 1.0])
    inaccurate = np.array([1.0, 0.20])

    with pytest.raises(
        RuntimeError,
        match="external_true_residual_above_tolerance",
    ) as exc_info:
        require_true_residual(
            matrix,
            rhs,
            inaccurate,
            tolerance=1.0e-8,
            backend_reason=2,
            backend_iterations=3,
        )

    assert exc_info.value.diagnostics["backend_reported_converged"] is True
    assert exc_info.value.diagnostics["external_true_relative_residual"] > 1.0e-8


def test_tensor_mesh_edge_constants_are_discrete_gradients_of_coordinates():
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.solvers.petsc_ams import tensor_mesh_ams_auxiliary_space

    mesh = TensorMesh([[(2.0, 3)], [(3.0, 2)], [(5.0, 2)]], x0="CCC")
    gradient, constants = tensor_mesh_ams_auxiliary_space(mesh)

    assert gradient.shape == (mesh.n_edges, mesh.n_nodes)
    assert len(constants) == 3
    for axis, vector in enumerate(constants):
        expected = gradient @ np.asarray(mesh.nodes[:, axis], dtype=float)
        np.testing.assert_allclose(vector, expected, rtol=0.0, atol=0.0)
        assert vector.shape == (mesh.n_edges,)
        assert np.linalg.norm(vector) > 0.0


def test_petsc_hypre_ams_solves_small_tensor_curl_curl_mass_system():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.solvers.petsc_ams import (
        PetscHypreAmsSolver,
        tensor_mesh_ams_auxiliary_space,
    )

    mesh = TensorMesh([[(1.0, 4)], [(1.0, 3)], [(1.0, 3)]], x0="CCC")
    curl = mesh.edge_curl.tocsr()
    face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    edge_mass = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).tocsr()
    matrix = (curl.T @ face_mass @ curl + 0.25 * edge_mass).tocsr()
    gradient, constants = tensor_mesh_ams_auxiliary_space(mesh)
    exact = np.sin(np.arange(mesh.n_edges, dtype=float) + 0.25)
    rhs = matrix @ exact

    solver = PetscHypreAmsSolver(
        matrix,
        nodal_gradient=gradient,
        edge_constant_vectors=constants,
        tolerance=1.0e-8,
        internal_tolerance=1.0e-11,
        maxiter=500,
    )
    computed = solver.solve(rhs)
    solver.destroy()

    relative_residual = np.linalg.norm(rhs - matrix @ computed) / np.linalg.norm(rhs)
    assert relative_residual <= 1.0e-8
    assert solver.last_diagnostics["backend_reported_converged"] is True
    assert solver.last_diagnostics["external_true_relative_residual"] <= 1.0e-8
    assert solver.last_diagnostics["pc_type"] == "hypre_ams"


def test_petsc_hypre_ams_solves_medium_tensor_curl_curl_mass_system():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.solvers.petsc_ams import (
        PetscHypreAmsSolver,
        tensor_mesh_ams_auxiliary_space,
    )

    mesh = TensorMesh([np.ones(16), np.ones(18), np.ones(20)], x0="CCC")
    assert mesh.n_cells == 5760
    assert mesh.n_edges == 19270
    curl = mesh.edge_curl.tocsr()
    face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    edge_mass = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).tocsr()
    matrix = (curl.T @ face_mass @ curl + 0.25 * edge_mass).tocsr()
    gradient, constants = tensor_mesh_ams_auxiliary_space(mesh)
    exact = np.sin(np.arange(mesh.n_edges, dtype=float) + 0.25)
    rhs = matrix @ exact

    solver = PetscHypreAmsSolver(
        matrix,
        nodal_gradient=gradient,
        edge_constant_vectors=constants,
        tolerance=1.0e-8,
        internal_tolerance=1.0e-11,
        maxiter=500,
    )
    computed = solver.solve(rhs)
    solver.destroy()

    relative_residual = np.linalg.norm(rhs - matrix @ computed) / np.linalg.norm(rhs)
    assert relative_residual <= 1.0e-8
    assert solver.last_diagnostics["backend_reported_converged"] is True
    assert solver.last_diagnostics["external_true_relative_residual"] <= 1.0e-8


def test_petsc_hypre_ams_solves_gauge_stabilized_ampere_initialization_system():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.solvers.petsc_ams import (
        PetscHypreAmsSolver,
        tensor_mesh_ams_auxiliary_space,
    )

    mesh = TensorMesh([np.ones(6), np.ones(5), np.ones(4)], x0="CCC")
    curl = mesh.edge_curl.tocsr()
    gradient = mesh.nodal_gradient.tocsr()
    face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    matrix = (curl.T @ face_mass @ curl + gradient @ gradient.T).tocsr()
    auxiliary_gradient, constants = tensor_mesh_ams_auxiliary_space(mesh)
    exact = np.sin(np.arange(mesh.n_edges, dtype=float) + 0.25)
    rhs = matrix @ exact

    solver = PetscHypreAmsSolver(
        matrix,
        nodal_gradient=auxiliary_gradient,
        edge_constant_vectors=constants,
        tolerance=1.0e-8,
        internal_tolerance=1.0e-11,
        maxiter=1000,
    )
    computed = solver.solve(rhs)
    solver.destroy()

    relative_residual = np.linalg.norm(rhs - matrix @ computed) / np.linalg.norm(rhs)
    assert np.all(np.isfinite(computed))
    assert relative_residual <= 1.0e-8
    assert solver.last_diagnostics["backend_reason"] > 0
    assert solver.last_diagnostics["backend_reported_converged"] is True


def test_medium_tdem_petsc_initialization_passes_algebraic_and_physical_gates():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.ip import DebyeIPModel
    from atem3d.simulation import TDEMIPSimulation
    from atem3d.sources import GroundedWireSource, StepOffWaveform

    mesh = TensorMesh([np.ones(20), np.ones(18), np.ones(16)], x0="CCC")
    source = GroundedWireSource(
        start=(-5.0, 0.0, 0.0),
        end=(5.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    simulation = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[1.0e-3],
        sources=[source],
        initialization_solver="petsc_hypre",
        initialization_tolerance=1.0e-8,
        initialization_internal_tolerance=1.0e-11,
        initialization_maxiter=1000,
    )

    electric = simulation.initial_electric_field()
    magnetic = simulation.initial_magnetic_flux_density(electric)

    assert mesh.n_cells == 5760
    assert mesh.n_edges == 19270
    assert np.all(np.isfinite(electric))
    assert np.all(np.isfinite(magnetic))
    assert [
        diagnostic["phase"]
        for diagnostic in simulation.initialization_solver_diagnostics
    ] == ["dc_electric", "ampere_magnetic"]
    assert all(
        diagnostic["backend_reason"] > 0
        and diagnostic["backend_reported_converged"] is True
        and diagnostic["external_true_relative_residual"] <= 1.0e-8
        and diagnostic["balance_relative_residual"] <= 1.0e-8
        for diagnostic in simulation.initialization_solver_diagnostics
    )


def test_petsc_ams_constructor_releases_partial_native_state(monkeypatch):
    pytest.importorskip("petsc4py")

    import atem3d.solvers.petsc_ams as petsc_ams

    class NativeMatrixProbe:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    allocated = NativeMatrixProbe()
    calls = {"count": 0}

    def fail_on_gradient(_matrix, _petsc):
        calls["count"] += 1
        if calls["count"] == 1:
            return allocated, (np.array([0, 1]), np.array([0]), np.array([1.0]))
        raise RuntimeError("synthetic gradient allocation failure")

    monkeypatch.setattr(petsc_ams, "_petsc_aij_from_csr", fail_on_gradient)

    with pytest.raises(RuntimeError, match="synthetic gradient allocation failure"):
        petsc_ams.PetscHypreAmsSolver(
            sp.eye(3, format="csr"),
            nodal_gradient=sp.eye(3, format="csr"),
            edge_constant_vectors=(np.ones(3), np.ones(3), np.ones(3)),
        )

    assert allocated.destroyed is True


def test_petsc_ams_native_state_is_released_in_dependency_order():
    from atem3d.solvers.petsc_ams import PetscHypreAmsSolver

    release_order = []

    class NativeProbe:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            release_order.append(self.name)

    solver = object.__new__(PetscHypreAmsSolver)
    solver._ksp = NativeProbe("ksp")
    solver._constant_vecs = (NativeProbe("constant_x"), NativeProbe("constant_y"))
    solver._petsc_gradient = NativeProbe("gradient")
    solver._petsc_matrix = NativeProbe("matrix")
    solver._gradient_csr_buffers = (np.array([1.0]),)
    solver._matrix_csr_buffers = (np.array([1.0]),)

    solver._destroy_native_state(suppress_errors=False)

    assert release_order == ["ksp", "constant_y", "constant_x", "gradient", "matrix"]
    assert solver._ksp is None
    assert solver._constant_vecs == ()
    assert solver._petsc_gradient is None
    assert solver._petsc_matrix is None
    assert solver._gradient_csr_buffers == ()
    assert solver._matrix_csr_buffers == ()


def test_petsc_ams_solve_destroys_first_vector_when_second_allocation_fails():
    from atem3d.solvers.petsc_ams import PetscHypreAmsSolver

    class VectorProbe:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    first_vector = VectorProbe()

    class MatrixProbe:
        def __init__(self):
            self.calls = 0

        def createVecRight(self):
            self.calls += 1
            if self.calls == 1:
                return first_vector
            raise RuntimeError("synthetic second vector allocation failure")

    solver = object.__new__(PetscHypreAmsSolver)
    solver._petsc_matrix = MatrixProbe()

    with pytest.raises(RuntimeError, match="synthetic second vector allocation failure"):
        solver._solve_once(np.ones(2))

    assert first_vector.destroyed is True


def test_petsc_ams_solve_cleans_both_vectors_without_masking_body_failure():
    from atem3d.solvers.petsc_ams import PetscHypreAmsSolver

    class VectorProbe:
        def __init__(self, *, get_array_error=None, destroy_error=None):
            self.get_array_error = get_array_error
            self.destroy_error = destroy_error
            self.destroyed = False

        def getArray(self):
            if self.get_array_error is not None:
                raise self.get_array_error
            return np.zeros(2)

        def destroy(self):
            self.destroyed = True
            if self.destroy_error is not None:
                raise self.destroy_error

    body_error = RuntimeError("synthetic solve body failure")
    b = VectorProbe(
        get_array_error=body_error,
        destroy_error=RuntimeError("synthetic b cleanup failure"),
    )
    x = VectorProbe()
    vectors = iter((b, x))

    class MatrixProbe:
        @staticmethod
        def createVecRight():
            return next(vectors)

    solver = object.__new__(PetscHypreAmsSolver)
    solver._petsc_matrix = MatrixProbe()

    with pytest.raises(RuntimeError, match="synthetic solve body failure") as exc_info:
        solver._solve_once(np.ones(2))

    assert exc_info.value is body_error
    assert b.destroyed is True
    assert x.destroyed is True


def test_tdem_simulation_petsc_ams_step_matches_sparse_direct_solution():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh
    from scipy.sparse.linalg import spsolve

    from atem3d.ip import DebyeIPModel
    from atem3d.simulation import TDEMIPSimulation

    mesh = TensorMesh([[(1.0, 4)], [(1.0, 3)], [(1.0, 3)]], x0="CCC")
    simulation = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[1.0e-3],
        linear_solver="petsc_ams",
        cg_preconditioner="hypre_ams",
        cg_tolerance=1.0e-8,
        cg_maxiter=500,
    )
    matrix = simulation.system_matrix(0)
    rhs = np.cos(np.arange(mesh.n_edges, dtype=float) + 0.5)

    computed = simulation._solver_for_step(0)(rhs)
    expected = spsolve(matrix.tocsc(), rhs)

    np.testing.assert_allclose(computed, expected, rtol=1.0e-7, atol=1.0e-10)
    assert np.linalg.norm(rhs - matrix @ computed) / np.linalg.norm(rhs) <= 1.0e-8
    assert simulation._petsc_ams_solver is not None
    simulation.close()


def test_tdem_petsc_ams_cache_holds_only_current_time_step_operator():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.ip import DebyeIPModel
    from atem3d.simulation import TDEMIPSimulation

    mesh = TensorMesh([[(1.0, 3)], [(1.0, 3)], [(1.0, 3)]], x0="CCC")
    simulation = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[1.0e-3, 2.0e-3],
        linear_solver="petsc_ams",
        cg_preconditioner="hypre_ams",
    )

    first = simulation._solver_for_step(0)
    assert simulation._solver_for_step(0) is first
    second = simulation._solver_for_step(1)

    assert second is not first
    assert first.destroyed is True
    assert simulation._factor_cache == {}
    simulation.close()
    assert second.destroyed is True
    assert simulation._petsc_ams_solver is None


def test_tdem_run_records_each_external_residual_and_releases_petsc_solver():
    pytest.importorskip("petsc4py")
    pytest.importorskip("discretize")

    from discretize import TensorMesh

    from atem3d.ip import DebyeIPModel
    from atem3d.simulation import TDEMIPSimulation

    mesh = TensorMesh([[(1.0, 3)], [(1.0, 3)], [(1.0, 3)]], x0="CCC")
    simulation = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[1.0e-3, 1.0e-3, 2.0e-3],
        linear_solver="petsc_ams",
        cg_preconditioner="hypre_ams",
    )

    simulation.run_data_only()

    assert simulation._petsc_ams_solver is None
    assert [item["step_index"] for item in simulation.linear_solver_diagnostics] == [
        0,
        1,
        2,
    ]
    assert all(
        item["external_true_relative_residual"] <= 1.0e-8
        for item in simulation.linear_solver_diagnostics
    )
