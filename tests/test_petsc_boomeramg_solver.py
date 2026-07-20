import numpy as np
import pytest
import scipy.sparse as sp


def test_boomeramg_zero_rhs_is_an_exact_mode_without_backend_claim():
    from atem3d.solvers.petsc_boomeramg import PetscHypreBoomerAmgSolver

    solver = object.__new__(PetscHypreBoomerAmgSolver)
    solver.destroyed = False
    solver.matrix = sp.eye(3, format="csr")
    solver.ksp_type = "cg"
    solver.tolerance = 1.0e-8
    solver.internal_tolerance = 1.0e-11
    solver.last_diagnostics = None

    solution = solver.solve(np.zeros(3))

    np.testing.assert_array_equal(solution, np.zeros(3))
    assert solver.last_diagnostics["solve_mode"] == "exact_zero_rhs"
    assert solver.last_diagnostics["backend_reason"] == 0
    assert solver.last_diagnostics["backend_reported_converged"] is False
    assert solver.last_diagnostics["backend_iterations"] == 0
    assert solver.last_diagnostics["external_true_relative_residual"] == 0.0


def test_boomeramg_external_gate_failure_names_actual_solver():
    from atem3d.solvers.petsc_ams import require_true_residual

    with pytest.raises(RuntimeError) as exc_info:
        require_true_residual(
            sp.eye(2, format="csr"),
            np.ones(2),
            np.zeros(2),
            tolerance=1.0e-8,
            backend_reason=2,
            backend_iterations=1,
            solver="petsc_ksp_hypre_boomeramg",
            diagnostic_schema="atem3d.petsc-boomeramg-convergence-diagnostic",
        )

    assert exc_info.value.diagnostics["solver"] == "petsc_ksp_hypre_boomeramg"
    assert exc_info.value.diagnostics["diagnostic_schema"] == (
        "atem3d.petsc-boomeramg-convergence-diagnostic"
    )


def test_petsc_hypre_boomeramg_solves_small_spd_system_with_external_residual():
    pytest.importorskip("petsc4py")

    from atem3d.solvers.petsc_boomeramg import PetscHypreBoomerAmgSolver

    matrix = sp.diags(
        [
            -np.ones(63),
            2.5 * np.ones(64),
            -np.ones(63),
        ],
        offsets=[-1, 0, 1],
        format="csr",
    )
    exact = np.sin(np.arange(64, dtype=float) + 0.25)
    rhs = matrix @ exact
    solver = PetscHypreBoomerAmgSolver(
        matrix,
        tolerance=1.0e-8,
        internal_tolerance=1.0e-11,
        maxiter=500,
        refinement_steps=2,
        ksp_type="cg",
    )

    computed = solver.solve(rhs)
    diagnostic = dict(solver.last_diagnostics)
    solver.destroy()

    assert np.all(np.isfinite(computed))
    assert diagnostic["solver"] == "petsc_ksp_hypre_boomeramg"
    assert diagnostic["solve_mode"] == "petsc_ksp"
    assert diagnostic["backend_reason"] > 0
    assert diagnostic["backend_reported_converged"] is True
    assert diagnostic["external_true_relative_residual"] <= 1.0e-8
    assert solver.destroyed is True


def test_petsc_boomeramg_native_state_is_released_in_dependency_order():
    from atem3d.solvers.petsc_boomeramg import PetscHypreBoomerAmgSolver

    release_order = []

    class NativeProbe:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            release_order.append(self.name)

    solver = object.__new__(PetscHypreBoomerAmgSolver)
    solver._ksp = NativeProbe("ksp")
    solver._petsc_matrix = NativeProbe("matrix")
    solver._matrix_csr_buffers = (np.array([1.0]),)

    solver._destroy_native_state(suppress_errors=False)

    assert release_order == ["ksp", "matrix"]
    assert solver._ksp is None
    assert solver._petsc_matrix is None
    assert solver._matrix_csr_buffers == ()


def test_petsc_boomeramg_solve_destroys_allocated_vector_on_partial_failure():
    from atem3d.solvers.petsc_boomeramg import PetscHypreBoomerAmgSolver

    class VectorProbe:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    first = VectorProbe()

    class MatrixProbe:
        def __init__(self):
            self.calls = 0

        def createVecRight(self):
            self.calls += 1
            if self.calls == 1:
                return first
            raise RuntimeError("synthetic second vector allocation failure")

    solver = object.__new__(PetscHypreBoomerAmgSolver)
    solver._petsc_matrix = MatrixProbe()

    with pytest.raises(RuntimeError, match="synthetic second vector allocation failure"):
        solver._solve_once(np.ones(2))

    assert first.destroyed is True
