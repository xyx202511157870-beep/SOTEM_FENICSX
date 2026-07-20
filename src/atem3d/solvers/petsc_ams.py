"""Serial PETSc/HYPRE AMS adapter for Discretize edge systems.

The adapter keeps the SciPy matrix as the acceptance oracle.  PETSc's
convergence reason is necessary but never sufficient: every returned solution
must also pass an unpreconditioned ``||b - A x|| / ||b||`` gate evaluated by
SciPy/NumPy.
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp


class PetscAmsConvergenceError(RuntimeError):
    """Fail-closed PETSc/AMS error carrying machine-readable diagnostics."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostics = dict(diagnostic)
        super().__init__(
            "PETSc/HYPRE solver failed convergence gate: "
            + json.dumps(
                self.diagnostics,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer excluding bool")
    return int(value)


def _csr_float64(matrix: Any, *, name: str) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    if result.ndim != 2 or not np.all(np.isfinite(result.data)):
        raise ValueError(f"{name} must be a finite two-dimensional sparse matrix")
    return result


def tensor_mesh_ams_auxiliary_space(mesh: Any) -> tuple[sp.csr_matrix, tuple[np.ndarray, ...]]:
    """Return the TensorMesh discrete gradient and constant edge fields."""

    gradient = _csr_float64(mesh.nodal_gradient, name="nodal_gradient")
    nodes = np.asarray(mesh.nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape != (gradient.shape[1], 3):
        raise ValueError("tensor mesh nodes must have shape (n_nodes, 3)")
    constants = tuple(
        np.ascontiguousarray(gradient @ nodes[:, axis], dtype=np.float64)
        for axis in range(3)
    )
    if any(vector.shape != (gradient.shape[0],) for vector in constants):
        raise ValueError("edge constant vectors must match the gradient row count")
    if any(not np.all(np.isfinite(vector)) for vector in constants):
        raise ValueError("edge constant vectors must be finite")
    return gradient, constants


def true_relative_residual(matrix: sp.spmatrix, rhs: Any, solution: Any) -> float:
    """Evaluate the unpreconditioned residual using SciPy/NumPy only."""

    rhs_array = np.asarray(rhs, dtype=float).reshape(-1)
    solution_array = np.asarray(solution, dtype=float).reshape(-1)
    residual_norm = float(np.linalg.norm(rhs_array - matrix @ solution_array))
    rhs_norm = float(np.linalg.norm(rhs_array))
    if rhs_norm == 0.0:
        return 0.0 if residual_norm == 0.0 else float("inf")
    return residual_norm / rhs_norm


def require_true_residual(
    matrix: sp.spmatrix,
    rhs: Any,
    solution: Any,
    *,
    tolerance: float,
    backend_reason: int,
    backend_iterations: int,
    solver: str = "petsc_ksp_hypre_ams",
    diagnostic_schema: str = "atem3d.petsc-ams-convergence-diagnostic",
) -> float:
    """Fail closed unless both PETSc and the external residual accept a solve."""

    tolerance = _positive_finite(tolerance, "tolerance")
    relative_residual = true_relative_residual(matrix, rhs, solution)
    backend_converged = int(backend_reason) > 0
    external_converged = bool(
        np.isfinite(relative_residual) and relative_residual <= tolerance
    )
    if backend_converged and external_converged:
        return float(relative_residual)
    reason = (
        "backend_failed"
        if not backend_converged
        else "external_true_residual_above_tolerance"
    )
    diagnostic = {
        "diagnostic_schema": str(diagnostic_schema),
        "diagnostic_schema_version": 1,
        "reason": reason,
        "solver": str(solver),
        "backend_reason": int(backend_reason),
        "backend_reported_converged": backend_converged,
        "backend_iterations": int(backend_iterations),
        "external_true_relative_residual": (
            float(relative_residual) if np.isfinite(relative_residual) else "inf"
        ),
        "external_tolerance": tolerance,
    }
    raise PetscAmsConvergenceError(diagnostic)


def _petsc_aij_from_csr(matrix: sp.csr_matrix, PETSc):
    indptr = np.ascontiguousarray(matrix.indptr, dtype=PETSc.IntType)
    indices = np.ascontiguousarray(matrix.indices, dtype=PETSc.IntType)
    data = np.ascontiguousarray(matrix.data, dtype=PETSc.ScalarType)
    result = None
    try:
        result = PETSc.Mat().createAIJ(
            size=matrix.shape,
            csr=(indptr, indices, data),
            comm=PETSc.COMM_SELF,
        )
        result.assemble()
    except BaseException:
        if result is not None:
            try:
                result.destroy()
            except BaseException:
                # Preserve the allocation/assembly failure that caused cleanup.
                pass
        raise
    # petsc4py may borrow user CSR storage.  Return and retain the exact arrays
    # for at least as long as the PETSc Mat exists.
    return result, (indptr, indices, data)


class PetscHypreAmsSolver:
    """Reusable serial KSP with a HYPRE AMS auxiliary-space preconditioner."""

    def __init__(
        self,
        matrix: Any,
        *,
        nodal_gradient: Any,
        edge_constant_vectors: Sequence[Any],
        tolerance: float = 1.0e-8,
        internal_tolerance: float | None = None,
        maxiter: int = 2000,
        refinement_steps: int = 2,
        ksp_type: str = "gmres",
    ) -> None:
        try:
            from petsc4py import PETSc  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - exercised outside PETSc env
            raise RuntimeError("PetscHypreAmsSolver requires petsc4py with HYPRE") from err

        self.matrix = _csr_float64(matrix, name="matrix")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square")
        self.nodal_gradient = _csr_float64(
            nodal_gradient,
            name="nodal_gradient",
        )
        if self.nodal_gradient.shape[0] != self.matrix.shape[0]:
            raise ValueError("nodal_gradient rows must match the edge matrix size")
        if len(edge_constant_vectors) != 3:
            raise ValueError("edge_constant_vectors must contain x, y, and z vectors")
        self.edge_constant_vectors = tuple(
            np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)
            for vector in edge_constant_vectors
        )
        if any(
            vector.shape != (self.matrix.shape[0],)
            or not np.all(np.isfinite(vector))
            for vector in self.edge_constant_vectors
        ):
            raise ValueError("edge constant vectors must be finite and match the matrix size")

        self.tolerance = _positive_finite(tolerance, "tolerance")
        default_internal = min(1.0e-11, self.tolerance * 1.0e-3)
        self.internal_tolerance = _positive_finite(
            default_internal if internal_tolerance is None else internal_tolerance,
            "internal_tolerance",
        )
        if self.internal_tolerance > self.tolerance:
            raise ValueError("internal_tolerance must not exceed the external tolerance")
        self.maxiter = _positive_integer(maxiter, "maxiter")
        if (
            isinstance(refinement_steps, bool)
            or not isinstance(refinement_steps, Integral)
            or refinement_steps < 0
        ):
            raise ValueError("refinement_steps must be a nonnegative integer excluding bool")
        self.refinement_steps = int(refinement_steps)
        self.ksp_type = str(ksp_type).strip().lower()
        if self.ksp_type not in {"cg", "gmres"}:
            raise ValueError("ksp_type must be 'cg' or 'gmres'")

        self._PETSc = PETSc
        self.last_diagnostics: dict[str, Any] | None = None
        self.destroyed = False
        self._petsc_matrix = None
        self._matrix_csr_buffers = ()
        self._petsc_gradient = None
        self._gradient_csr_buffers = ()
        self._constant_vecs: list[Any] | tuple[Any, ...] = []
        self._ksp = None
        try:
            self._petsc_matrix, self._matrix_csr_buffers = _petsc_aij_from_csr(
                self.matrix, PETSc
            )
            self._petsc_gradient, self._gradient_csr_buffers = _petsc_aij_from_csr(
                self.nodal_gradient, PETSc
            )
            for values in self.edge_constant_vectors:
                self._constant_vecs.append(self._make_edge_vector(values))
            self._constant_vecs = tuple(self._constant_vecs)
            self._ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            self._ksp.setOperators(self._petsc_matrix)
            self._ksp.setType(self.ksp_type)
            self._ksp.setTolerances(
                rtol=self.internal_tolerance,
                atol=0.0,
                max_it=self.maxiter,
            )
            pc = self._ksp.getPC()
            pc.setType("hypre")
            pc.setHYPREType("ams")
            pc.setHYPREDiscreteGradient(self._petsc_gradient)
            pc.setHYPRESetEdgeConstantVectors(*self._constant_vecs)
            self._ksp.setInitialGuessNonzero(False)
            self._ksp.setUp()
        except BaseException:
            # Preserve the construction failure while still releasing every
            # PETSc object which was successfully allocated before it.
            self._destroy_native_state(suppress_errors=True)
            self.destroyed = True
            raise

    def _make_edge_vector(self, values: np.ndarray):
        vector = self._petsc_matrix.createVecRight()
        try:
            vector.getArray()[:] = values
            vector.assemble()
            return vector
        except BaseException:
            vector.destroy()
            raise

    def _destroy_native_state(self, *, suppress_errors: bool) -> None:
        errors: list[BaseException] = []

        def release(native: Any) -> None:
            if native is None:
                return
            try:
                native.destroy()
            except BaseException as err:  # pragma: no cover - native failure
                errors.append(err)

        # KSP/PC retain references to every auxiliary object below it.
        release(self._ksp)
        self._ksp = None
        for vector in reversed(self._constant_vecs):
            release(vector)
        self._constant_vecs = ()
        release(self._petsc_gradient)
        self._petsc_gradient = None
        release(self._petsc_matrix)
        self._petsc_matrix = None
        self._gradient_csr_buffers = ()
        self._matrix_csr_buffers = ()
        if errors and not suppress_errors:
            raise errors[0]

    def _solve_once(self, rhs: np.ndarray) -> tuple[np.ndarray, int, int]:
        b = None
        x = None
        body_error = None
        try:
            b = self._petsc_matrix.createVecRight()
            x = self._petsc_matrix.createVecRight()
            b.getArray()[:] = rhs
            b.assemble()
            x.set(0.0)
            self._ksp.setInitialGuessNonzero(False)
            self._ksp.solve(b, x)
            return (
                np.asarray(x.getArray(readonly=True), dtype=float).copy(),
                int(self._ksp.getConvergedReason()),
                int(self._ksp.getIterationNumber()),
            )
        except BaseException as err:
            body_error = err
            raise
        finally:
            cleanup_errors = []
            # Release in reverse allocation order and attempt both releases,
            # even when one native destroy operation itself fails.
            for vector in (x, b):
                if vector is None:
                    continue
                try:
                    vector.destroy()
                except BaseException as err:  # pragma: no cover - native failure
                    cleanup_errors.append(err)
            if cleanup_errors and body_error is None:
                raise cleanup_errors[0]

    def solve(self, rhs: Any) -> np.ndarray:
        if self.destroyed:
            raise RuntimeError("PetscHypreAmsSolver has been destroyed")
        rhs_array = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
        if rhs_array.shape != (self.matrix.shape[0],):
            raise ValueError("rhs size must match the matrix")
        if not np.all(np.isfinite(rhs_array)):
            raise ValueError("rhs must be finite")
        if not np.any(rhs_array):
            solution = np.zeros_like(rhs_array)
            self.last_diagnostics = {
                "solver": "petsc_ksp_hypre_ams",
                "pc_type": "hypre_ams",
                "backend_reason": 0,
                "backend_reported_converged": True,
                "backend_iterations": 0,
                "external_true_relative_residual": 0.0,
                "external_tolerance": self.tolerance,
                "internal_tolerance": self.internal_tolerance,
                "residual_replacement_steps": 0,
            }
            return solution

        solution, backend_reason, iterations = self._solve_once(rhs_array)
        total_iterations = iterations
        replacements = 0
        relative_residual = true_relative_residual(self.matrix, rhs_array, solution)
        while (
            backend_reason > 0
            and relative_residual > self.tolerance
            and replacements < self.refinement_steps
        ):
            residual = np.ascontiguousarray(rhs_array - self.matrix @ solution)
            correction, backend_reason, iterations = self._solve_once(residual)
            solution += correction
            total_iterations += iterations
            replacements += 1
            relative_residual = true_relative_residual(self.matrix, rhs_array, solution)

        self.last_diagnostics = {
            "solver": "petsc_ksp_hypre_ams",
            "ksp_type": self.ksp_type,
            "pc_type": "hypre_ams",
            "backend_reason": int(backend_reason),
            "backend_reported_converged": bool(backend_reason > 0),
            "backend_iterations": int(total_iterations),
            "external_true_relative_residual": float(relative_residual),
            "external_tolerance": self.tolerance,
            "internal_tolerance": self.internal_tolerance,
            "residual_replacement_steps": int(replacements),
        }
        try:
            relative_residual = require_true_residual(
                self.matrix,
                rhs_array,
                solution,
                tolerance=self.tolerance,
                backend_reason=backend_reason,
                backend_iterations=total_iterations,
            )
        except PetscAmsConvergenceError as err:
            self.last_diagnostics.update(err.diagnostics)
            raise
        self.last_diagnostics["external_true_relative_residual"] = float(
            relative_residual
        )
        return solution

    __call__ = solve

    def destroy(self) -> None:
        if self.destroyed:
            return
        try:
            self._destroy_native_state(suppress_errors=False)
        finally:
            self.destroyed = True
