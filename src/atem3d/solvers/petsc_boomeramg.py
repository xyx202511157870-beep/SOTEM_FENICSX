"""Serial PETSc/HYPRE BoomerAMG adapter for scalar SPD systems."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from .petsc_ams import (
    PetscAmsConvergenceError,
    _csr_float64,
    _petsc_aij_from_csr,
    _positive_finite,
    _positive_integer,
    require_true_residual,
    true_relative_residual,
)


class PetscHypreBoomerAmgSolver:
    """Reusable serial KSP with HYPRE BoomerAMG and a SciPy residual gate."""

    def __init__(
        self,
        matrix: Any,
        *,
        tolerance: float = 1.0e-8,
        internal_tolerance: float | None = None,
        maxiter: int = 2000,
        refinement_steps: int = 2,
        ksp_type: str = "cg",
    ) -> None:
        try:
            from petsc4py import PETSc  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - exercised outside PETSc env
            raise RuntimeError(
                "PetscHypreBoomerAmgSolver requires petsc4py with HYPRE"
            ) from err

        self.matrix = _csr_float64(matrix, name="matrix")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square")
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
            raise ValueError(
                "refinement_steps must be a nonnegative integer excluding bool"
            )
        self.refinement_steps = int(refinement_steps)
        self.ksp_type = str(ksp_type).strip().lower()
        if self.ksp_type not in {"cg", "gmres"}:
            raise ValueError("ksp_type must be 'cg' or 'gmres'")

        self._PETSc = PETSc
        self.last_diagnostics: dict[str, Any] | None = None
        self.destroyed = False
        self._petsc_matrix = None
        self._matrix_csr_buffers = ()
        self._ksp = None
        try:
            self._petsc_matrix, self._matrix_csr_buffers = _petsc_aij_from_csr(
                self.matrix,
                PETSc,
            )
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
            pc.setHYPREType("boomeramg")
            self._ksp.setInitialGuessNonzero(False)
            self._ksp.setUp()
        except BaseException:
            self._destroy_native_state(suppress_errors=True)
            self.destroyed = True
            raise

    def _destroy_native_state(self, *, suppress_errors: bool) -> None:
        errors: list[BaseException] = []
        for native in (self._ksp, self._petsc_matrix):
            if native is None:
                continue
            try:
                native.destroy()
            except BaseException as err:  # pragma: no cover - native failure
                errors.append(err)
        self._ksp = None
        self._petsc_matrix = None
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
            raise RuntimeError("PetscHypreBoomerAmgSolver has been destroyed")
        rhs_array = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
        if rhs_array.shape != (self.matrix.shape[0],):
            raise ValueError("rhs size must match the matrix")
        if not np.all(np.isfinite(rhs_array)):
            raise ValueError("rhs must be finite")
        if not np.any(rhs_array):
            solution = np.zeros_like(rhs_array)
            self.last_diagnostics = {
                "solver": "petsc_ksp_hypre_boomeramg",
                "solve_mode": "exact_zero_rhs",
                "ksp_type": self.ksp_type,
                "pc_type": "hypre_boomeramg",
                "backend_reason": 0,
                "backend_reported_converged": False,
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
            relative_residual = true_relative_residual(
                self.matrix,
                rhs_array,
                solution,
            )

        self.last_diagnostics = {
            "solver": "petsc_ksp_hypre_boomeramg",
            "solve_mode": "petsc_ksp",
            "ksp_type": self.ksp_type,
            "pc_type": "hypre_boomeramg",
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
                solver="petsc_ksp_hypre_boomeramg",
                diagnostic_schema=(
                    "atem3d.petsc-boomeramg-convergence-diagnostic"
                ),
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
