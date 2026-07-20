# Canonical SimPEG PETSc Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace canonical SOTEM sparse-direct initialization with fail-closed PETSc/HYPRE solvers while preserving direct compatibility.

**Architecture:** A scalar PETSc/BoomerAMG solver handles the gauge-reduced DC potential matrix, and the existing edge PETSc/AMS solver handles the gauge-stabilized vector-potential matrix. `TDEMIPSimulation` selects the path explicitly, records initialization diagnostics, validates physical balances, and destroys native resources locally.

**Tech Stack:** Python, NumPy, SciPy sparse, PETSc/petsc4py, HYPRE BoomerAMG/AMS, Discretize, pytest.

---

### Task 1: Lock configuration and provenance

**Files:**
- Modify: `tests/test_sotem_simpeg_adapter.py`
- Modify: `src/atem3d/sotem_simpeg_adapter.py`
- Modify: `src/atem3d/config.py`

- [ ] Change the canonical-config test to require an explicit PETSc initialization mapping with fixed tolerance, internal tolerance, iteration limit, and backend names.
- [ ] Run `pytest tests/test_sotem_simpeg_adapter.py::test_public_z_down_geometry_is_mapped_once_to_internal_z_up -q` and confirm it fails because the adapter still reports sparse direct.
- [ ] Pass the initialization mapping through `build_simulation` while preserving `direct` when the mapping is absent.
- [ ] Update adapter metadata to name BoomerAMG and AMS, then rerun the focused test.

### Task 2: Add scalar PETSc/BoomerAMG solve

**Files:**
- Create: `src/atem3d/solvers/petsc_boomeramg.py`
- Create: `tests/test_petsc_boomeramg_solver.py`

- [ ] Write tests requiring positive backend convergence, SciPy `A @ x` residual at most `1e-8`, finite output, and native matrix/vector/KSP cleanup on success and failure.
- [ ] Run the new tests and confirm they fail because the solver does not exist.
- [ ] Implement a serial PETSc AIJ/KSP wrapper with HYPRE BoomerAMG, iterative residual replacement, `require_true_residual`, and idempotent destruction.
- [ ] Run the new focused tests in WSL and confirm they pass.

### Task 3: Route and validate both initialization systems

**Files:**
- Modify: `src/atem3d/simulation.py`
- Modify: `tests/test_solver_core.py`
- Modify: `tests/test_petsc_ams_solver.py`

- [ ] Write tests that require PETSc initialization to solve the reduced nodal DC system and the gauge-stabilized edge Ampere system, record two diagnostics, pass divergence/Ampere balance gates, and destroy both local solvers.
- [ ] Run the focused tests and confirm they fail while initialization still calls `spsolve`.
- [ ] Add the explicit initialization selector and local `try/finally` PETSc solver lifetimes. Reuse the full tensor-mesh AMS auxiliary data for the edge matrix and the gauge-reduced nodal coordinates for the scalar matrix.
- [ ] Compute external algebraic residuals using the exact SciPy matrices and physical balance residuals using the existing discrete operators; fail closed above `1e-8`.
- [ ] Rerun the focused tests and direct-compatibility tests.

### Task 4: Expose diagnostics and verify the adapter

**Files:**
- Modify: `src/atem3d/sotem_simpeg_adapter.py`
- Modify: `tests/test_sotem_simpeg_adapter.py`

- [ ] Write tests requiring exactly two initialization diagnostics and rejecting missing, nonfinite, unconverged, excessive-algebraic-residual, or excessive-balance-residual records.
- [ ] Run the tests and confirm the adapter currently omits those diagnostics.
- [ ] Validate and return deep-copied initialization diagnostics separately from transient diagnostics.
- [ ] Run all adapter and solver-focused tests.

### Task 5: Real WSL verification and commit

**Files:**
- No production files beyond Tasks 1-4.

- [ ] Run a small real PETSc initialization and compare fields/balances against direct.
- [ ] Run a medium tensor-mesh initialization under `/usr/bin/time -v` and a bounded cgroup, recording peak memory, backend reasons, iterations, and SciPy residuals.
- [ ] Run the complete relevant pytest files in the FEniCSx WSL environment and `git diff --check`.
- [ ] Review the diff against every design requirement and commit only the scoped files.
