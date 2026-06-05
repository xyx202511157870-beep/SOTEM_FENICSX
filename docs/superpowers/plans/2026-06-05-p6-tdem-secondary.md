# P6 TDEM Secondary Stepper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pure time-stepping kernels for no-IP and IP primary-secondary TDEM equations, with zero-contrast regression tests.

**Architecture:** Add `atem3d.solvers.tdem_secondary` as a NumPy core. It computes the current-history RHS densities and delegates the actual FEM solve to an injected secondary solver callback. This isolates the BE physics from DOLFINx matrix assembly.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: Secondary Stepper Core

**Files:**
- Create: `src/atem3d/solvers/tdem_secondary.py`
- Test: `tests/test_secondary_zero_contrast.py`

- [ ] **Step 1: Write failing tests**

Test no-IP zero contrast over multiple variable time steps, IP zero-delta equivalence, and nonzero contrast RHS passed to the injected solver.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_secondary_zero_contrast.py`
Expected: FAIL because `atem3d.solvers.tdem_secondary` does not exist.

- [ ] **Step 3: Implement stepper**

Implement `SecondaryState`, `secondary_step_noip`, and `secondary_step_ip`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_secondary_zero_contrast.py tests/test_dc_initialization.py tests/test_prony.py`
Expected: PASS.

- [ ] **Step 5: Update report and commit**

Commit with implemented modules, tests, validation command, and known limitations.
