# P5 DC Secondary Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable DC secondary initialization kernel for primary-secondary solvers, with zero-contrast regression tests.

**Architecture:** Add `atem3d.solvers.dc_secondary` as a pure NumPy core that can later be called from a DOLFINx assembly layer. It accepts an optional linear solver callback for the scalar secondary potential problem; if contrast is zero it returns the exact zero secondary field without solving.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: DC Secondary Core

**Files:**
- Create: `src/atem3d/solvers/__init__.py`
- Create: `src/atem3d/solvers/dc_secondary.py`
- Test: `tests/test_dc_initialization.py`

- [ ] **Step 1: Write failing tests**

Test zero contrast, nonzero contrast with injected solver, IP memory initialization `chi0=Ep0+Es0`, and `deltaJ0`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_dc_initialization.py`
Expected: FAIL because `atem3d.solvers.dc_secondary` does not exist.

- [ ] **Step 3: Implement core**

Implement `DCSecondaryInitialization`, `initialize_dc_secondary`, and helper validation.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_dc_initialization.py tests/test_prony.py tests/test_primary_provider.py`
Expected: PASS.

- [ ] **Step 5: Update report and commit**

Commit with implemented modules, tests, validation command, and known limitations.
