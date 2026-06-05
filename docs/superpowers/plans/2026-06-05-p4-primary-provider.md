# P4 Primary Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a primary-field provider abstraction for primary-secondary solvers while keeping pytest independent of empymod.

**Architecture:** Create `atem3d.primary` as a small provider package. `ZeroPrimaryProvider` and `CachedPrimaryProvider` are pure NumPy implementations; `EmpymodPrimaryProvider` is a delayed-import skeleton so importing the package does not require empymod.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: Provider API

**Files:**
- Create: `src/atem3d/primary/__init__.py`
- Create: `src/atem3d/primary/base.py`
- Create: `src/atem3d/primary/zero.py`
- Create: `src/atem3d/primary/cache.py`
- Create: `src/atem3d/primary/empymod_provider.py`
- Test: `tests/test_primary_provider.py`

- [ ] **Step 1: Write failing tests**

Write tests for zero fields, cached interpolation/exact lookup, shape validation, and delayed empymod import behavior.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_primary_provider.py`
Expected: FAIL because `atem3d.primary` does not exist.

- [ ] **Step 3: Implement providers**

Implement `PrimaryFieldProvider`, `ZeroPrimaryProvider`, `CachedPrimaryProvider`, and `EmpymodPrimaryProvider`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_primary_provider.py tests/test_prony.py`
Expected: PASS.

- [ ] **Step 5: Update report and commit**

Commit with implemented modules, tests, validation command, and known limitations.
