# P7 Three-Component Validation Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add no-IP/IP three-component validation smoke artifact generation for predictions, references, errors, summaries, diagnostics, and plots.

**Architecture:** Add a pure artifact writer in `atem3d.validation_3comp` that uses existing robust error metrics. It is backend-agnostic: DOLFINx/empymod/1D callers can supply arrays later.

**Tech Stack:** Python, NumPy, matplotlib, pytest.

---

### Task 1: Artifact Writer

**Files:**
- Create: `src/atem3d/validation_3comp.py`
- Test: `tests/test_noip_3comp_validation_smoke.py`
- Test: `tests/test_ip_3comp_validation_smoke.py`

- [ ] **Step 1: Write failing tests**

Test required output files, summary fields, 5% threshold line plots, and IP Prony metadata.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_noip_3comp_validation_smoke.py tests/test_ip_3comp_validation_smoke.py`
Expected: FAIL because `atem3d.validation_3comp` does not exist.

- [ ] **Step 3: Implement artifact writer**

Implement `ThreeComponentValidationInput` and `write_three_component_validation_artifacts`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_noip_3comp_validation_smoke.py tests/test_ip_3comp_validation_smoke.py tests/test_error_metric_floor.py`
Expected: PASS.

- [ ] **Step 5: Update report and commit**

Commit with implemented modules, tests, validation command, and known limitations.
