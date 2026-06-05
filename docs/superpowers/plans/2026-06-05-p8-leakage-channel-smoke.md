# P8 Leakage Channel Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add testable material mapping and irregular leakage-channel cell marking utilities for complex-terrain scenarios.

**Architecture:** Add `atem3d.materials.material_map` for marker-to-material arrays and leakage-channel marker application. Add `atem3d.examples.leakage_channel` to build a small synthetic scenario that later DOLFINx/gmsh code can consume.

**Tech Stack:** Python, NumPy, pytest.

---

### Task 1: Material Map and Leakage Channel Smoke

**Files:**
- Create: `src/atem3d/materials/material_map.py`
- Modify: `src/atem3d/materials/__init__.py`
- Create: `src/atem3d/examples/__init__.py`
- Create: `src/atem3d/examples/leakage_channel.py`
- Test: `tests/test_complex_terrain_leakage_smoke.py`

- [ ] **Step 1: Write failing tests**

Test marker material arrays, channel marker assignment for a polyline/radius, and example diagnostics.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_complex_terrain_leakage_smoke.py`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement utilities**

Implement `CellMaterialMap`, `mark_leakage_channel`, `apply_leakage_channel_marker`, and `build_leakage_channel_example`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_complex_terrain_leakage_smoke.py tests/test_prony.py`
Expected: PASS.

- [ ] **Step 5: Update report and commit**

Commit with implemented modules, tests, validation command, and known limitations.
