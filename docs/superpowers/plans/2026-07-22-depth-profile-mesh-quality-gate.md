# Depth-profile mesh-quality gate implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail the existing local tetrahedron preflight when any configured receiver-depth diagnostic point lies in an unacceptable cell.

**Architecture:** Reuse one collective point-to-quality-cell selector for the formal receiver and all depth-profile points. Publish explicit required selection names in the quality artifact so the unchanged fixed thresholds fail closed when a depth selection is missing or poor.

**Tech Stack:** Python 3.10, DOLFINx geometry, mpi4py collectives, pytest.

---

### Task 1: Fail closed on declared depth selections

**Files:**
- Modify: `tests/test_dolfinx_mesh_quality_gate.py`
- Modify: `dolfinx/sotem_pipeline.py:2469-2508`

- [ ] **Step 1: Write the failing gate test**

Add a test that builds the three existing passing selections, declares a fourth
`receiver_depth_profile_000` selection in `required_selections`, omits it from
`selections`, and asserts that `_apply_local_mesh_quality_gate` returns
`passed=False` with that exact name in `failed_selections`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py::test_local_mesh_quality_gate_fails_when_declared_depth_selection_is_missing -q
```

Expected: FAIL because the gate currently ignores `required_selections`.

- [ ] **Step 3: Implement explicit required selections**

Keep the three fixed names and append unique string names from
`summary.get("required_selections", ())`. Evaluate every resulting name with
the existing finite/positive/0.01/100 checks. Preserve backward compatibility
for summaries without the new field.

- [ ] **Step 4: Run focused gate tests and verify GREEN**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py -k local_mesh_quality_gate -q
```

Expected: all selected tests pass.

### Task 2: Add every configured depth point to the preflight

**Files:**
- Modify: `tests/test_dolfinx_mesh_quality_gate.py`
- Modify: `dolfinx/sotem_pipeline.py:2509-2618`

- [ ] **Step 1: Write the failing integration test**

Use a one-rank fake mesh and monkeypatch the source, collision, interface and
quality-summary seams. Configure depths `(300.0, 400.0)` and assert that
`diagnose_local_mesh_quality` creates selections
`receiver_depth_profile_000` and `receiver_depth_profile_001`, declares both as
required, and records points `(receiver_x, receiver_y, -300.0)` and
`(receiver_x, receiver_y, -400.0)`.

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py::test_local_mesh_quality_diagnoses_every_depth_profile_point -q
```

Expected: FAIL because depth selections are absent.

- [ ] **Step 3: Implement the shared selector and depth summaries**

Extract the existing formal-receiver collision/nearest-center logic into
`_quality_cells_for_point(msh, point) -> (cells, global_collision_count,
selection_mode)`. Use it for the formal receiver and each validated depth.
Build indexed selection keys, per-depth metadata, and an ordered
`required_selections` list. Bump the additive artifact schema to
`sotem_local_tetra_quality/v2`.

- [ ] **Step 4: Run the focused suites and verify GREEN**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py tests/test_dolfinx_receiver_depth_profile.py tests/test_dolfinx_model_consistency.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Prove the preserved real mesh is rejected for the right reason**

Load the preserved factor-2/Rx-5 m mesh on one rank, use its existing source
diagnostics or a read-only depth-cell audit, and require the 300 m depth
selection to report `min(3r/R) < 0.01` and `max(R/(3r)) > 100`. Do not alter
the mesh or the ongoing run artifact.

- [ ] **Step 6: Run final checks and commit**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx*.py -q
git diff --check
git status --short
```

Commit only the production code, tests, and this plan.
