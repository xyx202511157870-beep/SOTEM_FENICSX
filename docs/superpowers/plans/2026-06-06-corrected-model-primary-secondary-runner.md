# Corrected Model Primary-Secondary Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the corrected source/receiver model through the real DOLFINx primary-secondary path and write no-IP/IP full-window validation artifacts against empymod/1D references.

**Architecture:** Add a small runner layer that consumes `CorrectedModelValidationConfig`, builds the existing DOLFINx primary-secondary forward operator, uses `EmpymodPrimaryProvider` for transient and DC primary samples, and writes the existing `validation_3comp` artifacts. Keep generated meshes, NPZ/HDF5 files, and plots under ignored `dolfinx/runs/`.

**Tech Stack:** Python, DOLFINx/FEniCSx in WSL, NumPy, empymod, pytest, existing `atem3d` CLI.

---

### Task 1: Corrected-Model Runner Spec

**Files:**
- Modify: `src/atem3d/corrected_model.py`
- Test: `tests/test_corrected_model_validation.py`

- [ ] **Step 1: Write failing tests**

Add a test that the corrected-model spec includes enough fields for a runner:
`source_start`, `source_end`, `receiver`, `current`, `observation_times`,
`components=["Ex", "Ey", "dBzdt"]`, `validation_scope="corrected_model_full"`,
and separate `noip`/`ip` material metadata.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_corrected_model_validation.py
```

Expected: FAIL only for missing runner metadata.

- [ ] **Step 3: Implement minimal metadata**

Extend `build_corrected_model_case_specs` without changing the canonical
geometry:

```python
spec["runner"] = {
    "backend": "dolfinx_primary_secondary",
    "reference": "empymod",
    "components": ["Ex", "Ey", "dBzdt"],
    "output_root": str(output_root),
}
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q tests/test_corrected_model_validation.py
```

Expected: PASS.

### Task 2: Pure Runner Orchestration

**Files:**
- Create: `src/atem3d/corrected_model_runner.py`
- Modify: `src/atem3d/cli.py`
- Test: `tests/test_corrected_model_runner.py`

- [ ] **Step 1: Write failing tests**

Use injected `forward_runner` and `reference_runner` functions so the test does
not require DOLFINx or empymod. Assert that the runner writes:
`predictions.csv`, `reference_empymod_or_1d.csv`, `errors.csv`,
`error_summary.json`, `comparison_3comp.png`, `error_curves_3comp.png`,
`diagnostics.json`, and `run_config_resolved.yaml`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_corrected_model_runner.py
```

Expected: FAIL because `atem3d.corrected_model_runner` does not exist.

- [ ] **Step 3: Implement pure orchestration**

Implement:

```python
def run_corrected_model_validation(config, output_dir, *, case_type, forward_runner=None, reference_runner=None):
    ...
```

The function must call existing `write_three_component_validation_artifacts`
with `validation_scope="corrected_model_full"` and must not import DOLFINx at
module import time.

- [ ] **Step 4: Add CLI**

Add:

```bash
tdem-ip-forward corrected-model-run CONFIG.yaml
```

The CLI should accept `--case noip|ip|both` and `--output-root`.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python -m pytest -q tests/test_corrected_model_runner.py tests/test_corrected_model_validation.py tests/test_empymod_validation.py
```

Expected: PASS.

### Task 3: DOLFINx Backend Adapter

**Files:**
- Modify: `dolfinx/sotem_pipeline.py`
- Modify: `src/atem3d/corrected_model_runner.py`
- Test: `tests/test_dolfinx_primary_secondary_forward_smoke.py`

- [ ] **Step 1: Write failing smoke**

Add a WSL-only smoke that builds a very small mesh, uses
`EmpymodPrimaryProvider`, and calls `_make_dolfinx_primary_secondary_forward_operator`
for one or two observation times. Assert finite `Ex`, `Ey`, and `dBzdt`.

- [ ] **Step 2: Verify RED in WSL**

Run in WSL:

```bash
python -m pytest -q tests/test_dolfinx_primary_secondary_forward_smoke.py
```

Expected: FAIL only because the corrected-model backend adapter is missing.

- [ ] **Step 3: Implement backend adapter**

Add a function that creates the DOLFINx operator, samples primary fields on
Nedelec interpolation points, runs the existing primary-secondary core, and
returns a `(n_times, 3)` receiver table.

- [ ] **Step 4: Verify GREEN in WSL**

Run:

```bash
python -m pytest -q tests/test_dolfinx_primary_secondary_forward_smoke.py
```

Expected: PASS. Then run `wsl --shutdown` from Windows and verify `wsl -l -v`
shows Ubuntu stopped.

### Task 4: Full-Window Artifact Run

**Files:**
- Modify: `IMPLEMENTATION_REPORT.md`

- [ ] **Step 1: Run no-IP full-window corrected model**

Run in WSL with memory-safe mesh/time settings under `dolfinx/runs/`:

```bash
tdem-ip-forward corrected-model-run corrected_model_validation_spec.json --case noip --output-root dolfinx/runs/corrected_model_full
```

- [ ] **Step 2: Run IP full-window corrected model**

Run:

```bash
tdem-ip-forward corrected-model-run corrected_model_validation_spec.json --case ip --output-root dolfinx/runs/corrected_model_full
```

- [ ] **Step 3: Run final acceptance report**

Run:

```bash
tdem-ip-forward acceptance-report dolfinx/runs/corrected_model_full/acceptance.yaml
```

Expected: exit code `0` only if both no-IP and IP have
`final_acceptance_passed=true`. If it fails, record failed components, failed
times, and diagnostics in `IMPLEMENTATION_REPORT.md`.

- [ ] **Step 4: Shut down WSL**

Run from Windows:

```powershell
wsl --shutdown
wsl -l -v
```

Expected: Ubuntu is `Stopped`.

- [ ] **Step 5: Commit**

Commit code and report changes only. Do not commit generated `.msh`, `.npz`,
`.h5`, or plot artifacts.
