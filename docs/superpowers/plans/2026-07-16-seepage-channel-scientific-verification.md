# Seepage Channel Scientific Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the approved 60 x 1 x 1 m seepage-channel result reproducible and scientifically gated, run all required validation cases, and generate the formal Word report only after every gate passes.

**Architecture:** Merge the thin-model and magnetic/report branches into one variant-aware pipeline. A canonical serialized model contract and SHA-256 fingerprint flow through every solver and artifact. A verification module evaluates zero contrast, contrast/volume trends, three-level spatial/time convergence, parity, cross-solver agreement, and COMSOL 3D agreement; the Word builder fails closed unless the resulting summary passes.

**Tech Stack:** Python 3.12, NumPy, SciPy, SimPEG, empymod, DOLFINx/FEniCSx, gmsh, COMSOL batch/API, pytest, python-docx, LibreOffice, Poppler.

---

### Task 1: Integrate the divergent thin-model and magnetic-report branches

**Files:**
- Modify: `src/atem3d/seepage_channel_model.py`
- Modify: `src/atem3d/seepage_channel_validation.py`
- Modify: `tools/run_seepage_channel_benchmark.py`
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Modify: `tools/build_seepage_channel_word_report.py`
- Create: `examples/seepage_channel_100m_5rx_simpeg_thin_background.yaml`
- Create: `examples/seepage_channel_100m_5rx_simpeg_thin_channel.yaml`
- Create: `tools/run_fenicsx_seepage_thin_background.sh`
- Create: `tools/run_fenicsx_seepage_thin_channel.sh`
- Test: `tests/test_seepage_channel_model.py`
- Test: `tests/test_seepage_channel_aggregation.py`
- Test: `tests/test_seepage_channel_orchestrator.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`
- Test: `tests/test_seepage_channel_word_report.py`

- [ ] Add failing tests that request `model_for_variant("thin_60x1x1")`, require 60 x 1 x 1 m audit geometry, and reject a plot/report manifest containing 60 x 10 x 10 m bounds.
- [ ] Run the focused tests and confirm failures are caused by missing variant support.
- [ ] Merge `codex/thin-seepage-channel` without committing, resolve conflicts by preserving current magnetic/report features and thin variant inputs, then make the minimum code changes required by the tests.
- [ ] Run all seepage-channel tests and confirm the integrated code path passes.
- [ ] Commit the integrated reproducible thin-model pipeline.

### Task 2: Canonical model fingerprint and fail-closed artifact contract

**Files:**
- Create: `src/atem3d/seepage_verification.py`
- Modify: `src/atem3d/seepage_channel_model.py`
- Modify: `src/atem3d/seepage_channel_validation.py`
- Modify: `tools/run_seepage_channel_benchmark.py`
- Test: `tests/test_seepage_verification.py`

- [ ] Write a failing test for a deterministic JSON contract and SHA-256 fingerprint:

```python
def test_model_fingerprint_is_deterministic_and_geometry_sensitive():
    thin = model_for_variant("thin_60x1x1")
    assert model_fingerprint(thin) == model_fingerprint(thin)
    assert model_fingerprint(thin) != model_fingerprint(model_for_variant("default"))
```

- [ ] Write a failing test that feeds mixed fingerprints to the aggregator and expects `ModelContractMismatch`.
- [ ] Run the tests and observe the expected missing-function failures.
- [ ] Implement canonical sorted JSON serialization, fingerprint propagation, and mismatch validation at every aggregation boundary.
- [ ] Re-run focused and seepage-channel tests, then commit.

### Task 3: Verification metrics and report gate

**Files:**
- Modify: `src/atem3d/seepage_verification.py`
- Modify: `tools/build_seepage_channel_word_report.py`
- Test: `tests/test_seepage_verification.py`
- Test: `tests/test_seepage_channel_word_report.py`

- [ ] Add failing unit tests for strong-signal masks excluding Rx3, normalized zero-contrast residual, monotone anomaly energy, three-level convergence, even/odd parity, and percentile-based cross-solver errors.
- [ ] Add a failing Word test proving the builder refuses missing or failed `verification_summary.json`.
- [ ] Implement pure NumPy metric functions and a fail-closed `build_verification_summary` function with evidence paths.
- [ ] Add `--allow-unverified-draft` only for diagnostic drafts; keep the default formal path closed.
- [ ] Re-run tests and commit.

### Task 4: Generate solver matrices for physical-limit and convergence tests

**Files:**
- Create: `src/atem3d/seepage_case_matrix.py`
- Create: `tools/run_seepage_verification_matrix.py`
- Modify: `tools/run_seepage_channel_benchmark.py`
- Test: `tests/test_seepage_case_matrix.py`
- Test: `tests/test_seepage_channel_orchestrator.py`

- [ ] Write failing tests asserting cases for conductivity `[0.01, 0.02, 0.1, 1.0]`, cross-sections `[1, 2, 10]` m, spatial levels `[0.5, 0.25, 0.125]` m, and time-step factors `[1, 0.5, 0.25]`.
- [ ] Write a failing test that every case inherits the canonical thin source/receiver/time contract and declares its controlled variable explicitly.
- [ ] Implement deterministic case IDs, config generation, resumable execution, provenance hashes, and a dry-run manifest.
- [ ] Run matrix/orchestrator tests and commit.

### Task 5: Run SimPEG and FEniCSx validation matrices

**Files:**
- Generate: `output/seepage_channel_100m_5rx_60x1x1/verification_runs/**`
- Generate: `output/seepage_channel_100m_5rx_60x1x1/verification_summary_open3d.json`

- [ ] Execute zero-contrast cases for both solvers and verify the normalized residual gate.
- [ ] Execute conductivity and volume sweeps; retain raw signed values and evaluate monotone anomaly-energy gates.
- [ ] Execute coarse/medium/fine spatial runs and baseline/half/quarter time-step runs for background and channel.
- [ ] Run the thin-model FEniCSx magnetic-operator audit; select the operator only if the formal parity gates pass.
- [ ] Aggregate SimPEG/FEniCSx results and evaluate convergence, parity, volume audit, and cross-solver gates.
- [ ] If a gate fails, diagnose the specific solver/configuration, add a failing regression test, fix one root cause, and rerun only invalidated cases.

### Task 6: Add and run the independent COMSOL 3D anomaly reference

**Files:**
- Create: `COMSOL/seepage_channel_3d/` solver assets required by the chosen COMSOL batch/API route
- Create: `tools/run_comsol_seepage_channel_3d.py`
- Modify: `src/atem3d/seepage_verification.py`
- Test: `tests/test_comsol_seepage_channel_3d.py`
- Generate: `output/seepage_channel_100m_5rx_60x1x1/comsol_3d/**`

- [ ] Write a failing contract test requiring the same source, four formal receivers plus center diagnostic, 60 x 1 x 1 m bounds, conductivities, times, components, and model fingerprint.
- [ ] Implement a batch-safe COMSOL model/run adapter that never overwrites the user-provided reference MPH in place.
- [ ] Run COMSOL background, zero-contrast, and channel cases; export raw `Ex`, `Hz`, and `dBzdt` at all required points/times.
- [ ] Validate sample count, finite values, zero contrast, model fingerprint, and strong-signal agreement with SimPEG/FEniCSx.
- [ ] Commit the COMSOL adapter and tests; keep generated binary results out of source commits unless already permitted by repository policy.

### Task 7: Final aggregation, figures, Word, and visual QA

**Files:**
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Modify: `tools/build_seepage_channel_word_report.py`
- Generate: `output/seepage_channel_100m_5rx_60x1x1/verification_summary.json`
- Generate: `output/seepage_channel_100m_5rx_60x1x1/*.png`
- Generate: `output/doc/seepage_channel_100m_5rx_60x1x1_verified_report.docx`
- Generate temporarily: `tmp/docs/seepage_channel_verified/**`

- [ ] Run the final aggregator and assert every mandatory gate has `available=true` and `pass=true`.
- [ ] Regenerate model, total-field decay, signed anomaly, relative anomaly, profile, sweep, convergence, parity, and three-solver comparison figures from verified arrays only.
- [ ] Build the formal Word through the default fail-closed path and verify its model fingerprint matches the final summary.
- [ ] Render DOCX to PDF and page PNGs; inspect every page at full resolution and fix clipping, overlap, unreadable legends, blank pages, or encoding defects.
- [ ] Delete temporary render outputs after final approval while retaining the verified DOCX and source figures.

### Task 8: Completion audit

**Files:**
- Verify all source, tests, generated evidence, and final document above.

- [ ] Run the complete relevant pytest surface, Python compilation, and shell syntax checks.
- [ ] Re-run the final verification command fresh and record zero failed mandatory gates.
- [ ] Re-read this plan and the approved design requirement by requirement; map each requirement to authoritative evidence.
- [ ] Confirm the git diff contains only task-scoped changes and no user-owned unrelated changes.
- [ ] Use `superpowers:finishing-a-development-branch`, then report the verified artifact paths and limitations.
