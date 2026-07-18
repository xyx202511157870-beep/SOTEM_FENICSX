# Remove COMSOL From Seepage Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the verified seepage-channel release depend only on SimPEG, FEniCSx, and empymod, then regenerate the Word report after every retained gate passes.

**Architecture:** Reuse `build_open3d_summary` as the final fail-closed gate set, remove COMSOL loading from formal plotting, and update the report manifest and narrative to a two-solver finite-anomaly comparison. Preserve standalone COMSOL implementation code while deleting its task outputs.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, python-docx, pytest, LibreOffice rendering.

---

### Task 1: Final aggregation contract

**Files:**
- Modify: `tests/test_seepage_final_aggregation.py`
- Modify: `src/atem3d/seepage_final_aggregation.py`

- [ ] Write a test asserting final required gates equal `OPEN3D_REQUIRED_GATES`, contain no `comsol_` names, and can pass with only a passing open-3D summary.
- [ ] Run `python -m pytest tests/test_seepage_final_aggregation.py -q` and confirm the new test fails because COMSOL gates are still required.
- [ ] Replace the COMSOL aggregation with a direct fail-closed delegation to `build_open3d_summary` while preserving `require_pass` behavior.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Formal plotting contract

**Files:**
- Modify: `tests/test_verified_seepage_plots.py`
- Modify: `tools/plot_verified_seepage_report.py`

- [ ] Write tests asserting the manifest contains `verified_two_solver_anomaly.png`, excludes the obsolete three-solver file, and plot generation never requires `comsol_3d`.
- [ ] Run `python -m pytest tests/test_verified_seepage_plots.py -q` and confirm failure against the current COMSOL-dependent implementation.
- [ ] Remove COMSOL series loading and rename `_plot_three_solver` to `_plot_two_solver`.
- [ ] Re-run the focused plot tests and confirm they pass.

### Task 3: Word report contract

**Files:**
- Modify: `tests/test_verified_seepage_word_report.py`
- Modify: `tools/build_verified_seepage_word_report.py`

- [ ] Write tests asserting the report figure manifest uses the two-solver figure and that generated document text explicitly excludes COMSOL from scope without claiming COMSOL results.
- [ ] Run `python -m pytest tests/test_verified_seepage_word_report.py -q` and confirm failure against the current four-method report.
- [ ] Update the cover, solver table, conclusions, limitations, evidence table, and figure caption.
- [ ] Re-run the focused Word tests and confirm they pass.

### Task 4: Real final verification and figures

**Files:**
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/verification_summary.json`
- Regenerate: formal `verified_*.png` files in the same output directory.

- [ ] Delete obsolete COMSOL-dependent summary, plots, and Word artifacts from the task output only.
- [ ] Run `python tools/aggregate_seepage_verification.py --stage final --output-root <output> --require-pass` and require exit code 0.
- [ ] Inspect the JSON and prove every required gate is available and passing.
- [ ] Run `python tools/plot_verified_seepage_report.py --result-dir <output>` and verify all expected figures exist and are non-empty.

### Task 5: Word generation and visual QA

**Files:**
- Regenerate: final `.docx` under the task output directory.
- Generate: page render PNGs in a dedicated render-check directory.

- [ ] Run the verified Word builder only after Task 4 passes.
- [ ] Render the document with the bundled document renderer.
- [ ] Inspect every rendered page for clipping, mojibake, missing figures, stale COMSOL references, and z-axis direction.
- [ ] Correct any layout or text defect and repeat render inspection.

### Task 6: Completion audit

**Files:**
- Verify all modified source, tests, generated artifacts, and output directories.

- [ ] Run focused and relevant regression tests.
- [ ] Search the formal output and generated Word text for stale COMSOL result claims.
- [ ] Confirm no COMSOL process, task output, or OOC scratch remains.
- [ ] Review Git diff and commit the narrowly scoped change.

