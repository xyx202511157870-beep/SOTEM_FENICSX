# 60 x 1 x 1 m Seepage Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and formally solve a 60 x 1 x 1 m seepage-channel variant with SimPEG, empymod, and full-domain FEniCSx, then generate the same Chinese Word report format.

**Architecture:** Parameterize the existing benchmark around a channel-size/model-contract argument while preserving the original default. Add thin-channel-specific SimPEG configurations and FEniCSx launch scripts, write outputs to an isolated result root, and reuse the existing aggregation/plot/report contracts.

**Tech Stack:** Python, NumPy, SimPEG, empymod, DOLFINx/FEniCSx, gmsh, pytest, python-docx, LibreOffice/Poppler.

---

### Task 1: Thin model contract

**Files:**
- Modify: `src/atem3d/seepage_channel_model.py`
- Modify: `tests/test_seepage_channel_model.py`

- [ ] Add a failing test asserting a named `thin_60x1x1` model has center `(0,0,20)`, size `(60,1,1)`, volume `60`, and z-up bounds `(-20.5,-19.5)`.
- [ ] Run `python -m pytest tests/test_seepage_channel_model.py -q` and confirm failure.
- [ ] Add a model factory/registry that preserves `MODEL` as the 60 x 10 x 10 m default and exposes the thin variant.
- [ ] Re-run the test and commit the contract.

### Task 2: Accurate SimPEG and FEniCSx inputs

**Files:**
- Create: `examples/seepage_channel_100m_5rx_simpeg_thin_background.yaml`
- Create: `examples/seepage_channel_100m_5rx_simpeg_thin_channel.yaml`
- Create: `tools/run_fenicsx_seepage_thin_background.sh`
- Create: `tools/run_fenicsx_seepage_thin_channel.sh`
- Modify: `tests/test_config_conductivity_boxes.py`
- Modify: `tests/test_seepage_channel_fenicsx_command.py`

- [ ] Add failing tests for exact `[-0.5,0.5]` y bounds, `[-20.5,-19.5]` z-up bounds, 0.25 m local cells, and five explicit FEniCSx receivers.
- [ ] Run the two tests and confirm failure.
- [ ] Add aligned 0.25 m SimPEG y/z mesh segments and the 60 x 1 x 1 m conductivity box.
- [ ] Add FEniCSx scripts with `--conductivity-box-bounds=-30,30;-0.5,0.5;-20.5,-19.5`, `--conductivity-box-mesh-size 0.25`, the unchanged time settings, and all five receiver arguments.
- [ ] Re-run the tests and commit solver inputs.

### Task 3: Variant-aware aggregation, plotting, and orchestration

**Files:**
- Modify: `src/atem3d/seepage_channel_validation.py`
- Modify: `tools/run_seepage_channel_benchmark.py`
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Modify: `tests/test_seepage_channel_aggregation.py`
- Modify: `tests/test_seepage_channel_orchestrator.py`
- Modify: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add failing tests proving `--variant thin_60x1x1` selects thin configs/scripts, writes the thin channel audit, and plots bounds from `model_audit.json` rather than the default global model.
- [ ] Run the three tests and confirm failure.
- [ ] Thread a benchmark model/variant through audit generation and aggregation while leaving the original default behavior unchanged.
- [ ] Make the plotter consume the stored audit geometry.
- [ ] Extend the resumable job plan with the thin solver inputs and isolated output root.
- [ ] Re-run tests and commit the variant-aware pipeline.

### Task 4: Formal three-solver run

**Files:**
- Generate: `output/seepage_channel_100m_5rx_60x1x1/**`

- [ ] Reuse or regenerate the unchanged empymod background reference under the thin result root.
- [ ] Run thin SimPEG background and channel models and normalize each to `(5,31,3)`.
- [ ] Run thin FEniCSx background and channel models in the full domain; verify five `explicit_full_domain` provenance values and no mirror job.
- [ ] Run aggregation and plotting; retain every raw value and signed channel delta even if a target is not met.
- [ ] Generate `benchmark_manifest.json` last and validate hashes.

### Task 5: Same-format Word report

**Files:**
- Modify: `tools/build_seepage_channel_word_report.py`
- Modify: `tests/test_seepage_channel_word_report.py`
- Generate: `output/doc/100m长导线60x1x1m渗漏通道三算法正演对比报告.docx`

- [ ] Add a failing test that builds from the thin audit and finds `60 x 1 x 1 m`, `explicit_full_domain`, and the no-symmetry statement.
- [ ] Parameterize report text/tables from `model_audit.json` and FEniCSx summaries rather than hard-coded 60 x 10 x 10 m values.
- [ ] Build the formal Chinese report.
- [ ] Render it with `render_docx.py` (or LibreOffice plus Poppler fallback), inspect every page, and fix any clipping/overlap.

### Task 6: Completion verification

**Files:**
- Verify all files above and formal artifacts.

- [ ] Run all seepage-channel tests and Python/shell syntax checks.
- [ ] Assert all five formal payloads are structurally valid; each 3D case has 465 finite values and FEniCSx provenance is explicit full-domain.
- [ ] Assert the audit contains center 20 m, size 60 x 1 x 1 m, conductivity 1 S/m, and a finite discrete-volume audit.
- [ ] Verify the DOCX opens, contains the required text and figures, and all rendered pages are non-empty.
- [ ] Commit only task-scoped source/test/document changes; preserve unrelated worktree changes.

## Plan self-review

- Every design requirement maps to Tasks 1-6.
- No placeholders or deferred implementation steps remain.
- The variant name, output root, model dimensions, coordinate conversion, and report filename are consistent across tasks.
