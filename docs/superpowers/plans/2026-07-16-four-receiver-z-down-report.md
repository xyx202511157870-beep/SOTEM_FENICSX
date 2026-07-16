# Four-receiver Z-down Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the formal benchmark report with physical depth drawn downward and only Rx1, Rx2, Rx4, and Rx5 presented.

**Architecture:** Preserve the completed five-receiver numerical artifacts and introduce a single reporting-index constant in the plotting layer. Apply that selection consistently to geometry, response, error, and Word-report presentation while leaving solver/audit provenance intact.

**Tech Stack:** Python, NumPy, Matplotlib, python-docx, pytest, Microsoft Word PDF export, Poppler.

---

### Task 1: Lock the plotting behavior with failing tests

**Files:**
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add a test that calls `plot_model_geometry`, intercepts the figure before saving, and asserts the 3D z axis and both section depth axes are inverted and both receiver scatters contain four points.
- [ ] Add a test that calls `_plot_response_grid` and `_plot_error_grid` with five-receiver arrays and asserts every plotted label excludes `Rx3` while retaining `Rx1`, `Rx2`, `Rx4`, and `Rx5`.
- [ ] Run `pytest tests/test_seepage_channel_plots_manifest.py -q` and confirm the new tests fail because the current plotting layer still draws five receivers and the 3D z axis is not inverted.

### Task 2: Implement the four-receiver plotting view

**Files:**
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add `REPORT_RECEIVER_INDICES = (0, 1, 3, 4)` next to the figure constants.
- [ ] Select those indices in `plot_model_geometry`, label the marker set as four reporting receivers, and call `axis_3d.invert_zaxis()`.
- [ ] Iterate over `REPORT_RECEIVER_INDICES` in `_plot_response_grid` and `_plot_error_grid`, using a four-color palette.
- [ ] Run `pytest tests/test_seepage_channel_plots_manifest.py -q` and confirm all plot tests pass.

### Task 3: Lock and implement the Word-report behavior

**Files:**
- Modify: `tests/test_seepage_channel_word_report.py`
- Modify: `tools/build_seepage_channel_word_report.py`

- [ ] Change the Word test fixture expectations so that the report contains the four retained receiver labels, does not embed `rx3_absolute_residual.png`, and describes the four-point reporting view plus five-point raw provenance.
- [ ] Run `pytest tests/test_seepage_channel_word_report.py -q` and confirm failure against the existing Rx3-focused report.
- [ ] Filter the receiver table to `(0, 1, 3, 4)`, update model captions and observation-system wording, and distinguish the formal four-point report from raw five-point solver data.
- [ ] Remove the center-point residual row, figure, and prose from `add_magnetic_stability_section`; retain `biot_rate`, `Rx2/Rx4`, `Rx1/Rx5`, full-domain, and no-mirroring information.
- [ ] Run `pytest tests/test_seepage_channel_word_report.py -q` and confirm it passes.

### Task 4: Regenerate and inspect deliverables

**Files:**
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/*.png`
- Regenerate: `output/doc/seepage_channel_100m_magnetic_stability_report.docx`
- Regenerate: `output/doc/seepage_channel_100m_magnetic_stability_report_preview.pdf`

- [ ] Run `python tools/plot_seepage_channel_benchmark.py output/seepage_channel_100m_5rx_60x1x1`.
- [ ] Run `python tools/build_seepage_channel_word_report.py --result-dir output/seepage_channel_100m_5rx_60x1x1 --output output/doc/seepage_channel_100m_magnetic_stability_report.docx`.
- [ ] Export the DOCX to PDF, render all PDF pages to PNG, and inspect every page for clipping, blank pages, old five-point language, and center-point figures.
- [ ] Run the two focused pytest files, `git diff --check`, and a text extraction check over the final DOCX.
- [ ] Copy the verified DOCX, preview PDF, and revised report figures to the main workspace output directory.

