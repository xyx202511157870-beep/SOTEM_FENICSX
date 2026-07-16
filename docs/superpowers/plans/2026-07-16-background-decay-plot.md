# Background Conventional Decay Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Figure 2 to positive absolute-amplitude log-log decay curves while preserving signed numerical artifacts.

**Architecture:** Add an explicit magnitude-decay option to the shared response-grid function and enable it only for the background-response call. Update the Word caption without changing solver data or signed channel plots.

**Tech Stack:** Python, NumPy, Matplotlib, python-docx, pytest, Microsoft Word PDF export, Poppler.

---

### Task 1: Add a failing plotting test

**Files:**
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add a test that passes signed five-receiver values to `_plot_response_grid(..., magnitude_decay=True)` and asserts every y axis is `log`, every plotted y value is positive, and the original array remains unchanged.
- [ ] Run `python -m pytest tests/test_seepage_channel_plots_manifest.py::test_background_response_uses_absolute_log_decay -q` and confirm failure because `magnitude_decay` is not accepted.

### Task 2: Implement the background-only decay mode

**Files:**
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add a keyword-only `magnitude_decay: bool = False` parameter to `_plot_response_grid`.
- [ ] When enabled, plot `np.abs(values[receiver_index, :, component_index])`, set y scale to `log`, and use absolute-magnitude unit labels.
- [ ] Pass `magnitude_decay=True` only in the `background_response` call.
- [ ] Run `python -m pytest tests/test_seepage_channel_plots_manifest.py -q` and confirm all tests pass.

### Task 3: Update and regenerate the Word report

**Files:**
- Modify: `tools/build_seepage_channel_word_report.py`
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/background_response.png`
- Regenerate: `output/doc/seepage_channel_100m_magnetic_stability_report.docx`

- [ ] Change Figure 2 caption to identify absolute-amplitude decay curves.
- [ ] Regenerate plots and the DOCX, export PDF, render all pages, and inspect layout.
- [ ] Run the ten seepage-channel test files and `git diff --check`.
- [ ] Copy the verified background plot, DOCX, and preview PDF to the main workspace output directory.

