# Channel Conventional Decay Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display Figures 4 and 5 as absolute-amplitude log-log decay curves while retaining signed raw artifacts.

**Architecture:** Reuse the already tested `magnitude_decay` response-grid mode at the two remaining formal response call sites. Update Word headings and captions only; error and audit plots remain unchanged.

**Tech Stack:** Python, NumPy, Matplotlib, python-docx, pytest, Microsoft Word PDF export, Poppler.

---

### Task 1: Add failing call-site tests

**Files:**
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Add a source-contract test asserting the background, channel, and channel-delta `_plot_response_grid` calls each pass `magnitude_decay=True`.
- [ ] Run the focused test and confirm failure because only the background call currently enables the mode.

### Task 2: Enable channel magnitude decay

**Files:**
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Test: `tests/test_seepage_channel_plots_manifest.py`

- [ ] Pass `magnitude_decay=True` in the `channel_response` and `channel_delta` calls.
- [ ] Change the channel-delta plot title from signed response to absolute-amplitude anomaly decay.
- [ ] Run the plot tests and confirm all pass.

### Task 3: Update Word wording and regenerate

**Files:**
- Modify: `tests/test_seepage_channel_word_report.py`
- Modify: `tools/build_seepage_channel_word_report.py`
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/channel_response.png`
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/channel_delta.png`
- Regenerate: `output/doc/seepage_channel_100m_magnetic_stability_report.docx`

- [ ] Add failing Word source assertions for the two new absolute-amplitude captions and removal of the old signed heading.
- [ ] Update section 4.2/4.3 captions and heading.
- [ ] Regenerate DOCX/PDF, render all pages, and inspect each page.
- [ ] Run all seepage-channel tests, `git diff --check`, and copy verified artifacts to the main output directory.

