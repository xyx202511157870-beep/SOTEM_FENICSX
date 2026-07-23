# Song Cross-Code Word Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and visually verify a Chinese Word report documenting the completed Song no-IP/IP comparison among FEniCSx, empymod, and SimPEG, including method-specific parameters and only evidence-backed conclusions.

**Architecture:** A reproducible Python builder reads the immutable WSL result artifacts and repository case files, validates required hashes/metrics, renders report-sized scientific figures, and assembles an A4 DOCX with `python-docx`. A separate verification script extracts the final DOCX text and checks required values; LibreOffice/Poppler rendering supplies page-by-page visual QA.

**Tech Stack:** Python 3, NumPy, pandas, matplotlib, python-docx, LibreOffice headless rendering, Poppler/PDF image rendering.

---

## File structure

- Create: `scripts/reports/build_song_cross_code_word_report.py` - evidence loading, validation, figure generation, and DOCX assembly.
- Create: `scripts/reports/verify_song_cross_code_word_report.py` - machine checks for document text, figures, tables, and key numerical values.
- Create: `output/doc/Song文献模型_FEniCSx_empymod_SimPEG_三维时域正演对比报告.docx` - final deliverable.
- Create temporarily: `tmp/docs/song_cross_code_report/` - generated PNG/PDF/page renders; remove after final checks except the final rendered PDF when useful for inspection.
- Read only: `.tmp/final-results/comparison_summary.json`, `.tmp/final-results/cross_code_comparison.png`, `benchmarks/sotem/song2025_layered_pair.yaml`, FEniCSx WSL result directories, and SimPEG WSL result directories.

### Task 1: Build the evidence loader and fail-closed checks

**Files:**
- Create: `scripts/reports/build_song_cross_code_word_report.py`

- [ ] **Step 1: Define immutable evidence paths and required keys**

Define constants for the local summary, case YAML, FEniCSx no-IP/IP directories, SimPEG no-IP/IP JSON files, and the exact reference. Require `Ex`, `Hz`, and `dBzdt` for all four solver variants and require 21 observations from `1e-5` to `1e-3` seconds.

- [ ] **Step 2: Add numerical integrity assertions**

Assert the published maxima before any figure or document is written:

```python
EXPECTED_MAX = {
    "fenicsx_noip": {"Ex": 0.01036429269070334, "Hz": 0.014124974665132994, "dBzdt": 0.03301669355706037},
    "fenicsx_ip": {"Ex": 0.01993756177376695, "Hz": 0.010906898025984593, "dBzdt": 0.02687928048666212},
    "simpeg_noip": {"Ex": 0.032101056843179654, "Hz": 0.006303089535895287, "dBzdt": 0.034527646600017095},
    "simpeg_ip": {"Ex": 0.07420634839988084, "Hz": 0.006542826240444855, "dBzdt": 0.034454769506365814},
}
```

Use `numpy.testing.assert_allclose(..., rtol=0, atol=5e-12)` so a stale or altered payload fails closed.

- [ ] **Step 3: Extract method-specific parameters from evidence**

Read FEniCSx `run_config_resolved.yaml`, diagnostics, mesh preflight and magnetic initialization; read empymod settings from the resolved config and source quadrature audit; read SimPEG mesh, time, solver, material-fit and runtime fields from `result.json`. Do not infer missing values.

- [ ] **Step 4: Run the loader in evidence-only mode**

Run:

```powershell
python scripts/reports/build_song_cross_code_word_report.py --check-only
```

Expected: exit code 0 and a summary stating 21 observations, the four formal solver variants, and one retained failure (`simpeg_ip/Ex`).

### Task 2: Generate report-sized scientific figures

**Files:**
- Modify: `scripts/reports/build_song_cross_code_word_report.py`
- Create temporarily: `tmp/docs/song_cross_code_report/*.png`

- [ ] **Step 1: Draw the evidence-backed Song model schematic**

Draw a two-dimensional section with the 1 km line source, 500 m receiver offset, surface, 0-300 m parameter interval, lower half-space, `z` down, 10 A ideal step-off, and the label `示意图，非比例绘制`. Do not draw a dam or the user's proposed apparatus.

- [ ] **Step 2: Generate no-IP response and error figures**

Create separate three-panel figures for absolute responses and relative errors. Preserve every observation, use logarithmic time, retain the 5% gate, and label empymod as the finite-wire layered reference.

- [ ] **Step 3: Generate IP response and error figures**

Use the same axes and styles as the no-IP figures. Mark the SimPEG `Ex` value at 10 us as a failed point without changing its value.

- [ ] **Step 4: Generate polarization-effect and maximum-error figures**

Plot `(IP/noIP - 1) * 100` for each formal component and solver, then create a grouped maximum-error chart with a visible 5% line. Include the exact published effect-error values in the figure caption data structure.

- [ ] **Step 5: Export figures and inspect dimensions**

Export 300 dpi PNGs with embedded fonts and minimum 8 pt final-size text. Confirm each output is non-empty and at least 1800 pixels wide.

### Task 3: Assemble the Chinese Word report

**Files:**
- Modify: `scripts/reports/build_song_cross_code_word_report.py`
- Create: `output/doc/Song文献模型_FEniCSx_empymod_SimPEG_三维时域正演对比报告.docx`

- [ ] **Step 1: Define document styles and page geometry**

Set A4 portrait pages, 22-25 mm margins, Chinese body and heading fonts, numbered headings, consistent captions, repeating table headers, and page numbers.

- [ ] **Step 2: Write the model and method sections**

Explain the step-off grounded-source physics, the definitions and units of `Ex`, `Hz`, and `dBzdt`, and why an induction coil corresponds to `dB/dt`. State that `Hz` is an internal state/independent validation quantity.

- [ ] **Step 3: Add the three method-parameter tables**

Include separate FEniCSx, empymod, and SimPEG tables. Each row must identify the parameter, actual value, evidence file, and interpretation. Separate common physical parameters from method-specific numerical parameters.

- [ ] **Step 4: Add response, error, and IP-effect interpretation**

Insert the overview and report-sized figures. Explain the maximum and terminal errors, the retained SimPEG IP `Ex` failure, FEniCSx/SimPEG cross-code differences, and the effect errors without presenting SimPEG as a universal failure.

- [ ] **Step 5: Add validity and limitation sections**

State that the validated window is `10 us` to `1 ms`, `Ey` is diagnostic-only, the source is ideal step-off, and the result does not validate a real dam, real coil transfer function, a 100 m electrode array, or a 1 s window.

- [ ] **Step 6: Build the DOCX**

Run:

```powershell
python scripts/reports/build_song_cross_code_word_report.py
```

Expected: final DOCX exists under `output/doc/`, opens with `python-docx`, and contains all required tables and eight figure captions.

### Task 4: Add machine verification for the final report

**Files:**
- Create: `scripts/reports/verify_song_cross_code_word_report.py`

- [ ] **Step 1: Implement required-text and numerical checks**

Open the DOCX with `python-docx`, concatenate paragraphs and table cells, and assert the presence of:

```text
1.036429%, 1.412497%, 3.301669%
1.993756%, 1.090690%, 2.687928%
3.210106%, 0.630309%, 3.452765%
7.420635%, 0.654283%, 3.445477%
```

Also require `10 μs`, `1 ms`, `5%`, `Ey`, `diagnostic-only` or its Chinese equivalent, and the explicit engineering-scope limitation.

- [ ] **Step 2: Check document structure**

Require at least 8 figures, 6 tables, 10 heading paragraphs, and non-empty core/app properties. Reject unresolved drafting markers or replacement tokens.

- [ ] **Step 3: Run verification**

Run:

```powershell
python scripts/reports/verify_song_cross_code_word_report.py output/doc/Song文献模型_FEniCSx_empymod_SimPEG_三维时域正演对比报告.docx
```

Expected: `PASS` with the figure, table, heading and key-value counts.

### Task 5: Render and visually inspect every page

**Files:**
- Create temporarily: `tmp/docs/song_cross_code_report/rendered/`
- Modify if required: `scripts/reports/build_song_cross_code_word_report.py`

- [ ] **Step 1: Render DOCX to PDF and PNG**

Use the bundled document renderer or LibreOffice headless conversion, then render every PDF page to PNG.

- [ ] **Step 2: Inspect every rendered page**

Check at 100% scale for clipped figures, unreadable legends, broken Chinese glyphs, table overflow, orphan headings, blank pages, and captions separated from figures.

- [ ] **Step 3: Fix and rerender**

Adjust widths, page breaks, font sizes or table column widths in the builder, regenerate the DOCX, and repeat until every page is acceptable.

- [ ] **Step 4: Re-run machine verification**

Run the verification script again after the final render. Expected: `PASS` and unchanged key scientific values.

### Task 6: Final repository and artifact checks

**Files:**
- Final: `output/doc/Song文献模型_FEniCSx_empymod_SimPEG_三维时域正演对比报告.docx`
- Final source: `scripts/reports/build_song_cross_code_word_report.py`
- Final verifier: `scripts/reports/verify_song_cross_code_word_report.py`

- [ ] **Step 1: Remove unrelated temporary files**

Delete generated scratch files under `tmp/docs/song_cross_code_report/` that are not needed for the final audit; preserve only render evidence required for final inspection during the session.

- [ ] **Step 2: Check git scope**

Run `git status --short` and confirm no unrelated files are staged or modified.

- [ ] **Step 3: Run final verification commands**

Run the builder check-only mode, the DOCX verifier, and `git diff --check`. All must exit 0.

- [ ] **Step 4: Report the deliverable**

Provide a clickable absolute path to the final DOCX and briefly state the page count, included figures/tables, and verification result.
