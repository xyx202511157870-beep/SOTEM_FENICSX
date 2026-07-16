# Channel Anomaly Diagnostic Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add relative-anomaly decay, signed-anomaly, and representative-time spatial-profile figures to the seepage-channel benchmark and Word report without changing solver data.

**Architecture:** Add two small numerical helpers and two dedicated plotting functions to the existing benchmark plot module. Reuse `_plot_response_grid` for the signed figure, keep all calculations derived from `benchmark_results.npz`, and insert the three new artifacts into the existing report results section.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, python-docx, pytest, Microsoft Word/LibreOffice PDF rendering, Poppler.

---

### Task 1: Relative-anomaly calculation

**Files:**
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Modify: `tools/plot_seepage_channel_benchmark.py`

- [ ] **Step 1: Write the failing numerical-helper test**

```python
def test_relative_anomaly_percent_uses_pointwise_background_with_finite_floor() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    background = np.asarray([[[2.0], [0.0]]])
    delta = np.asarray([[[1.0], [0.25]]])
    result = plots.relative_anomaly_percent(delta, background)

    assert result[0, 0, 0] == 50.0
    assert np.isfinite(result).all()
    assert result[0, 1, 0] == 100.0 * 0.25 / (2.0e-12)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_seepage_channel_plots_manifest.py::test_relative_anomaly_percent_uses_pointwise_background_with_finite_floor -q`

Expected: FAIL with `AttributeError` because `relative_anomaly_percent` does not exist.

- [ ] **Step 3: Implement the helper**

```python
def relative_anomaly_percent(delta: np.ndarray, background: np.ndarray) -> np.ndarray:
    delta_values = np.asarray(delta, dtype=float)
    background_values = np.asarray(background, dtype=float)
    if delta_values.shape != background_values.shape:
        raise ValueError("delta and background must have matching shapes")
    scale = np.max(np.abs(background_values), axis=1, keepdims=True)
    floor = np.maximum(scale * 1.0e-12, np.finfo(float).tiny)
    denominator = np.maximum(np.abs(background_values), floor)
    return 100.0 * np.abs(delta_values) / denominator
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_seepage_channel_plots_manifest.py::test_relative_anomaly_percent_uses_pointwise_background_with_finite_floor -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the numerical helper**

Run:

```powershell
git add tests/test_seepage_channel_plots_manifest.py tools/plot_seepage_channel_benchmark.py
git commit -m "feat: calculate channel relative anomaly"
```

### Task 2: Three diagnostic figures

**Files:**
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Modify: `tools/plot_seepage_channel_benchmark.py`

- [ ] **Step 1: Write failing tests for gate selection and plot contracts**

```python
def test_representative_profile_indices_select_nearest_output_gates() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    times = np.asarray([1.0e-5, 1.0e-4, 3.162e-4, 1.0e-3, 1.0e-2])
    assert plots._representative_profile_indices(times) == (0, 2, 4)


def test_channel_diagnostic_plot_source_contains_new_formal_artifacts() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    assert "channel_relative_anomaly" in plots.FIGURE_STEMS
    assert "channel_delta_signed" in plots.FIGURE_STEMS
    assert "channel_relative_anomaly_profiles" in plots.FIGURE_STEMS


def test_signed_anomaly_plot_preserves_negative_values(tmp_path: Path, monkeypatch) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}
    monkeypatch.setattr(plots, "_save_figure", lambda fig, output_root, stem: captured.setdefault("fig", fig))
    times = np.asarray([1.0e-5, 1.0e-4])
    values = np.ones((5, 2, 3), dtype=float)
    values[:, 1, :] = -0.5
    plots._plot_response_grid(
        tmp_path,
        "channel_delta_signed",
        times,
        {"FEniCSx": values},
        title="signed",
        magnitude_decay=False,
    )
    fig = captured["fig"]
    try:
        assert all(axis.get_yscale() == "symlog" for axis in fig.axes)
        assert any(np.any(np.asarray(line.get_ydata()) < 0.0) for line in fig.axes[0].lines)
    finally:
        plt.close(fig)


def test_relative_anomaly_profiles_exclude_rx3(tmp_path: Path, monkeypatch) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}
    monkeypatch.setattr(plots, "_save_figure", lambda fig, output_root, stem: captured.setdefault("fig", fig))
    times = np.asarray([1.0e-5, 3.162e-4, 1.0e-2])
    receiver_locations = np.column_stack((np.zeros(5), [-20.0, -10.0, 0.0, 10.0, 20.0], np.zeros(5)))
    relative = np.ones((5, 3, 3), dtype=float)
    plots._plot_relative_anomaly_profiles(
        tmp_path,
        times,
        receiver_locations,
        {"FEniCSx": relative},
    )
    fig = captured["fig"]
    try:
        for axis in fig.axes:
            for line in axis.lines:
                np.testing.assert_array_equal(line.get_xdata(), [-20.0, -10.0, 10.0, 20.0])
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_seepage_channel_plots_manifest.py -k "representative_profile or channel_diagnostic or signed" -q`

Expected: FAIL because the new helpers, plot function, and figure stems do not exist.

- [ ] **Step 3: Implement gate selection and diagnostic plotting**

```python
PROFILE_TARGET_TIMES = (1.0e-5, 3.162e-4, 1.0e-2)


def _representative_profile_indices(times: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(times, dtype=float)
    return tuple(int(np.argmin(np.abs(np.log(values) - np.log(target)))) for target in PROFILE_TARGET_TIMES)
```

Implement `_plot_relative_anomaly_grid` as a three-panel log-log time plot with `relative anomaly (%)` y labels. Implement `_plot_relative_anomaly_profiles` as a three-panel plot using physical receiver y positions, the three selected gates, method line styles, gate colors, and `REPORT_RECEIVER_INDICES` only.

```python
def _plot_relative_anomaly_grid(
    output_root: Path,
    times: np.ndarray,
    series: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPORT_RECEIVER_INDICES)))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for color_index, receiver_index in enumerate(REPORT_RECEIVER_INDICES):
                axis.plot(
                    times,
                    values[receiver_index, :, component_index],
                    color=colors[color_index],
                    ls={"SimPEG": "--", "FEniCSx": "-"}[method],
                    lw=1.1,
                    label=f"{method} Rx{receiver_index + 1}" if component_index == 0 else None,
                )
        axis.set(xscale="log", yscale="log", xlabel="time (s)", ylabel="relative anomaly (%)", title=component)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Channel relative anomaly: |channel - background| / |background|")
    _save_figure(fig, output_root, "channel_relative_anomaly")


def _plot_relative_anomaly_profiles(
    output_root: Path,
    times: np.ndarray,
    receiver_locations: np.ndarray,
    series: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    gate_indices = _representative_profile_indices(times)
    y_positions = np.asarray(receiver_locations)[list(REPORT_RECEIVER_INDICES), 1]
    gate_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gate_indices)))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for gate_color, gate_index in zip(gate_colors, gate_indices):
                axis.plot(
                    y_positions,
                    values[list(REPORT_RECEIVER_INDICES), gate_index, component_index],
                    color=gate_color,
                    ls={"SimPEG": "--", "FEniCSx": "-"}[method],
                    marker="o",
                    lw=1.1,
                    label=f"{method}, {times[gate_index]:.3g} s" if component_index == 0 else None,
                )
        axis.set(xlabel="receiver y (m)", ylabel="relative anomaly (%)", title=component)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Channel relative-anomaly profiles at representative times")
    _save_figure(fig, output_root, "channel_relative_anomaly_profiles")
```

Extend `FIGURE_STEMS` with:

```python
"channel_relative_anomaly",
"channel_delta_signed",
"channel_relative_anomaly_profiles",
```

In `generate_plots`, calculate SimPEG and FEniCSx relative arrays, call the two dedicated relative-anomaly plotting functions, and call `_plot_response_grid` with `magnitude_decay=False` for `channel_delta_signed`.

```python
relative_series = {
    "SimPEG": relative_anomaly_percent(arrays["simpeg_delta"], arrays["simpeg_background"]),
    "FEniCSx": relative_anomaly_percent(arrays["fenicsx_delta"], arrays["fenicsx_background"]),
}
_plot_relative_anomaly_grid(output_root, times, relative_series)
_plot_response_grid(
    output_root,
    "channel_delta_signed",
    times,
    {"SimPEG": arrays["simpeg_delta"], "FEniCSx": arrays["fenicsx_delta"]},
    title="Signed channel-minus-background anomaly",
    magnitude_decay=False,
)
_plot_relative_anomaly_profiles(
    output_root,
    times,
    arrays["receiver_locations"],
    relative_series,
)
```

- [ ] **Step 4: Run the full plot-test file and verify GREEN**

Run: `python -m pytest tests/test_seepage_channel_plots_manifest.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the plotting feature**

Run:

```powershell
git add tests/test_seepage_channel_plots_manifest.py tools/plot_seepage_channel_benchmark.py
git commit -m "feat: add channel anomaly diagnostic plots"
```

### Task 3: Word report integration

**Files:**
- Modify: `tests/test_seepage_channel_word_report.py`
- Modify: `tools/build_seepage_channel_word_report.py`

- [ ] **Step 1: Add failing Word source-contract assertions**

```python
def test_report_source_adds_channel_anomaly_diagnostic_figures() -> None:
    source = Path("tools/build_seepage_channel_word_report.py").read_text(encoding="utf-8")
    for filename in (
        "channel_relative_anomaly.png",
        "channel_delta_signed.png",
        "channel_relative_anomaly_profiles.png",
    ):
        assert f'result_dir / "{filename}"' in source
    assert "100 x |F_channel - F_background| / |F_background|" in source
    assert 'result_dir / "channel_delta_error.png", "图 9' in source
```

- [ ] **Step 2: Run the Word test and verify RED**

Run: `python -m pytest tests/test_seepage_channel_word_report.py -q`

Expected: FAIL because the new images and report text are absent.

- [ ] **Step 3: Insert the figures and explanations**

In `add_results`, keep Figures 2-5 unchanged, then add:

```python
add_heading(document, "4.4 通道相对异常百分比", level=2, page_break=True)
add_body(document, "相对异常定义为 100 x |F_channel - F_background| / |F_background|；分母仅在零值处采用数值下限保护。")
add_figure(document, result_dir / "channel_relative_anomaly.png", "图 6  SimPEG 与 FEniCSx 通道相对异常百分比衰减曲线")
add_heading(document, "4.5 通道带符号异常响应", level=2, page_break=True)
add_figure(document, result_dir / "channel_delta_signed.png", "图 7  保留极性和过零信息的通道带符号异常")
add_heading(document, "4.6 典型时刻接收点空间剖面", level=2, page_break=True)
add_figure(document, result_dir / "channel_relative_anomaly_profiles.png", "图 8  三个典型时刻的四接收点相对异常空间剖面")
```

Move the existing channel-delta algorithm-error figure into section 4.7 and caption it as Figure 9. Add one short paragraph explaining that total-field similarity is caused by background dominance and does not mean the channel is absent.

- [ ] **Step 4: Run the Word test and relevant plot tests**

Run: `python -m pytest tests/test_seepage_channel_word_report.py tests/test_seepage_channel_plots_manifest.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit report-source integration**

Run:

```powershell
git add tests/test_seepage_channel_word_report.py tools/build_seepage_channel_word_report.py
git commit -m "docs: add channel anomaly diagnostics to report"
```

### Task 4: Regenerate and verify artifacts

**Files:**
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/channel_relative_anomaly.png`
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/channel_delta_signed.png`
- Regenerate: `output/seepage_channel_100m_5rx_60x1x1/channel_relative_anomaly_profiles.png`
- Regenerate: `output/doc/seepage_channel_100m_magnetic_stability_report.docx`

- [ ] **Step 1: Generate the plots and report**

Run:

```powershell
python tools/plot_seepage_channel_benchmark.py output/seepage_channel_100m_5rx_60x1x1
python tools/build_seepage_channel_word_report.py --result-dir output/seepage_channel_100m_5rx_60x1x1 --output output/doc/seepage_channel_100m_magnetic_stability_report.docx
```

Expected: both commands exit 0 and the six new PNG/PDF artifacts exist.

- [ ] **Step 2: Render the DOCX and inspect every page**

Run:

```powershell
python C:\Users\paidaxin\.codex\skills\doc\scripts\render_docx.py output\doc\seepage_channel_100m_magnetic_stability_report.docx --output_dir tmp\docs\channel_anomaly_report
```

Inspect every generated PNG at full resolution. Correct any clipped axes, unreadable captions, orphaned headings, or overlapping content, then rerun the same command.

- [ ] **Step 3: Run final automated verification**

Run:

```powershell
python -m pytest tests/test_seepage_channel_plots_manifest.py tests/test_seepage_channel_word_report.py tests/test_seepage_channel_aggregation.py tests/test_seepage_channel_validation.py -q
git diff --check
git status --short
```

Expected: all selected tests pass, `git diff --check` prints nothing, and only intended generated artifacts remain untracked or modified.

- [ ] **Step 4: Copy the verified report and figures to the main workspace output**

Run:

```powershell
$result = 'output\seepage_channel_100m_5rx_60x1x1'
$mainResult = '..\..\output\seepage_channel_100m_5rx_60x1x1'
foreach ($name in @('channel_relative_anomaly.png','channel_relative_anomaly.pdf','channel_delta_signed.png','channel_delta_signed.pdf','channel_relative_anomaly_profiles.png','channel_relative_anomaly_profiles.pdf')) {
    Copy-Item -LiteralPath (Join-Path $result $name) -Destination (Join-Path $mainResult $name) -Force
}
Copy-Item -LiteralPath 'output\doc\seepage_channel_100m_magnetic_stability_report.docx' -Destination '..\..\output\doc\seepage_channel_100m_magnetic_stability_report.docx' -Force
Get-FileHash -Algorithm SHA256 (Join-Path $result 'channel_relative_anomaly.png'), (Join-Path $mainResult 'channel_relative_anomaly.png')
Get-FileHash -Algorithm SHA256 'output\doc\seepage_channel_100m_magnetic_stability_report.docx', '..\..\output\doc\seepage_channel_100m_magnetic_stability_report.docx'
```

Expected: each source/destination pair has matching SHA-256 values.

- [ ] **Step 5: Commit tracked source and documentation changes if any remain**

Run `git status --short`, stage only files listed in this plan, and commit with a narrowly scoped message.
