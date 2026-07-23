# Zhou 2020 Figure and Reference-Stability Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading signed-log Zhou 2020 plots with literature-style absolute-magnitude figures, publish a reproducible `dBz/dt` reference-transform stability audit, and regenerate a scientifically accurate DOCX report without rerunning the formal FEniCSx fields.

**Architecture:** Add a small, pure evidence module that computes transform-spread diagnostics and fail-closed status records from signed arrays. A thin audit CLI produces immutable JSON/NPZ artifacts from the existing formal run and an independently transformed empymod reference. The plotting and DOCX scripts consume those artifacts; they do not recalculate acceptance status or silently remove samples.

**Tech Stack:** Python 3.11, NumPy, empymod, matplotlib, pytest, python-docx, ZIP/XML package checks, Windows scientific Conda, WSL2 FEniCSx Conda.

---

## File map and responsibility boundaries

- Create `src/atem3d/zhou2020_reference_stability.py`
  - Pure validation and metric functions.
  - No matplotlib, DOCX, CLI parsing, or filesystem publishing.
- Create `scripts/audit_zhou2020_reference_stability.py`
  - Real empymod DLF/QWE execution and atomic JSON/NPZ publication.
  - Reads the existing formal run; never modifies `strict_comparison.json`.
- Create `tests/test_zhou2020_reference_stability.py`
  - Unit tests for sign changes, stable-window detection, fail-closed QWE status, and preservation of all samples.
- Create `tests/test_audit_zhou2020_reference_stability.py`
  - Thin CLI/artifact contract tests using injected arrays, without expensive empymod calls.
- Create `tests/test_plot_zhou2020_strict_validation.py`
  - Axis-scale, absolute-value, no-PDF, shaded-window, and Debye 4-vs-16 tests.
- Create `tests/test_zhou2020_validation_report.py`
  - Hermetic DOCX language, embedded-image, ZIP integrity, and forbidden-claim tests.
- Modify `scripts/plot_zhou2020_strict_validation.py`
  - Literature-style absolute plots, signed linear diagnostic, audit-aware gate summary, Debye pole-count sensitivity.
- Modify `scripts/build_zhou2020_validation_report.py`
  - Consume audit evidence and replace unsupported pass/fail wording.
- Preserve without staging:
  - `dolfinx/run_sotem_benchmark.py`
  - `tests/test_run_sotem_benchmark.py`
- Preserve as immutable evidence:
  - `generated/validation/zhou2020_grounded_wire/runs/20260723T062004Z_zhou_strict_v2/comparisons/S1T1B1/strict_comparison.json`

## Task 1: Pure reference-stability evidence model

**Files:**
- Create: `src/atem3d/zhou2020_reference_stability.py`
- Create: `tests/test_zhou2020_reference_stability.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zhou2020_reference_stability.py` with these public-contract tests:

```python
import numpy as np
import pytest

from atem3d.zhou2020_reference_stability import (
    build_reference_stability_audit,
    first_stable_sample,
    sign_change_count,
)


def test_sign_change_count_ignores_exact_zero_samples():
    assert sign_change_count([1.0, 0.0, -2.0, -3.0, 0.0, 4.0]) == 2


def test_first_stable_sample_requires_consecutive_signal_over_spread():
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.array([0.1, 0.1, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    default = direct + np.array([0.2, -0.2, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02])
    separate = direct + np.array([-0.2, 0.2, -0.2, -0.02, -0.02, -0.02, -0.02, -0.02])

    index = first_stable_sample(
        times,
        np.column_stack([default, separate, direct]),
        signal_to_spread=3.0,
        consecutive=3,
    )

    assert index == 3
    assert times[index] == pytest.approx(times[3])


def test_audit_is_inconclusive_when_qwe_did_not_converge_and_retains_samples():
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.array([1e-14, 2e-14, 3e-14, 1e-11, 2e-11, 3e-11, 4e-11, 5e-11])
    default = direct.copy()
    default[:3] = [1e-10, -1e-10, 1e-10]
    separate = direct * 1.01
    fenicsx = direct * 1.05

    result = build_reference_stability_audit(
        times=times,
        default_dlf=default,
        separate_total_qwe=separate,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=fenicsx,
        signal_to_spread=3.0,
        consecutive=3,
    )

    assert result["schema"] == "atem3d.zhou2020.reference-stability/v1"
    assert result["status"] == "inconclusive"
    assert result["qwe"]["converged"] is False
    assert result["all_samples_retained"] is True
    assert result["sample_count"] == len(times)
    assert result["default_dlf"]["sign_changes_first20"] == 2
    assert result["stable_window"]["start_s"] == pytest.approx(times[3])
    assert result["fenicsx_vs_direct_qwe"]["relative_l2_full"] == pytest.approx(0.05)


def test_audit_never_promotes_nonconverged_qwe_to_passed():
    times = np.geomspace(1.0e-4, 1.0e-2, 6)
    values = np.geomspace(1.0e-12, 1.0e-9, 6)

    result = build_reference_stability_audit(
        times=times,
        default_dlf=values,
        separate_total_qwe=values,
        direct_frequency_qwe=values,
        direct_qwe_converged=False,
        fenicsx_increment=values,
        consecutive=2,
    )

    assert result["status"] == "inconclusive"
    assert result["formal_gate_decision"] is None
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_reference_stability.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'atem3d.zhou2020_reference_stability'`.

- [ ] **Step 3: Implement the pure evidence module**

Create `src/atem3d/zhou2020_reference_stability.py` with:

```python
"""Reference-transform stability evidence for the Zhou 2020 benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA = "atem3d.zhou2020.reference-stability/v1"


def sign_change_count(values) -> int:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("values must be a finite one-dimensional array")
    signs = np.sign(array)
    signs = signs[signs != 0.0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("reference norm must be positive")
    return float(np.linalg.norm(values - reference) / denominator)


def _validated_inputs(times, *arrays) -> tuple[np.ndarray, list[np.ndarray]]:
    time_values = np.asarray(times, dtype=float)
    if (
        time_values.ndim != 1
        or time_values.size < 2
        or not np.isfinite(time_values).all()
        or np.any(time_values <= 0.0)
        or np.any(np.diff(time_values) <= 0.0)
    ):
        raise ValueError("times must be finite, positive, and strictly increasing")
    validated = []
    for values in arrays:
        array = np.asarray(values, dtype=float)
        if array.shape != time_values.shape or not np.isfinite(array).all():
            raise ValueError("each response must be finite and match times")
        validated.append(array)
    return time_values, validated


def first_stable_sample(
    times,
    candidates,
    *,
    signal_to_spread: float = 3.0,
    consecutive: int = 5,
) -> int:
    time_values = np.asarray(times, dtype=float)
    matrix = np.asarray(candidates, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != time_values.size or matrix.shape[1] < 2:
        raise ValueError("candidates must have shape (n_times, n_methods>=2)")
    if consecutive < 1 or signal_to_spread <= 0.0:
        raise ValueError("consecutive and signal_to_spread must be positive")
    centre = np.median(matrix, axis=1)
    spread = np.max(matrix, axis=1) - np.min(matrix, axis=1)
    stable = np.abs(centre) >= signal_to_spread * spread
    for index in range(0, stable.size - consecutive + 1):
        if bool(np.all(stable[index : index + consecutive])):
            return index
    raise ValueError("no stable consecutive window found")


def build_reference_stability_audit(
    *,
    times,
    default_dlf,
    separate_total_qwe,
    direct_frequency_qwe,
    direct_qwe_converged: bool,
    fenicsx_increment,
    signal_to_spread: float = 3.0,
    consecutive: int = 5,
) -> dict[str, Any]:
    time_values, arrays = _validated_inputs(
        times,
        default_dlf,
        separate_total_qwe,
        direct_frequency_qwe,
        fenicsx_increment,
    )
    default, separate, direct, fenicsx = arrays
    stable_index = first_stable_sample(
        time_values,
        np.column_stack([default, separate, direct]),
        signal_to_spread=signal_to_spread,
        consecutive=consecutive,
    )
    return {
        "schema": SCHEMA,
        "status": "inconclusive" if not direct_qwe_converged else "audited",
        "formal_gate_decision": None,
        "all_samples_retained": True,
        "sample_count": int(time_values.size),
        "default_dlf": {
            "sign_changes_all": sign_change_count(default),
            "sign_changes_first20": sign_change_count(default[:20]),
        },
        "qwe": {"converged": bool(direct_qwe_converged)},
        "stable_window": {
            "start_index": int(stable_index),
            "start_s": float(time_values[stable_index]),
            "signal_to_spread": float(signal_to_spread),
            "consecutive": int(consecutive),
        },
        "transform_difference": {
            "default_dlf_vs_direct_qwe_relative_l2_full": _relative_l2(default, direct),
            "default_dlf_vs_direct_qwe_relative_l2_first20": _relative_l2(
                default[:20], direct[:20]
            ),
        },
        "fenicsx_vs_direct_qwe": {
            "relative_l2_full": _relative_l2(fenicsx, direct),
            "relative_l2_stable_window": _relative_l2(
                fenicsx[stable_index:], direct[stable_index:]
            ),
        },
    }
```

- [ ] **Step 4: Run the unit tests and verify GREEN**

Run the Step 2 command again.

Expected: `4 passed`.

- [ ] **Step 5: Commit only the new module and tests**

```powershell
git add -- src/atem3d/zhou2020_reference_stability.py tests/test_zhou2020_reference_stability.py
git diff --cached --check
git commit -m "test: define Zhou reference stability evidence"
```

Expected: the commit contains exactly two files.

## Task 2: Reproducible DLF/QWE audit publisher

**Files:**
- Create: `scripts/audit_zhou2020_reference_stability.py`
- Create: `tests/test_audit_zhou2020_reference_stability.py`
- Modify: `src/atem3d/zhou2020_reference_stability.py`

- [ ] **Step 1: Write the failing artifact-contract test**

Create `tests/test_audit_zhou2020_reference_stability.py`:

```python
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_zhou2020_reference_stability.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("zhou_reference_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_publish_audit_writes_json_and_npz_without_modifying_strict_json(tmp_path):
    module = _load_script()
    run = tmp_path / "run"
    comparison = run / "comparisons/S1T1B1"
    comparison.mkdir(parents=True)
    strict = comparison / "strict_comparison.json"
    strict.write_text('{"status":"failed_with_reproducible_evidence"}\n', encoding="utf-8")
    before = strict.read_bytes()
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.geomspace(1.0e-12, 1.0e-9, 8)

    module.publish_audit(
        run=run,
        output=tmp_path / "audit",
        times=times,
        default_dlf=direct * 1.02,
        separate_total_qwe=direct * 1.01,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=direct * 1.05,
        consecutive=2,
    )

    assert strict.read_bytes() == before
    payload = json.loads((tmp_path / "audit/reference_stability.json").read_text("utf-8"))
    arrays = np.load(tmp_path / "audit/reference_stability.npz")
    assert payload["status"] == "inconclusive"
    assert payload["input_sha256"]["strict_comparison.json"]
    assert arrays["time_s"].shape == (8,)
    assert set(arrays.files) == {
        "time_s",
        "default_dlf",
        "separate_total_qwe",
        "direct_frequency_qwe",
        "fenicsx_increment",
    }
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_audit_zhou2020_reference_stability.py -q
```

Expected: the script does not exist or `publish_audit` is absent.

- [ ] **Step 3: Add atomic artifact publication**

Add `sha256_file`, `_atomic_write_json`, `_atomic_write_npz`, and `publish_audit` to the new script. `publish_audit` must:

```python
def publish_audit(
    *,
    run: Path,
    output: Path,
    times,
    default_dlf,
    separate_total_qwe,
    direct_frequency_qwe,
    direct_qwe_converged: bool,
    fenicsx_increment,
    consecutive: int = 5,
) -> dict:
    strict = run / "comparisons/S1T1B1/strict_comparison.json"
    before = sha256_file(strict)
    audit = build_reference_stability_audit(
        times=times,
        default_dlf=default_dlf,
        separate_total_qwe=separate_total_qwe,
        direct_frequency_qwe=direct_frequency_qwe,
        direct_qwe_converged=direct_qwe_converged,
        fenicsx_increment=fenicsx_increment,
        consecutive=consecutive,
    )
    audit["input_sha256"] = {"strict_comparison.json": before}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_npz(
        output / "reference_stability.npz",
        time_s=np.asarray(times),
        default_dlf=np.asarray(default_dlf),
        separate_total_qwe=np.asarray(separate_total_qwe),
        direct_frequency_qwe=np.asarray(direct_frequency_qwe),
        fenicsx_increment=np.asarray(fenicsx_increment),
    )
    _atomic_write_json(output / "reference_stability.json", audit)
    if sha256_file(strict) != before:
        raise RuntimeError("strict_comparison.json changed during audit publication")
    return audit
```

Use temporary files in `output.parent` followed by `os.replace`; set `allow_nan=False`.

- [ ] **Step 4: Add real empymod audit execution**

Implement these CLI arguments:

```python
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--srcpts", type=int, default=17)
```

The real execution must:

1. Load default DLF from `run/reference/empymod_ip.csv - empymod_noip.csv`.
2. Load FEniCSx increment from the two `predictions.csv` files.
3. Compute `separate_total_qwe` by transforming IP and no-IP totals separately with:

```python
QWE = {
    "rtol": 1.0e-8,
    "atol": 1.0e-20,
    "nquad": 51,
    "maxint": 1000,
    "pts_per_dec": 60,
}
```

4. Compute `direct_frequency_qwe` by subtracting complex IP/no-IP frequency responses before `empymod.model.tem`.
5. Preserve the boolean returned by `tem`; do not replace `False` with a successful status.
6. Call `publish_audit`.

- [ ] **Step 5: Run tests and the formal audit**

Run unit tests:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_reference_stability.py tests/test_audit_zhou2020_reference_stability.py -q
```

Expected: all tests pass.

Run the real audit:

```powershell
$run = "generated/validation/zhou2020_grounded_wire/runs/20260723T062004Z_zhou_strict_v2"
$out = "output/zhou2020_strict_validation/reference_audit"
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe scripts/audit_zhou2020_reference_stability.py `
  --run $run `
  --case benchmarks/sotem/zhou2020_grounded_wire.yaml `
  --output $out
```

Expected evidence:

- `status == "inconclusive"`;
- `qwe.converged == false`;
- default DLF first-20 sign changes equal 10;
- stable window starts close to `5.77e-4 s`;
- DLF/direct-QWE full-window difference close to 6.44%;
- DLF/direct-QWE first-20 difference close to 93.65%;
- FEniCSx/direct-QWE full-window sensitivity close to 9.44%.

- [ ] **Step 6: Commit the publisher and tests**

```powershell
git add -- src/atem3d/zhou2020_reference_stability.py scripts/audit_zhou2020_reference_stability.py tests/test_audit_zhou2020_reference_stability.py
git diff --cached --check
git commit -m "feat: publish Zhou reference transform audit"
```

## Task 3: Literature-style figures and signed stability diagnostic

**Files:**
- Create: `tests/test_plot_zhou2020_strict_validation.py`
- Modify: `scripts/plot_zhou2020_strict_validation.py`

- [ ] **Step 1: Write plotting-contract tests**

Create tests that import the script with `importlib.util` and verify:

```python
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_zhou2020_strict_validation.py"


def _load_plotter():
    spec = importlib.util.spec_from_file_location("zhou_plotter_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_data():
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    base = np.geomspace(1.0e-7, 1.0e-9, 8)
    return {
        "time_s": times,
        "Ex_V_per_m": -base,
        "Hz_A_per_m": 2.0 * base,
        "dBzdt_T_per_s": -3.0 * base,
    }


def _write_verification_npz(path, scale):
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    values = np.column_stack(
        [scale * times, 2.0 * scale * times, -3.0 * scale * times]
    )
    np.savez_compressed(
        path,
        times=times,
        components=np.array(["Ex", "Hz", "dBzdt"]),
        fem=values,
    )


def test_positive_magnitude_masks_only_exact_zeros():
    plotter = _load_plotter()
    values = np.array([-2.0, 0.0, 3.0])
    plotted = plotter._positive_magnitude(values)
    np.testing.assert_allclose(plotted.compressed(), [2.0, 3.0])
    np.testing.assert_allclose(values, [-2.0, 0.0, 3.0])


def test_total_field_axes_are_log_log_and_never_symlog(tmp_path):
    plotter = _load_plotter()
    noip_fem = _sample_data()
    noip_ref = _sample_data()
    ip_fem = {name: values.copy() for name, values in noip_fem.items()}
    ip_ref = {name: values.copy() for name, values in noip_ref.items()}
    fig = plotter.plot_total_fields(noip_fem, ip_fem, noip_ref, ip_ref, None)
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert all(ax.get_yscale() == "log" for ax in fig.axes)
    assert "absolute magnitude" in fig._suptitle.get_text().lower()
    plt.close(fig)


def test_signed_reference_panel_uses_linear_y_and_shades_unstable_window():
    plotter = _load_plotter()
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    arrays = {
        "time_s": times,
        "default_dlf": np.array([1, -1, 1, 2, 3, 4, 5, 6]) * 1.0e-10,
        "direct_frequency_qwe": np.geomspace(1.0e-13, 1.0e-9, 8),
        "fenicsx_increment": np.geomspace(1.1e-13, 1.1e-9, 8),
    }
    audit = {
        "status": "inconclusive",
        "qwe": {"converged": False},
        "default_dlf": {"sign_changes_first20": 2},
        "stable_window": {"start_s": float(times[3])},
    }
    fig = plotter.plot_reference_stability(arrays, audit, None)
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[1].get_yscale() == "linear"
    assert fig.axes[1].get_xscale() == "log"
    assert len(fig.axes[0].patches) >= 1
    assert "signed diagnostic" in fig.axes[1].get_title().lower()
    plt.close(fig)


def test_debye_metric_is_four_relative_to_sixteen_not_empymod(tmp_path):
    plotter = _load_plotter()
    noip_path = tmp_path / "noip.npz"
    ip4_path = tmp_path / "ip4.npz"
    ip16_path = tmp_path / "ip16.npz"
    _write_verification_npz(noip_path, 1.0)
    _write_verification_npz(ip4_path, 1.9)
    _write_verification_npz(ip16_path, 2.0)
    result, fig = plotter.plot_debye_order_diagnostic(
        noip_path, ip4_path, ip16_path, None
    )
    assert "debye_4_vs_16_relative_l2" in result["comparison"]["Ex"]
    assert "debye_4_relative_l2" not in result["comparison"]["Ex"]
    assert all(ax.get_yscale() == "log" for ax in fig.axes)
    plt.close(fig)


def test_save_all_does_not_create_pdf(tmp_path):
    plotter = _load_plotter()
    fig = plt.figure()
    plotter._save_all(fig, tmp_path / "figure")
    assert (tmp_path / "figure.svg").exists()
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.tiff").exists()
    assert not (tmp_path / "figure.pdf").exists()
    plt.close(fig)
```

Functions accept `stem: Path | None`; when `stem is None`, return the open figure for inspection instead of writing or closing it.

- [ ] **Step 2: Verify RED**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_plot_zhou2020_strict_validation.py -q
```

Expected failures: missing `_positive_magnitude`, `symlog` axes, PDF output, and old Debye keys.

- [ ] **Step 3: Implement absolute-magnitude total-field plotting**

Add:

```python
def _positive_magnitude(values: np.ndarray) -> np.ma.MaskedArray:
    magnitude = np.abs(np.asarray(values, dtype=float))
    return np.ma.masked_equal(magnitude, 0.0, copy=True)


def _finish_or_save(fig: plt.Figure, stem: Path | None) -> plt.Figure | None:
    fig.tight_layout()
    if stem is None:
        return fig
    _save_all(fig, stem)
    plt.close(fig)
    return None
```

In `plot_total_fields`, plot `_positive_magnitude(f)` and `_positive_magnitude(r)`, use:

```python
ax.set_xscale("log")
ax.set_yscale("log")
fig.suptitle(
    "Absolute-magnitude total fields: literature-style log-log comparison",
    y=0.995,
)
```

Do not change the signed arrays used by `strict_comparison.json`.

- [ ] **Step 4: Replace Figure 3 with the reference-stability audit**

Implement:

```python
def plot_reference_stability(arrays: dict[str, np.ndarray], audit: dict, stem):
    times = arrays["time_s"]
    unstable_end = audit["stable_window"]["start_s"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax in axes:
        ax.axvspan(times[0], unstable_end, color="#D9D9D9", alpha=0.55)
    for name, label, style in (
        ("default_dlf", "default DLF", "--"),
        ("direct_frequency_qwe", "direct-frequency QWE", "-"),
        ("fenicsx_increment", "FEniCSx", "-"),
    ):
        axes[0].plot(times, _positive_magnitude(arrays[name]), style, label=label)
        axes[1].plot(times, arrays[name], style, label=label)
    axes[0].set(xscale="log", yscale="log", title="(a) Absolute magnitude")
    axes[1].set_xscale("log")
    axes[1].set_yscale("linear")
    axes[1].set_title("(b) Signed diagnostic")
    return _finish_or_save(fig, stem)
```

The annotation must state:

- `reference-transform unstable`;
- default DLF first-20 sign changes;
- QWE `converged=False`;
- `status=inconclusive`;
- all samples retained.

- [ ] **Step 5: Make the gate summary audit-aware**

Keep formal total-field L2 bars. Do not display Ex/Hz/`dBzdt` IP-increment bars as ordinary red pass/fail bars. Add a grey hatched evidence block:

```text
IP increments: sensitivity only
Ex 16.32%, Hz 61.26%, dBz/dt 11.40% (default DLF)
reference stability not established -> inconclusive
```

Keep a separate red warning for the independent Hz false reversal at about `0.022 s`.

- [ ] **Step 6: Convert Debye Figure 5 to internal pole-count sensitivity**

Remove `reference_noip_path` and `reference_ip_path` from the function signature. For every component:

```python
change = np.linalg.norm(delta4 - delta16) / np.linalg.norm(delta16)
result["comparison"][component] = {
    "debye_4_vs_16_relative_l2": float(change),
}
```

Plot only `|delta4|` and `|delta16|` on log-log axes. Expected actual values are approximately:

- Ex: 9.20%;
- Hz: 9.84%;
- `dBz/dt`: 6.40%.

Label the figure `pole-count sensitivity`, not `cross-code validation`.

- [ ] **Step 7: Remove PDF export and wire the new audit inputs**

Change `_save_all` to write only SVG, PNG, and TIFF.

Add:

```python
parser.add_argument("--reference-audit", type=Path, required=True)
```

Load:

```python
audit = json.loads(
    (args.reference_audit / "reference_stability.json").read_text(encoding="utf-8")
)
audit_arrays = dict(np.load(args.reference_audit / "reference_stability.npz"))
```

Generate `fig03_reference_stability` instead of the old misleading signed-IP-increment figure.

- [ ] **Step 8: Run tests and figure-source preflight**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_plot_zhou2020_strict_validation.py -q
C:\Users\paidaxin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  C:\Users\paidaxin\.codex\skills\nature-figure\scripts\validate_figure.py `
  scripts/plot_zhou2020_strict_validation.py --backend python --strict
```

Expected: pytest passes; static preflight reports ready with no strict warning.

- [ ] **Step 9: Commit plotting changes**

```powershell
git add -- scripts/plot_zhou2020_strict_validation.py tests/test_plot_zhou2020_strict_validation.py
git diff --cached --check
git commit -m "fix: use literature-style Zhou response figures"
```

## Task 4: Scientifically accurate DOCX language and package tests

**Files:**
- Create: `tests/test_zhou2020_validation_report.py`
- Modify: `scripts/build_zhou2020_validation_report.py`

- [ ] **Step 1: Write a hermetic report test**

The test must build a temporary report from synthetic JSON and five simple PNGs, monkeypatch `_git_state`, then inspect the DOCX with `python-docx` and `zipfile`:

```python
import importlib.util
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_zhou2020_validation_report.py"


def _load_report_script():
    spec = importlib.util.spec_from_file_location("zhou_report_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metric(value, gate, passed):
    return {"relative_l2": value, "gate": gate, "passed": passed}


def _make_report_fixture(tmp_path):
    root = tmp_path
    run = root / "generated/run"
    comparison = run / "comparisons/S1T1B1"
    reference = run / "reference"
    figures = root / "figures"
    audit_dir = figures / "reference_audit"
    comparison.mkdir(parents=True)
    reference.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    total = {
        variant: {
            component: _metric(0.03, 0.05, True)
            for component in ("Ex", "Hz", "dBzdt")
        }
        for variant in ("noip", "ip")
    }
    metrics = {
        "status": "failed_with_reproducible_evidence",
        "total_field": total,
        "ip_increment": {
            "Ex": _metric(0.1632, 0.10, False),
            "Hz": _metric(0.6126, 0.10, False),
            "dBzdt": _metric(0.1140, 0.10, False),
        },
        "zero_crossings": {
            "noip": {
                "Hz": {
                    "prediction": [0.022],
                    "reference": [],
                    "passed": False,
                }
            },
            "ip": {
                "Hz": {
                    "prediction": [0.022],
                    "reference": [],
                    "passed": False,
                },
                "Ex": {
                    "prediction": [0.0576942],
                    "reference": [0.0582826],
                    "max_relative_time_error": 0.010096,
                    "passed": True,
                },
            },
        },
    }
    (comparison / "strict_comparison.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (reference / "empymod_srcpts_convergence.json").write_text(
        json.dumps({"max_relative_difference": 0.001}),
        encoding="utf-8",
    )
    audit = {
        "status": "inconclusive",
        "qwe": {"converged": False},
        "default_dlf": {"sign_changes_first20": 10},
        "stable_window": {"start_s": 5.77e-4},
        "transform_difference": {
            "default_dlf_vs_direct_qwe_relative_l2_full": 0.0644,
            "default_dlf_vs_direct_qwe_relative_l2_first20": 0.9365,
        },
        "fenicsx_vs_direct_qwe": {
            "relative_l2_full": 0.0944,
            "relative_l2_stable_window": 0.0939,
        },
    }
    (audit_dir / "reference_stability.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    diagnostic = {
        "comparison": {
            "Ex": {"debye_4_vs_16_relative_l2": 0.0920},
            "Hz": {"debye_4_vs_16_relative_l2": 0.0984},
            "dBzdt": {"debye_4_vs_16_relative_l2": 0.0640},
        }
    }
    (figures / "debye_order_diagnostic.json").write_text(
        json.dumps(diagnostic), encoding="utf-8"
    )
    for name in (
        "fig01_model_contract",
        "fig02_total_fields",
        "fig03_reference_stability",
        "fig04_gate_summary",
        "fig05_debye_order_diagnostic",
    ):
        fig, ax = plt.subplots(figsize=(2.0, 1.0))
        ax.plot([0.0, 1.0], [0.0, 1.0])
        fig.savefig(figures / f"{name}.png", dpi=80)
        plt.close(fig)
    return root, run, figures


def _document_text(path):
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return "\n".join([*paragraphs, *cells])


def test_report_uses_absolute_plot_and_inconclusive_reference_language(tmp_path, monkeypatch):
    report = _load_report_script()
    root, run, figures = _make_report_fixture(tmp_path)
    monkeypatch.setattr(report, "_git_state", lambda root: ("test-commit", False))
    output = tmp_path / "report.docx"

    report.build_report(root, run, figures, output)

    text = _document_text(output)
    assert "绝对值双对数" in text
    assert "绝对值不提供符号信息" in text
    assert "reference-transform unstable" in text
    assert "inconclusive" in text
    assert "0.022 s" in text
    assert "9.44%" in text
    assert "QWE" in text
    assert "4 项相对 16 项" in text
    assert "dBz/dt 极化模块严格通过" not in text
    assert "Ex、Hz 与 dBz/dt 均未同时满足 10% 严格门槛" not in text
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
        assert len(media) >= 5
```

The fixture writes:

- `strict_comparison.json` with the preserved 16.32%, 61.26%, and 11.40% values;
- `reference_stability.json` with 9.44%, `converged=false`, and `status=inconclusive`;
- `debye_order_diagnostic.json` with 9.20%, 9.84%, and 6.40%;
- `empymod_srcpts_convergence.json`;
- blank `fig01` through `fig05` PNGs.

- [ ] **Step 2: Verify RED**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_validation_report.py -q
```

Expected: required audit language is absent and the old forbidden sentence is present.

- [ ] **Step 3: Load and validate the reference audit in the report**

At the beginning of `build_report`, add:

```python
reference_audit = json.loads(
    (figures / "reference_audit/reference_stability.json").read_text(
        encoding="utf-8"
    )
)
if reference_audit["status"] != "inconclusive":
    raise ValueError("this report revision expects an inconclusive QWE audit")
if reference_audit["qwe"]["converged"] is not False:
    raise ValueError("QWE convergence evidence changed; re-review the report wording")
```

This prevents stale prose if the numerical evidence changes later.

- [ ] **Step 4: Revise the metric table and summary**

Change `_metric_rows` to accept `reference_audit`. Keep no-IP/IP total-field decisions. Render each IP-increment row as:

```text
敏感性数值（不判定）
```

Replace the summary statement with:

```text
总场及 Ex 反号提供部分支持；Hz 约 0.022 s 的伪反号仍是独立失败证据。
独立 IP 增量所用 empymod 弱差分未通过参考变换稳定性审计，因此其状态为
inconclusive，不能据 11.40% 宣称失败，也不能据 QWE 敏感性结果 9.44% 宣称通过。
```

- [ ] **Step 5: Replace figure captions and Section 5.1**

Use these exact meanings:

- Figure 2: `纵轴为响应绝对值的双对数坐标；绝对值不提供符号信息。`
- Figure 3: `dBz/dt 独立 IP 增量参考稳定性审计；灰区为 reference-transform unstable，全部样本保留。`
- Figure 5: `同网格、同时间步下 4 项相对 16 项 Debye 的 pole-count sensitivity。`

Section 5.1 must report:

- early total field near `4e-7 T/s`;
- early increment near `1e-14` to `1e-10 T/s`;
- default DLF first-20 sign changes = 10;
- stable boundary near `5.77e-4 s`;
- DLF/direct-QWE differences 6.44% full and 93.65% first 20;
- FEniCSx/direct-QWE sensitivity 9.44% full and 9.39% stable window;
- QWE global convergence remains false.

- [ ] **Step 6: Revise Debye and conclusion sections**

The Debye table headings become:

```text
分量 | 4 项相对 16 项内部变化 | 解释
```

The conclusion must preserve:

- total-field L2 values;
- Ex zero-crossing agreement;
- Hz false reversal at about `0.022 s`;
- original weak-increment values as reproducible sensitivity evidence;
- overall algorithm status is not fully passed.

It must not call the weak increments a formal strict failure because the reference is unstable.

- [ ] **Step 7: Run DOCX tests**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_validation_report.py -q
```

Expected: all report tests pass.

- [ ] **Step 8: Commit report changes**

```powershell
git add -- scripts/build_zhou2020_validation_report.py tests/test_zhou2020_validation_report.py
git diff --cached --check
git commit -m "docs: correct Zhou validation evidence language"
```

## Task 5: Regenerate figures and DOCX from existing formal fields

**Files:**
- Generate: `output/zhou2020_strict_validation/fig01_model_contract.{svg,png,tiff}`
- Generate: `output/zhou2020_strict_validation/fig02_total_fields.{svg,png,tiff}`
- Generate: `output/zhou2020_strict_validation/fig03_reference_stability.{svg,png,tiff}`
- Generate: `output/zhou2020_strict_validation/fig04_gate_summary.{svg,png,tiff}`
- Generate: `output/zhou2020_strict_validation/fig05_debye_order_diagnostic.{svg,png,tiff}`
- Generate: `output/zhou2020_strict_validation/debye_order_diagnostic.json`
- Generate: `output/doc/Zhou2020接地线源TEM-IP_FEniCSx_empymod严格验证报告.docx`

- [ ] **Step 1: Hash the immutable formal evidence before regeneration**

```powershell
$run = "generated/validation/zhou2020_grounded_wire/runs/20260723T062004Z_zhou_strict_v2"
Get-FileHash "$run/comparisons/S1T1B1/strict_comparison.json" -Algorithm SHA256
Get-FileHash "$run/fenicsx/noip/S1T1B1/verification_data.npz" -Algorithm SHA256
Get-FileHash "$run/fenicsx/ip/S1T1B1/verification_data.npz" -Algorithm SHA256
```

Save the three values in the execution log.

- [ ] **Step 2: Regenerate all figures without PDF**

```powershell
$run = "generated/validation/zhou2020_grounded_wire/runs/20260723T062004Z_zhou_strict_v2"
$compute = "generated/validation/zhou2020_grounded_wire/compute"
$figures = "output/zhou2020_strict_validation"
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe scripts/plot_zhou2020_strict_validation.py `
  --run $run `
  --compute-root $compute `
  --reference-audit "$figures/reference_audit" `
  --output $figures
```

Expected:

- no `.pdf` is created;
- Figure 2 contains only positive log y axes;
- Figure 3 visibly shades the early unstable window;
- Figure 5 contains only 4- and 16-Debye curves.

- [ ] **Step 3: Regenerate the DOCX only**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe scripts/build_zhou2020_validation_report.py `
  --root . `
  --run $run `
  --figures $figures `
  --output "output/doc/Zhou2020接地线源TEM-IP_FEniCSx_empymod严格验证报告.docx"
```

Expected: DOCX exists and no user-delivery PDF is generated.

- [ ] **Step 4: Re-hash immutable evidence**

Repeat Step 1 and assert all three SHA-256 values are unchanged.

- [ ] **Step 5: Run package and language checks**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -c @'
from docx import Document
from pathlib import Path
import zipfile

path = Path(r"output/doc/Zhou2020接地线源TEM-IP_FEniCSx_empymod严格验证报告.docx")
with zipfile.ZipFile(path) as archive:
    assert archive.testzip() is None
    media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert len(media) >= 5
doc = Document(path)
text = "\n".join(
    [p.text for p in doc.paragraphs]
    + [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
)
for required in (
    "绝对值双对数",
    "绝对值不提供符号信息",
    "reference-transform unstable",
    "inconclusive",
    "0.022 s",
    "9.44%",
    "4 项相对 16 项",
):
    assert required in text, required
assert "dBz/dt 极化模块严格通过" not in text
print("docx_package_ok", len(doc.paragraphs), len(doc.tables), len(media))
'@
```

- [ ] **Step 6: Inspect every scientific figure**

Open the five PNGs at original detail and verify:

- labels are readable;
- no negative logarithmic tick labels exist in Figures 2 and 5;
- Figure 3 panel b crosses zero on a linear y axis;
- the grey unstable region is visible and labeled;
- legends do not cover data;
- no curve smoothing or sample removal is visible.

Because the user explicitly rejected PDF delivery and local LibreOffice startup is broken, DOCX acceptance does not depend on PDF conversion. Preserve the existing report layout, verify its package and structure programmatically, and disclose any residual page-layout risk if no native DOCX renderer is available.

## Task 6: Cross-environment regression and final evidence review

**Files:**
- Test all files changed in Tasks 1-4.
- Do not modify or stage unrelated dirty files.

- [ ] **Step 1: Run focused Windows tests**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest `
  tests/test_zhou2020_reference_stability.py `
  tests/test_audit_zhou2020_reference_stability.py `
  tests/test_plot_zhou2020_strict_validation.py `
  tests/test_zhou2020_validation_report.py `
  tests/test_zhou2020_metrics.py `
  tests/test_zhou2020_reference.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run focused WSL2 tests**

```powershell
wsl.exe -e bash -lc "cd '/mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解/.worktrees/zhou2020-strict-validation' && source /home/paidaxin/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx && PYTHONPATH=src python -m pytest tests/test_zhou2020_reference_stability.py tests/test_audit_zhou2020_reference_stability.py tests/test_plot_zhou2020_strict_validation.py tests/test_zhou2020_validation_report.py tests/test_zhou2020_metrics.py tests/test_zhou2020_reference.py -q"
```

Expected: all tests pass; if `python-docx` is absent in WSL, record that dependency gap and run the non-DOCX subset there rather than changing the environment silently.

- [ ] **Step 3: Verify repository scope**

```powershell
git status --short
git diff --check
git log --oneline -6
```

Expected:

- only planned files are included in the new commits;
- `dolfinx/run_sotem_benchmark.py` and `tests/test_run_sotem_benchmark.py` remain unstaged;
- generated scientific evidence still reports the Hz false reversal;
- QWE non-convergence is still visible;
- no PDF is present under the new figure output.

- [ ] **Step 4: Final verification-before-completion review**

Read the final JSON, figure captions, and DOCX conclusion side-by-side. Confirm:

```text
total fields: formal signed metrics retained
Ex zero crossing: formal result retained
Hz 0.022 s false reversal: explicit failure retained
weak IP increments: inconclusive because reference stability failed
9.44% QWE result: sensitivity only, not pass
11.40% DLF result: preserved, not formal failure
Debye 4 vs 16: internal sensitivity only
```

Only after this review report completion to the user.
