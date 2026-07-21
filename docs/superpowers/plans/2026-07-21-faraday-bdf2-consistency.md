# Faraday BDF2 Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance receiver `Hz` with the same variable-step BDF2 formula used by the electric-field solve, after a backward-Euler warmup through the first observation, so no-IP BDF2 validation is both discretely consistent and startup-stable.

**Architecture:** Add one pure BDF2 Faraday recurrence beside the existing backward-Euler helper, retain one older receiver state in the existing FETD loop, and isolate the backward-Euler-to-BDF2 switch in a pure selector.  A BDF2-requested run uses backward Euler through its first reported observation and switches both electric and Faraday states on the next internal step.  All backward-Euler/Cole--Cole paths remain unchanged; BDF2 resume remains rejected by the existing configuration gate.

**Tech Stack:** Python, NumPy, pytest, DOLFINx/PETSc in WSL, empymod validation artifacts.

---

### Task 1: Pure BDF2 receiver recurrence

**Files:**
- Modify: `tests/test_dolfinx_biot_receiver.py:265`
- Modify: `dolfinx/sotem_pipeline.py:6024`

- [ ] **Step 1: Write the failing constant-step recurrence test**

```python
def test_faraday_receiver_hz_update_uses_bdf2_coefficients():
    sp = _load_pipeline_module()
    coefficients = sp._bdf2_step_coefficients(dt=1.0, previous_dt=1.0)

    updated = sp._advance_faraday_receiver_hz_bdf2(
        previous_hz=2.0,
        older_hz=1.0,
        dbzdt_new=6.0e-6,
        coefficients=coefficients,
        mu=2.0e-6,
    )

    assert updated == pytest.approx((3.0 + 2.0 * 2.0 - 0.5 * 1.0) / 1.5)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_biot_receiver.py::test_faraday_receiver_hz_update_uses_bdf2_coefficients -q
```

Expected: FAIL with `AttributeError` for the missing BDF2 Faraday helper.

- [ ] **Step 3: Implement the minimal pure recurrence**

Add beside `_advance_faraday_receiver_hz`:

```python
def _advance_faraday_receiver_hz_bdf2(
    *,
    previous_hz: float,
    older_hz: float,
    dbzdt_new: float,
    coefficients: dict[str, float],
    mu: float = 1.2566370614359173e-6,
) -> float:
    """Variable-step BDF2 receiver Hz update from dBz/dt = -curl(E)."""

    mu_value = float(mu)
    values = {
        "previous_hz": float(previous_hz),
        "older_hz": float(older_hz),
        "dbzdt_new": float(dbzdt_new),
        "lhs": float(coefficients["lhs"]),
        "old": float(coefficients["old"]),
        "older": float(coefficients["older"]),
    }
    if not math.isfinite(mu_value) or mu_value <= 0.0:
        raise ValueError("mu must be positive")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("BDF2 Faraday state and coefficients must be finite")
    if values["lhs"] <= 0.0:
        raise ValueError("BDF2 Faraday lhs coefficient must be positive")
    return (
        values["dbzdt_new"] / mu_value
        + values["old"] * values["previous_hz"]
        + values["older"] * values["older_hz"]
    ) / values["lhs"]
```

- [ ] **Step 4: Verify GREEN and existing backward-Euler tests**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_biot_receiver.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the recurrence**

```powershell
git add tests/test_dolfinx_biot_receiver.py dolfinx/sotem_pipeline.py
git commit -m "fix: advance Faraday Hz with BDF2 coefficients"
```

### Task 2: Wire BDF2 magnetic history into FETD

**Files:**
- Modify: `tests/test_dolfinx_biot_receiver.py:265`
- Modify: `dolfinx/sotem_pipeline.py:6800-7210`

- [ ] **Step 1: Add a failing two-step history test**

```python
def test_bdf2_faraday_history_advances_initial_then_previous_state():
    sp = _load_pipeline_module()
    mu = 2.0e-6
    initial_hz = 1.0
    first_hz, older_hz = sp._advance_faraday_receiver_hz_state(
        previous_hz=initial_hz,
        older_hz=None,
        dbzdt_new=2.0e-6,
        dt=1.0,
        step_method="backward_euler",
        bdf2_coefficients=None,
        mu=mu,
    )
    coefficients = sp._bdf2_step_coefficients(dt=1.0, previous_dt=1.0)

    second_hz, next_older_hz = sp._advance_faraday_receiver_hz_state(
        previous_hz=first_hz,
        older_hz=older_hz,
        dbzdt_new=4.0e-6,
        dt=1.0,
        step_method="bdf2",
        bdf2_coefficients=coefficients,
        mu=mu,
    )

    assert first_hz == pytest.approx(2.0)
    assert older_hz == pytest.approx(initial_hz)
    assert second_hz == pytest.approx((2.0 + 2.0 * first_hz - 0.5 * initial_hz) / 1.5)
    assert next_older_hz == pytest.approx(first_hz)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the named test with pytest.  Expected: FAIL with `AttributeError` for the missing state-transition helper.

- [ ] **Step 3: Implement and wire the state transition**

Add the pure transition helper:

```python
def _advance_faraday_receiver_hz_state(
    *, previous_hz, older_hz, dbzdt_new, dt, step_method,
    bdf2_coefficients, mu=1.2566370614359173e-6,
):
    if step_method == "backward_euler":
        updated = _advance_faraday_receiver_hz(
            previous_hz=previous_hz, dbzdt_new=dbzdt_new, dt=dt, mu=mu
        )
    elif step_method == "bdf2":
        if older_hz is None or bdf2_coefficients is None:
            raise ValueError("BDF2 Faraday update requires older Hz and coefficients")
        updated = _advance_faraday_receiver_hz_bdf2(
            previous_hz=previous_hz,
            older_hz=older_hz,
            dbzdt_new=dbzdt_new,
            coefficients=bdf2_coefficients,
            mu=mu,
        )
    else:
        raise ValueError("Faraday step_method must be 'backward_euler' or 'bdf2'")
    return updated, float(previous_hz)
```

Add `faraday_receiver_hz_older = None` beside the current Faraday state.  Replace the unconditional backward-Euler call in the loop with:

```python
faraday_receiver_hz, faraday_receiver_hz_older = _advance_faraday_receiver_hz_state(
    previous_hz=float(faraday_receiver_hz),
    older_hz=faraday_receiver_hz_older,
    dbzdt_new=float(faraday_step_record["dBzdt"]),
    dt=dt,
    step_method="bdf2" if use_bdf2 else "backward_euler",
    bdf2_coefficients=bdf2_coeffs,
)
```

- [ ] **Step 4: Run focused regression tests**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_biot_receiver.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_partial_forward.py tests/test_dolfinx_transient_operator_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit loop wiring**

```powershell
git add tests/test_dolfinx_biot_receiver.py dolfinx/sotem_pipeline.py
git commit -m "fix: keep BDF2 history for Faraday receiver"
```

### Task 3: Runtime verification and no-IP rerun

**Files:**
- Verify: `dolfinx/sotem_pipeline.py`
- Generate: external WSL run directory outside the repository

- [ ] **Step 1: Run static and full tests**

Run `python -m py_compile dolfinx/sotem_pipeline.py`, `bash -n benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh`, `git diff --check`, then the full pytest suite in `/home/paidaxin/miniconda3/envs/fenicsx`.

Expected: zero syntax errors and all tests pass.

- [ ] **Step 2: Preserve and stop the failed BE/T4 baseline**

Allow the current run to save its 0.01 s output checkpoint, archive its log with a failure-specific name, then interrupt only tmux session `sotem_noip_t4_resume1`.  Verify no IP process exists.

- [ ] **Step 3: Launch BDF2/T4 no-IP with the same mesh and physics**

Use a fresh output root, the same mesh seed, `RUN_CASES=noip`, `OUTPUT_INTERVAL_SUBSTEPS=4`, `MIN_STEPS_BEFORE_FIRST_OBSERVATION=16`, and replace only `--time-method theta --time-theta 1` with `--time-method bdf2 --time-theta 1`.  Do not enable IP.

- [ ] **Step 4: Compare every completed output to empymod**

At each checkpoint, load `forward_partial.npz` and the existing independent empymod reference.  Exclude `Ey`; report current and maximum relative errors for `Ex`, `Hz`, and `dBzdt` without scaling, shifting, smoothing, dropping samples, or changing the 5% threshold.

- [ ] **Step 5: Apply the no-IP gate**

Only a complete 51-time artifact with all three formal components below 5%, converged solvers, valid provenance, and no non-finite values passes.  If it passes, run the prepared current-version SimPEG `S0T2B0` no-IP case.  If it fails, preserve the evidence and return to spatial/time convergence diagnosis; do not launch IP.

### Task 4: Replace the disproved single-step BDF2 startup

**Files:**
- Modify: `tests/test_dolfinx_model_consistency.py`
- Modify: `dolfinx/sotem_pipeline.py`
- Verify: `docs/superpowers/specs/2026-07-21-faraday-bdf2-consistency-design.md`

- [ ] **Step 1: Write the failing startup-boundary test**

```python
def test_bdf2_starts_after_first_output_step():
    sp = _load_pipeline_module()

    assert not sp._should_use_bdf2_step(
        time_method="bdf2", step=0, first_output_step=15
    )
    assert not sp._should_use_bdf2_step(
        time_method="bdf2", step=15, first_output_step=15
    )
    assert sp._should_use_bdf2_step(
        time_method="bdf2", step=16, first_output_step=15
    )
    assert not sp._should_use_bdf2_step(
        time_method="theta", step=16, first_output_step=15
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run in WSL:

```bash
cd /home/paidaxin/codex-worktree-live-lei
PYTHONPATH="$PWD/src:$PWD" /home/paidaxin/miniconda3/envs/fenicsx/bin/python \
  -m pytest tests/test_dolfinx_model_consistency.py::test_bdf2_starts_after_first_output_step -q
```

Expected: FAIL with `AttributeError` for the missing selector.

- [ ] **Step 3: Implement the pure selector and wire the loop**

Add beside the time-step coefficient helpers:

```python
def _should_use_bdf2_step(
    *, time_method: str, step: int, first_output_step: int
) -> bool:
    """Return whether a step is past the backward-Euler startup interval."""

    return (
        str(time_method).strip().lower() == "bdf2"
        and int(step) > int(first_output_step)
    )
```

Before the loop, derive:

```python
first_output_step = int(schedule["output_step_indices"][0])
```

Replace the current BDF2 selection predicate with:

```python
use_bdf2 = (
    _should_use_bdf2_step(
        time_method=time_method,
        step=step,
        first_output_step=first_output_step,
    )
    and E_older is not None
    and previous_step_dt is not None
)
```

This keeps both histories advancing during the warmup and prevents any BDF2
step until the first reported observation has been produced.

- [ ] **Step 4: Verify GREEN and focused regressions**

Run in WSL:

```bash
cd /home/paidaxin/codex-worktree-live-lei
PYTHONPATH="$PWD/src:$PWD" /home/paidaxin/miniconda3/envs/fenicsx/bin/python \
  -m pytest tests/test_dolfinx_model_consistency.py \
  tests/test_dolfinx_biot_receiver.py \
  tests/test_dolfinx_partial_forward.py \
  tests/test_dolfinx_transient_operator_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the startup fix**

```bash
git add tests/test_dolfinx_model_consistency.py dolfinx/sotem_pipeline.py
git commit -m "fix: damp BDF2 through first observation"
```

- [ ] **Step 6: Rerun only through the first observation before committing to a full solve**

Launch the unchanged no-IP P2/T4 case in a fresh external run root with
`TIME_METHOD=bdf2`.  Stop and preserve evidence if the first-observation
formal errors do not return to the established backward-Euler neighborhood
(`Ex` about 2.36%, `Hz` about 0.06%, `dBzdt` about 1.49%).  Continue the full
51-time solve only if this startup gate succeeds.  Do not launch IP.
