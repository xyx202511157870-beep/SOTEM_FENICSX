# Terminal Receiver Depth Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional MPI-safe terminal receiver-depth profile that samples the already-computed FEniCSx field at configured depths and publishes an independent atomic CSV.

**Architecture:** Extend `PipelineConfig` and the CLI with a validated tuple of positive depths. At the final observation only, derive point-receiver configurations at the same horizontal location, call the existing collective receiver evaluator on the current fields, and publish calculated values without touching the primary response or Faraday state.

**Tech Stack:** Python 3.10, dataclasses, csv/io, DOLFINx 0.8, mpi4py, pytest.

---

### Task 1: Configuration and CLI contract

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:130-260`
- Modify: `dolfinx/sotem_pipeline.py:12280-12455`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Write failing validation tests**

```python
def test_receiver_depth_profile_depths_are_validated():
    assert sp._validated_receiver_depth_profile_depths(
        sp.PipelineConfig(receiver_depth_profile_depths=(300.0, 400.0, 500.0, 600.0))
    ) == (300.0, 400.0, 500.0, 600.0)
    for values in ((0.0,), (-1.0,), (300.0, 300.0), (400.0, 300.0), (float("nan"),)):
        with pytest.raises(ValueError, match="receiver_depth_profile_depths"):
            sp._validated_receiver_depth_profile_depths(
                sp.PipelineConfig(receiver_depth_profile_depths=values)
            )


def test_receiver_depth_profile_default_is_disabled():
    config = sp.PipelineConfig()
    assert config.receiver_depth_profile_depths == ()
    assert sp._validated_receiver_depth_profile_depths(config) == ()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_model_consistency.py -k receiver_depth_profile -q
```

Expected: fail because the dataclass field and validation helper do not exist.

- [ ] **Step 3: Add minimal configuration and validation**

Add to `PipelineConfig`:

```python
receiver_depth_profile_depths: tuple[float, ...] = ()

def receiver_depth_profile_csv(self) -> Path:
    return self.workdir / "receiver_depth_profile.csv"
```

Add:

```python
def _validated_receiver_depth_profile_depths(config: PipelineConfig) -> tuple[float, ...]:
    values = tuple(float(value) for value in config.receiver_depth_profile_depths)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("receiver_depth_profile_depths must contain finite positive depths")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("receiver_depth_profile_depths must be unique and strictly increasing")
    return values
```

Call the helper from model validation. Add:

```python
parser.add_argument(
    "--receiver-depth-profile-depths",
    type=_parse_float_csv,
    default=(),
    help="Comma-separated positive depths for terminal same-field receiver diagnostics.",
)
```

Propagate `receiver_depth_profile_depths=args.receiver_depth_profile_depths` to
the `PipelineConfig` constructor.

- [ ] **Step 4: Run focused tests**

Run the Step 2 command. Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: validate terminal receiver depths"
```

### Task 2: Same-field evaluation and atomic artifact

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:5323-5485`
- Modify: `dolfinx/sotem_pipeline.py:9783-9795`
- Create: `tests/test_dolfinx_receiver_depth_profile.py`

- [ ] **Step 1: Write failing evaluator and artifact tests**

```python
def test_terminal_depth_profile_reuses_fields_and_negative_z(monkeypatch, tmp_path):
    config = sp.PipelineConfig(
        workdir=tmp_path,
        receiver=(1.0, 2.0, -0.1),
        receiver_depth_profile_depths=(300.0, 400.0),
    )
    seen = []

    def fake_evaluate(E, dbdt, msh, eval_config):
        seen.append((E, dbdt, eval_config.receiver, eval_config.receiver_type))
        return {"Ex": eval_config.receiver[2], "Ey": 0.0, "dBzdt": -eval_config.receiver[2],
                "sample_count": 1, "candidate_count_min": 1, "candidate_count_max": 1,
                "candidate_count_mean": 1.0, "multi_candidate_sample_count": 0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate)
    msh = SimpleNamespace(comm=SimpleNamespace(rank=0))
    E, dbdt = object(), object()
    rows = sp._evaluate_terminal_receiver_depth_profile(
        E, dbdt, msh, config, time_obs=7.943282347242813e-4
    )
    assert [item[2] for item in seen] == [(1.0, 2.0, -300.0), (1.0, 2.0, -400.0)]
    assert all(item[0] is E and item[1] is dbdt and item[3] == "point" for item in seen)
    assert [row["depth_m"] for row in rows] == [300.0, 400.0]
```

Add a writer test that reads the result with `csv.DictReader`, verifies column
and depth order, and confirms no temporary file remains.

- [ ] **Step 2: Run new tests to verify failure**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_receiver_depth_profile.py -q
```

Expected: fail because evaluator and writer helpers do not exist.

- [ ] **Step 3: Implement evaluator and writer**

Use `dataclasses.replace` and the existing evaluator:

```python
def _evaluate_terminal_receiver_depth_profile(E, dbdt, msh, config, *, time_obs):
    rows = []
    x, y, _ = config.receiver
    for depth in _validated_receiver_depth_profile_depths(config):
        diag_config = replace(config, receiver=(float(x), float(y), -float(depth)), receiver_type="point")
        rec = evaluate_receivers(E, dbdt, msh, diag_config)
        rows.append({
            "time_obs": float(time_obs), "depth_m": float(depth),
            "receiver_x": float(x), "receiver_y": float(y), "receiver_z": -float(depth),
            "Ex": float(rec["Ex"]), "Ey": float(rec["Ey"]), "dBzdt": float(rec["dBzdt"]),
            "sample_count": int(rec.get("sample_count", 0)),
            "candidate_count_min": int(rec.get("candidate_count_min", 0)),
            "candidate_count_max": int(rec.get("candidate_count_max", 0)),
            "candidate_count_mean": float(rec.get("candidate_count_mean", math.nan)),
            "multi_candidate_sample_count": int(rec.get("multi_candidate_sample_count", 0)),
        })
    return rows
```

Serialize with `csv.DictWriter` into `io.StringIO` and publish through the
existing `_atomic_write_text`. Return without writing when rows are empty or
`comm.rank != 0`.

- [ ] **Step 4: Run focused tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_receiver_depth_profile.py
git commit -m "feat: evaluate terminal receiver depth profile"
```

### Task 3: Integrate only at the terminal observation

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:7420-7770`
- Modify: `tests/test_dolfinx_receiver_depth_profile.py`

- [ ] **Step 1: Write a failing terminal-only test**

```python
def test_depth_profile_is_terminal_only():
    outputs = frozenset({15, 19, 23})
    assert not sp._is_terminal_depth_profile_step(15, outputs, (300.0,))
    assert not sp._is_terminal_depth_profile_step(19, outputs, (300.0,))
    assert sp._is_terminal_depth_profile_step(23, outputs, (300.0,))
    assert not sp._is_terminal_depth_profile_step(23, outputs, ())
```

- [ ] **Step 2: Run the test to verify failure**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_receiver_depth_profile.py -k terminal_only -q
```

Expected: fail because `_is_terminal_depth_profile_step` does not exist.

- [ ] **Step 3: Implement and integrate**

```python
def _is_terminal_depth_profile_step(step, output_step_indices, depths):
    return bool(depths) and bool(output_step_indices) and int(step) == max(output_step_indices)
```

After the primary receiver record is complete, call the profile evaluator only
when this predicate is true, then call the root-only atomic writer. Do not add
profile values to primary rows, receiver-shape diagnostics, Faraday state, or
solver history.

- [ ] **Step 4: Run surrounding suites**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_receiver_depth_profile.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_partial_forward.py tests/test_dolfinx_validation_artifacts.py -q
```

Expected: all pass with existing primary artifacts unchanged.

- [ ] **Step 5: Commit**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_receiver_depth_profile.py
git commit -m "feat: publish terminal depth profile"
```

### Task 4: Real MPI proof and scientific gate

**Files:**
- Modify: `docs/validation/2026-07-22-song-noip-spatial-underresolution.md`

- [ ] **Step 1: Run a one-observation eight-rank smoke**

Use the accepted 153,264-cell Song mesh controls, one 10 us observation, and
`--receiver-depth-profile-depths 300,400,500,600`. Require exit 0, four CSV
rows, finite values, and unchanged primary Ex/Hz/dBzdt versus the existing
eight-rank baseline within `2e-6` relatively.

- [ ] **Step 2: Run the 20-observation eight-rank depth gate**

Use the existing 10 us to 0.794328 ms BDF2/T4 command with receiver mesh size
5 m and diffusion factor 2. Add only the depth-profile flag. Require the same
surface errors within parallel tolerance and a terminal four-depth CSV.

- [ ] **Step 3: Compute independent empymod depth errors**

For every CSV row, construct an otherwise identical reference configuration
with receiver z equal to `-depth_m`, evaluate the finite-source empymod Ex and
dBz/dt at the recorded terminal time, and write a separate analysis table.
Require each 300/400/500/600 m dBz/dt error below its preserved baseline value
`(11.5005%, 14.4834%, 14.5694%, 38.0580%)` and their maximum to fall by at
least 25%.

- [ ] **Step 4: Record evidence and run final verification**

Append exact surface and depth tables, artifact paths, runtime, MPI rank count,
and the gate decision. Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx*.py -q
git diff --check
git status --short
```

Expected: all relevant tests pass; only the intended validation document is
modified after code commits.

- [ ] **Step 5: Commit**

```bash
git add docs/validation/2026-07-22-song-noip-spatial-underresolution.md
git commit -m "docs: verify Song terminal depth profile"
```
