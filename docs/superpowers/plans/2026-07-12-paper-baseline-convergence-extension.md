# Paper Baseline Convergence Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, and independently verify the five additional FEniCSx solves needed to accept or reject the 12 km, 0.5 percent time-step layered model as the publication baseline.

**Architecture:** Preserve the completed first-stage study and add a separate stage-two level builder whose axis views point to five canonical generated runs and two immutable prior runs. Extend the existing thin runner with study selection, run de-duplication, mesh identity checks, and preflight records; extend the pure evaluator with baseline and large-domain empymod evidence plus deterministic acceptance artifacts. Long solves remain sequential and resumable through the existing FEniCSx pipeline.

**Tech Stack:** Python 3.12, NumPy, meshio, Matplotlib, pytest, FEniCSx/DOLFINx under WSL, PETSc/HYPRE AMS, empymod, SHA-256 artifact identity.

---

## File Structure

- Modify `src/atem3d/layered_convergence.py`: stage-two immutable levels, canonical run identity, mesh hashing, metadata, gates, and acceptance record.
- Modify `dolfinx/run_layered_convergence_study.py`: stage selection, canonical-run de-duplication, preflight enforcement, resume, and evaluation orchestration.
- Modify `tests/test_layered_convergence.py`: pure definitions, reuse graph, axis isolation, mesh identity, metadata, and runner contracts.
- Modify `tests/test_dolfinx_e_solver_preconditioner_refresh.py`: retain the AMS recreation regression test as part of the stage-two verification set.
- Generate `output/publication_validation/convergence/layered_resistive_offset100_stage2`: manifests, meshes, checkpoints, forward outputs, reports, figures, and independent audit.

### Task 1: Define The Stage-Two Levels And Exact Five-Run Graph

**Files:**
- Modify: `src/atem3d/layered_convergence.py:21`
- Modify: `tests/test_layered_convergence.py:29`

- [ ] **Step 1: Write the failing stage-two level test**

```python
from atem3d.layered_convergence import build_paper_baseline_convergence_levels


def test_paper_baseline_levels_match_approved_stage_two_design(tmp_path):
    prior = tmp_path / "stage1"
    layered = tmp_path / "layered"
    output = tmp_path / "stage2"
    levels = build_paper_baseline_convergence_levels(layered, output, prior)

    assert [(x.level_id, x.max_internal_dt, x.max_internal_dt_fraction)
            for x in levels["time"]] == [
        ("coarse", 2.5e-5, 0.01),
        ("standard", 1.25e-5, 0.005),
        ("fine", 6.25e-6, 0.0025),
    ]
    assert [(x.level_id, x.x_extent, x.earth_depth, x.air_height)
            for x in levels["domain"]] == [
        ("small", 6000.0, 6000.0, 600.0),
        ("standard", 12000.0, 12000.0, 1200.0),
        ("large", 24000.0, 24000.0, 2400.0),
    ]
    assert [(x.source_mesh_size, x.receiver_mesh_size)
            for x in levels["mesh"]] == [
        (12.0, 9.0), (8.0, 6.0), (6.0, 4.5),
    ]
    generated = {
        x.run_id
        for axis in levels.values()
        for x in axis
        if x.existing_run_dir is None
    }
    assert generated == {
        "baseline_12km_dt005_mesh8_6",
        "time_fine_12km_dt0025_mesh8_6",
        "domain_large_24km_dt005_mesh8_6",
        "mesh_coarse_12km_dt005_mesh12_9",
        "mesh_fine_12km_dt005_mesh6_4p5",
    }
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/test_layered_convergence.py::test_paper_baseline_levels_match_approved_stage_two_design -q`

Expected: FAIL because `build_paper_baseline_convergence_levels` and `run_id` do not exist.

- [ ] **Step 3: Add canonical run identity to the immutable level**

```python
@dataclass(frozen=True)
class ConvergenceLevel:
    axis: str
    level_id: str
    run_id: str
    x_extent: float
    y_extent: float
    earth_depth: float
    air_height: float
    far_field_mesh_size: float
    source_mesh_size: float
    receiver_mesh_size: float
    max_internal_dt: float
    max_internal_dt_fraction: float
    workdir: Path
    existing_run_dir: Path | None = None
    reuse_mesh_path: Path | None = None
```

Update the first-stage `level()` helper to default `run_id` to `f"{axis}_{level_id}"` so its behavior and reports remain unchanged.

- [ ] **Step 4: Implement the stage-two builder**

```python
def build_paper_baseline_convergence_levels(
    layered_root: Path,
    output_root: Path,
    prior_convergence_root: Path,
) -> dict[str, tuple[ConvergenceLevel, ...]]:
    case_id = "resistive_basement_rho1000_offset100"
    time_coarse = Path(layered_root) / "domain12000" / case_id
    domain_small = Path(prior_convergence_root) / "time" / "fine"
    locked_mesh = time_coarse / "verification_mesh.msh"
    runs = Path(output_root) / "runs"

    def level(axis: str, level_id: str, run_id: str, **overrides):
        values = dict(
            axis=axis,
            level_id=level_id,
            run_id=run_id,
            x_extent=12000.0,
            y_extent=12000.0,
            earth_depth=12000.0,
            air_height=1200.0,
            far_field_mesh_size=750.0,
            source_mesh_size=8.0,
            receiver_mesh_size=6.0,
            max_internal_dt=1.25e-5,
            max_internal_dt_fraction=0.005,
            workdir=runs / run_id,
        )
        values.update(overrides)
        return ConvergenceLevel(**values)

    baseline = dict(run_id="baseline_12km_dt005_mesh8_6", reuse_mesh_path=locked_mesh)
    return {
        "time": (
            level("time", "coarse", "existing_12km_dt01", existing_run_dir=time_coarse,
                  max_internal_dt=2.5e-5, max_internal_dt_fraction=0.01),
            level("time", "standard", **baseline),
            level("time", "fine", "time_fine_12km_dt0025_mesh8_6",
                  max_internal_dt=6.25e-6, max_internal_dt_fraction=0.0025,
                  reuse_mesh_path=locked_mesh),
        ),
        "mesh": (
            level("mesh", "coarse", "mesh_coarse_12km_dt005_mesh12_9",
                  source_mesh_size=12.0, receiver_mesh_size=9.0),
            level("mesh", "standard", **baseline),
            level("mesh", "fine", "mesh_fine_12km_dt005_mesh6_4p5",
                  source_mesh_size=6.0, receiver_mesh_size=4.5),
        ),
        "domain": (
            level("domain", "small", "existing_6km_dt005", existing_run_dir=domain_small,
                  x_extent=6000.0, y_extent=6000.0, earth_depth=6000.0, air_height=600.0),
            level("domain", "standard", **baseline),
            level("domain", "large", "domain_large_24km_dt005_mesh8_6",
                  x_extent=24000.0, y_extent=24000.0,
                  earth_depth=24000.0, air_height=2400.0),
        ),
    }
```

- [ ] **Step 5: Write and run axis-isolation tests**

```python
def test_stage_two_axes_change_only_the_intended_parameters(tmp_path):
    levels = build_paper_baseline_convergence_levels(
        tmp_path / "layered", tmp_path / "stage2", tmp_path / "stage1"
    )
    physical = lambda x: (
        x.x_extent, x.y_extent, x.earth_depth, x.air_height,
        x.far_field_mesh_size, x.source_mesh_size, x.receiver_mesh_size,
        x.max_internal_dt, x.max_internal_dt_fraction,
    )
    time = [physical(x) for x in levels["time"]]
    mesh = [physical(x) for x in levels["mesh"]]
    domain = [physical(x) for x in levels["domain"]]
    assert all(row[:7] == time[1][:7] for row in time)
    assert all(row[:5] + row[7:] == mesh[1][:5] + mesh[1][7:] for row in mesh)
    assert all(row[4:] == domain[1][4:] for row in domain)


def test_stage_two_pipeline_arguments_lock_solver_and_observation_contract(tmp_path):
    levels = build_paper_baseline_convergence_levels(
        tmp_path / "layered", tmp_path / "stage2", tmp_path / "stage1"
    )
    baseline = levels["time"][1]
    args = build_pipeline_command_arguments(baseline)
    option = lambda name: args[args.index(name) + 1]
    assert option("--t-min") == "1e-05"
    assert option("--max-internal-dt") == "1.25e-05"
    assert option("--max-internal-dt-fraction") == "0.005"
    assert option("--rtol") == "1e-07"
    assert option("--atol") == "1e-12"
    assert option("--memory-limit-gb") == "24"
    assert option("--stop-after-outputs") == "25"
```

Run: `python -m pytest tests/test_layered_convergence.py -q`

Expected: all existing first-stage and new stage-two definition tests PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: define paper baseline convergence levels"
```

### Task 2: Enforce Mesh Identity And Publication Preflight

**Files:**
- Modify: `src/atem3d/layered_convergence.py:567`
- Modify: `dolfinx/run_layered_convergence_study.py:29`
- Modify: `tests/test_layered_convergence.py:429`

- [ ] **Step 1: Write failing hash and preflight tests**

```python
def test_manifest_records_locked_mesh_path_and_sha256(tmp_path):
    mesh = tmp_path / "locked.msh"
    mesh.write_bytes(b"publication-mesh")
    level = ConvergenceLevel(
        axis="time", level_id="fine", run_id="fine", x_extent=12000.0,
        y_extent=12000.0, earth_depth=12000.0, air_height=1200.0,
        far_field_mesh_size=750.0, source_mesh_size=8.0,
        receiver_mesh_size=6.0, max_internal_dt=6.25e-6,
        max_internal_dt_fraction=0.0025, workdir=tmp_path / "run",
        reuse_mesh_path=mesh,
    )
    manifest = convergence_level_manifest(level)
    assert manifest["reuse_mesh"]["path"] == str(mesh)
    assert manifest["reuse_mesh"]["sha256"] == hashlib.sha256(b"publication-mesh").hexdigest()


def test_preflight_rejects_mesh_over_24gb_or_missing_diagnostics(tmp_path):
    mesh = tmp_path / "verification_mesh.msh"
    mesh.write_bytes(b"mesh")
    with pytest.raises(ValueError, match="source coverage"):
        validate_publication_preflight(
            mesh_path=mesh,
            diagnostics={"estimated_memory_gb": 10.0, "receiver_found": True},
            memory_limit_gb=24.0,
        )
    with pytest.raises(ValueError, match="24 GB"):
        validate_publication_preflight(
            mesh_path=mesh,
            diagnostics={
                "estimated_memory_gb": 24.1,
                "source_coverage_passed": True,
                "receiver_found": True,
                "source_divergence_passed": True,
            },
            memory_limit_gb=24.0,
        )
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_layered_convergence.py -k "manifest_records_locked or preflight_rejects" -q`

Expected: FAIL because hash metadata and `validate_publication_preflight` are missing.

- [ ] **Step 3: Implement streaming SHA-256 and strict preflight**

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_publication_preflight(
    *, mesh_path: Path, diagnostics: dict, memory_limit_gb: float = 24.0
) -> dict:
    reasons = []
    if not Path(mesh_path).is_file():
        reasons.append("mesh_missing")
    if not diagnostics.get("source_coverage_passed", False):
        reasons.append("source coverage failed")
    if not diagnostics.get("receiver_found", False):
        reasons.append("receiver location failed")
    if not diagnostics.get("source_divergence_passed", False):
        reasons.append("source divergence failed")
    estimated = float(diagnostics.get("estimated_memory_gb", math.inf))
    if not math.isfinite(estimated) or estimated > memory_limit_gb:
        reasons.append("estimated memory exceeds 24 GB")
    if reasons:
        raise ValueError("; ".join(reasons))
    return {
        "passed": True,
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "estimated_memory_gb": estimated,
        "memory_limit_gb": memory_limit_gb,
    }
```

Adapt diagnostic field names only after inspecting the actual mesh-only artifacts; map them once in the runner and keep the pure validator contract above unchanged.

- [ ] **Step 4: Record and verify the locked time-axis mesh**

The runner writes `preflight.json` after mesh-only execution. For levels with `reuse_mesh_path`, compare the SHA-256 in the current manifest, the preflight record, and the actual reused file before launching the forward solve. A mismatch exits nonzero with `locked_mesh_hash_mismatch`.

Run: `python -m pytest tests/test_layered_convergence.py -k "mesh or preflight" -q`

Expected: PASS, including all existing mesh-level tests.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "feat: enforce convergence mesh preflight"
```

### Task 3: Select Stage Two, De-Duplicate Shared Runs, And Preserve Resume Semantics

**Files:**
- Modify: `dolfinx/run_layered_convergence_study.py:105`
- Modify: `tests/test_layered_convergence.py:429`

- [ ] **Step 1: Write failing runner tests**

```python
def test_stage_two_dry_run_emits_exactly_five_canonical_commands(tmp_path):
    layered, prior, output = _make_stage_two_reuse_fixture(tmp_path)
    result = _run_study_cli(
        "--study", "paper-baseline", "--layered-root", str(layered),
        "--prior-convergence-root", str(prior), "--output-root", str(output),
        "--mode", "full", "--dry-run",
    )
    commands = list((output / "runs").glob("*/command.txt"))
    assert len(commands) == 5
    assert result.stdout.count("RUN_ID=") == 5
    assert result.stdout.count("REUSE_RUN_ID=") == 2


def test_stage_two_shared_baseline_is_processed_once(tmp_path):
    layered, prior, output = _make_stage_two_reuse_fixture(tmp_path)
    result = _run_study_cli(
        "--study", "paper-baseline", "--layered-root", str(layered),
        "--prior-convergence-root", str(prior), "--output-root", str(output),
        "--axis", "time", "--axis", "mesh", "--axis", "domain",
        "--mode", "full", "--dry-run",
    )
    assert result.stdout.count("RUN_ID=baseline_12km_dt005_mesh8_6") == 1
```

`_make_stage_two_reuse_fixture` creates the locked 12 km mesh, completed time-coarse files, and completed 6 km/0.5 percent files required by the builder; it does not create generated stage-two outputs.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_layered_convergence.py -k "stage_two_dry_run or shared_baseline" -q`

Expected: FAIL because the CLI has no study selection or run de-duplication.

- [ ] **Step 3: Add explicit study selection without changing legacy defaults**

```python
parser.add_argument(
    "--study", choices=("stage1", "paper-baseline"), default="stage1"
)
parser.add_argument(
    "--prior-convergence-root",
    type=Path,
    default=ROOT / "output" / "publication_validation" / "convergence"
    / "layered_resistive_offset100",
)

if args.study == "paper-baseline":
    levels = build_paper_baseline_convergence_levels(
        args.layered_root, args.output_root, args.prior_convergence_root
    )
else:
    levels = build_convergence_levels(args.layered_root, args.output_root)
```

Keep the legacy default so all completed first-stage commands and tests remain reproducible. The production stage-two command always supplies `--study paper-baseline` and the stage-two output root explicitly.

- [ ] **Step 4: Process canonical run IDs once**

```python
processed_run_ids: set[str] = set()
for axis_name in selected_axes:
    for level in levels[axis_name]:
        if level.run_id in processed_run_ids:
            print(f"SHARED_RUN={level.run_id}")
            continue
        processed_run_ids.add(level.run_id)
        # Existing manifest, checkpoint, mesh/full, and dry-run logic follows.
```

Write `axis_levels/<axis>/<level>/level_pointer.json` before de-duplication so every axis level records its canonical run ID, resolved run directory, source type, and manifest SHA-256.

- [ ] **Step 5: Re-run resume and postprocess tests**

Run: `python -m pytest tests/test_layered_convergence.py -k "runner" -q`

Expected: PASS for legacy remaining-output resume, direct 25-row postprocess, existing-level pointers, stage-two five commands, and shared-baseline de-duplication.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "feat: orchestrate paper baseline convergence runs"
```

### Task 4: Capture Cumulative Runtime And KSP Evidence

**Files:**
- Modify: `src/atem3d/layered_convergence.py:305`
- Modify: `tests/test_layered_convergence.py:294`
- Verify: `dolfinx/sotem_pipeline.py:3476`
- Verify: `tests/test_dolfinx_e_solver_preconditioner_refresh.py`

- [ ] **Step 1: Write failing metadata tests**

```python
def test_read_run_metadata_reports_output_ksp_statistics(tmp_path):
    run_dir = _make_metadata_fixture(tmp_path)
    np.savez(
        run_dir / "forward_partial.npz",
        internal_solver_steps=np.arange(9),
        solver_iterations=np.array([12, 15, 9]),
        solver_reasons=np.array([2, 2, 2]),
        solver_residuals=np.array([1e-9, 2e-9, 8e-10]),
    )
    metadata = read_run_metadata(run_dir)
    assert metadata["ksp_output_solve_count"] == 3
    assert metadata["ksp_iterations_median"] == 12.0
    assert metadata["ksp_iterations_max"] == 15
    assert metadata["ksp_all_converged"] is True


def test_read_run_metadata_sums_completed_forward_attempts(tmp_path):
    run_dir = _make_metadata_fixture(tmp_path)
    (run_dir / "timing_events.jsonl").write_text(
        '\n'.join([
            '{"event":"forward_done","seconds":100.5}',
            '{"event":"forward_done","seconds":20.25}',
        ]) + '\n', encoding="utf-8"
    )
    assert read_run_metadata(run_dir)["forward_runtime_seconds"] == pytest.approx(120.75)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_layered_convergence.py -k "ksp_statistics or completed_forward_attempts" -q`

Expected: the KSP assertion FAILS while the cumulative completed-attempt runtime assertion protects existing behavior.

- [ ] **Step 3: Add KSP statistics from `forward_partial.npz`**

```python
iterations = np.asarray(payload["solver_iterations"], dtype=int)
reasons = np.asarray(payload["solver_reasons"], dtype=int)
residuals = np.asarray(payload["solver_residuals"], dtype=float)
ksp = {
    "ksp_output_solve_count": int(iterations.size),
    "ksp_iterations_median": float(np.median(iterations)),
    "ksp_iterations_max": int(np.max(iterations)),
    "ksp_residual_max": float(np.max(residuals)),
    "ksp_all_converged": bool(np.all(reasons > 0)),
}
```

Reject mismatched array lengths, nonfinite residuals, negative iterations, and nonpositive convergence reasons in a completed publication run. Merge `ksp` into the existing mesh, step-count, memory, and runtime metadata.

- [ ] **Step 4: Verify AMS recreation remains covered**

Run: `python -m pytest tests/test_dolfinx_e_solver_preconditioner_refresh.py tests/test_sotem_pipeline_schedule.py -q`

Expected: PASS; the test proves each changed time-step matrix gets a new KSP/AMS instance while the discrete gradient and edge constants are retained.

- [ ] **Step 5: Run metadata tests and commit**

Run: `python -m pytest tests/test_layered_convergence.py -k "metadata" -q`

Expected: PASS.

```powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: report convergence solver evidence"
```

### Task 5: Evaluate Stage Two And Write The Baseline Acceptance Record

**Files:**
- Modify: `src/atem3d/layered_convergence.py:620`
- Modify: `tests/test_layered_convergence.py:344`

- [ ] **Step 1: Write failing acceptance tests**

```python
def test_stage_two_summary_reports_baseline_and_large_empymod_gates(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=True)
    summary = evaluate_convergence_study(
        levels, study_id="layered_resistive_offset100_stage2"
    )
    assert summary["study_passed"] is True
    assert summary["candidate_baseline"]["run_id"] == "baseline_12km_dt005_mesh8_6"
    assert summary["candidate_baseline"]["accepted_for_paper_figures"] is True
    assert summary["candidate_baseline"]["external_reference_gate"]["publication_gate_passed"] is True
    domain = next(x for x in summary["axes"] if x["axis"] == "domain")
    assert domain["large_external_reference_gate"]["publication_gate_passed"] is True


def test_baseline_is_rejected_when_any_axis_fails(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=False)
    summary = evaluate_convergence_study(
        levels, study_id="layered_resistive_offset100_stage2"
    )
    assert summary["candidate_baseline"]["accepted_for_paper_figures"] is False
    assert summary["study_passed"] is False
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_layered_convergence.py -k "baseline_and_large or baseline_is_rejected" -q`

Expected: FAIL because the evaluator does not accept a study ID or emit candidate-baseline evidence.

- [ ] **Step 3: Extend the evaluator without changing thresholds**

```python
def evaluate_convergence_study(
    levels: dict[str, tuple[ConvergenceLevel, ...]],
    *,
    selected_axes: tuple[str, ...] = ("time", "mesh", "domain"),
    study_id: str = "layered_resistive_offset100",
) -> dict:
    # Existing pairwise metrics and unchanged gates run first.
    baseline = levels["time"][1]
    baseline_dir = resolved_run_dir(baseline)
    baseline_external = evaluate_errors_csv(baseline_dir / "errors.csv")
    all_axes_present = set(selected_axes) == {"time", "mesh", "domain"}
    accepted = all_axes_present and complete_count == 3 and passed_count == 3
    return {
        "study_id": study_id,
        "study_passed": accepted,
        "candidate_baseline": {
            "run_id": baseline.run_id,
            "run_dir": str(baseline_dir),
            "accepted_for_paper_figures": accepted,
            "external_reference_gate": baseline_external,
            **read_run_metadata(baseline_dir),
        },
        # Existing coordinate, counts, and axes keys remain.
    }
```

The domain gate continues to use the 24 km result. The baseline empymod gate is reported but cannot substitute for a failed convergence axis. Median/RMS/maximum limits remain 1/2/5 percent for time and mesh, monotonicity tolerance remains 0.1 percentage point, and the large empymod gate remains 5/5/10 percent.

- [ ] **Step 4: Write deterministic `baseline_acceptance.json`**

Extend `write_convergence_reports` to write the JSON-safe `candidate_baseline` object plus the three axis gates to `baseline_acceptance.json`. It must contain mesh counts, mesh SHA-256, internal step count, KSP statistics, runtime, empymod metrics, blocking reasons, and the exact coordinate convention.

```python
acceptance = {
    "study_id": summary["study_id"],
    "coordinate_convention": summary["coordinate_convention"],
    "accepted_for_paper_figures": summary["study_passed"],
    "candidate_baseline": summary["candidate_baseline"],
    "axis_gates": [
        {
            "axis": axis["axis"],
            "status": axis["status"],
            "passed": axis["passed"],
            "blocking_reasons": axis["blocking_reasons"],
        }
        for axis in summary["axes"]
    ],
}
```

- [ ] **Step 5: Verify reports, incomplete studies, and failed gates**

Run: `python -m pytest tests/test_layered_convergence.py -q`

Expected: PASS, including deterministic JSON/CSV/Markdown, publication-resolution figures, incomplete-axis exit status, failed-gate reporting, and baseline acceptance tests.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: gate paper baseline convergence evidence"
```

### Task 6: Run Five Solves And Independently Audit The Evidence

**Files:**
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/**`
- Create if the audit is reusable: `dolfinx/audit_layered_convergence.py`
- Create if the audit is reusable: `tests/test_layered_convergence_audit.py`

- [ ] **Step 1: Run the focused pre-solve test suite**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_e_solver_preconditioner_refresh.py tests/test_dolfinx_partial_forward.py tests/test_sotem_pipeline_schedule.py -q
```

Expected: all tests PASS before any expensive solve.

- [ ] **Step 2: Generate and inspect all five mesh preflights**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --mode mesh
```

Expected: five unique `RUN_ID` records; every `preflight.json` has `passed=true`; source coverage, receiver location, source divergence, and memory checks pass; baseline and time-fine mesh hashes equal the completed 12 km mesh hash; no estimate exceeds 24 GB.

- [ ] **Step 3: Check host memory before each solve**

Run:

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='FreeMemoryGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
```

Expected: free physical memory exceeds the run estimate by at least 6 GB. Execute one canonical run at a time; do not start two runs merely to reduce wall-clock time.

- [ ] **Step 4: Run the shared 12 km baseline and resume until complete**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis time --level standard --mode full
```

Expected: `verification_data.npz`, `errors.csv`, `forward_partial.npz`, and 25 effective outputs. If interrupted, rerun the exact command; the runner requests only missing outputs or directly postprocesses a completed 25-row checkpoint.

- [ ] **Step 5: Run the time-fine 12 km solve**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis time --level fine --mode full
```

Expected: complete 25-output result on the same locked mesh SHA-256 as the shared baseline.

- [ ] **Step 6: Run the 24 km domain solve**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis domain --level large --mode full
```

Expected: complete 25-output result; final empymod median/RMS/maximum are evaluated against the unchanged 5/5/10 percent publication gate.

- [ ] **Step 7: Run coarse and fine local-mesh solves**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis mesh --level coarse --level fine --mode full
```

Expected: both canonical runs complete with their own passing mesh preflight and 25 effective outputs.

- [ ] **Step 8: Evaluate all axes from disk**

Run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --mode evaluate
```

Expected for acceptance: exit code 0, `CONVERGENCE_COMPLETE=3`, `CONVERGENCE_PASSED=3`, all thresholds unchanged, and `baseline_acceptance.json` has `accepted_for_paper_figures=true`.

- [ ] **Step 9: Independently recompute every metric**

Load each resolved `verification_data.npz` directly, select the first 25 `dBzdt` samples, apply the same `1e-6 * peak(abs(empymod))` amplitude floor, and calculate pairwise median, RMS, and maximum percentages without importing `atem3d.layered_convergence`. Compare every result to `convergence_summary.json` with `rtol=1e-12` and `atol=1e-12`. Recompute both baseline and 24 km empymod gates directly from `errors.csv`.

Expected: print `INDEPENDENT_RECOMPUTE_OK`; any mismatch exits nonzero and preserves all artifacts.

- [ ] **Step 10: Inspect figures at original resolution**

Open `convergence_curves.png` and `convergence_differences.png` at original resolution. Verify readable labels, all nine axis-level traces, visible pairwise curves, correct log-time axis, no clipping, and no omitted late-time sample.

- [ ] **Step 11: Run final related regression suite**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_e_solver_preconditioner_refresh.py tests/test_dolfinx_partial_forward.py tests/test_sotem_pipeline_schedule.py -q
```

Expected: all tests PASS after the numerical artifacts are complete.

- [ ] **Step 12: Apply the non-negotiable failure rule**

If any stage-two axis fails, retain the report and all checkpoints. Extend only the failing axis to the next physical level: halve the fine time controls again, refine local sizes beyond 6/4.5 m, or add a domain beyond 24 km while holding 750 m far-field size and 0.5 percent time control. Do not change thresholds, remove samples, or relabel the candidate baseline as accepted.

Passing Task 6 establishes only the non-polarizable layered FEniCSx baseline. Start separate approved specifications for the COMSOL three-dimensional cross-check and nonzero-IP benchmark before making a full paper-readiness claim.
