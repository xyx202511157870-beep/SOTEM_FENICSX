# Layered Convergence Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and execute a reproducible three-level convergence study for time step, source/receiver-local mesh size, and computational-domain size using the hardest completed layered FEniCSx benchmark.

**Architecture:** A new pure-Python module owns immutable level definitions, command construction, artifact loading, metrics, gates, and deterministic reports. A thin DOLFINx runner creates manifests, reuses baseline artifacts and locked meshes, resumes matching checkpoints, and delegates forward solves to the unchanged sotem_pipeline.py.

**Tech Stack:** Python 3.12, NumPy, meshio, Matplotlib, pytest, FEniCSx/DOLFINx under WSL, empymod, PETSc/HYPRE AMS.

---

## File Structure

- Create src/atem3d/layered_convergence.py for specifications, metrics, gates, metadata, and reports.
- Create dolfinx/run_layered_convergence_study.py for mesh, full, and evaluate orchestration.
- Create tests/test_layered_convergence.py for pure-module and CLI-contract tests.
- Generate output/publication_validation/convergence/layered_resistive_offset100 without committing generated outputs.

### Task 1: Define Three-Level Specifications And Isolated Commands

**Files:**
- Create: src/atem3d/layered_convergence.py
- Create: tests/test_layered_convergence.py

- [ ] **Step 1: Write the failing level-definition test**

~~~python
from atem3d.layered_convergence import build_convergence_levels

def test_levels_match_approved_design(tmp_path):
    levels = build_convergence_levels(tmp_path / "layered", tmp_path / "out")
    assert [(x.level_id, x.max_internal_dt, x.max_internal_dt_fraction)
            for x in levels["time"]] == [
        ("coarse", 5e-5, 0.02),
        ("standard", 2.5e-5, 0.01),
        ("fine", 1.25e-5, 0.005),
    ]
    assert [(x.source_mesh_size, x.receiver_mesh_size)
            for x in levels["mesh"]] == [(12.0, 9.0), (8.0, 6.0), (6.0, 4.5)]
    assert [(x.x_extent, x.far_field_mesh_size)
            for x in levels["domain"]] == [
        (3000.0, 750.0), (6000.0, 750.0), (12000.0, 750.0)
    ]
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py::test_levels_match_approved_design -q

Expected: FAIL because atem3d.layered_convergence does not exist.

- [ ] **Step 3: Implement immutable level definitions**

~~~python
@dataclass(frozen=True)
class ConvergenceLevel:
    axis: str
    level_id: str
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
~~~

build_convergence_levels returns:
- time coarse/standard/fine with 50/25/12.5 microsecond maxima and 2/1/0.5 percent relative limits;
- mesh coarse/standard/fine with 12/8/6 m source and 9/6/4.5 m receiver targets;
- domain small/standard/large with 3000/6000/12000 m extents and a fixed 750 m far-field target.

The standard level points to the completed domain6000 result. Domain large points to domain12000. Time coarse/fine reuse the exact domain6000 verification_mesh.msh.

- [ ] **Step 4: Write failing command-isolation tests**

~~~python
from atem3d.layered_convergence import build_pipeline_command_arguments

def pairs(args):
    return dict(zip(args[0::2], args[1::2], strict=True))

def test_time_coarse_changes_only_time_and_reuses_mesh(tmp_path):
    level = build_convergence_levels(tmp_path / "layered", tmp_path / "out")["time"][0]
    args = build_pipeline_command_arguments(level)
    assert pairs(args[:-4])["--max-internal-dt"] == "5e-05"
    assert pairs(args[:-4])["--max-internal-dt-fraction"] == "0.02"
    assert "--reuse-mesh" in args
    assert args[args.index("--stop-after-outputs") + 1] == "25"
~~~

- [ ] **Step 5: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: FAIL because build_pipeline_command_arguments is missing.

- [ ] **Step 6: Implement command construction**

Start from build_pipeline_arguments for the existing resistive rho1000 offset100 case. Replace only the approved axis parameters. Append --stop-after-outputs 25. Append --reuse-mesh only for generated time levels. Keep waveform, material, geometry, receiver method, boundary condition, PETSc tolerances, and empymod settings unchanged.

- [ ] **Step 7: Verify GREEN**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: all Task 1 tests PASS.

- [ ] **Step 8: Commit only Task 1 files**

~~~powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: define layered convergence levels"
~~~

### Task 2: Load Responses And Compute Pairwise Metrics

**Files:**
- Modify: src/atem3d/layered_convergence.py
- Modify: tests/test_layered_convergence.py

- [ ] **Step 1: Write failing response and metric tests**

~~~python
def response(values):
    return ConvergenceResponse(
        times=np.array([1e-5, 1e-4, 1e-3]),
        dbzdt=np.asarray(values, dtype=float),
        reference=np.array([1.0, 0.1, 0.01]),
    )

def test_compare_responses_reports_metrics():
    result = compare_responses(response([1.1, .11, .011]), response([1.0, .1, .01]))
    assert result["sample_count"] == 3
    assert result["median_percent"] == pytest.approx(10.0)
    assert result["rms_percent"] == pytest.approx(10.0)
    assert result["max_percent"] == pytest.approx(10.0)

def test_compare_rejects_mismatched_grids():
    fine = response([1.0, .1, .01])
    coarse = ConvergenceResponse(fine.times * 1.01, fine.dbzdt, fine.reference)
    with pytest.raises(ValueError, match="observation grids"):
        compare_responses(coarse, fine)

def test_compare_excludes_below_floor():
    fine = ConvergenceResponse(
        np.array([1e-5, 1e-4, 1e-3]),
        np.array([1.0, .1, 1e-7]),
        np.array([1.0, .1, 1e-7]),
    )
    coarse = ConvergenceResponse(fine.times, np.array([1.01, .102, 1.0]), fine.reference)
    result = compare_responses(coarse, fine)
    assert result["sample_count"] == 2
    assert result["max_percent"] == pytest.approx(2.0)
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: FAIL because response APIs are missing.

- [ ] **Step 3: Implement loading and metrics**

ConvergenceResponse contains aligned one-dimensional times, dBzdt, and empymod reference arrays. load_response reads verification_data.npz keys times, fem, empymod, and components, selecting dBzdt. Validation rejects fewer than three samples, nonfinite values, non-increasing or near-duplicate log times, and mismatched grids.

compare_responses applies abs(reference) >= 1e-6 * peak(abs(reference)), rejects a zero fine response inside that window, and returns sample count, excluded count, floor, masked times, pointwise relative differences, and median/RMS/max percentages using the fine or larger response as denominator.

- [ ] **Step 4: Verify GREEN and commit**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: all tests PASS.

~~~powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: evaluate layered convergence responses"
~~~

### Task 3: Axis Gates And Reproducibility Metadata

**Files:**
- Modify: src/atem3d/layered_convergence.py
- Modify: tests/test_layered_convergence.py

- [ ] **Step 1: Write failing axis-gate tests**

~~~python
def test_time_gate_passes_small_ordered_refinement():
    result = evaluate_axis_metrics("time", {
        "coarse_to_standard": {"rms_percent": 3.0},
        "standard_to_fine": {
            "median_percent": .5, "rms_percent": 1.0, "max_percent": 2.0
        },
    })
    assert result["passed"] is True

def test_mesh_gate_rejects_non_decreasing_rms():
    result = evaluate_axis_metrics("mesh", {
        "coarse_to_standard": {"rms_percent": .5},
        "standard_to_fine": {
            "median_percent": .5, "rms_percent": 1.0, "max_percent": 2.0
        },
    })
    assert "mesh_rms_not_decreasing" in result["blocking_reasons"]

def test_domain_gate_requires_large_external_pass():
    result = evaluate_axis_metrics("domain", {
        "small_to_standard": {"rms_percent": 12.0},
        "standard_to_large": {"rms_percent": 8.0},
    }, large_external_passed=True)
    assert result["passed"] is True
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: FAIL because evaluate_axis_metrics is missing.

- [ ] **Step 3: Implement exact approved gates**

Time and mesh standard-to-fine must have median <=1 percent, RMS <=2 percent, maximum <=5 percent. Coarse-to-standard RMS plus 0.1 percentage point must be at least standard-to-fine RMS. Domain small-to-standard RMS plus 0.1 percentage point must be at least standard-to-large RMS, and the large external empymod gate must pass. Return deterministic blocking reason identifiers.

- [ ] **Step 4: Write failing metadata fixture test**

Create a four-node, one-tetrahedron mesh plus synthetic verification_data.npz and timing_events.jsonl. Assert:
- nodes = 4;
- tetrahedra = 1;
- first-order Nedelec DOFs = 6 unique edges;
- internal step count = 3;
- summed forward runtime = 12.5 seconds;
- estimated memory equals cells_blocks * 2.85e-5 + nodes * 1.5e-6 GB.

- [ ] **Step 5: Implement metadata extraction**

Use meshio for nodes and tetrahedra, sorted unique tetrahedron edges for Nedelec DOFs, internal_solver_steps for step count, finite forward_done.seconds JSONL records for runtime, and the same memory formula as sotem_pipeline.py. Count tetrahedron, triangle, and line blocks in cells_blocks.

- [ ] **Step 6: Verify GREEN and commit**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: all tests PASS.

~~~powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: gate layered convergence axes"
~~~

### Task 4: Deterministic Reports And Figures

**Files:**
- Modify: src/atem3d/layered_convergence.py
- Modify: tests/test_layered_convergence.py

- [ ] **Step 1: Write failing report tests**

~~~python
def test_report_writer_emits_publication_artifacts(tmp_path):
    write_convergence_reports(tmp_path, synthetic_summary())
    for name in (
        "convergence_summary.json",
        "convergence_summary.csv",
        "convergence_report.md",
        "convergence_curves.png",
        "convergence_differences.png",
    ):
        assert (tmp_path / name).is_file()
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py::test_report_writer_emits_publication_artifacts -q

Expected: FAIL because write_convergence_reports is missing.

- [ ] **Step 3: Implement reports**

JSON omits internal NumPy arrays. CSV has one row per pairwise comparison with axis, comparison ID, counts, median/RMS/max, axis pass, and reasons. Markdown contains a paper-ready table, model and coordinate convention, reused paths, software versions, amplitude floor, and exact gates. Curves use log-log abs(dBzdt); differences use semilog-x with 1, 2, and 5 percent lines.

- [ ] **Step 4: Extend tests**

Parse JSON and CSV, assert Markdown includes every threshold, assert PNG dimensions are at least 1200 by 800, and call the writer twice to verify byte-identical JSON/CSV/Markdown.

- [ ] **Step 5: Verify GREEN and commit**

Run: python -m pytest tests/test_layered_convergence.py -q

Expected: all tests PASS.

~~~powershell
git add -- src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: report layered convergence evidence"
~~~

### Task 5: Study Runner

**Files:**
- Create: dolfinx/run_layered_convergence_study.py
- Modify: tests/test_layered_convergence.py

- [ ] **Step 1: Write failing dry-run CLI test**

~~~python
def test_runner_dry_run_writes_manifest(tmp_path):
    result = subprocess.run([
        sys.executable, "dolfinx/run_layered_convergence_study.py",
        "--output-root", str(tmp_path / "convergence"),
        "--layered-root", str(tmp_path / "layered"),
        "--axis", "time", "--level", "coarse",
        "--mode", "full", "--dry-run",
    ], check=True, capture_output=True, text=True)
    assert "RUN_AXIS=time" in result.stdout
    assert "RUN_LEVEL=coarse" in result.stdout
    assert (tmp_path / "convergence/time/coarse/case_spec.json").is_file()
    assert (tmp_path / "convergence/time/coarse/command.txt").is_file()
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_layered_convergence.py::test_runner_dry_run_writes_manifest -q

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement CLI**

Arguments:
- --output-root and --layered-root paths;
- repeatable --axis time, mesh, or domain;
- repeatable --level;
- --mode mesh, full, or evaluate;
- --force-mesh, --rerun, and --dry-run.

Generated levels write case_spec.json and command.txt, invoke the same WSL FEniCSx environment, append --checkpoint-forward, and append --resume-forward only when checkpoint and manifest match. Existing standard/large levels write existing_run.json pointers and are never copied or recomputed. Evaluate requires all three levels and exits zero only when each selected axis passes.

- [ ] **Step 4: Add contract tests**

Test existing-run pointers, checkpoint manifest matching, incomplete evaluate nonzero exit, unknown axis/level errors, and deterministic commands.

- [ ] **Step 5: Verify GREEN and existing regressions**

Run:

~~~powershell
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_e_solver_preconditioner_refresh.py tests/test_sotem_pipeline_schedule.py -q
~~~

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 5 files**

~~~powershell
git add -- dolfinx/run_layered_convergence_study.py src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: orchestrate layered convergence study"
~~~

### Task 6: Execute And Independently Verify

**Files:**
- Generate: output/publication_validation/convergence/layered_resistive_offset100/**
- Do not commit generated outputs.

- [ ] **Step 1: Generate and audit new meshes**

Run:

~~~powershell
python dolfinx/run_layered_convergence_study.py --axis mesh --mode mesh
python dolfinx/run_layered_convergence_study.py --axis domain --level small --mode mesh
~~~

Expected: memory preflight passes, source segments are covered, receiver is finite, and source divergence residual is below 1e-10.

- [ ] **Step 2: Run time convergence**

Run: python dolfinx/run_layered_convergence_study.py --axis time --mode full

Expected: coarse and fine finish 25 outputs; standard points to the existing 6000 m result.

- [ ] **Step 3: Run mesh convergence**

Run: python dolfinx/run_layered_convergence_study.py --axis mesh --mode full

Expected: coarse and fine finish 25 outputs; standard points to the existing result.

- [ ] **Step 4: Run domain convergence**

Run: python dolfinx/run_layered_convergence_study.py --axis domain --level small --mode full

Expected: small finishes 25 outputs; standard and large point to completed 6000 m and 12000 m results.

- [ ] **Step 5: Evaluate all axes**

Run: python dolfinx/run_layered_convergence_study.py --mode evaluate

Expected: CONVERGENCE_COMPLETE=3 and CONVERGENCE_PASSED=3.

- [ ] **Step 6: Independently recompute metrics**

Use a separate Python command to load every verification_data.npz, apply abs(empymod) >= 1e-6 * peak(abs(empymod)), recompute pairwise median/RMS/max, and assert convergence_summary.json agreement at 1e-12 relative tolerance.

- [ ] **Step 7: Inspect figures**

Use the local image viewer to verify readable legends, correct log axes, gate lines, no clipping, and distinct curves.

- [ ] **Step 8: Run final verification**

~~~powershell
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_e_solver_preconditioner_refresh.py tests/test_sotem_pipeline_schedule.py -q
git diff --check -- src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
~~~

Expected: all tests PASS and no whitespace errors.
