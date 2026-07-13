# 20 GB Publication Guard Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the independent-review gaps in the approved 20 GB layered-convergence workflow before generating the 18 km mesh or claiming publication acceptance.

**Architecture:** Keep the approved scientific model unchanged. Enforce the workstation ceiling at the immutable contract boundary, pass one authoritative 14 GB effective limit to the pipeline, and wrap every formal FEniCSx solve in a host-wide lease plus continuous COMSOL monitoring. Derive report and audit resource status from run-associated mesh, preflight, lease, and live-monitor evidence instead of trusting caller-supplied summary fields.

**Tech Stack:** Python 3.12, pytest, pathlib, meshio, NumPy, subprocess, Windows/WSL process control, cross-platform advisory file locking.

---

## File Map

- Add `src/atem3d/solver_resource_guard.py`: reusable host-wide advisory solver lease.
- Modify `src/atem3d/layered_convergence.py`: cap memory contracts, align the pipeline safety fraction, gate stage-two acceptance on run evidence, and report exact domain/resource metadata.
- Modify `dolfinx/run_layered_convergence_study.py`: record tetrahedra, hold the solver lease from launch check through exit, continuously monitor COMSOL, terminate on a late conflict, and persist the final live-monitor result.
- Modify `dolfinx/audit_layered_convergence.py`: enforce independent 20/14 GB maxima, rehash/recount meshes, recompute estimates, and verify run-associated live evidence.
- Modify `tests/test_layered_convergence.py`: adversarial coverage for every review finding.
- Track `src/atem3d/publication_validation.py` and `tests/test_layered_publication_validation.py`: make a clean checkout reproduce the imported benchmark API and its tests.

### Task 1: Make The Existing Benchmark Dependency Reproducible

**Files:**
- Add to Git: `src/atem3d/publication_validation.py`
- Add to Git: `tests/test_layered_publication_validation.py`

- [ ] **Step 1: Reproduce the clean-tree dependency gap**

Run:

```powershell
git ls-files --error-unmatch src/atem3d/publication_validation.py
```

Expected: nonzero exit because the imported module is absent from the commit tree.

- [ ] **Step 2: Stage only the existing dependency and focused tests**

```powershell
git add -- src/atem3d/publication_validation.py tests/test_layered_publication_validation.py
```

- [ ] **Step 3: Run the focused benchmark tests**

```powershell
python -m pytest -q tests/test_layered_publication_validation.py tests/test_layered_convergence.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git commit -m "test: track layered publication benchmark"
```

### Task 2: Enforce The Hard Ceiling And One Effective Solver Limit

**Files:**
- Modify: `tests/test_layered_convergence.py`
- Modify: `src/atem3d/layered_convergence.py`
- Modify: `dolfinx/audit_layered_convergence.py`

- [ ] **Step 1: Write failing contract and command tests**

Add cases proving that `PublicationMemoryContract(24, 6)` and
`PublicationMemoryContract(20, 5)` raise `ValueError`, while the existing stricter
`PublicationMemoryContract(19, 5)` remains valid. Extend the command assertion:

```python
arguments = build_pipeline_command_arguments(
    baseline,
    memory_limit_gb=PublicationMemoryContract().solver_memory_limit_gb,
)
assert _option_value(arguments, "--memory-limit-gb") == "14"
assert _option_value(arguments, "--memory-safety-fraction") == "1"
```

Add an audit fixture with `24/6/18` and assert that the audit rejects it even when
the summary and preflight agree.

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
python -m pytest -q tests/test_layered_convergence.py -k "memory_contract or safety_fraction or audit_rejects_contract_above_hardware_ceiling"
```

Expected: failures show that upward contracts are accepted, the safety option is
missing, and the audit trusts the raised ceiling.

- [ ] **Step 3: Implement the minimal ceiling and safety-fraction changes**

In `PublicationMemoryContract.__post_init__`, reject total memory above 20 GB and
derived solver memory above 14 GB. In `build_pipeline_command_arguments`, append
`--memory-safety-fraction 1` whenever the publication runner supplies an explicit
memory limit. In the independent audit, enforce the same absolute maxima without
importing the implementation constants.

- [ ] **Step 4: Run focused and full convergence tests**

```powershell
python -m pytest -q tests/test_layered_convergence.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/atem3d/layered_convergence.py dolfinx/audit_layered_convergence.py tests/test_layered_convergence.py
git commit -m "fix: enforce publication memory ceiling"
```

### Task 3: Hold A Host-Wide Solver Lease And Monitor Late COMSOL Starts

**Files:**
- Create: `src/atem3d/solver_resource_guard.py`
- Modify: `dolfinx/run_layered_convergence_study.py`
- Modify: `tests/test_layered_convergence.py`

- [ ] **Step 1: Write failing lease and runtime-monitor tests**

Use a real temporary lock file to prove a second `HostSolverLease` fails while the
first is held and succeeds after release. Launch a real sleeping Python child and
provide a COMSOL-name callback that changes from `[]` to
`["comsolmphserver.exe"]`; assert that the monitor terminates the child and reports
`comsol_started_during_fenicsx`.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest -q tests/test_layered_convergence.py -k "solver_lease or late_comsol"
```

Expected: import or attribute failures because the lease and monitored runner do
not exist.

- [ ] **Step 3: Implement the advisory lease**

Create `HostSolverLease` using nonblocking `msvcrt.locking` on Windows and
`fcntl.flock` elsewhere. Store owner PID, label, and UTC acquisition time in the
locked file. Hold the file descriptor for the context lifetime and release it in
`finally`; contention must fail closed with the recorded owner metadata.

- [ ] **Step 4: Implement monitored formal execution**

For paper-baseline forward solves only:

1. Acquire the host-wide lease before collecting the launch snapshot.
2. Write `live_resource_check.json` with run ID, mesh hash, estimate, lease path,
   launch status, and `runtime_monitor_passed=null`.
3. Start the WSL command with `subprocess.Popen` and poll every two seconds.
4. If COMSOL appears, terminate the process, wait, kill only if needed, record the
   detected names plus whether the latest checkpoint exists, and fail the run.
5. On normal exit, require return code zero and update the evidence with
   `runtime_monitor_passed=true` and completion time.
6. Release the lease only after the child exits and evidence is flushed.

Mesh-only and postprocess-only commands retain normal `subprocess.run` behavior.

- [ ] **Step 5: Run focused and full convergence tests**

```powershell
python -m pytest -q tests/test_layered_convergence.py
```

Expected: all tests pass and no sleeping test child remains.

- [ ] **Step 6: Commit**

```powershell
git add -- src/atem3d/solver_resource_guard.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "fix: serialize and monitor publication solvers"
```

### Task 4: Gate Reports On Complete Resource Evidence

**Files:**
- Modify: `dolfinx/run_layered_convergence_study.py`
- Modify: `src/atem3d/layered_convergence.py`
- Modify: `tests/test_layered_convergence.py`

- [ ] **Step 1: Write failing evidence/report tests**

Extend fixture preflights with `tetrahedra`, and create passing launch/runtime
snapshots for the baseline and large-domain runs. Assert that removing either
snapshot makes `study_passed` and `accepted_for_paper_figures` false. Assert that
the public summary contains a passing `resource_evidence` block and that the
Markdown report explicitly lists 6/12/18 km, nodes, tetrahedra, estimate, solver
limit, and live-monitor status.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest -q tests/test_layered_convergence.py -k "resource_evidence or report_lists_domain or tetrahedra_preflight"
```

Expected: failures show tetrahedra are absent and scientific gates can pass
without launch/runtime evidence.

- [ ] **Step 3: Record tetrahedra and derive evaluation evidence**

Count tetrahedra independently from aggregate retained cells in the runner
preflight. In stage-two evaluation, compare the actual mesh metadata against each
required baseline/large preflight and live snapshot. Return explicit blocking
reasons instead of raising so incomplete studies still produce diagnostic reports.
Include the level extents in every level row.

- [ ] **Step 4: Gate acceptance and expand the report**

Require `resource_evidence.passed` in `study_passed`. Include the evidence in
`baseline_acceptance.json`. Add a flat run-evidence Markdown table with axis,
level, domain extents, nodes, tetrahedra, estimated GB, solver-limit GB, and
runtime-monitor state.

- [ ] **Step 5: Run the full convergence tests**

```powershell
python -m pytest -q tests/test_layered_convergence.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "fix: gate publication reports on resource evidence"
```

### Task 5: Make The Independent Audit Recompute Resource Evidence

**Files:**
- Modify: `dolfinx/audit_layered_convergence.py`
- Modify: `tests/test_layered_convergence.py`

- [ ] **Step 1: Write failing tamper tests**

Generate a valid fixture report, then separately alter `mesh_sha256`, `nodes`,
`tetrahedra`, `cells_blocks`, `estimated_memory_gb`, and the live snapshot mesh
hash. Each alteration must make the audit exit nonzero. A missing or nonpassing
runtime monitor must also fail.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest -q tests/test_layered_convergence.py -k "audit_rejects_tampered_resource or audit_requires_live"
```

Expected: current audit incorrectly succeeds for at least one tampered fixture.

- [ ] **Step 3: Recompute and compare**

In the audit script, independently hash each required mesh, read it with meshio,
recount nodes/retained cells/tetrahedra, and recompute
`cells * 2.85e-5 + nodes * 1.5e-6`. Compare those values with the preflight and
summary metadata using explicit finite/tolerance checks. Verify that each live
snapshot belongs to the same run and mesh, passed at launch, passed throughout
runtime, had no COMSOL process, and had available memory at least equal to the
estimate.

- [ ] **Step 4: Run focused and broad verification**

```powershell
python -m pytest -q tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_partial_forward.py
python -m py_compile src/atem3d/layered_convergence.py src/atem3d/solver_resource_guard.py dolfinx/run_layered_convergence_study.py dolfinx/audit_layered_convergence.py
git diff --check
```

Expected: all tests and compilation pass; `git diff --check` is clean.

- [ ] **Step 5: Commit**

```powershell
git add -- dolfinx/audit_layered_convergence.py tests/test_layered_convergence.py
git commit -m "fix: independently audit resource evidence"
```

### Task 6: Regenerate Evidence And Re-Review Before 18 km

**Files:**
- Regenerate: four existing stage-two `preflight.json` files
- Generate later: run-associated baseline/large live evidence

- [ ] **Step 1: Request a second independent code review**

Review from the pre-remediation base through the new head. Critical and Important
findings must be resolved before any 18 km mesh or formal solve.

- [ ] **Step 2: Wait for every external COMSOL process to exit**

Require both the parent batch and `comsolmphserver.exe` to be absent and available
physical memory to remain stable for two observations.

- [ ] **Step 3: Regenerate the four existing preflights**

Run the approved dry-run mesh commands for time standard/fine and mesh
coarse/fine. Assert the 20/6/14 contract, tetrahedron count, mesh hash, and estimate
for each artifact.

- [ ] **Step 4: Run the 18 km mesh-only preflight**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis domain --level large --mode mesh
```

Expected: no time stepping; the exact estimate is at most 14 GB. If it exceeds 14
GB, stop and preserve the evidence without changing physics or thresholds.

## Completion Boundary

This remediation is complete only when the second review has no unresolved
Critical or Important findings, the focused/broad tests pass from tracked files,
the four existing preflights contain independently checkable tetrahedron evidence,
and the 18 km mesh-only preflight is allowed by the fixed 20/6/14 GB contract.
