# MPI-Coordinated Formal Run Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let all ranks of one no-checkpoint FEniCSx MPI job enter a single formally locked work directory and produce globally consistent point-receiver values while preserving cross-process writer exclusion.

**Architecture:** Add a communicator-aware context manager in `dolfinx/sotem_pipeline.py`. Rank 0 owns the existing dependency-free lock, broadcasts acquisition status, all ranks synchronize before rank 0 releases it, and the CLI uses this context around `_main_locked`. Receiver candidates are evaluated on owned cells and gathered across ranks before the existing collapse rule is applied. Root-only operations are coordinated, while all MPI checkpoint modes fail closed.

**Tech Stack:** Python context managers, mpi4py communicator interface, pytest, MPICH, FEniCSx/DOLFINx, PETSc/HYPRE.

---

### Task 1: Specify coordinated lock behavior with failing tests

**Files:**
- Modify: `tests/test_dolfinx_model_consistency.py`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Add fake communicator and lock helpers**

Add small test-local helpers that expose `rank`, `size`, `bcast`, and `barrier`, plus a recording context manager. The root communicator returns the value it broadcasts; a peer communicator returns a preloaded root payload. Record events so ordering can be asserted without launching MPI.

- [ ] **Step 2: Add the root-only acquisition test**

```python
def test_coordinated_formal_run_lock_only_root_touches_lock_and_releases_after_barrier(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    comm = _RecordingComm(rank=0, size=8, events=events)
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda path: _RecordingContext(events, f"lock:{path}"),
    )

    with sp._coordinated_formal_run_lock(tmp_path, comm):
        events.append("body")

    assert events == [
        f"lock:{tmp_path}:enter",
        "bcast:0",
        "body",
        "barrier",
        f"lock:{tmp_path}:exit",
    ]
```

- [ ] **Step 3: Add peer and acquisition-failure tests**

Test that a peer never calls `_formal_run_lock`, enters after a successful root payload, and raises `RuntimeError("another writer holds the run lock")` when its broadcast payload reports root failure. Test the root path broadcasts the same failure before raising.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest \
  tests/test_dolfinx_model_consistency.py \
  -k 'coordinated_formal_run_lock' -vv
```

Expected: FAIL because `_coordinated_formal_run_lock` does not exist.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_dolfinx_model_consistency.py
git commit -m "test: specify MPI-coordinated run lock"
```

### Task 2: Implement the minimal coordinated lock

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:289`
- Modify: `dolfinx/sotem_pipeline.py:11685`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Add the context manager**

Add `@contextmanager def _coordinated_formal_run_lock(workdir, comm)` beside `_formal_run_lock`. Rank 0 calls the existing lock context manager and converts acquisition success to `None` or failure to a string. Broadcast the payload from root 0. Raise the original root exception on rank 0 and a `RuntimeError` carrying the same message on peers. After successful entry, execute `comm.barrier()` in `finally` before rank 0 exits the underlying lock.

- [ ] **Step 2: Route the CLI through the coordinated lock**

Change only the writer branch in `main`:

```python
if is_writer:
    with _coordinated_formal_run_lock(workdir, MPI.COMM_WORLD):
        return _main_locked(raw_argv)
```

Leave `postprocess_saved_forward` reentrant use of `_formal_run_lock` unchanged.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all coordinated lock tests PASS.

- [ ] **Step 4: Run existing lock regression tests**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest \
  tests/test_dolfinx_model_consistency.py \
  -k 'formal_run_lock or competing_process or writer_lock' -vv
```

Expected: PASS, including cross-process exclusion and exception release.

- [ ] **Step 5: Commit the implementation**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "fix: coordinate formal run lock across MPI ranks"
```

### Task 3: Make point receivers and root-only stages MPI-safe

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:4720-5050`
- Modify: `dolfinx/sotem_pipeline.py:12070-12170`
- Modify: `tests/test_dolfinx_model_consistency.py`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Write failing receiver aggregation tests**

Add tests for a pure `_gather_receiver_sample_candidates` helper using fake communicators. Require deterministic rank-order concatenation, empty-rank tolerance, and a serial pass-through. Add a focused `evaluate_receivers` test proving cells whose local index is not below `index_map.size_local` are excluded as ghosts.

- [ ] **Step 2: Verify receiver tests fail for the missing helper**

Run the focused tests with `-k 'gather_receiver_sample_candidates or receiver_excludes_ghost_cells'`. Expected: FAIL because the collective helper and owned-cell filter are absent.

- [ ] **Step 3: Implement collective receiver aggregation**

Filter local collision cells to owned cells. Evaluate local field values, use `comm.allgather` for the tiny per-sample payload, concatenate non-empty rank payloads, and feed the existing aggregation functions. Preserve serial ordering and globally empty hard failure.

- [ ] **Step 4: Replace the blanket serial guard with checkpoint guards**

For communicator size greater than one, reject `checkpoint_forward`, `resume_forward`, and positive `stop_after_outputs` before mesh generation. Permit the no-checkpoint path.

- [ ] **Step 5: Coordinate root-only operations**

Generate/reuse the mesh on rank 0, propagate failure to peers, then barrier before `load_mesh`. Publish the reference-source quadrature audit only on rank 0. Keep final formal artifact publication root-only.

- [ ] **Step 6: Run focused and existing MPI-guard tests**

Run the receiver tests and all tests matching `formal_run_lock`, `competing_process`, `writer_lock`, or `mpi_forward`. Expected: PASS.

- [ ] **Step 7: Commit the implementation**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: enable no-checkpoint MPI FEniCSx runs"
```

### Task 4: Prove real MPI entry and benchmark numerical reproducibility

**Files:**
- Runtime artifact only: `/home/paidaxin/codex-sotem-song-p2-mpi8-bench2-<commit>/song-noip-first2`

- [ ] **Step 1: Run a real 8-rank lock smoke test**

Use the Conda MPICH launcher, never `/usr/bin/mpiexec`:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/mpiexec -n 8 \
  /home/paidaxin/miniconda3/envs/fenicsx/bin/python \
  dolfinx/sotem_pipeline.py --workdir <fresh-dir> --check-env-only --no-install
```

Expected: all ranks exit zero without lock contention.

- [ ] **Step 2: Run the two-observation benchmark without checkpointing**

Use the exact production mesh, source, material, boundary, receiver, quadrature, BDF2/BE warmup, and first two observation times. Omit `--checkpoint-forward`, `--resume-forward`, and `--stop-after-outputs` because distributed checkpointing is not implemented in this stage.

- [ ] **Step 3: Compare MPI and serial response arrays**

Load both `verification_data.npz` files with `allow_pickle=False`. Require exact component and time arrays. For Ex, Hz, and dBz/dt, report absolute and relative differences for both observations. Do not compare Ey for acceptance.

- [ ] **Step 4: Compare elapsed time and memory**

Read `/usr/bin/time -v` evidence from each `run.log`. Report wall-clock speedup and peak RSS. A switch to the full 8-rank run requires numerical agreement within solver tolerance and material wall-clock improvement.

- [ ] **Step 5: Run the complete relevant test suite**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest \
  tests/test_dolfinx_model_consistency.py \
  tests/test_dolfinx_partial_forward.py -q
```

Expected: PASS.

### Task 5: Design distributed BDF2 checkpointing after MPI benchmark passes

**Files:**
- Create: `docs/superpowers/specs/2026-07-22-mpi-bdf2-checkpoint-design.md`

- [ ] **Step 1: Record benchmark decision**

If numerical reproducibility or speedup fails, preserve the benchmark and do not proceed. If both pass, document the selected rank count and evidence paths.

- [ ] **Step 2: Specify checkpoint state and atomic publication**

The follow-up design must include per-rank `E_old`, `E_older`, Debye memories, local vector ownership metadata, `previous_step_dt`, `previous_time`, both Faraday Hz histories, solver log, common rows, MPI size, schedule fingerprint, producer identity, generation manifest, and fail-closed same-rank-count restore.

- [ ] **Step 3: Keep full-run switch gated**

Do not terminate the serial full run or launch a formal MPI full run until the two-observation benchmark and checkpoint design are complete.
