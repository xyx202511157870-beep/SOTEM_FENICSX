# MPI-Coordinated Formal Run Lock Design

## Purpose

Allow the FEniCSx CLI to enter one formal run directory under multiple MPI ranks without weakening the existing single-writer guarantee. This first stage only enables an MPI numerical benchmark without forward checkpointing; distributed checkpoint/restart is a separate follow-up design.

## Considered approaches

1. Disable the formal lock for MPI. Rejected because concurrent writers could corrupt formal evidence.
2. Give every rank a different work directory. Rejected because one physical solve must publish one coherent result set.
3. Let rank 0 own the existing lock and collectively broadcast acquisition status. Selected because it preserves the current lock semantics and gives every rank the same success or failure outcome.

## Design

Add an MPI-aware context manager around the existing `run_lock` primitive. Rank 0 attempts to acquire the lock and broadcasts a serializable success/failure payload. Non-root ranks never touch the lock file. All ranks enter the protected region only after a successful broadcast. On exit, all ranks synchronize before rank 0 releases the lock, preventing a later writer from entering while peer ranks still publish collective results.

The helper accepts an MPI-like communicator so its coordination can be unit tested without launching MPI. The CLI uses `MPI.COMM_WORLD`. A rank-0 acquisition error is reconstructed as a `RuntimeError` with the original message on every rank. No fallback disables the lock.

## Scope and failure policy

- Preserve serial behavior exactly when communicator size is one.
- Preserve fail-closed behavior when another process holds the run lock.
- Do not enable MPI forward checkpoints in this stage. MPI benchmarks omit `--checkpoint-forward`.
- Do not change mesh, equations, receiver definitions, time schedule, tolerances, or acceptance thresholds.

## Verification

- Unit tests prove only rank 0 invokes the underlying lock.
- Unit tests prove a root lock failure is raised on root and peers.
- Unit tests prove the exit barrier occurs before root lock release.
- Existing run-lock and model-consistency tests remain green.
- A real 8-rank two-observation FEniCSx run must complete and match the serial response within numerical reproducibility bounds before replacing the serial full run.
