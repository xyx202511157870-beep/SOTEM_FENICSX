# MPI-Coordinated Formal Run Lock Design

## Purpose

Allow the FEniCSx CLI to run a no-checkpoint numerical benchmark under multiple MPI ranks without weakening the existing single-writer guarantee or changing receiver semantics. Distributed checkpoint/restart is a separate follow-up design.

## Considered approaches

1. Disable the formal lock for MPI. Rejected because concurrent writers could corrupt formal evidence.
2. Give every rank a different work directory. Rejected because one physical solve must publish one coherent result set.
3. Let rank 0 own the existing lock and collectively broadcast acquisition status. Selected because it preserves the current lock semantics and gives every rank the same success or failure outcome.

## Design

Add an MPI-aware context manager around the existing `run_lock` primitive. Rank 0 attempts to acquire the lock and broadcasts a serializable success/failure payload. Non-root ranks never touch the lock file. All ranks enter the protected region only after a successful broadcast. On exit, all ranks synchronize before rank 0 releases the lock, preventing a later writer from entering while peer ranks still publish collective results.

The helper accepts an MPI-like communicator so its coordination can be unit tested without launching MPI. The CLI uses `MPI.COMM_WORLD`. A rank-0 acquisition error is reconstructed as a `RuntimeError` with the original message on every rank. No fallback disables the lock.

Point receiver extraction becomes a collective operation. Each rank evaluates only cells it owns, excluding ghost-cell duplicates. The communicator gathers the tiny candidate arrays for each receiver sample; every rank concatenates candidates in deterministic rank order and applies the existing `median`, `mean`, `nearest_center`, or `shallowest` collapse. A globally empty candidate set remains a hard failure. The formal benchmark continues to use `median`.

Mesh generation is a root action followed by collective error propagation and a barrier before distributed mesh loading. Existing seeded meshes are still accepted. Reference-source audit publication is root-only. Existing final validation artifacts are already root-only.

MPI runs fail before mesh or solver work if `--checkpoint-forward`, `--resume-forward`, or `--stop-after-outputs` is requested. This preserves the serial checkpoint format instead of allowing corrupt multi-rank files. MPI+BDF2 checkpointing remains a separate design.

## Scope and failure policy

- Preserve serial behavior exactly when communicator size is one.
- Preserve fail-closed behavior when another process holds the run lock.
- Do not enable MPI forward checkpoints in this stage. MPI benchmark commands omit checkpoint flags, and the CLI rejects them if supplied.
- Do not change mesh, equations, receiver definitions, time schedule, tolerances, or acceptance thresholds.

## Verification

- Unit tests prove only rank 0 invokes the underlying lock.
- Unit tests prove a root lock failure is raised on root and peers.
- Unit tests prove the exit barrier occurs before root lock release.
- Unit tests prove receiver candidates from multiple ranks are combined once, ghost candidates are excluded by ownership filtering, and a globally empty receiver still fails.
- CLI tests prove MPI checkpoint flags fail before mesh generation while a no-checkpoint MPI path is no longer rejected solely because communicator size exceeds one.
- Existing run-lock and model-consistency tests remain green.
- A real 8-rank two-observation FEniCSx run must complete and match the serial response within numerical reproducibility bounds before replacing the serial full run.
