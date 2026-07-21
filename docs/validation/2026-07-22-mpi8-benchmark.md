# MPI source assembly and eight-rank benchmark

Date: 2026-07-22

## Scope

This benchmark tested whether the validated serial Song no-IP configuration
could be accelerated safely with MPI. It did not change the physical model,
time schedule, mesh controls, receiver definition, or acceptance thresholds.

## Correctness changes

- One formal run lock is held by rank 0 and its status is broadcast.
- Mesh generation and reference-audit publication are root-only collective
  stages.
- Point-receiver candidates are gathered from owned cells across all ranks.
- Exact source-line/tetrahedron intersections are computed on owned cells,
  gathered globally, partitioned into non-overlapping atomic intervals, and
  assembled only by the selected owner rank.
- Point-electrode loads use one globally selected owned cell, preventing the
  factor-of-two endpoint load observed at a partition boundary.
- Checkpoint/resume remains fail-closed for MPI runs; only no-checkpoint MPI
  execution is currently eligible.

## Verification evidence

The serial unit/regression suites passed:

```text
pytest tests/test_dolfinx_exact_line_segments.py \
       tests/test_dolfinx_source_gradient_space.py \
       tests/test_dolfinx_model_consistency.py \
       tests/test_dolfinx_runtime_report.py -q

result: 277 passed
```

The real two-rank DOLFINx source test passed for Nedelec orders 1 and 2:

```text
mpiexec -n 2 python -m pytest \
  tests/test_dolfinx_source_gradient_space.py::test_cross_cell_exact_segments_satisfy_raw_de_rham_and_reverse_orientation -q

result: 4 passed (two parameter cases on each rank)
```

The production-size eight-rank run reached the source preflight with:

```text
global source intervals: 81
global quadrature points: 243
union coverage fraction: 1.0
charge-conservation divergence residual: 1.053241e-13
```

This replaces the earlier fail-closed `serial_assembly_required` result and
the reproduced factor-of-two endpoint residual (`0.5`) from duplicate MPI
point-load insertion.

## Performance result

The two-observation production-size benchmark used eight MPI ranks and the
same approximately 520,084 global Nedelec degrees of freedom as the serial
case. All eight ranks remained CPU-active and memory use was safe, with no
swap. The run did not publish a response after 1:19:34 and was terminated
deliberately because it was already more than twice the 33:55 serial
first-observation baseline while competing with the authoritative serial
full-window run.

The resulting PETSc/MPI exit status 59 records the deliberate SIGTERM, not a
spontaneous solver crash. No numerical-response equivalence claim is made for
this incomplete benchmark.

Artifact directory (WSL):

```text
/home/paidaxin/codex-sotem-song-p2-mpi8-bench2-fb864e8/song-noip-first2
```

## Decision

MPI source assembly is now covered by serial and two-rank correctness tests,
but the current eight-rank production configuration is not an effective
acceleration path and must not replace the running serial formal validation.
Before another full benchmark, use isolated 2/4/8-rank short tests to tune the
PETSc/HYPRE solver and avoid recomputing reusable DC/H0 initialization data.
Full-response MPI output remains unverified until a completed run is compared
component-by-component with the serial response.
