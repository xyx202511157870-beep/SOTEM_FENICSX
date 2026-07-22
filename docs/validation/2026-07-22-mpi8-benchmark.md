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

## Post-fix four-rank/eight-rank benchmark

The earlier incomplete eight-rank result above is retained as failure evidence.
After the collective-boundary, distributed-vector, point-receiver and
Biot-Savart geometry fixes in commit `9e0796e`, a new controlled benchmark
changed only the MPI rank count from four to eight. Both runs used the same
153,264-cell mesh, 984,394 global Nedelec-order-2 degrees of freedom, 16 BDF2
steps and one 10 us observation. OpenMP and BLAS thread counts were fixed at
one per rank and MPI ranks were bound to cores.

Artifacts:

```text
/home/paidaxin/codex-sotem-song-p2-rx5-depth711-first1-mpi4-9e0796e/
song-noip-rx5-depth711-first1

/home/paidaxin/codex-sotem-song-p2-rx5-depth711-first1-mpi8-9e0796e/
song-noip-rx5-depth711-first1
```

| Metric | Four ranks | Eight ranks |
| --- | ---: | ---: |
| Wall time | 30:11.99 | 20:05.39 |
| CPU utilization | 399% | 797% |
| Swap operations | 0 | 0 |
| Exit status | 0 | 0 |

The measured four-to-eight-rank speedup is 1.503x, corresponding to a 33.48%
wall-time reduction. The extra partitioning increased the first/last transient
KSP iterations from 72/90 to 84/93, so the speedup is useful but sublinear.

The formal response components agree to parallel-reduction precision:

| Component | Four ranks | Eight ranks | Relative difference |
| --- | ---: | ---: | ---: |
| Ex | 9.0549337003e-4 | 9.0549362593e-4 | 2.8261e-7 |
| Hz | -2.2161012221e-3 | -2.2161011866e-3 | 1.6010e-8 |
| dBz/dt | 4.4570780035e-6 | 4.4570825647e-6 | 1.0234e-6 |

`Ey` remains diagnostic-only and differed by 4.3973e-5 relatively. Against
the finite-source empymod reference, the eight-rank errors were 1.38735% for
Ex, 0.02845% for Hz and 0.61618% for dBz/dt. These are single-observation
results and do not establish full-window no-IP or IP validation.

The host exposes 20 physical cores and 20 logical processors on a hybrid Intel
Core Ultra 7 265K (8 performance cores plus 12 efficiency cores). Eight bound
MPI ranks are selected for subsequent long runs because this matches the
performance-core count, preserves memory headroom and avoids synchronous MPI
load imbalance between performance and efficiency cores. Increasing past eight
ranks requires a separate benchmark rather than an assumption of linear
scaling.
