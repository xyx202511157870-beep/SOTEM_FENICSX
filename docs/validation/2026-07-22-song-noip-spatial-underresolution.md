# Song no-IP spatial-underresolution diagnosis

Date: 2026-07-22

## Scope

This note records why the 51-observation Song no-IP BDF2/T4 run was stopped
after 20 completed observations.  The run remains a failed diagnostic artifact;
it is not an accepted FEniCSx response and its threshold was not changed.

Baseline artifact:

```text
/home/paidaxin/codex-sotem-song-p2-faraday-h0q8-bdf2warm16-time4-full-450cbc3/
failure_snapshot_spatial_underresolution_20260722
```

The process was deliberately interrupted after 3:35:18 wall time.  The final
`/usr/bin/time` exit status 1 therefore records the interrupt, not a spontaneous
solver failure.

## Formal-receiver evidence

At the last completed observation, 0.794328 ms, all formal components were
still below the fixed 5% empymod gate, but no full-window pass can be inferred
from this partial result.  `Ey` is a symmetry-null component and is excluded.

| Component | Error at 0.794328 ms | Maximum error in 20 samples | Time of maximum |
| --- | ---: | ---: | ---: |
| Ex | 2.141327% | 4.742749% | 0.398107 ms |
| Hz | 4.737264% | 4.737264% | 0.794328 ms |
| dBz/dt | 3.701705% | 4.925044% | 0.199526 ms |

The previously completed H0 quadrature audit had already reduced the q8-to-q10
initial-Hz difference to approximately 0.0035%, so the growing transient error
cannot be attributed to the magnetic initialization constant.

## Mesh evidence

The baseline used `diffusion_refinement_factor=0`, which fixes the fine
diffusion box at radius 1000 m and depth 500 m even though the requested window
extends to 1 s.  For a 100 ohm-m non-magnetic earth,

```text
L(t) = sqrt(2 rho t / mu0)
```

and the pipeline audit recommends a radius and depth of at least `2 L(t)`.
The recommended depth first exceeds 500 m at
`t = 0.3926990817 ms`; the Ex error maximum occurs at 0.398107 ms.

Direct inspection of the tetrahedra within 1 km of the source center showed the
following maximum-edge statistics:

| Depth interval | Cell count | Median maximum edge | 95th percentile |
| --- | ---: | ---: | ---: |
| 0-100 m | 3059 | 113.73 m | 473.25 m |
| 100-300 m | 532 | 192.47 m | 869.16 m |
| 300-500 m | 101 | 234.22 m | 1327.06 m |
| 500-700 m | 17 | 929.72 m | 1096.73 m |
| 700-1000 m | 7 | 1699.50 m | 2465.43 m |

The field checkpoint at 0.794328 ms was then evaluated at additional receiver
depths without advancing or modifying the solution.  Each value was compared
with a matching finite-source empymod response:

| Receiver depth | Ex error | dBz/dt error |
| ---: | ---: | ---: |
| 0.1 m | 2.1413% | 3.7017% |
| 100 m | 2.9506% | 3.5116% |
| 200 m | 3.6331% | 1.6222% |
| 300 m | 2.2765% | 11.5005% |
| 400 m | 2.5760% | 14.4834% |
| 500 m | 5.0646% | 14.5694% |
| 600 m | 3.8303% | 38.0580% |
| 800 m | 62.8419% | 58.1357% |

This spatial profile directly disproves the assumption that the baseline mesh
can support a credible 1 s response, even though the one surface receiver had
not yet crossed the formal threshold.

## Single-variable convergence test

A fresh 20-observation run changes only the diffusion-coverage control from
factor 0 to factor 2.  Its effective terminal time is 0.794328 ms, giving a
fine-box depth of 711.115 m while retaining the 80 m target mesh size, all
source/receiver controls, BDF2/T4 schedule, material model, domain and receiver.

```text
/home/paidaxin/codex-sotem-song-p2-spatial-factor2-t20-abe73a5/
song-noip-factor2-t20
```

The new mesh has 80,620 tetrahedra and 525,074 global Nedelec-order-2 degrees
of freedom, versus 79,836 tetrahedra and 520,084 degrees of freedom in the
baseline.  Between 500 and 700 m depth, its median maximum edge is 192.24 m
instead of 929.72 m.

The hypothesis test failed at its first 10 us observation and was stopped.

| Component | Baseline error | Factor-2 error |
| --- | ---: | ---: |
| Ex | 2.362747% | 2.629679% |
| Hz | 0.060523% | 0.050844% |
| dBz/dt | 1.489341% | 7.280626% |

The factor-2 dBz/dt result violates both the fixed 5% formal gate and the
predeclared requirement that no formal-component maximum worsen by more than
0.5 percentage point.  The failed artifact is preserved at:

```text
/home/paidaxin/codex-sotem-song-p2-spatial-factor2-t20-abe73a5/
failure_snapshot_firstpoint_dbdzdt_7p28pct_20260722
```

This disproves the assumption that regenerating a deeper Gmsh box isolates
only late-depth coverage.  The source-hit-cell quality distributions are
identical, but the receiver lies in a different single tetrahedron.  In both
meshes the point is only 0.1 m below the air-earth interface and has about 99%
of its barycentric weight on the surface vertex `(0, -500, 0)`.  The receiver
cell changed as follows:

| Quantity | Baseline | Factor 2 |
| --- | ---: | ---: |
| quality `3r/R` | 0.223026 | 0.193691 |
| aspect `R/(3r)` | 4.483781 | 5.162868 |
| maximum edge | 90.779 m | 69.152 m |
| selected-cell center depth | 18.213 m | 10.120 m |

At 10 us the diffusion length is about 39.9 m, so both receiver cells provide
only order-one vertical resolution.  On the factor-2 checkpoint, changing the
receiver from a point to 2-10 m disk or volume averages left the dBz/dt error
between 7.28% and 7.84%; simple averaging therefore does not repair the bias.
The next single-variable test keeps the baseline diffusion box and changes
only `receiver_mesh_size` from 20 m to 10 m through the first observation.

## Performance correction used by the rerun

Commit `abe73a5` makes the transient-operator cache insensitive to the few-ULP
endpoint noise that propagates into otherwise equivalent BDF2 coefficients.
The regression test reproduces the production time steps.  Replaying the 88
baseline step signatures predicts 51 cache reuses instead of 17, without
changing any matrix coefficients, right-hand sides, fields or acceptance
criteria.

## Receiver-10 m first-observation rerun

The next single-variable run reduced `receiver_mesh_size` from 20 m to 10 m
and retained the original factor-0 diffusion box.  It completed the 10 us
observation on one MPI rank with 97,303 tetrahedra and 630,562 global
Nedelec-order-2 degrees of freedom:

```text
/home/paidaxin/codex-sotem-song-p2-rx10-first1-06b5437/
song-noip-rx10-first1
```

The solve took 47:34.66 wall time.  The formal-component errors against the
finite-source empymod reference were:

| Component | Receiver-20 m baseline | Receiver-10 m rerun |
| --- | ---: | ---: |
| Ex | 2.362747% | 1.731027% |
| Hz | 0.060523% | 0.040424% |
| dBz/dt | 1.489341% | 2.140215% |

The Ex and Hz errors improved, whereas dBz/dt changed non-monotonically but
remained below the unchanged 5% gate.  This is one 10 us smoke observation,
not evidence for the requested full time window.

## Four-rank MPI equivalence and performance

The first four-rank attempt exposed MPI-only defects rather than a physics
change: ranks without local boundary facets skipped collective work, PETSc
vector assembly was not called by empty ranks, diagnostics treated local tag
counts as global counts, and the Biot-Savart quadrature indexed distributed
geometry with topology vertex ids.  The last issue corrupted the Faraday
initial constant while leaving Ex and dBz/dt nearly unchanged.  The corrected
quadrature uses the geometry dofmap, and all small-vector Biot contributions
are summed across ranks.

A strict four-rank BDF2 rerun used the same physical model, mesh, 16 internal
steps and one 10 us observation as the serial receiver-10 m case:

```text
/home/paidaxin/codex-sotem-song-p2-rx10-first1-mpi4-srcboot/
song-noip-rx10-first1-mpi4-geomfix-bdf2
```

The corrected initial `Hz` differs from the serial value by
`8.25e-13` relatively.  Final serial-versus-four-rank response differences
are numerical parallel-reduction effects only:

| Component | Serial | Four ranks | Relative difference |
| --- | ---: | ---: | ---: |
| Ex | 9.0856304916e-4 | 9.0856216701e-4 | 9.7093e-7 |
| Ey (diagnostic only) | -2.5238601907e-5 | -2.5239173812e-5 | 2.2660e-5 |
| Hz | -2.2163665220e-3 | -2.2163665343e-3 | 5.5454e-9 |
| dBz/dt | 4.3887338181e-6 | 4.3887318274e-6 | 4.5360e-7 |

The four-rank run took 15:47.00 wall time with 399% CPU utilization, compared
with 47:34.66 serial, for a 3.014x measured speedup.  `/usr/bin/time` reported
a 2.48 GB maximum resident set and no swapping for the MPI launcher process.
This establishes a usable four-core path for subsequent local validation, but
does not change the failed/incomplete status of the 1 s no-IP window or the
still-pending IP validation.
