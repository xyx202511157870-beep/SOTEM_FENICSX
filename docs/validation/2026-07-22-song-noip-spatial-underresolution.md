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

## Internal 300 m layer-interface root cause and mesh-only repair

The paired Song no-IP/IP command retains a 300 m model boundary with equal
100 ohm-m resistivities so both branches can use one mesh.  Direct inspection
showed that the three-dimensional 80 m Gmsh `Box` field did not constrain this
internal OCC surface.  Inside 1000 m horizontal radius, the old factor-2/Rx-5 m
mesh had only 11 interface triangles; every maximum edge exceeded 600 m and the
largest was 1356.37 m.  The triangle covering `(0, -500, -300)` produced two
colliding tetrahedra with the following fixed quality metrics:

| Quantity | Minimum/maximum |
| --- | ---: |
| `3r/R` minimum | 0.00416838 |
| `R/(3r)` maximum | 239.901 |
| maximum cell edge | 1356.37 m |

Commit `6a2253a` extends the fail-closed local mesh preflight to every configured
terminal depth point.  The preserved old mesh now fails only
`receiver_depth_profile_000` (300 m); its 400/500/600 m selections pass.

A mesh-only candidate then embedded a deterministic 80 m point lattice into the
physical 300 m layer surface without changing material boundaries, equations,
receivers or acceptance thresholds:

```text
/home/paidaxin/codex-sotem-song-layer-interface-lattice-mesh-candidate
```

It generated in 5.165 s with 37,259 Gmsh nodes and 219,333 physically tagged
tetrahedra.  The DOLFINx companion contains the same 219,333 tetrahedra; the
larger HXT progress count included 5,906 outer-boundary triangles and was not a
missing-cell discrepancy.  The Nedelec-order-2 memory preflight estimate is
26.70 GB against the existing 30.4 GB usable limit.

| Gate | Candidate result |
| --- | ---: |
| 300 m interface triangles within 1000 m | 1980 |
| interface maximum edge | 80.0 m |
| triangles above the conservative 160 m gate | 0 |
| 300 m point minimum `3r/R` | 0.614925 |
| 400 m point minimum `3r/R` | 0.863643 |
| 500 m point minimum `3r/R` | 0.687601 |
| 600 m point minimum `3r/R` | 0.816489 |

The mesh-only spatial hypothesis therefore passes.  This is approval to run a
single-variable transient test after the ongoing preserved diagnostic finishes;
it is not yet evidence that the response itself passes empymod or SimPEG.

## Preserved old-interface transient and depth baseline

The 20-observation factor-2/Rx-5 m run with the unrefined internal interface
subsequently completed rather than being inferred from the earlier partial
run:

```text
/home/paidaxin/codex-sotem-song-p2-rx5-depth711-profile-t20-mpi8-17d6e49/
song-noip-rx5-depth711-profile-t20
```

Eight core-bound MPI ranks completed all 91 internal steps in 2:13:58 wall
time with 799% reported CPU usage, zero swaps and exit status 0.  All KSP
reasons were positive convergence reason 2.  At the formal surface receiver,
the maximum finite-source empymod errors over 10 us--0.794328 ms were:

| Component | Maximum error | Time of maximum | Terminal error |
| --- | ---: | ---: | ---: |
| Ex | 2.010899% | 0.316228 ms | 0.248736% |
| Hz | 0.605442% | 0.398107 ms | 0.333860% |
| dBz/dt | 3.585368% | 0.794328 ms | 3.585368% |

This is a surface-receiver smoke pass under the unchanged 5% gate, not a full
three-dimensional validation.  The terminal field was evaluated at the four
predeclared depth points and compared separately with 9-point finite-source
empymod values whose 17-point quadrature audit had already converged:

| Depth | Ex error | Ex absolute error | dBz/dt error | dBz/dt absolute error |
| ---: | ---: | ---: | ---: | ---: |
| 300 m | 2.045661% | 5.8364e-6 V/m | 17.022524% | 1.8771e-7 T/s |
| 400 m | 1.184657% | 2.1409e-6 V/m | 7.261189% | 6.6230e-8 T/s |
| 500 m | 4.050165% | 3.4698e-6 V/m | 3.791788% | 2.5925e-8 T/s |
| 600 m | 52.810828% | 6.3876e-6 V/m | 4.574333% | 2.1370e-8 T/s |

The 600 m Ex reference is only `1.2095e-5 V/m`, so its large relative error
is reported together with the absolute error rather than hidden.  More
decisively, the 300 m dBz/dt error is 17.02% and its two candidate-cell centers
remain about 130--134 m from the requested point.  The completed response
therefore confirms that the old internal-interface mesh cannot support a
credible depth field even while the surface curve remains below 5%.

The approved single-variable v3 rerun reuses the mesh-only candidate hashes and
changes only the internal-interface surface refinement.  Its isolated artifact
directory is:

```text
/home/paidaxin/codex-sotem-song-layer-interface-v3-p2-rx5-depth711-profile-t20-mpi8-72461f5/
song-noip-rx5-depth711-profile-t20
```

No response conclusion is recorded for that candidate until its full transient
and terminal depth profile complete.

### Initial DC solver compatibility gate

The first v3 transient startup failed closed before time stepping because the
initial scalar-potential path hard-coded conjugate gradients.  On the larger
v3 mesh PETSc returned `DIVERGED_INDEFINITE_MAT` (`reason=-10`) after three
iterations, even though the DC conductivity form is expected to be positive
after its single gauge constraint.  The failure evidence is preserved at:

```text
/home/paidaxin/codex-sotem-song-layer-interface-v3-p2-rx5-depth711-profile-t20-mpi8-72461f5/
failure_snapshot_initial_dc_cg_20260722
```

An independent mesh audit found 219,333 positive-volume tetrahedra, no zero or
inverted cells and a worst Jacobian condition number of 20.90; the old mesh was
worse at approximately 1599.5.  Direct assembly of the 296,399 by 296,399 DC
matrix then showed symmetry at tolerance `1e-12`, strictly positive diagonal
entries and positive quadratic forms for three deterministic distributed test
vectors.  Solver variants on that same matrix isolated the algorithmic
difference:

| KSP / PC | PETSc reason | Iterations | True relative residual |
| --- | ---: | ---: | ---: |
| CG / Hypre BoomerAMG | -10 | 3 | 8.6594e-3 |
| GMRES / Hypre BoomerAMG | 2 | 9 | 1.5336e-8 |
| CG / GAMG | -8 | 5 | 1.1932 |
| CG / Jacobi | -10 | 57 | 4.2574e-2 |

The production DC path was changed to use the already-configured pipeline KSP
type, whose default is GMRES, instead of overriding it to CG.  A production
function rerun on the real v3 mesh converged in nine iterations with reason 2
and produced a finite initial-field vector.  This repairs a solver-compatibility
failure; it does not by itself establish response accuracy.

## Refined-interface v3 transient result

The corrected v3 no-IP run used the approved interface-refined mesh, the same
20 observations through 0.794328 ms, eight core-bound MPI ranks, and no change
to the fixed physical acceptance threshold:

```text
/home/paidaxin/codex-sotem-song-layer-interface-v3-p2-rx5-depth711-profile-t20-mpi8-340a030/
song-noip-rx5-depth711-profile-t20
```

It completed all 91 internal BDF2 steps in 1:00:55 wall time with 799% CPU
usage, zero swap and positive KSP convergence reason 2 throughout.  The mesh
has 219,333 tetrahedra and 1,401,922 global Nedelec-order-2 degrees of freedom.
At the formal surface receiver, the maximum errors against the same 9-point
finite-source empymod reference were:

| Component | Refined-interface maximum | Old-interface maximum | Time of new maximum |
| --- | ---: | ---: | ---: |
| Ex | 0.677062% | 2.010899% | 0.010000 ms |
| Hz | 0.169503% | 0.605442% | 0.794328 ms |
| dBz/dt | 1.355536% | 3.585368% | 0.010000 ms |

The terminal depth-profile comparison changed as follows:

| Depth | New Ex error | Old Ex error | New dBz/dt error | Old dBz/dt error |
| ---: | ---: | ---: | ---: | ---: |
| 300 m | 0.247120% | 2.045661% | 0.028324% | 17.022524% |
| 400 m | 0.413626% | 1.184657% | 0.054198% | 7.261189% |
| 500 m | 0.546922% | 4.050165% | 0.352672% | 3.791788% |
| 600 m | 2.853020% | 52.810828% | 0.187363% | 4.574333% |

The selected-cell-center distance at 300 m fell from about 131.88 m to
23.64 m, and at 600 m from 29.53 m to 4.55 m.  The simultaneous surface and
depth improvements validate the internal-interface spatial diagnosis.  This
artifact remains a short-window no-IP spatial validation, not a full 1 s or
IP validation; its generated `error_summary.json` records that limitation.

## Interior-receiver magnetic-initialization quadrature repair

The v3 run also exposed an independent initialization issue.  Before the
repair, the total initial receiver Hz changed non-monotonically with nominal
tetrahedron quadrature degree: q8 gave `-2.25081716674e-3 A/m`, q10 gave
`-2.22018898159e-3 A/m`, and q14 returned to `-2.25072737909e-3 A/m`.  At q10,
a quadrature point landed only 0.0119014 m from the receiver inside its
containing tetrahedron.  This was a near-singular point-sampling artifact, not
a DC or Faraday time-stepping error.

The production Biot-Savart recovery now uses a radial Duffy transform with
adaptive face refinement in receiver-containing cells.  Neighboring cells
whose maximum edge is not small relative to their receiver distance use
longest-edge adaptive tetrahedron subdivision with all evaluation points kept
inside the original finite element cell.  Far cells retain vectorized standard
tetrahedron quadrature.

The real 219,333-cell mesh was then re-solved on eight MPI ranks and audited
without changing the DC field.  The repaired initial values are:

| Degree | Conductive Hz (A/m) | Total Hz (A/m) |
| ---: | ---: | ---: |
| 2 | 2.108788081073e-8 | -2.250769702512e-3 |
| 4 | 1.549652564516e-8 | -2.250775293867e-3 |
| 6 | 1.538817925300e-8 | -2.250775402214e-3 |
| 8 | 1.542244005309e-8 | -2.250775367953e-3 |
| 10 | 1.542001241246e-8 | -2.250775370380e-3 |
| 12 | 1.541890726309e-8 | -2.250775371486e-3 |
| 14 | 1.541901298671e-8 | -2.250775371380e-3 |

Thus q8 through q14 agree within approximately `1.6e-9` relative in total Hz,
despite the unchanged q10 minimum raw quadrature-point distance of 0.0119014 m.
The test suite includes independent high-accuracy reference integrals for both
an interior receiver and a near external receiver, plus production-path tests
that ensure containing and neighboring cells take the adaptive routes.
