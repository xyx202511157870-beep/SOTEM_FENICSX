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

## Backward-Euler time-refinement result on the repaired mesh

The paired no-IP/IP validation must use backward Euler because the current
Debye-memory implementation does not support BDF2.  A same-mesh no-IP T4 run
therefore followed the BDF2 spatial result.  It used 16 steps before the first
observation and four steps between subsequent observations:

```text
/home/paidaxin/codex-sotem-song-v3-paired-be-noip-mpi8-243643f-r2/
song-noip-be-t20
```

Eight bound MPI ranks completed 92 internal steps in 1:00:24 with 799% CPU,
positive KSP reason 2 throughout and no swap.  The source gates and reference
source-quadrature audit passed, but the unchanged 5% physical gate failed at
the final 0.794328 ms sample:

| Component | T4 maximum error | Peak-normalized maximum | Result |
| --- | ---: | ---: | --- |
| Ex | 1.495500% | 0.766018% | pass |
| Hz | 2.433253% | 0.635374% | pass |
| dBz/dt | 5.403222% | 1.351917% | fail |

The failed point was retained; neither the threshold nor the data were
modified.  A single-variable T8 rerun reused byte-identical Gmsh and DOLFINx
mesh files and changed only the number of steps between observations from four
to eight:

```text
/home/paidaxin/codex-sotem-song-v3-paired-be-noip-mpi12-t8-243643f/
song-noip-be-t20
```

Twelve bound ranks completed 168 internal steps in 1:23:04 with 1199% CPU,
positive KSP reason 2, 28--29 iterations at every output and no swap.  Its
formal errors were:

| Component | T8 maximum error | T8 RMS error | Peak-normalized maximum |
| --- | ---: | ---: | ---: |
| Ex | 0.764704% | 0.309310% | 0.673767% |
| Hz | 1.208454% | 0.421809% | 0.317258% |
| dBz/dt | 2.726604% | 0.999534% | 1.351512% |

All three formal components pass the fixed 5% short-window gate.  The repaired
initial Hz differs between the eight-rank T4 and twelve-rank T8 partitions by
only about `1.3e-12` relatively, so the improvement is not an initialization
constant or MPI partition artifact.  The T4-to-T8 reduction at the failed late
sample, together with the already passing same-mesh BDF2 result, identifies
backward-Euler temporal truncation as the remaining T4 error source.

This is still a 10 us--0.794328 ms smoke window.  The generated artifact
correctly leaves `final_acceptance_passed=false` because the requested 1 s
window is not covered.

## Paired Cole--Cole IP result on the repaired mesh

The paired IP run changed only the earth constitutive model.  It reused the
same 219,333-tetrahedron, 1,401,922-DOF mesh, backward-Euler T8 schedule and
12-rank layout as the passing no-IP run.  Both mesh contracts contain the
same hashes:

```text
Gmsh:   000ce6e7c575a0182dbd4e23702b86cbed1bf0b2cc2870b57e6e360cba856e73
DOLFINx: 6329ef86a3562d4d8ee8229a3b46e7e67d3efef766fea5a9301314e0fbea4814
```

The production artifact is:

```text
/home/paidaxin/codex-sotem-song-v3-paired-be-ip-d8-mpi12-t8-243643f/
song-ip-be-t20
```

Eight positive Debye terms approximate the requested Cole--Cole law
(`rho0=100 ohm m`, `m=0.3`, `tau=1 s`, `c=0.3`) with relative L2 error
`0.452638%`, below the unchanged 1% material-fit gate.  Four, five and six
terms had respectively failed that gate at `2.87468%`, `1.83974%` and
`1.15461%`; eight terms are therefore the first tested passing level, not a
post-hoc response fit.

Twelve bound MPI ranks completed 168 internal steps in 1:23:15 with 1199%
CPU, positive KSP reason 2 throughout and no swap.  The source, source-line
quadrature, material-fit and 5% short-window physical gates all passed:

| Component | Maximum error | RMS error | Peak-normalized maximum |
| --- | ---: | ---: | ---: |
| Ex | 1.996575% | 1.614270% | 1.958912% |
| Hz | 0.903835% | 0.294480% | 0.306963% |
| dBz/dt | 2.035163% | 0.889179% | 1.796783% |

The IP and no-IP q8 initial values agree to machine precision
(`-0.002250775367955675` and `-0.0022507753679556746` A/m), as required
because the Cole--Cole DC conductivity is the same.  Within the IP run, q8
and q10 Biot--Savart initialization differ by only about `1.1e-9`
relatively.  This rules out an IP-dependent initial magnetic constant.

The polarization effect was also evaluated as `(IP-noIP)/noIP` rather than
only comparing each absolute response independently:

| Component | FEM effect at 10 us | Exact effect at 10 us | FEM effect at 0.794 ms | Exact effect at 0.794 ms | Maximum effect difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ex | -43.2909% | -41.9438% | -27.3932% | -26.3489% | 1.3570 percentage points |
| Hz | +0.4745% | +0.4674% | +33.4101% | +33.8129% | 0.4028 percentage points |
| dBz/dt | -28.7042% | -29.0357% | +23.9868% | +24.8270% | 0.8985 percentage points |

Ey remains diagnostic-only by design.  Its FEM peak is 0.168% of the Ex
peak, while the symmetric 1D reference is numerically zero.  It is retained
in the artifacts and plots but excluded from the fixed formal component set.

Like the paired no-IP result, this establishes only internal short-window
validation over 10 us--0.794328 ms.  The artifact intentionally reports
`final_acceptance_passed=false` because neither the corrected-model full
scope nor the required 1 s endpoint is covered.

## Independent SimPEG space/time audit

The canonical SimPEG S0/T4 short-window run was not accepted as a third
reference.  At 10 us its errors against the same finite-wire exact reference
were 20.0008% in Ex, 0.9491% in Hz and 7.7022% in dBz/dt.  Increasing only the
first-observation time subdivision from T4 to T16 reduced the errors to
12.9443%, 0.9766% and 2.8520%, respectively, but Ex still failed.  This showed
that both time and space resolution mattered.

The S1/T16 first-observation audit used 159,424 cells and 496,218 edges.  Its
initial Ampere solve required 2088 iterations, which exposed the old hard
2000-iteration cutoff.  Commit `cfbf889` increased only the initialization
iteration limit to 4000; it retained the `1e-8` external and `1e-11` internal
residual tolerances.  The converged true residual was `7.5645e-9`, and all 16
transient solves had positive reason 2 with maximum true residual
`9.7592e-12`.

```text
/home/paidaxin/codex-song-simpeg-firstpoint-s1t16-243643f/result.json
```

| Component | S0/T16 error | S1/T16 error | S1 result |
| --- | ---: | ---: | --- |
| Ex | 12.944349% | 3.210106% | pass |
| Hz | 0.976590% | 0.333085% | pass |
| dBz/dt | 2.851962% | 1.380265% | pass |

An S2/T16 attempt with 250,344 cells was not converted into a response point:
the initial Ampere solve reached 4000 iterations with true residual
`4.0935e-6`, above the same `1e-8` gate.  Its failure log is preserved at:

```text
/home/paidaxin/codex-song-simpeg-firstpoint-s2t16-243643f.failure-maxiter4000-residual4e-6.log
```

Thus S1/T16 is the first independently converged SimPEG level below the fixed
5% first-observation gate; S2 is a solver-convergence failure, not evidence of
further spatial convergence.

The complete 20-observation S1/T16 no-IP run subsequently finished in
2:41:56 with exit status zero, 99% CPU, peak RSS 2.72 GiB and no swap:

```text
/home/paidaxin/codex-song-simpeg-short20-s1t16-cfbf889/result.json
```

It used the same 159,424-cell, 496,218-edge S1 mesh and performed 320
transient solves.  All backend reasons were positive reason 2, the maximum
transient iteration count was 97 and the maximum external true relative
residual was `9.7592e-12`.  The DC and Ampere initialization true residuals
were respectively `2.0903e-11` and `7.5645e-9`, both below their unchanged
`1e-8` gates.

| Component | Maximum SimPEG error | Time at maximum | Terminal error |
| --- | ---: | ---: | ---: |
| Ex | 3.210106% | 10 us | 0.021277% |
| Hz | 0.630309% | 0.316228 ms | 0.380335% |
| dBz/dt | 3.452765% | 0.063096 ms | 0.300406% |

All formal components pass the fixed 5% gate across the whole short window.
On the identical 20-point time axis, the maximum SimPEG--FEniCSx difference
scaled by the exact-reference amplitude is 3.887927% in Ex, 1.588789% in Hz
and 3.462634% in dBz/dt.  The corresponding RMS differences are 1.517972%,
0.737656% and 2.150435%.  This is independent agreement between two different
three-dimensional discretizations, not merely two comparisons produced by
the FEniCSx pipeline.

### SimPEG IP failure and time-refinement diagnosis

The corresponding S1/T16 IP run completed all 320 transient solves in
2:15:02 with exit status zero, peak RSS 2.94 GiB and no swap:

```text
/home/paidaxin/codex-song-simpeg-short20-ip-s1t16-cfbf889/result.json
```

Its DC and Ampere initialization true residuals were `2.1460e-11` and
`7.6031e-9`.  Every transient solve had positive backend reason 2, with a
maximum of 82 iterations and maximum external true residual `2.2345e-11`.
The 16-term Debye approximation had relative L2 error `0.0102433%`, zero DC
conductivity residual and strictly positive amplitudes.  Solver convergence
and the material approximation are therefore not explanations for the
response failure.

| Component | Maximum SimPEG IP error | Time at maximum | Terminal error | Result |
| --- | ---: | ---: | ---: | --- |
| Ex | 7.420635% | 10 us | 0.750796% | fail |
| Hz | 0.654283% | 0.501187 ms | 0.538677% | pass |
| dBz/dt | 3.445477% | 0.1 ms | 0.000859% | pass |

The failure is retained.  At 10 us the no-IP Ex error is already signed
`-3.210106%`; the SimPEG polarization effect is `-44.4693%` versus the exact
`-41.9438%`, adding another same-sign 2.5255 percentage-point discrepancy.
The two contributions combine to the signed IP Ex error `-7.420635%`.

Two same-mesh, same-material, same-solver single-observation runs varied only
the number of time steps before 10 us:

```text
/home/paidaxin/codex-song-simpeg-firstpoint-ip-s1t8-cfbf889/result.json
/home/paidaxin/codex-song-simpeg-firstpoint-ip-s1t32-cfbf889/result.json
```

| Component | T8 error | T16 error | T32 error |
| --- | ---: | ---: | ---: |
| Ex | 7.802149% | 7.420635% | 7.320007% |
| Hz | 0.336603% | 0.337439% | 0.337828% |
| dBz/dt | 0.680039% | 0.848914% | 0.889146% |

Ex improves monotonically, but the improvement from T16 to T32 is only
0.100628 percentage points.  A conservative first-order Richardson
extrapolation gives about 7.22% at zero time-step size; using the observed
order of about 1.92 gives about 7.28%.  Both remain well above the unchanged
5% gate.  Thus the accepted explanation is an S1 spatial/common-mode SimPEG
Ex error, not insufficient T16 temporal resolution.  S2 cannot yet provide a
spatial-convergence point because its initialization fails the independent
`1e-8` residual gate, as recorded above.

## User-selected 1 ms validation window

The required engineering window was subsequently narrowed explicitly to
`1e-5 <= t <= 1e-3 s`; extending the calculation to 1 s is not required for
this validation decision.  The 21 logarithmically spaced observation times
include both endpoints.  The domain/refinement audit at 1 ms used a diffusion
length of 398.942 m, a recommended resolved radius/depth of 797.885 m, a
resolved radius of 1000 m, and a resolved depth of 797.885 m; the 25 km outer
domain is therefore not the limiting factor in this window.

The no-IP FEniCSx run is preserved at:

```text
/home/paidaxin/codex-sotem-song-be-noip-mpi12-t8-1ms-5d03d6e/song-noip-be-t21-1ms
```

It used 12 MPI ranks, 221,164 distributed tetrahedra, 1,413,482 second-order
Nedelec unknowns and 176 backward-Euler internal steps.  It finished with
exit status zero in 1:28:34 at 1199% CPU, with no swap.  Every transient KSP
solve had positive reason 2.

| Component | Maximum FEniCSx error | Time at maximum | RMS error | Result |
| --- | ---: | ---: | ---: | --- |
| Ex | 1.036429% | 1 ms | 0.378103% | pass |
| Hz | 1.412497% | 1 ms | 0.514213% | pass |
| dBz/dt | 3.301669% | 1 ms | 1.212791% | pass |

All three formal components pass the unchanged 5% physical-error gate over
the complete user-selected window.  Ey remains diagnostic-only by design.
The generated `error_summary.json` still reports
`final_acceptance_passed=false` solely because the older acceptance-contract
metadata requires a 1 s endpoint and `corrected_model_full` scope.  Its
`strict_error_gate_passed` and `physical_error_gate_passed` fields are both
true.  This legacy scope flag is not reinterpreted as a response failure.

The paired IP run reused the no-IP mesh byte-for-byte.  The Gmsh mesh,
DOLFINx mesh and mesh-contract SHA-256 values were respectively
`ee6cb662dcbbc83cf83d63c66f634da37aef1232aff37799d25b2e69e88a07f1`,
`e26bbdae69ef4c659d0a26db2480d78664677fd812bebfd51da9a680235f20a3`
and `f46625d7ea671cfb0bedbd4100b709e29c8ec2126f5b7f0b1c1af73c4e26e13f`
for both cases.  It is preserved at:

```text
/home/paidaxin/codex-sotem-song-be-ip-d8-mpi12-t8-1ms-5d03d6e/song-ip-be-t21-1ms
```

The eight-term positive Debye approximation had relative L2 error
`0.452638%`, below the fixed 1% material-fit gate.  The run completed all 176
backward-Euler steps with positive KSP reason 2, exit status zero, 1198% CPU,
1:28:48 elapsed time and no swap.

| Component | Maximum FEniCSx IP error | Time at maximum | RMS error | Result |
| --- | ---: | ---: | ---: | --- |
| Ex | 1.993756% | 19.953 us | 1.574857% | pass |
| Hz | 1.090690% | 1 ms | 0.373077% | pass |
| dBz/dt | 2.687928% | 1 ms | 1.047552% | pass |

The paired IP case therefore also passes the unchanged 5% physical-error
gate over the entire user-selected window.  As in the no-IP case, the legacy
1 s/scope metadata keeps `final_acceptance_passed=false`; both strict and
physical error gates are true.

The first attempt with safety fraction 0.95 was rejected before any solve
because the IP memory estimate was 33.37 GB versus a 33.25 GB usable limit.
That failed preflight is preserved at:

```text
/home/paidaxin/codex-sotem-song-be-ip-d8-mpi12-t8-1ms-5d03d6e.failure-preflight-33.37gt33.25
```

The completed run changed only the safety fraction to 0.96, making the limit
33.60 GB.  Observed execution used no swap, so this was a documented resource
preflight adjustment rather than a numerical-tolerance relaxation.

### Final 1 ms SimPEG and cross-code comparison

The independently discretized SimPEG S1/T16 runs used the same 159,424-cell,
496,218-edge mesh (`09bd7d3e...70c3`), the same 21-point time hash
(`ce1f5f5b...f4741`) and 336 transient solves in each case.  Their atomic
results are preserved at:

```text
/home/paidaxin/codex-song-simpeg-1ms-noip-s1t16-5d03d6e/result.json
/home/paidaxin/codex-song-simpeg-1ms-ip-s1t16-5d03d6e/result.json
```

The no-IP audit payload recorded 2:56:10 compute elapsed time; outer process
wall time was 3:06:21, with exit status zero and no swap.  All transient
solves had positive reason 2, with maximum true relative residual
`9.7592e-12`.  Its DC and Ampere initialization residuals were `2.0903e-11`
and `7.5645e-9`.

| Component | Maximum SimPEG no-IP error | Time at maximum | 1 ms error | Result |
| --- | ---: | ---: | ---: | --- |
| Ex | 3.210106% | 10 us | 0.044328% | pass |
| Hz | 0.630309% | 0.316228 ms | 0.248385% | pass |
| dBz/dt | 3.452765% | 0.063096 ms | 0.511599% | pass |

The IP audit payload recorded 2:49:14 compute elapsed time; outer process wall
time was 2:58:58, again with exit status zero and no swap.  Its 336 transient
solves also all had positive reason 2; the maximum true relative residual was
`3.5731e-11`.  The DC and Ampere initialization residuals were `2.1460e-11`
and `7.6031e-9`.  Its 16-term positive Debye approximation retained relative
L2 error `0.0102433%` and zero DC residual.

| Component | Maximum SimPEG IP error | Time at maximum | 1 ms error | Result |
| --- | ---: | ---: | ---: | --- |
| Ex | 7.420635% | 10 us | 1.079199% | fail |
| Hz | 0.654283% | 0.501187 ms | 0.416578% | pass |
| dBz/dt | 3.445477% | 0.1 ms | 0.176290% | pass |

The 1 ms extension therefore does not alter the earlier diagnosis: SimPEG's
only formal failure is the early IP Ex point.  It is not caused by linear
solver convergence, material fitting or insufficient T16 time subdivision.

The four response arrays were compared on their common time axis with the
same 1%-of-peak denominator floor.  The maximum FEniCSx--SimPEG differences,
scaled by the exact-reference amplitude, were:

| Model | Ex | Hz | dBz/dt |
| --- | ---: | ---: | ---: |
| no-IP | 3.889398% | 1.660883% | 3.462542% |
| IP | 5.764285% | 1.507268% | 3.468952% |

The IP Ex pairwise difference exceeds 5% only at 10 us because the SimPEG IP
Ex result independently fails there; the FEniCSx IP Ex error is 1.66% at that
same time and its maximum over the window is 1.99%.

The polarization-effect error was also calculated as the maximum absolute
difference between solver and exact `(IP / no-IP - 1) * 100` curves:

| Solver | Ex effect error | Hz effect error | dBz/dt effect error |
| --- | ---: | ---: | ---: |
| FEniCSx | 1.356574 percentage points | 0.429457 pp | 0.898638 pp |
| SimPEG | 2.525546 percentage points | 0.228192 pp | 0.858592 pp |

Thus both solvers recover the polarization effect, but FEniCSx has the better
absolute Ex agreement in both no-IP and IP cases.  SimPEG has the better Hz
agreement; dBz/dt accuracy is comparable.

The machine-readable combined audit and rendered comparison are at:

```text
/home/paidaxin/codex-song-1ms-cross-code-5d03d6e/comparison_summary.json
/home/paidaxin/codex-song-1ms-cross-code-5d03d6e/cross_code_comparison.png
```

The relevant SimPEG adapter, validation CLI, runner and DOLFINx regression
suites were rerun together after these computations.  Pytest reached 100%
with exit status zero and two existing skipped tests.
