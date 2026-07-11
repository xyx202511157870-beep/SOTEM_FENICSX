# Paper Baseline Convergence Extension Design

## Objective

Establish one publication baseline for the layered grounded-source TDEM
benchmark that is demonstrably converged in time step, local mesh size, and
finite-domain size. The second-stage study must preserve the previously locked
physical model, coordinate convention, error window, and acceptance thresholds.

Physical coordinates use z=0 at the ground surface, underground positive, and
air negative. DOLFINx command-line z coordinates retain the opposite internal
sign already used by the completed runs.

## Evidence Requiring The Extension

The completed first-stage study is reproducible and internally consistent. Its
metrics were independently recomputed from the stored NPZ arrays and matched the
generated report exactly. The related 104-test suite passed.

- Local-mesh convergence passed: standard-to-fine median 0.2053 percent, RMS
  0.3481 percent, and maximum 0.9522 percent.
- Time convergence missed only the median gate: standard-to-fine median 1.3066
  percent, RMS 1.3825 percent, and maximum 2.2631 percent.
- Domain convergence was not asymptotic over 3/6/12 km. The 3-to-6 km RMS
  response change was 0.5944 percent, while the 6-to-12 km change was 7.9468
  percent.
- The 12 km result independently passed the empymod gate with median 0.8286
  percent, RMS 2.7648 percent, and maximum 9.2164 percent. This indicates that
  12 km is materially better than 6 km at late time, rather than an invalid
  outlier that should be discarded.

The first-stage thresholds must not be relaxed to convert these findings into a
pass. Instead, the convergence study is recentered around the more accurate
discretization indicated by the evidence.

## Candidate Publication Baseline

The second-stage standard model is:

- Horizontal extent: 12000 m by 12000 m.
- Earth depth: 12000 m.
- Air height: 1200 m.
- Far-field mesh size: 750 m.
- Source-local mesh size and refinement radius: 8 m and 40 m.
- Receiver-local mesh size and refinement radius: 6 m and 30 m.
- Maximum internal time step: 12.5 microseconds.
- Relative internal time-step limit through 1 ms: 0.5 percent.
- Observation window: the first 25 logarithmic outputs, from 10 microseconds
  through 2.511886 ms.
- CG/HYPRE AMS tolerances: relative 1e-7 and absolute 1e-12.

Source geometry, receiver geometry, two-layer resistivity model, waveform,
formulation, natural outer boundary, receiver recovery, and empymod settings are
identical to the completed hardest-case benchmark.

## Second-Stage Axes

### Time-Step Axis

Use the same locked 12 km standard mesh for all levels.

| Level | Maximum internal dt | Relative dt limit through 1 ms | Source |
| --- | ---: | ---: | --- |
| coarse | 25 microseconds | 1 percent | reuse completed 12 km run |
| standard | 12.5 microseconds | 0.5 percent | new shared baseline run |
| fine | 6.25 microseconds | 0.25 percent | new run |

The standard and fine runs must reuse the exact completed 12 km mesh. Mesh
identity is verified by path and file hash in the study metadata.

### Domain Axis

Use the standard 8/6 m local mesh targets, 750 m far-field target, and 0.5
percent time-step controls for every level.

| Level | Horizontal extent | Earth depth | Air height | Source |
| --- | ---: | ---: | ---: | --- |
| small | 6000 m | 6000 m | 600 m | reuse completed 6 km, 0.5 percent run |
| standard | 12000 m | 12000 m | 1200 m | shared new baseline run |
| large | 24000 m | 24000 m | 2400 m | new run |

The 24 km mesh must retain the 750 m far-field target so domain expansion is not
confounded with mesh coarsening.

### Local-Mesh Axis

Use the 12 km domain and 0.5 percent time-step controls for every level.

| Level | Source-local size | Receiver-local size | Source |
| --- | ---: | ---: | --- |
| coarse | 12 m | 9 m | new run |
| standard | 8 m | 6 m | shared new baseline run |
| fine | 6 m | 4.5 m | new run |

Refinement radii remain 40 m at the source and 30 m at the receiver. Every new
mesh must pass source coverage, receiver location, source-divergence, and 24 GB
per-run memory preflight checks before forward modeling.

## Reuse And New Runs

Exactly five new forward runs are required:

1. 12 km domain, standard mesh, 0.5 percent time step. This is shared by all
   three axes as the candidate publication baseline.
2. 12 km domain, standard mesh, 0.25 percent time step.
3. 24 km domain, standard mesh, 0.5 percent time step.
4. 12 km domain, coarse local mesh, 0.5 percent time step.
5. 12 km domain, fine local mesh, 0.5 percent time step.

The completed 12 km, 1 percent run supplies the time-coarse level. The completed
6 km, 0.5 percent run supplies the domain-small level. No completed result may be
reused when its non-axis parameters differ from the table above.

## Acceptance Gates

Thresholds are unchanged from the first-stage approved design.

For time and local-mesh convergence:

- Standard-to-fine median relative difference <= 1 percent.
- Standard-to-fine RMS relative difference <= 2 percent.
- Standard-to-fine maximum relative difference <= 5 percent.
- Coarse-to-standard RMS must be at least standard-to-fine RMS minus a 0.1
  percentage-point numerical tolerance.

For domain convergence:

- The 12-to-24 km RMS response change must be no greater than the 6-to-12 km RMS
  change plus a 0.1 percentage-point numerical tolerance.
- The 24 km run must pass the existing empymod publication gate: median and RMS
  <= 5 percent and maximum <= 10 percent over the 25 effective samples.

The candidate 12 km, 0.5 percent standard result is accepted for paper figures
only when all three axes pass. Its own empymod metrics are reported alongside the
convergence metrics even though the domain gate is controlled by the 24 km run.

## Runtime And Recovery

New runs write a forward checkpoint every 10 internal steps. A resumed command
requests only the remaining outputs needed to reach 25 total rows. A checkpoint
that already contains 25 rows is postprocessed directly and must not resume time
stepping.

HYPRE AMS is recreated when the time-step matrix changes while the discrete
gradient and edge constant vectors are retained. This is a solver-efficiency
change only; equations, tolerances, and converged fields are unchanged.

Runs may execute concurrently only when the sum of mesh memory estimates remains
below 24 GB and the host retains at least 6 GB of uncommitted memory headroom.
Failed or interrupted artifacts are preserved for diagnosis and resume.

## Outputs

Write second-stage artifacts below:

`output/publication_validation/convergence/layered_resistive_offset100_stage2`

Required outputs are:

- A manifest, exact command, mesh identity, checkpoint, and final solver
  artifacts for every generated level.
- `convergence_summary.json`, `convergence_summary.csv`, and
  `convergence_report.md`.
- Publication-resolution response and pairwise-difference figures.
- A baseline acceptance record containing convergence gates, empymod metrics,
  mesh statistics, internal step counts, KSP statistics, and runtime data.

Generated numerical artifacts remain outside source control unless already
tracked by project policy.

## Verification

Before any long solve, tests must cover:

- The revised level definitions and exact five-run reuse graph.
- Isolation of each axis so only its intended parameter changes.
- Locked-mesh identity for the time axis.
- Checkpoint remaining-output semantics and direct partial postprocessing.
- Incomplete-study and failed-gate reporting.

After all runs finish:

1. Run the evaluator from disk.
2. Independently recompute every pairwise metric directly from stored NPZ arrays.
3. Verify all report values against the independent calculation.
4. Inspect both publication figures at original resolution.
5. Run the full related unit and integration test suite.

If the second-stage study fails, preserve the result and extend the failing axis
to the next finer or larger physical level. Do not lower a threshold, omit a
late-time point, or select a passing subset.

## Publication Sequence After Convergence

Passing this second-stage study establishes the non-polarizable layered FEniCSx
baseline. The remaining publication validation then proceeds to the already
planned COMSOL three-dimensional cross-check and a nonzero-IP benchmark. Neither
later validation can replace a failed second-stage convergence axis.
