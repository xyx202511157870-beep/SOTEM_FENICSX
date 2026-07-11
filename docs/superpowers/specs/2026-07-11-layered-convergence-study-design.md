# Layered FEniCSx Convergence Study Design

## Objective

Produce publication-grade, reproducible evidence that the validated FEniCSx
time-domain electromagnetic response is independently converged with respect to
the internal time step, source/receiver-local mesh size, and finite computational
domain size.

The study uses the most demanding completed layered benchmark: a 50 m grounded
electric bipole above a two-layer earth with a 100 ohm m, 100 m thick upper layer,
a 1000 ohm m basement, and a receiver 100 m perpendicular to the source midpoint.
The physical coordinate convention is z=0 at the ground surface, underground
positive, and air negative. DOLFINx command-line z coordinates use the opposite
internal sign.

## Scope

The convergence study covers `dBz/dt` from 10 microseconds through 2.511886 ms,
the complete 25-sample effective-amplitude window defined by
`abs(reference) >= 1e-6 * peak(abs(reference))`. Samples below that amplitude
floor remain available in the completed baseline curves but do not control the
convergence gate.

Each axis is varied independently. Source geometry, receiver geometry, material
properties, source waveform, formulation, boundary condition, receiver recovery,
linear tolerance, observation times, and empymod reference settings remain fixed.

## Baseline

The standard discretization is:

- Domain: 6000 m by 6000 m horizontally, 6000 m earth depth, 600 m air height.
- Far-field mesh size: 750 m.
- Source-local mesh size and refinement radius: 8 m and 40 m.
- Receiver-local mesh size and refinement radius: 6 m and 30 m.
- Maximum internal time step: 25 microseconds.
- Relative internal time-step limit through 1 ms: 1 percent of elapsed time.
- Observation window: 10 microseconds to 10 ms with factor 10^0.1 spacing.
- Linear solver: CG with HYPRE AMS, relative tolerance 1e-7 and absolute
  tolerance 1e-12.

The existing 6000 m standard result is reused rather than recomputed.

## Three-Level Experiments

### Time-Step Convergence

Use the same locked 6000 m standard mesh for all three runs.

| Level | Maximum internal dt | Relative dt limit through 1 ms |
| --- | ---: | ---: |
| coarse | 50 microseconds | 2 percent |
| standard | 25 microseconds | 1 percent |
| fine | 12.5 microseconds | 0.5 percent |

The expected internal step counts through 2.511886 ms are 363, 593, and 1073.

### Local-Mesh Convergence

Keep the 6000 m domain, 750 m far-field size, standard time-step controls, and
refinement radii fixed. Change only the source- and receiver-local target sizes.

| Level | Source-local size | Receiver-local size |
| --- | ---: | ---: |
| coarse | 12 m | 9 m |
| standard | 8 m | 6 m |
| fine | 6 m | 4.5 m |

Each generated mesh must pass the source-segment coverage, receiver-location,
source divergence, and 24 GB memory preflight audits before a forward solve starts.

### Domain Convergence

Keep local mesh sizes and standard time-step controls fixed.

| Level | Horizontal extent | Earth depth | Air height | Far-field size |
| --- | ---: | ---: | ---: | ---: |
| small | 3000 m | 3000 m | 600 m | 750 m |
| standard | 6000 m | 6000 m | 600 m | 750 m |
| large | 12000 m | 12000 m | 1200 m | 750 m |

The 6000 m and 12000 m standard runs are reused. Only the 3000 m run is new.
Using the same 750 m far-field target prevents mesh coarsening from being confused
with domain expansion.

## Metrics And Gates

All pairwise response differences use the fine or larger run as denominator and
are evaluated only on the 25-sample effective-amplitude window.

For time and local-mesh convergence:

- Fine-to-standard median relative difference must not exceed 1 percent.
- Fine-to-standard RMS relative difference must not exceed 2 percent.
- Fine-to-standard maximum relative difference must not exceed 5 percent.
- Coarse-to-standard RMS must be greater than or equal to standard-to-fine RMS,
  allowing a numerical comparison tolerance of 0.1 percentage point.

For domain convergence:

- The response-change RMS must decrease from small-to-standard to
  standard-to-large, allowing a numerical comparison tolerance of 0.1 percentage
  point.
- The large-domain run must independently pass the empymod gate: median and RMS
  error at most 5 percent and maximum error at most 10 percent.
- Domain sensitivity is reported rather than hidden when the standard domain
  fails the external-reference gate.

Every run records observation values, cell count, node count, Nedelec degrees of
freedom, internal step count, KSP iteration statistics, wall-clock time, and peak
memory estimate.

## Architecture

Add a focused convergence module under `src/atem3d` containing immutable level
specifications, command construction, completed-run loading, pairwise metrics,
gate evaluation, and JSON/CSV report generation. Add a thin DOLFINx runner that
creates manifests and commands, reuses locked meshes where required, resumes
checkpoints, and supports `mesh`, `full`, and `evaluate` modes.

The forward solver remains unchanged. Existing publication-validation helpers are
reused for the physical model, amplitude floor, and external-reference metrics.

## Outputs

Write convergence artifacts below
`output/publication_validation/convergence/layered_resistive_offset100`:

- One directory per axis and level, with `case_spec.json`, `command.txt`, solver
  artifacts, and the locked mesh or explicit mesh-reuse reference.
- `convergence_summary.json` and `convergence_summary.csv` containing run metadata,
  pairwise metrics, gate results, and blocking reasons.
- `convergence_curves.png` with the three response curves for each axis.
- `convergence_differences.png` with pairwise relative differences and gate lines.
- `convergence_report.md` containing a paper-ready table and reproducibility notes.

Generated outputs are not source-controlled unless already allowed by the project
ignore policy.

## Failure Handling

- Reject missing, nonfinite, nonmonotonic, near-duplicate, or mismatched observation
  grids before computing metrics.
- Reject comparisons with fewer than three effective-amplitude samples.
- Mark an axis incomplete when a required level lacks final artifacts.
- Preserve failed runs and their diagnostics; never select a passing subset in
  place of the specified three levels.
- Resume interrupted forward runs only from a checkpoint whose field shape and
  recorded model manifest match the requested level.

## Verification

Unit tests cover level definitions, command isolation, locked-mesh reuse, time-grid
validation, amplitude-floor masking, pairwise metrics, convergence ordering,
external-reference gating, incomplete-run reporting, and deterministic CSV/JSON
serialization. The implementation test suite must pass before any long run begins.
After all runs finish, the evaluator is rerun from disk and its reported metrics
are independently recomputed from the stored NPZ/CSV arrays.
