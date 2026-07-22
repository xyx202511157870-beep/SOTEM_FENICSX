# FEniCSx terminal depth-profile diagnostic design

Date: 2026-07-22

## Goal

Evaluate the already-computed terminal FEniCSx electric field and curl-derived
`dBz/dt` at additional receiver depths without changing the forward state,
formal surface receiver, mesh, time schedule, equations, or acceptance gates.
This supplies the 300/400/500/600 m evidence required by the graded-diffusion
mesh hypothesis gate.

## Selected approach

Add an optional `receiver_depth_profile_depths` tuple to `PipelineConfig` and a
matching `--receiver-depth-profile-depths` comma-separated CLI option. An empty
tuple preserves current behavior exactly.

At the final requested observation only, reuse the current `E_new` and the
already-computed curl field. For each configured positive depth, derive a
diagnostic receiver at `(receiver_x, receiver_y, -depth)` and call the existing
MPI-safe point receiver evaluator. The diagnostic must not update Faraday
state, Debye memories, the primary response rows, or solver history.

## Artifact contract

Rank 0 writes `receiver_depth_profile.csv` with one row per requested depth and
the following columns:

- `time_obs`
- `depth_m`
- `receiver_x`, `receiver_y`, `receiver_z`
- `Ex`, `Ey`, `dBzdt`
- receiver candidate-count and selected-cell geometry diagnostics already
  returned by `evaluate_receivers`

The artifact contains calculated FEniCSx values only. Independent empymod
values and errors are added by a separate analysis step so the solver does not
mix calculated output with reference data.

## Validation and failure behavior

- Depth values must be finite, strictly positive, unique, and strictly
  increasing after CLI parsing.
- A diagnostic point outside the distributed mesh is a hard error.
- Every MPI rank participates in each receiver evaluation; only rank 0 writes
  the CSV.
- The CSV is written atomically after all depth evaluations succeed.
- The primary `predictions.csv`, `errors.csv`, `error_summary.json`, and formal
  component gates remain unchanged.

## Tests

1. Configuration validation rejects non-finite, non-positive, duplicate, or
   non-increasing depths.
2. CLI parsing propagates the depth tuple without changing an empty default.
3. A unit test supplies fake fields and confirms each diagnostic uses the same
   terminal field with the expected negative-z receiver coordinates.
4. Artifact tests verify deterministic column order, depth order, atomic
   publication, and root-only writing.
5. Existing DOLFINx and validation tests must continue to pass.

## Rejected alternatives

Distributed full-field checkpointing would be more general but expands the
scope into MPI file layout and restart compatibility. Four independent forward
runs avoid code changes but cost roughly eight additional hours and would not
guarantee identical remeshing. The terminal diagnostic is the narrowest way to
obtain the required same-field evidence.
