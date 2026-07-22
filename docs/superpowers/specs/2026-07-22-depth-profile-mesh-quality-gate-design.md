# Depth-profile mesh-quality gate design

Date: 2026-07-22

## Problem

The terminal receiver-depth profile samples the existing FEniCSx field without
changing the solve.  The local tetrahedron preflight currently covers the
source hit cells, the formal surface receiver, and the nearby interface patch,
but not the configured depth-profile points.  A real factor-2/Rx-5 m mesh
therefore passed preflight even though the 300 m diagnostic point collides with
two tetrahedra whose `3r/R` qualities are 0.0042 and 0.0056, below the fixed
0.01 threshold.

## Decision

Extend the existing local mesh-quality preflight to every configured terminal
depth-profile point.  Do not alter the mesh, receiver response, solver state,
quality thresholds, or scientific acceptance tolerances.

For each positive configured depth, form the point `(receiver_x, receiver_y,
-depth)`.  Select all owned colliding cells collectively.  If no rank reports a
collision, select the globally nearest cell center using the same deterministic
distance/rank/local-cell tie break as the formal receiver.  Summarize the cells
with the existing exact tetrahedron quality metric.

Each point receives a stable indexed selection key and metadata containing its
depth, coordinates, global collision count, selection mode, and local selected
cell ids.  The quality artifact records an explicit ordered
`required_selections` list.  The gate always requires the three existing
selections plus every listed depth-profile selection and fails closed if any is
missing, non-finite, non-positive, below `3r/R = 0.01`, or above
`R/(3r) = 100`.

## Alternatives rejected

1. Record poor depth-cell quality only in the terminal CSV.  This would spend
   the full solve time before discovering an invalid diagnostic and could still
   let downstream analysis treat the values as credible.
2. Gate every tetrahedron in the full domain.  Remote transition cells that do
   not affect the requested source/receiver evidence would reject otherwise
   useful meshes and substantially increase diagnostic volume.
3. Add the depth-profile points to Gmsh refinement fields.  The profile is an
   observational diagnostic and must not silently change the primary mesh or
   response merely because it is enabled.

## Testing and evidence

Unit tests first reproduce the missing-required-selection failure and verify
that a configured depth point is included in `diagnose_local_mesh_quality`.
The focused mesh-quality and depth-profile suites must then pass.  A real
one-rank preflight against the preserved factor-2/Rx-5 m mesh must reject the
300 m selection while identifying the existing low-quality cells.  The ongoing
20-observation run remains preserved as pre-fix diagnostic evidence.
