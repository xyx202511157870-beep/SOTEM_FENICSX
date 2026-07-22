# Layer-interface refinement design

Date: 2026-07-22

## Problem

The Song paired no-IP/IP mesh contains a physical model boundary at 300 m so
both branches can share one discretization.  In the no-IP branch the adjacent
resistivities are both 100 ohm-m, but deleting the boundary only there would
make the polarization effect include a mesh-change contribution.

The existing three-dimensional Gmsh `Box` size field does not sufficiently
constrain the triangulation of the internal OCC layer surface.  On the real
factor-2/Rx-5 m mesh, only 11 triangles cover the 1000 m-radius refined region
at `z=-300 m`; every one has a maximum edge above 600 m although the configured
target is 80 m.  The triangle covering the receiver column has a 1356 m edge
and creates tetrahedra with `3r/R` as low as 0.00417.

## Decision

For every configured layer interface that lies within the diffusion-refinement
depth, create a deterministic square point lattice across the diffusion box.
The lattice interval count is `ceil(2 * radius / mesh_size)`, so adjacent point
spacing never exceeds `diffusion_refinement_mesh_size`.  Embed these zero-
dimensional entities into the matching internal surface after OCC
synchronization.  They constrain only the existing layer surface and do not
create a new material boundary.

The point lattice is derived solely from physical layer depths and existing
diffusion-refinement controls.  Enabling or disabling the terminal diagnostic
therefore cannot change the mesh.  Interfaces deeper than the refined box are
left at the background resolution.  The mesh contract records the active
interface depths, radius, target spacing and lattice counts, and the generator
contract version changes so an old mesh cannot be silently reused.

## Alternatives rejected

1. Collapse equal-resistivity layers only for no-IP.  This would give the
   no-IP and IP branches different meshes and contaminate the polarization
   difference with discretization change.
2. Add a point only at each diagnostic receiver depth.  It would tailor the
   forward mesh to an optional observation and leave the rest of the internal
   surface severely under-resolved.
3. Tighten the volume `Box` field alone.  The preserved mesh proves that this
   field does not enforce the internal surface triangulation.

## Verification

Tests first require the deterministic lattice, contract identity and internal-
surface embedding.  A mesh-only real generation then must pass the fixed local
quality gate at all 300/400/500/600 m diagnostic points.  At the 300 m layer,
triangles inside the 1000 m refined radius must have maximum edge no greater
than twice the 80 m target; this conservative 160 m gate allows tetrahedral
surface-layout variation without accepting the old 600-1356 m failure.  Only
after this mesh gate passes may another expensive time-domain solve begin.
