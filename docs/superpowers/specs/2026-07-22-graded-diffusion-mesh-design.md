# Graded Diffusion Mesh Design

Date: 2026-07-22

## Goal

Generate a deterministic, memory-bounded tetrahedral mesh that resolves the
near-source field and the expanding late-time diffusion volume without placing
the finest mesh size throughout the complete 1 s diffusion domain.  The design
must support a credible FEniCSx-versus-empymod/SimPEG validation, not merely
produce a runnable mesh.

## Evidence and design gate

The fixed baseline box has radius 1000 m, depth 500 m and target size 80 m.
At 0.794328 ms, a read-only spatial profile of the FEniCSx checkpoint showed
dBz/dt errors of 11.5% at 300 m depth, 14.6% at 500 m, 38.1% at 600 m and
58.1% at 800 m.  The tetrahedral median maximum edge jumps from approximately
234 m in the 300-500 m interval to 930 m in the 500-700 m interval.

Implementation is gated by the running single-variable factor-2 short-window
test.  That test changes only the fine-box depth from 500 m to 711.115 m for a
0.794328 ms terminal time.  Production mesh changes proceed only if the same
receiver and spatial-profile errors move toward empymod.  A negative or
inconclusive result returns the investigation to time and element-size
convergence instead of adding mesh features.

## Considered approaches

### 1. Nested diffusion boxes (selected)

Use a sequence of Gmsh Box fields.  Radius/depth and target element size grow
geometrically.  The Min field combines these boxes with the existing source
and receiver fields.

This approach is selected because each level has an explicit physical extent,
mesh size and audit record.  It follows the existing Gmsh field architecture
and can be tested through pure level-generation functions before meshing.

### 2. Continuous distance-dependent size field

A MathEval field could grow the target size smoothly with radial distance and
depth.  It may produce smoother grading but is harder to audit, harder to make
different above and below the air-earth interface, and more sensitive to Gmsh
expression semantics.

### 3. Time-window remeshing

Separate early-, middle- and late-time meshes minimize degrees of freedom, but
require conservative transfer of Nedelec electric history and magnetic/Faraday
state between meshes.  That transfer would introduce a new correctness problem
before the single-mesh algorithm is validated, so it is out of scope.

## Configuration and level generation

`diffusion_refinement_mode` is explicit.  Its default is `single_box`, which
preserves existing diagnostic runs and the active short-window hypothesis
test.  The formal Song launcher selects `graded`.  No existing factor-positive
run silently changes mesh topology.

The existing `diffusion_refinement_factor` continues to define the required
outer coverage relative to

```text
Lmax = sqrt(2 * rho_max * effective_t_max / mu_earth)
```

For formal validation the factor is 2.0.  The new
`diffusion_refinement_growth` defaults to 2.0 and must be greater than 1 in
graded mode.  A pure level generator receives:

- inner radius: 1000 m;
- inner depth: 500 m;
- inner target size: `diffusion_refinement_mesh_size` (80 m in Song);
- growth ratio: `diffusion_refinement_growth`;
- required radius/depth: `factor * Lmax`;
- maximum target size: the existing 2500 m far-field size;
- air top: 200 m for every diffusion level.

Starting from the inner level, radius, depth and target size grow by the ratio
until both radius and depth cover the requirement.  The final level is clipped
to the required extent and its target size is capped at the far-field size.
Radius and depth are advanced independently so the 1000 m/500 m asymmetric
inner box cannot leave a shallow-depth gap.

For a 100 ohm-m earth at 1 s, the required 2L coverage is approximately
25.23 km.  The formal domain therefore increases from 25 km to 30 km in x, y
and earth depth.  Domain convergence later compares 30 km and 40 km; this
domain change is not mixed into the short-window factor test.

## Gmsh composition

Each level becomes a Box field with its own `VIn`.  `VOut` remains 2500 m.  A
transition thickness proportional to the level extent avoids an abrupt size
jump.  The final background field is the minimum of:

- source-line threshold;
- receiver-point threshold;
- receiver ball;
- every diffusion level.

The existing source, receiver, interface embedding and mesh-quality gate are
unchanged.  The implementation does not create anisotropic elements and does
not alter material tags or the physical domain topology.

## Audit and fail-closed behavior

The mesh contract records the ordered level list with radius, depth, top,
target size and transition thickness.  The late-diffusion audit reports:

- effective terminal time and Lmax;
- required and achieved radius/depth;
- domain radius/depth;
- number of levels and growth ratio;
- target cells per Lmax at the outer active level;
- coverage and domain pass/fail flags.

A formal benchmark launcher must reject factor zero, insufficient coverage, or
an insufficient domain before a multi-hour forward solve.  The generic CLI may
retain factor zero for explicit diagnostic runs, but reports them as
diagnostic-only and cannot publish a passing validation artifact.

The memory preflight estimates each shell separately from its volume and target
size before invoking the full mesh generator.  It remains conservative; the
existing post-generation DOF estimate is still authoritative.

## Verification

### Pure tests

- level extents and sizes grow monotonically;
- the final level covers both required radius and depth;
- target sizes never exceed the far-field cap;
- factor zero preserves the current diagnostic box;
- mesh-contract identity changes when any level-defining input changes;
- full-window factor 2 with a 25 km domain fails the domain audit;
- the 30 km Song configuration passes the coverage/domain audit.

### Mesh tests

- a small Gmsh smoke mesh contains every generated level;
- the median maximum-edge ratio across adjacent designed shells is at most 3;
- source-line coverage, de Rham charge residual and local tetra-quality gates
  remain unchanged.

### Numerical gates

1. The 20-observation factor-2 short-window run must reduce the absolute
   dBz/dt error at every 300-600 m profile point and reduce their maximum by at
   least 25%, without worsening any surface formal-component maximum by more
   than 0.5 percentage point.  The time schedule remains unchanged.
2. A full 51-observation no-IP run on the graded 30 km mesh must have finite,
   converged Ex, Hz and dBz/dt and remain below the fixed 5% empymod gate at
   every sample.  Ey remains excluded as the symmetry-null component.
3. A finer graded mesh, changing only the inner target size from 80 m to 60 m,
   must change Ex, Hz and dBz/dt by no more than 2% pointwise over the formal
   window and must not rely on shifting, scaling, smoothing or dropping
   samples.
4. A 40 km domain comparison, with the same graded-mesh controls, must change
   Ex, Hz and dBz/dt by no more than 1% pointwise, showing that the 30 km
   boundary does not control the accepted response.
5. Only after these gates pass may the current-version SimPEG no-IP comparison
   and paired IP/no-IP validation start.

## Non-goals

- No time-window remeshing or cross-mesh state transfer.
- No change to BDF2, Faraday integration, H0 recovery, source definition,
  receiver operator, material equations or acceptance threshold.
- No claim that diffusion-length coverage alone proves convergence; response,
  mesh and boundary convergence are all required.
