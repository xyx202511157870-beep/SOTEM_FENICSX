# Layer-interface refinement implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain every active internal layer surface to the configured diffusion-refinement scale before tetrahedralization.

**Architecture:** A pure helper derives active interface depths and a bounded-spacing square lattice from the existing diffusion box. Gmsh embeds the lattice points into matching OCC surfaces, while the mesh contract records the derived scheme and forces stale meshes to regenerate.

**Tech Stack:** Python 3.10, Gmsh OCC/mesh fields, meshio, NumPy, pytest.

---

### Task 1: Deterministic lattice and mesh contract

**Files:**
- Modify: `tests/test_dolfinx_mesh_quality_gate.py`
- Modify: `dolfinx/sotem_pipeline.py:73, 577-621, 1630-1685`

- [ ] **Step 1: Write a failing lattice test**

Configure layer depths `(300.0, 900.0)`, a factor-0 box with depth 500 m,
radius 1000 m, and target size 400 m. Require
`_layer_interface_refinement_lattice` to select only 300 m, create five
intervals and six axis coordinates from -1000 to 1000 m, and keep every spacing
at or below 400 m.

- [ ] **Step 2: Run the lattice test and verify RED**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py::test_layer_interface_refinement_lattice_has_bounded_spacing -q
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the pure helper**

Return `active_depths`, `radius`, `mesh_size`, `intervals_per_axis`,
`axis_coordinates`, and `points_per_interface`. Use
`ceil(2 * radius / mesh_size)` and validated normalized layer depths no deeper
than the diffusion box.

- [ ] **Step 4: Write and verify a failing contract test**

Require `_mesh_contract_identity` to include the lattice metadata without the
full coordinate array and require generator version
`sotem-air-earth-unembedded-probes/v3`. Run the named test and confirm the old
identity fails.

- [ ] **Step 5: Implement the contract change and verify GREEN**

Add the derived interface-refinement identity and the explicit surface-only
embedding policy. Run all mesh-contract and lattice tests.

### Task 2: Embed the lattice into internal layer surfaces

**Files:**
- Modify: `tests/test_dolfinx_mesh_quality_gate.py`
- Modify: `dolfinx/sotem_pipeline.py:1730-1875`

- [ ] **Step 1: Write a failing Gmsh recording test**

Extend the recording surface model with a `z=-300 m` surface. Generate a mesh
with a 300 m layer and 400 m lattice spacing. Require 36 point tags to be
embedded into that surface and no point to be embedded into a volume.

- [ ] **Step 2: Run the recording test and verify RED**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py::test_gmsh_embeds_diffusion_lattice_on_active_layer_interface -q
```

Expected: FAIL because only the `z=0` receiver cloud is embedded.

- [ ] **Step 3: Implement surface discovery and embedding**

Create OCC point clouds before synchronization, identify matching constant-z
surfaces after fragmentation, fail if an active layer has no surface, and
embed its point cloud into the matching two-dimensional entity. Do not add the
points to physical groups or volume embeddings.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py tests/test_dolfinx_mesh_refinement_config.py -q
```

Expected: all tests pass.

### Task 3: Mesh-only scientific gate

**Files:**
- Modify: `docs/validation/2026-07-22-song-noip-spatial-underresolution.md`

- [ ] **Step 1: Generate a fresh real candidate mesh**

Use the exact active Song factor-2/Rx-5 m geometry, 300 m equal-resistivity
layer and 80 m target. Stop before field assembly or time stepping.

- [ ] **Step 2: Audit the candidate mesh**

Require the fixed local tetra quality gate to pass at 300/400/500/600 m. On the
300 m interface within 1000 m horizontal radius, require every triangle maximum
edge to be at most 160 m. Record cell count, point count and mesh generation
wall time.

- [ ] **Step 3: Preserve failure or approve the next solve**

If either gate fails, preserve the mesh and do not launch a time-domain run. If
both pass, record the artifact path as the sole mesh candidate for the next
single-variable solve.

- [ ] **Step 4: Run regression and commit**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx*.py -q
git diff --check
git status --short
```

Commit only the intended generator, tests and validation evidence.
