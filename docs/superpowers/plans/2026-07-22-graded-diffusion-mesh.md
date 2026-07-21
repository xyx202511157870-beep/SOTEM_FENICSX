# Graded Diffusion Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, fail-closed graded diffusion mesh that can cover the 1 s Song no-IP diffusion volume within workstation memory and validate it by response, mesh and boundary convergence.

**Architecture:** Preserve `single_box` as the default and add an explicit `graded` mode that builds deterministic nested Gmsh Box fields from the effective terminal diffusion length. Pure functions generate and audit levels; the formal launcher opts into graded mode and rejects inadequate coverage before solving. Source, receiver, material, BDF2, Faraday and acceptance logic remain unchanged.

**Tech Stack:** Python 3.10, DOLFINx 0.8, Gmsh 4.15, PETSc/HYPRE AMS, NumPy, pytest, Bash, empymod.

---

### Task 1: Confirm the short-window hypothesis gate

**Files:**
- Verify: `/home/paidaxin/codex-sotem-song-p2-spatial-factor2-t20-abe73a5/song-noip-factor2-t20`
- Modify: `docs/validation/2026-07-22-song-noip-spatial-underresolution.md`

- [ ] **Step 1: Require the complete checkpoint**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python - <<'PY'
import numpy as np
p = "/home/paidaxin/codex-sotem-song-p2-spatial-factor2-t20-abe73a5/song-noip-factor2-t20/forward_checkpoint.npz"
with np.load(p, allow_pickle=True) as data:
    assert data["rows"].shape == (20, 4)
    assert int(data["completed_step"]) == 91
print("factor2_t20_complete")
PY
```

Expected: `factor2_t20_complete`.

- [ ] **Step 2: Apply the formal receiver gate**

Compare raw Ex, Hz and dBz/dt with the independent empymod array at
`/home/paidaxin/codex-sotem-song-p2-hybrid-full-953544d/song-noip-full/verification_data.npz`.
Exclude Ey only. Each factor-2 maximum must be no more than 0.5 percentage point
above the corresponding baseline maximum.

- [ ] **Step 3: Apply the depth-profile gate**

Evaluate the completed `e_old` at depths 300, 400, 500 and 600 m. Each absolute
dBz/dt error must be below its baseline value (11.5005%, 14.4834%, 14.5694%,
38.0580%), and their maximum must fall by at least 25%.

- [ ] **Step 4: Record the decision and commit**

If either gate fails, archive the result and stop this plan. If both pass,
append exact tables to the validation note and run:

```bash
git add docs/validation/2026-07-22-song-noip-spatial-underresolution.md
git commit -m "docs: confirm Song spatial underresolution"
```

### Task 2: Generate deterministic graded levels

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:78-115`
- Modify: `dolfinx/sotem_pipeline.py:562-608`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Write failing tests**

```python
def test_graded_diffusion_levels_cover_song_window_monotonically():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        observation_times=(1.0e-5, 1.0), rho_earth=100.0,
        diffusion_refinement_mode="graded",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_growth=2.0,
        diffusion_refinement_mesh_size=80.0,
        x_extent=30000.0, y_extent=30000.0,
        earth_depth=30000.0, air_height=30000.0,
    )
    levels = sp._diffusion_refinement_levels(config)
    required = 2.0 * sp._max_earth_diffusion_length(config)
    assert levels[0]["radius"] == 1000.0
    assert levels[0]["depth"] == 500.0
    assert levels[0]["mesh_size"] == 80.0
    assert levels[-1]["radius"] == pytest.approx(required)
    assert levels[-1]["depth"] == pytest.approx(required)
    assert all(a["radius"] < b["radius"] for a, b in zip(levels, levels[1:]))
    assert all(a["depth"] < b["depth"] for a, b in zip(levels, levels[1:]))
    assert all(a["mesh_size"] <= b["mesh_size"] <= 2500.0 for a, b in zip(levels, levels[1:]))


def test_single_box_mode_preserves_factor_positive_geometry():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        observation_times=(1.0e-5, 7.943282347242813e-4),
        rho_earth=100.0,
        diffusion_refinement_mode="single_box",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_mesh_size=80.0,
    )
    assert sp._diffusion_refinement_levels(config) == [sp._diffusion_refinement_box(config)]


def test_graded_mode_rejects_non_growing_ratio():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        diffusion_refinement_mode="graded",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_growth=1.0,
    )
    with pytest.raises(ValueError, match="diffusion_refinement_growth"):
        sp.validate_model_consistency(config)
```

- [ ] **Step 2: Verify RED**

Run the three named tests. Expected: missing fields/function failures.

- [ ] **Step 3: Implement the minimal configuration and generator**

Add to `PipelineConfig`:

```python
diffusion_refinement_mode: str = "single_box"
diffusion_refinement_growth: float = 2.0
require_validated_diffusion_mesh: bool = False
```

Add `_diffusion_refinement_levels(config)`. In `single_box`, return the current
box unchanged. In `graded`, require factor > 0 and growth > 1, start at
radius/depth/size 1000/500/80 (configured size), multiply each by growth, clip
radius and depth to `factor * Lmax`, cap size at 2500 m, and store
`transition_thickness = 0.5 * min(radius, depth)` for every level. Never shrink
the base near-field box when the requested window is short.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_model_consistency.py -q
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: generate graded diffusion mesh levels"
```

### Task 3: Audit and fail closed before formal work

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:584-608`
- Modify: `dolfinx/sotem_pipeline.py:678-760`
- Modify: `dolfinx/sotem_pipeline.py:12000-12260`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: Write failing gate tests**

```python
def _formal_song_config(sp, extent):
    return sp.PipelineConfig(
        observation_times=(1.0e-5, 1.0), rho_earth=100.0,
        diffusion_refinement_mode="graded",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_growth=2.0,
        x_extent=extent, y_extent=extent,
        earth_depth=extent, air_height=extent,
        require_validated_diffusion_mesh=True,
    )


def test_formal_gate_rejects_25km_song_domain():
    sp = _load_pipeline_module()
    with pytest.raises(ValueError, match="domain does not cover"):
        sp.validate_model_consistency(_formal_song_config(sp, 25000.0))


def test_formal_gate_accepts_30km_song_domain():
    sp = _load_pipeline_module()
    diagnostics = sp.validate_model_consistency(_formal_song_config(sp, 30000.0))
    audit = diagnostics["diffusion_refinement"]
    assert audit["coverage_passed"] is True
    assert audit["domain_passed"] is True
```

- [ ] **Step 2: Verify RED**

Run both tests. Expected: missing audit/gate behavior.

- [ ] **Step 3: Extend the audit and gate**

Return mode, growth, ordered levels, required/achieved radius and depth,
`coverage_passed`, and `domain_passed`. When
`require_validated_diffusion_mesh` is true, require graded mode, factor >= 2,
coverage pass and horizontal/earth/air domain pass; raise a specific
`ValueError` otherwise.

- [ ] **Step 4: Wire exact CLI arguments**

```python
parser.add_argument("--diffusion-refinement-mode", choices=("single_box", "graded"), default="single_box")
parser.add_argument("--diffusion-refinement-growth", type=float, default=2.0)
parser.add_argument("--require-validated-diffusion-mesh", action="store_true")
```

Pass all three into `PipelineConfig`.

- [ ] **Step 5: Verify and commit**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_model_consistency.py tests/test_dolfinx_runtime_report.py -q
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: fail closed on diffusion coverage"
```

### Task 4: Compose nested Gmsh fields and contract identity

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:1600-1640`
- Modify: `dolfinx/sotem_pipeline.py:1810-1845`
- Test: `tests/test_dolfinx_mesh_quality_gate.py`
- Test: `tests/test_dolfinx_mesh_refinement_config.py`

- [ ] **Step 1: Write failing contract and fake-Gmsh tests**

```python
def test_mesh_contract_records_graded_levels(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        workdir=tmp_path, observation_times=(1.0e-5, 1.0), rho_earth=100.0,
        diffusion_refinement_mode="graded",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_growth=2.0,
        x_extent=30000.0, y_extent=30000.0,
        earth_depth=30000.0, air_height=30000.0,
    )
    identity = sp._mesh_contract_identity(config)
    refinement = identity["refinement"]
    assert refinement["diffusion_mode"] == "graded"
    assert refinement["diffusion_growth"] == 2.0
    assert refinement["diffusion_levels"] == sp._diffusion_refinement_levels(config)
```

The fake-Gmsh test records `field.add`, `setNumber` and `setNumbers`; assert one
Box per level with matching bounds, `VIn` and `Thickness`.

- [ ] **Step 2: Verify RED**

Run both named tests. Expected: missing contract keys and field helper.

- [ ] **Step 3: Implement the helper**

```python
def _add_diffusion_refinement_fields(gmsh, config: PipelineConfig) -> list[int]:
    fields = []
    for level in _diffusion_refinement_levels(config):
        fields.append(_add_box_field(
            gmsh,
            -level["radius"], level["radius"],
            -level["radius"], level["radius"],
            -level["depth"], level["top"],
            level["mesh_size"], 2500.0,
            level.get("transition_thickness", level["radius"]),
        ))
    return fields
```

Use `[f_source, f_receiver, f_receiver_ball, *diffusion_fields]` in the Min
field. Record mode, growth and the exact ordered levels in the mesh contract.

- [ ] **Step 4: Verify and commit**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_quality_gate.py tests/test_dolfinx_mesh_refinement_config.py -q
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_mesh_quality_gate.py tests/test_dolfinx_mesh_refinement_config.py
git commit -m "feat: compose graded Gmsh diffusion fields"
```

### Task 5: Reject excessive meshes before Gmsh

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:1450-1540`
- Test: `tests/test_dolfinx_mesh_refinement_config.py`

- [ ] **Step 1: Write failing complexity tests**

```python
def test_graded_complexity_preflight_accepts_song_on_32gb():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        observation_times=(1.0e-5, 1.0), rho_earth=100.0,
        diffusion_refinement_mode="graded",
        diffusion_refinement_factor=2.0,
        diffusion_refinement_mesh_size=80.0,
        x_extent=30000.0, y_extent=30000.0, earth_depth=30000.0,
        memory_limit_gb=32.0,
    )
    audit = sp._graded_mesh_complexity_preflight(config)
    assert audit["ok"] is True
    assert audit["estimated_cells"] > 100_000


def test_uniform_80m_full_domain_is_rejected_before_meshing():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(memory_limit_gb=32.0)
    level = {"radius": 25232.0, "depth": 25232.0, "top": 200.0, "mesh_size": 80.0}
    with pytest.raises(MemoryError):
        sp._mesh_level_complexity_preflight(config, [level])
```

- [ ] **Step 2: Verify RED**

Run both tests. Expected: helper functions are absent.

- [ ] **Step 3: Implement the shell estimate**

For each ordered level compute box volume `(2r)^2 * (depth + top)`, subtract the
previous volume and estimate tetrahedra as `6 * shell_volume / mesh_size**3`.
Add a 100,000-cell source/far-field allowance, estimate nodes as cells/6, and
pass the result through the calibrated `_mesh_memory_preflight`. Call this
before Gmsh only in graded mode; retain the post-mesh gate as authoritative.

- [ ] **Step 4: Verify and commit**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_mesh_refinement_config.py -q
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_mesh_refinement_config.py
git commit -m "feat: preflight graded mesh memory"
```

### Task 6: Make the Song launcher explicitly formal

**Files:**
- Modify: `benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh`
- Create: `tests/test_song2025_fenicsx_runner.py`

- [ ] **Step 1: Write the failing launcher test**

```python
from pathlib import Path


def test_song_runner_requires_validated_graded_mesh():
    root = Path(__file__).resolve().parents[1]
    text = (root / "benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh").read_text()
    for token in (
        "--diffusion-refinement-mode graded",
        "--diffusion-refinement-factor 2",
        "--diffusion-refinement-growth 2",
        "--require-validated-diffusion-mesh",
        "--x-extent 30000",
        "--y-extent 30000",
        "--earth-depth 30000",
        "--air-height 30000",
    ):
        assert token in text
```

- [ ] **Step 2: Verify RED**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_song2025_fenicsx_runner.py -q
```

Expected: current 25 km/single-box runner fails.

- [ ] **Step 3: Change only mesh/domain arguments**

Add graded mode, factor 2, growth 2 and the formal gate. Change x/y/earth depth
and air height from 25 km to 30 km. Leave times, source, receiver, materials,
BDF2/T4, H0 and component gates unchanged.

- [ ] **Step 4: Verify and commit**

```bash
bash -n benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_song2025_fenicsx_runner.py -q
git add benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh tests/test_song2025_fenicsx_runner.py
git commit -m "feat: require graded mesh for Song validation"
```

### Task 7: Static and real-Gmsh verification

**Files:**
- Verify: all changed code
- Create: `docs/validation/2026-07-22-graded-song-mesh-smoke.md`

- [ ] **Step 1: Run static and focused checks**

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m py_compile dolfinx/sotem_pipeline.py
bash -n benchmarks/sotem/run_song2025_fenicsx_p2_t4_full.sh
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_model_consistency.py tests/test_dolfinx_mesh_refinement_config.py tests/test_dolfinx_mesh_quality_gate.py tests/test_dolfinx_runtime_report.py tests/test_song2025_fenicsx_runner.py -q
git diff --check
```

Expected: exit 0 and no failed tests.

- [ ] **Step 2: Generate a source-only 30 km graded Song mesh**

Use formal arguments plus `--source-only` in a fresh external workdir. Require
coverage/domain audits, source coverage 1, de Rham residual near `1e-13`, local
mesh quality and post-mesh memory below 30.4 GB.

- [ ] **Step 3: Audit actual shell sizes and commit evidence**

Bin tetrahedra by level using meshio. Require adjacent median maximum-edge
ratios <= 3. Record exact controls/counts/hashes and commit the report.

### Task 8: Numerical no-IP, mesh and boundary convergence

**Files:**
- Generate: external WSL artifacts
- Create: `docs/validation/2026-07-22-graded-song-noip-convergence.md`

- [ ] **Step 1: Run full 51-time 30 km/80 m no-IP**

Use BDF2/T4, graded factor 2/growth 2 and no IP. Require finite values, positive
PETSc reasons and raw pointwise empymod errors below 5% for Ex, Hz and dBz/dt.

- [ ] **Step 2: Run 60 m mesh convergence**

Change only inner target size 80 -> 60 m. Require raw pointwise differences
below 2% for Ex, Hz and dBz/dt.

- [ ] **Step 3: Run 40 km boundary convergence**

From accepted 80 m controls, change only x/y/earth depth and air height 30 ->
40 km. Require raw pointwise differences below 1% for all formal components.

- [ ] **Step 4: Publish or preserve failure**

Record exact commands, hashes, DOFs, solver logs, maxima and every failing
sample. Never shift, scale, smooth, drop points or change the 5% threshold.

### Task 9: Complete SimPEG and paired IP/no-IP validation

**Files:**
- Generate: current-version SimPEG no-IP artifact
- Generate: FEniCSx and reference IP/no-IP artifacts
- Update: fail-closed manifest and validation report

- [ ] **Step 1: Run memory-bounded SimPEG no-IP on identical Song geometry**

Lock source, receiver, 10 A, z convention, half-space, time origin and 51 times.
Keep below the 20 GiB cgroup limit and validate SimPEG against empymod before
using it as a third reference.

- [ ] **Step 2: Run paired no-IP/IP references and FEniCSx**

Use identical accepted mesh/receivers. For IP use top 300 m, `m=0.3`, `tau=1 s`,
`c=0.3`. Keep exact Cole-Cole empymod and Debye-fit diagnostics distinct.

- [ ] **Step 3: Publish provenance-backed final evidence**

Validate absolute IP responses and IP-minus-no-IP effects. Include hashes,
versions, coordinate/sign/unit conventions, convergence tables and the
separation between calculated output, internal numerical validation and
engineering/field validation.
