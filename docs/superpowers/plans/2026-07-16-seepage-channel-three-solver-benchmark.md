# Seepage Channel Three-Solver Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 60 m × 10 m × 10 m conductive seepage-channel cuboid 20 m below the current 100 m grounded wire, compute audited empymod background plus SimPEG and full-domain five-receiver FEniCSx responses, compare background/channel/delta responses, and deliver a rendered Chinese Word report.

**Architecture:** A shared immutable model contract owns physical coordinates and the channel box. SimPEG applies the box as a cell-centered conductivity override; FEniCSx applies the same box in z-up coordinates and evaluates all five receivers directly from one full-domain field solve. A separate validation layer preserves empymod as a 1D background-only reference, creates convergence/error artifacts, and feeds plots and the Word report.

**Tech Stack:** Python 3.11, NumPy, SciPy, SimPEG/discretize, empymod, FEniCSx 0.8, PETSc, Gmsh, PyYAML, HDF5, Matplotlib, python-docx, pytest, Windows Conda `Codex_simpeg_ATEM3D`, WSL Conda `fenicsx`.

---

## Execution preflight

The approved design is `docs/superpowers/specs/2026-07-15-seepage-channel-three-solver-benchmark-design.md`. The current 100 m/five-receiver model files are untracked in the active workspace, so this implementation must run in the current workspace rather than a new Git worktree. A new worktree from `HEAD` would omit the authoritative files `src/atem3d/four_way_validation.py`, `examples/four_way_100m_5rx_simpeg.yaml`, and the current `output/four_way_100m_5rx/` baseline.

Before every commit, stage only the paths listed in that task. Do not stage the existing unrelated modified/deleted files or generated `dolfinx/current_task_runs/` removals.

Use these command prefixes:

```powershell
$PY = 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe'
& $PY -m pytest <test-path> -v
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解 && /home/paidaxin/miniconda3/envs/fenicsx/bin/python ...'
```

### Task 1: Shared physical model and channel geometry contract

**Files:**
- Create: `src/atem3d/seepage_channel_model.py`
- Create: `tests/test_seepage_channel_model.py`

- [ ] **Step 1: Write the failing geometry tests**

```python
import numpy as np
import pytest

from atem3d.seepage_channel_model import ChannelBox, MODEL


def test_approved_channel_contract() -> None:
    assert MODEL.source_start == (-50.0, 0.0, 0.1)
    assert MODEL.source_end == (50.0, 0.0, 0.1)
    assert MODEL.receiver_locations == tuple(
        (0.0, y, -0.1) for y in (-20.0, -10.0, 0.0, 10.0, 20.0)
    )
    assert MODEL.channel.bounds == ((-30.0, 30.0), (-5.0, 5.0), (15.0, 25.0))
    assert MODEL.channel.volume_m3 == 6000.0
    assert MODEL.channel.conductivity_s_per_m == 1.0
    assert MODEL.times.shape == (31,)


def test_channel_mask_and_z_up_conversion() -> None:
    points = np.array([[0, 0, 20], [30, 5, 25], [31, 0, 20], [0, 0, -1]], float)
    np.testing.assert_array_equal(MODEL.channel.mask(points), [True, True, False, False])
    assert MODEL.channel.to_z_up_bounds() == ((-30.0, 30.0), (-5.0, 5.0), (-25.0, -15.0))


def test_channel_must_be_underground_and_parallel() -> None:
    with pytest.raises(ValueError, match="fully underground"):
        ChannelBox(center=(0, 0, 2), size=(60, 10, 10), conductivity_s_per_m=1.0)
    with pytest.raises(ValueError, match="parallel to x"):
        ChannelBox(center=(0, 0, 20), size=(10, 60, 10), conductivity_s_per_m=1.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_model.py -v
```

Expected: collection fails with `ModuleNotFoundError: atem3d.seepage_channel_model`.

- [ ] **Step 3: Implement the immutable model contract**

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChannelBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    conductivity_s_per_m: float

    def __post_init__(self) -> None:
        if any(float(value) <= 0.0 for value in self.size):
            raise ValueError("channel sizes must be positive")
        if self.size[0] <= self.size[1]:
            raise ValueError("channel long axis must be parallel to x")
        if self.bounds[2][0] <= 0.0:
            raise ValueError("channel must be fully underground in physical z-down coordinates")
        if self.conductivity_s_per_m <= 0.0:
            raise ValueError("channel conductivity must be positive")

    @property
    def bounds(self):
        return tuple(
            (float(center - size / 2.0), float(center + size / 2.0))
            for center, size in zip(self.center, self.size)
        )

    @property
    def volume_m3(self) -> float:
        return float(np.prod(self.size))

    def mask(self, physical_points) -> np.ndarray:
        points = np.asarray(physical_points, dtype=float)
        return np.logical_and.reduce(
            [(points[:, axis] >= low) & (points[:, axis] <= high)
             for axis, (low, high) in enumerate(self.bounds)]
        )

    def to_z_up_bounds(self):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        return ((xmin, xmax), (ymin, ymax), (-zmax, -zmin))


@dataclass(frozen=True)
class SeepageBenchmarkModel:
    source_start: tuple[float, float, float]
    source_end: tuple[float, float, float]
    receiver_locations: tuple[tuple[float, float, float], ...]
    channel: ChannelBox
    background_conductivity_s_per_m: float = 0.01
    air_conductivity_s_per_m: float = 1.0e-8
    current_a: float = 1.0

    @property
    def times(self) -> np.ndarray:
        return np.logspace(-5.0, -2.0, 31)


MODEL = SeepageBenchmarkModel(
    source_start=(-50.0, 0.0, 0.1),
    source_end=(50.0, 0.0, 0.1),
    receiver_locations=tuple((0.0, y, -0.1) for y in (-20.0, -10.0, 0.0, 10.0, 20.0)),
    channel=ChannelBox((0.0, 0.0, 20.0), (60.0, 10.0, 10.0), 1.0),
)
```

- [ ] **Step 4: Run the geometry tests and verify GREEN**

Run: `& $PY -m pytest tests/test_seepage_channel_model.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the shared contract**

```powershell
git add src/atem3d/seepage_channel_model.py tests/test_seepage_channel_model.py
git commit -m "feat: define seepage channel model contract"
```

### Task 2: SimPEG conductivity-box support and matching configs

**Files:**
- Modify: `src/atem3d/config.py:441-486`
- Create: `tests/test_config_conductivity_boxes.py`
- Create: `examples/seepage_channel_100m_5rx_simpeg_background.yaml`
- Create: `examples/seepage_channel_100m_5rx_simpeg_channel.yaml`

- [ ] **Step 1: Write failing tests for box override and resolution audit**

```python
import numpy as np
import pytest
from discretize import TensorMesh

from atem3d.config import _build_ip_model_properties


def test_conductivity_box_overrides_only_earth_cells() -> None:
    mesh = TensorMesh([np.ones(8) * 5, np.ones(8) * 2.5, np.ones(24) * 2.5], x0=(-20, -10, -30))
    model = {
        "coordinate_system": "z_up",
        "layers": [
            {"top": 1e9, "bottom": 0.0, "sigma_infinity": 1e-8},
            {"top": 0.0, "bottom": -1e9, "sigma_infinity": 0.01},
        ],
        "conductivity_boxes": [{
            "bounds": [[-15, 15], [-5, 5], [-25, -15]],
            "sigma_infinity": 1.0,
            "name": "seepage_channel",
            "minimum_cells_per_cross_section": 4,
        }],
    }
    sigma, terms = _build_ip_model_properties(mesh, model)
    centers = mesh.cell_centers
    mask = np.logical_and.reduce([
        (centers[:, 0] >= -15) & (centers[:, 0] <= 15),
        (centers[:, 1] >= -5) & (centers[:, 1] <= 5),
        (centers[:, 2] >= -25) & (centers[:, 2] <= -15),
    ])
    np.testing.assert_allclose(sigma[mask], 1.0)
    np.testing.assert_allclose(sigma[(centers[:, 2] < 0) & ~mask], 0.01)
    assert terms == []


def test_box_resolution_fails_closed() -> None:
    mesh = TensorMesh([np.ones(4) * 20, np.ones(4) * 5, np.ones(4) * 5], x0=(-40, -10, -30))
    with pytest.raises(ValueError, match="at least 4 cells"):
        _build_ip_model_properties(mesh, {
            "coordinate_system": "z_up",
            "sigma_infinity": 0.01,
            "conductivity_boxes": [{
                "bounds": [[-30, 30], [-5, 5], [-25, -15]],
                "sigma_infinity": 1.0,
                "minimum_cells_per_cross_section": 4,
            }],
        })
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `& $PY -m pytest tests/test_config_conductivity_boxes.py -v`

Expected: the first test sees only background conductivity because boxes are not parsed.

- [ ] **Step 3: Add box assignment after layer assignment**

```python
def _apply_conductivity_boxes(mesh, sigma_inf, delta_by_tau, model_cfg):
    centers = np.asarray(mesh.cell_centers, dtype=float)
    for box in model_cfg.get("conductivity_boxes", []):
        bounds = np.asarray(box["bounds"], dtype=float)
        if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("conductivity box bounds must have shape (3, 2) with upper > lower")
        mask = np.logical_and.reduce([
            (centers[:, axis] >= bounds[axis, 0]) & (centers[:, axis] <= bounds[axis, 1])
            for axis in range(3)
        ])
        if not np.any(mask):
            raise ValueError(f"conductivity box {box.get('name', '<unnamed>')} marks no cells")
        minimum = int(box.get("minimum_cells_per_cross_section", 1))
        for axis in (1, 2):
            count = np.unique(np.round(centers[mask, axis], 12)).size
            if count < minimum:
                raise ValueError(f"conductivity box requires at least {minimum} cells across axis {axis}")
        sigma_inf[mask] = float(box["sigma_infinity"])
        for values in delta_by_tau.values():
            values[mask] = 0.0
    return sigma_inf, delta_by_tau
```

Call `_apply_conductivity_boxes` in both scalar-model and layered-model branches before returning the `DebyeTerm` objects.

- [ ] **Step 4: Add the two YAML configurations**

Copy the current 100 m/five-receiver mesh, source, receiver and time-step blocks from `examples/four_way_100m_5rx_simpeg.yaml`. The background YAML keeps the two layers unchanged. The channel YAML adds exactly:

```yaml
model:
  require_layer_boundary_alignment: true
  layers:
    - {top: 1000000000.0, bottom: 0.0, sigma_infinity: 1.0e-8}
    - {top: 0.0, bottom: -1000000000.0, sigma_infinity: 0.01}
  conductivity_boxes:
    - name: seepage_channel
      bounds: [[-30.0, 30.0], [-5.0, 5.0], [-25.0, -15.0]]
      sigma_infinity: 1.0
      minimum_cells_per_cross_section: 4
```

Refine the central tensor-mesh widths so the channel y and z spans each contain at least four cells, while keeping identical meshes in the background and channel YAML files.

- [ ] **Step 5: Run tests and a config-build smoke**

Run:

```powershell
& $PY -m pytest tests/test_config_conductivity_boxes.py tests/test_solver_core.py -v
& $PY -c "from atem3d.config import load_config, build_simulation; [build_simulation(load_config(p)) for p in ('examples/seepage_channel_100m_5rx_simpeg_background.yaml','examples/seepage_channel_100m_5rx_simpeg_channel.yaml')]"
```

Expected: tests pass and both simulations build without solving.

- [ ] **Step 6: Commit SimPEG box support**

```powershell
git add src/atem3d/config.py tests/test_config_conductivity_boxes.py examples/seepage_channel_100m_5rx_simpeg_background.yaml examples/seepage_channel_100m_5rx_simpeg_channel.yaml
git commit -m "feat: add SimPEG seepage conductivity box"
```

### Task 3: FEniCSx receiver-set and channel configuration contracts

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:60-215`
- Create: `dolfinx/seepage_channel_full_domain.py`
- Create: `tests/test_seepage_channel_fenicsx_contract.py`

- [ ] **Step 1: Write failing pure-Python contract tests**

```python
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys


def load_pipeline():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location("seepage_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_receiver_set_is_explicit_and_preserves_legacy_receiver() -> None:
    module = load_pipeline()
    cfg = module.PipelineConfig(receiver_locations=((0, -20, 0.1), (0, 20, 0.1)))
    assert module._resolved_receiver_locations(cfg) == ((0.0, -20.0, 0.1), (0.0, 20.0, 0.1))
    assert module._resolved_receiver_locations(module.PipelineConfig(receiver=(1, 2, 3))) == ((1.0, 2.0, 3.0),)


def test_channel_box_is_z_up_and_auditable() -> None:
    module = load_pipeline()
    cfg = module.PipelineConfig(
        conductivity_box_bounds=((-30, 30), (-5, 5), (-25, -15)),
        conductivity_box_sigma=1.0,
        conductivity_box_name="seepage_channel",
    )
    audit = module._conductivity_box_config_audit(cfg)
    assert audit["bounds"] == [[-30.0, 30.0], [-5.0, 5.0], [-25.0, -15.0]]
    assert audit["theoretical_volume_m3"] == 6000.0
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_fenicsx_contract.py -v`

Expected: `PipelineConfig` rejects `receiver_locations` and conductivity-box fields.

- [ ] **Step 3: Add backward-compatible configuration fields and validation**

```python
@dataclass
class PipelineConfig:
    receiver: tuple[float, float, float] = (0.0, -300.0, -0.1)
    receiver_locations: tuple[tuple[float, float, float], ...] = ()
    conductivity_box_name: str = ""
    conductivity_box_bounds: tuple[tuple[float, float], ...] = ()
    conductivity_box_sigma: float = 0.0
    conductivity_box_mesh_size: float = 2.5


def _resolved_receiver_locations(config):
    raw = config.receiver_locations or (config.receiver,)
    locations = tuple(tuple(float(v) for v in point) for point in raw)
    if any(len(point) != 3 for point in locations):
        raise ValueError("each receiver location must contain x, y, z")
    if len(set(locations)) != len(locations):
        raise ValueError("receiver locations must be unique")
    return locations


def _conductivity_box_config_audit(config):
    bounds = np.asarray(config.conductivity_box_bounds, dtype=float)
    if bounds.size == 0:
        return {"enabled": False}
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("conductivity_box_bounds must have shape (3, 2)")
    if bounds[2, 1] >= 0.0:
        raise ValueError("conductivity box must remain below z=0 in z-up coordinates")
    return {
        "enabled": True,
        "name": str(config.conductivity_box_name),
        "bounds": bounds.tolist(),
        "sigma_s_per_m": float(config.conductivity_box_sigma),
        "theoretical_volume_m3": float(np.prod(bounds[:, 1] - bounds[:, 0])),
    }
```

- [ ] **Step 4: Add CLI parsing for repeated receivers and box arguments**

Use repeated `--receiver-location x,y,z` values; keep `--receiver-x/y/z` as the legacy fallback. Add:

```python
parser.add_argument("--receiver-location", action="append", default=[])
parser.add_argument("--conductivity-box-name", default="")
parser.add_argument("--conductivity-box-bounds", default="")
parser.add_argument("--conductivity-box-sigma", type=float, default=0.0)
parser.add_argument("--conductivity-box-mesh-size", type=float, default=2.5)
```

Parse `-30,30;-5,5;-25,-15` deterministically into the tuple-of-pairs field.

- [ ] **Step 5: Run contract and existing parser tests**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_fenicsx_contract.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_mesh_refinement_config.py -v
```

Expected: all tests pass; legacy single-receiver configuration remains valid.

- [ ] **Step 6: Commit configuration contracts**

```powershell
git add dolfinx/sotem_pipeline.py dolfinx/seepage_channel_full_domain.py tests/test_seepage_channel_fenicsx_contract.py
git commit -m "feat: define FEniCSx seepage receiver set"
```

### Task 4: FEniCSx shared mesh refinement and channel material override

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:1404-1775,2084-2148`
- Modify: `dolfinx/seepage_channel_full_domain.py`
- Create: `tests/test_seepage_channel_fenicsx_mesh_material.py`

- [ ] **Step 1: Write failing tests for all receiver refinement and material audit**

```python
import numpy as np
from dolfinx.seepage_channel_full_domain import box_mask, receiver_configs


def test_receiver_configs_cover_both_sides_without_mirroring() -> None:
    locations = ((0, -20, 0.1), (0, -10, 0.1), (0, 0, 0.1), (0, 10, 0.1), (0, 20, 0.1))
    configs = receiver_configs(locations)
    assert [cfg.receiver for cfg in configs] == list(locations)
    assert all(cfg.provenance == "explicit_full_domain" for cfg in configs)


def test_box_mask_and_volume_audit() -> None:
    centers = np.array([[0, 0, -20], [29, 4, -16], [0, 6, -20], [0, 0, 1]], float)
    volumes = np.array([1000, 1000, 1000, 1000], float)
    mask, audit = box_mask(centers, volumes, ((-30, 30), (-5, 5), (-25, -15)))
    np.testing.assert_array_equal(mask, [True, True, False, False])
    assert audit["local_cell_count"] == 2
    assert audit["local_discrete_volume_m3"] == 2000.0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_fenicsx_mesh_material.py -v`

Expected: imports fail because the helper functions do not exist.

- [ ] **Step 3: Implement pure receiver/box helpers**

```python
@dataclass(frozen=True)
class ReceiverEvaluationConfig:
    receiver: tuple[float, float, float]
    receiver_id: str
    provenance: str = "explicit_full_domain"


def receiver_configs(locations):
    return tuple(
        ReceiverEvaluationConfig(tuple(float(v) for v in location), f"Rx{index + 1}")
        for index, location in enumerate(locations)
    )


def box_mask(centers, volumes, bounds):
    centers = np.asarray(centers, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    mask = np.logical_and.reduce([
        (centers[:, axis] >= bounds[axis, 0]) & (centers[:, axis] <= bounds[axis, 1])
        for axis in range(3)
    ])
    return mask, {
        "local_cell_count": int(np.count_nonzero(mask)),
        "local_discrete_volume_m3": float(np.sum(volumes[mask])),
    }
```

- [ ] **Step 4: Refine all receivers and the box in one Gmsh model**

Change receiver point/cloud/surface refinement creation to iterate over `_resolved_receiver_locations(config)` using `replace(config, receiver=location)`. Add a Gmsh box-distance/threshold field covering `(-30,30) × (-5,5) × (-25,-15)` with `SizeMin=conductivity_box_mesh_size`. Combine source, receiver, diffusion and channel fields with a Gmsh `Min` field.

The generated mesh remains one full air-earth domain and one source line. Do not generate a half-domain mesh or apply a symmetry boundary.

- [ ] **Step 5: Override earth DG0 conductivity in the channel cells**

After ordinary air/earth/layer assignment:

```python
audit = _conductivity_box_config_audit(config)
if audit["enabled"]:
    centers, radii, volumes = _cell_centers_radii_volumes(msh)
    mask, local = box_mask(centers, volumes, config.conductivity_box_bounds)
    cells = np.flatnonzero(mask).astype(np.int32)
    if cells.size == 0:
        raise RuntimeError("conductivity box marks no local cells")
    _assign_dg0_by_cell(sigma, cells, config.conductivity_box_sigma)
    _assign_dg0_by_cell(sigma_inf, cells, config.conductivity_box_sigma)
    _assign_dg0_by_cell(rho, cells, 1.0 / config.conductivity_box_sigma)
```

Reduce cell count and discrete volume across MPI and store `materials["conductivity_box_audit"]` with theoretical volume, discrete volume, relative volume error, global cell count and conductivity extrema.

- [ ] **Step 6: Run pure tests and WSL mesh/material smoke**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_fenicsx_mesh_material.py tests/test_seepage_channel_fenicsx_contract.py -v
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解 && /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_seepage_channel_fenicsx_mesh_material.py -v'
```

Expected: all tests pass in both environments.

- [ ] **Step 7: Commit mesh and material support**

```powershell
git add dolfinx/sotem_pipeline.py dolfinx/seepage_channel_full_domain.py tests/test_seepage_channel_fenicsx_mesh_material.py
git commit -m "feat: mesh FEniCSx seepage channel volume"
```

### Task 5: FEniCSx direct five-receiver sampling in one field solve

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:4168-4398,7480-8365,9500-9740`
- Modify: `dolfinx/seepage_channel_full_domain.py`
- Create: `tests/test_seepage_channel_fenicsx_receivers.py`

- [ ] **Step 1: Write failing receiver-set tests using injected evaluators**

```python
from types import SimpleNamespace
from dolfinx.seepage_channel_full_domain import evaluate_receiver_set, records_to_array


def test_five_receiver_set_calls_evaluator_five_times() -> None:
    calls = []
    def evaluator(_E, _dbdt, _msh, config):
        calls.append(config.receiver)
        return {"Ex": config.receiver[1], "dBzdt": 2 * config.receiver[1], "Hz": 3 * config.receiver[1]}
    config = SimpleNamespace(
        receiver_locations=tuple((0.0, y, 0.1) for y in (-20, -10, 0, 10, 20)),
        receiver=(0.0, 0.0, 0.1),
    )
    records = evaluate_receiver_set(None, None, None, config, evaluator=evaluator)
    assert calls == list(config.receiver_locations)
    assert [row["provenance"] for row in records] == ["explicit_full_domain"] * 5
    assert records_to_array(records, ("Ex", "dBzdt", "Hz")).shape == (5, 3)
```

- [ ] **Step 2: Run the receiver test and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_fenicsx_receivers.py -v`

Expected: import fails for `evaluate_receiver_set`.

- [ ] **Step 3: Implement direct receiver-set evaluation**

```python
def evaluate_receiver_set(E, dbdt, msh, config, *, evaluator):
    import copy

    records = []
    for receiver in receiver_configs(_resolved_receiver_locations(config)):
        local_config = copy.copy(config)
        local_config.receiver = receiver.receiver
        local_config.receiver_locations = ()
        record = dict(evaluator(E, dbdt, msh, local_config))
        record.update({
            "receiver_id": receiver.receiver_id,
            "receiver_x_m": receiver.receiver[0],
            "receiver_y_m": receiver.receiver[1],
            "receiver_z_m": receiver.receiver[2],
            "provenance": receiver.provenance,
        })
        records.append(record)
    return records


def records_to_array(records, components):
    return np.asarray(
        [[float(record[component]) for component in components] for record in records],
        dtype=float,
    )
```

Use this function at every observation output inside `run_fetd_forward`. Preserve the single-receiver code path when `receiver_locations` is empty. For `biot_current` and `biot_rate`, compute the receiver-specific H and dB/dt for each receiver rather than attaching the legacy receiver's magnetic value to all rows.

- [ ] **Step 4: Store multi-receiver forward arrays and direct-source diagnostics**

For receiver-set runs, return:

```python
{
    "times": observation_times,
    "receiver_locations": np.asarray(_resolved_receiver_locations(config)),
    "components": np.asarray(("Ex", "dBzdt", "Hz")),
    "data": np.asarray(receiver_time_records),  # shape (5, 31, 3)
    "receiver_provenance": np.asarray(["explicit_full_domain"] * 5),
    "receiver_diagnostics": receiver_diagnostic_records,
}
```

Write `predictions_5rx.csv` in long form with one row per receiver/time and the columns `receiver_id,receiver_x_m,receiver_y_m,receiver_z_m,time_obs,Ex,dBzdt,Hz,provenance`.

- [ ] **Step 5: Add a regression guard against mirror generation**

```python
def test_full_domain_module_has_no_mirror_expansion() -> None:
    source = Path("dolfinx/seepage_channel_full_domain.py").read_text(encoding="utf-8")
    assert "mirror_crossline_values" not in source
    assert "output[0] = output[4]" not in source
    assert "explicit_full_domain" in source
```

- [ ] **Step 6: Run receiver tests and the affected FEniCSx unit surface**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_fenicsx_receivers.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_validation_artifacts.py -v
```

Expected: all tests pass, including legacy single-receiver artifacts.

- [ ] **Step 7: Commit direct five-receiver support**

```powershell
git add dolfinx/sotem_pipeline.py dolfinx/seepage_channel_full_domain.py tests/test_seepage_channel_fenicsx_receivers.py
git commit -m "feat: sample five FEniCSx receivers directly"
```

### Task 6: empymod/SimPEG result generation and provenance

**Files:**
- Create: `src/atem3d/seepage_channel_validation.py`
- Create: `tools/run_seepage_channel_benchmark.py`
- Create: `tests/test_seepage_channel_validation.py`

- [ ] **Step 1: Write failing tests for empymod metadata and solver result shapes**

```python
import numpy as np
import pytest
from atem3d.seepage_channel_validation import validate_result_payload


def test_empymod_is_background_only() -> None:
    payload = {
        "times": np.logspace(-5, -2, 31),
        "receiver_locations": np.zeros((5, 3)),
        "components": np.array(["Ex", "dBzdt", "Hz"]),
        "values": np.ones((5, 31, 3)),
        "background_only_1d": True,
    }
    validate_result_payload("empymod", payload)
    payload["background_only_1d"] = False
    with pytest.raises(ValueError, match="background_only_1d"):
        validate_result_payload("empymod", payload)


def test_all_results_require_465_finite_values() -> None:
    payload = {
        "times": np.logspace(-5, -2, 31),
        "receiver_locations": np.zeros((5, 3)),
        "components": np.array(["Ex", "dBzdt", "Hz"]),
        "values": np.ones((5, 31, 3)),
    }
    validate_result_payload("SimPEG", payload)
    payload["values"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_result_payload("SimPEG", payload)
```

- [ ] **Step 2: Run validation tests and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_validation.py -v`

Expected: module import fails.

- [ ] **Step 3: Implement payload validation and empymod background**

Reuse `run_empymod_reference` from `src/atem3d/four_way_validation.py` for the approved 100 m wire, five receivers, 31 times and three components. Save `empymod_background.npz` with:

```python
np.savez_compressed(
    output,
    times=MODEL.times,
    receiver_locations=np.asarray(MODEL.receiver_locations),
    components=np.asarray(("Ex", "dBzdt", "Hz")),
    values=values,
    background_only_1d=np.asarray(True),
    reference_role=np.asarray("layered_background_only"),
)
```

`validate_result_payload` must check exact shape `(5, 31, 3)`, finite values, exact times/components/coordinates, and the empymod role flag.

- [ ] **Step 4: Add explicit runner subcommands**

```text
python tools/run_seepage_channel_benchmark.py empymod --output-root output/seepage_channel_100m_5rx
python tools/run_seepage_channel_benchmark.py simpeg --case background --output-root output/seepage_channel_100m_5rx
python tools/run_seepage_channel_benchmark.py simpeg --case channel --output-root output/seepage_channel_100m_5rx
```

The SimPEG commands call `atem3d.cli run` with the two YAML files and write `simpeg_background.h5` / `simpeg_channel.h5`. After each run, load the HDF5 result, resample positive times to `MODEL.times`, validate the 465-value contract and write a JSON provenance sidecar with the config SHA-256.

- [ ] **Step 5: Run unit tests and a short empymod generation**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_validation.py tests/test_four_way_100m_5rx_validation.py -v
& $PY tools/run_seepage_channel_benchmark.py empymod --output-root tmp/seepage_channel_smoke
```

Expected: tests pass and `tmp/seepage_channel_smoke/empymod_background.npz` validates.

- [ ] **Step 6: Commit result generation**

```powershell
git add src/atem3d/seepage_channel_validation.py tools/run_seepage_channel_benchmark.py tests/test_seepage_channel_validation.py
git commit -m "feat: generate seepage benchmark solver inputs"
```

### Task 7: FEniCSx benchmark command and smoke run

**Files:**
- Modify: `dolfinx/seepage_channel_full_domain.py`
- Create: `tools/run_fenicsx_seepage_background.sh`
- Create: `tools/run_fenicsx_seepage_channel.sh`
- Create: `tests/test_seepage_channel_fenicsx_command.py`

- [ ] **Step 1: Write a failing command-contract test**

Test that both scripts use the same extents, source, five repeated receiver locations, time settings and solver settings; only the channel script includes:

```text
--conductivity-box-name seepage_channel
--conductivity-box-bounds=-30,30;-5,5;-25,-15
--conductivity-box-sigma 1.0
--conductivity-box-mesh-size 2.5
```

Also assert neither script contains `mirror`, `y10_baseline`, or `y20_baseline`.

- [ ] **Step 2: Run the command test and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_fenicsx_command.py -v`

Expected: scripts are missing.

- [ ] **Step 3: Create matching full-domain commands**

Base both scripts on `tools/run_fenicsx_y0_baseline.sh`, but use one work directory per physical case and add all five receivers:

```bash
--receiver-location 0,-20,0.1 \
--receiver-location 0,-10,0.1 \
--receiver-location 0,0,0.1 \
--receiver-location 0,10,0.1 \
--receiver-location 0,20,0.1
```

Keep domain, source, 31 observation times, natural boundary, E formulation, direct solver settings, 30 GB memory cap and checkpointing identical between background and channel.

- [ ] **Step 4: Run contract tests and a coarse source-only/mesh smoke**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_fenicsx_command.py tests/test_seepage_channel_fenicsx_receivers.py -v
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解 && bash tools/run_fenicsx_seepage_channel.sh --check-env-only'
```

If the shell wrapper cannot forward `--check-env-only`, run the generated Python command directly with `--check-env-only --force-mesh` and verify that mesh/material/receiver preflight completes without transient stepping.

- [ ] **Step 5: Commit FEniCSx commands**

```powershell
git add dolfinx/seepage_channel_full_domain.py tools/run_fenicsx_seepage_background.sh tools/run_fenicsx_seepage_channel.sh tests/test_seepage_channel_fenicsx_command.py
git commit -m "feat: add FEniCSx seepage benchmark commands"
```

### Task 8: Background, channel-delta, error and convergence aggregation

**Files:**
- Modify: `src/atem3d/seepage_channel_validation.py`
- Modify: `tools/run_seepage_channel_benchmark.py`
- Create: `tests/test_seepage_channel_aggregation.py`

- [ ] **Step 1: Write failing delta and strong-signal tests**

```python
import numpy as np
from atem3d.seepage_channel_validation import channel_delta, strong_signal_mask, summarize_convergence


def test_channel_delta_preserves_sign() -> None:
    background = np.array([[[2.0, -3.0, 4.0]]])
    channel = np.array([[[1.0, -1.0, 8.0]]])
    np.testing.assert_allclose(channel_delta(channel, background), [[[-1.0, 2.0, 4.0]]])


def test_strong_signal_mask_uses_component_peak_fraction() -> None:
    values = np.array([1.0, 0.04, 0.0, -0.2])
    np.testing.assert_array_equal(strong_signal_mask(values, 0.05), [True, False, False, True])


def test_convergence_median_is_reported_without_dropping_raw_values() -> None:
    coarse = np.array([1.0, 2.0, 0.0])
    refined = np.array([1.02, 1.90, 0.01])
    summary = summarize_convergence(coarse, refined, strong_mask=np.array([True, True, False]))
    assert summary["raw_count"] == 3
    assert summary["strong_count"] == 2
    assert summary["median_relative_change"] > 0.0
```

- [ ] **Step 2: Run aggregation tests and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_aggregation.py -v`

Expected: imports fail for new functions.

- [ ] **Step 3: Implement signed delta and documented masks**

Implement ordinary relative error with undefined zero references, a 5%-of-component-peak strong-signal mask, per-receiver/component/window statistics, and convergence summaries. Never overwrite or filter the raw `(5,31,3)` arrays.

- [ ] **Step 4: Write the canonical artifacts**

`aggregate` must write:

```text
benchmark_values.csv
background_errors.csv
channel_delta_values.csv
channel_delta_errors.csv
model_audit.json
convergence_summary.json
benchmark_summary.json
benchmark_results.npz
```

The long CSV key is `(method, case, receiver_id, time_s, component)`. Channel-to-empymod error rows are prohibited. `channel_delta_errors.csv` compares SimPEG delta only to FEniCSx delta and records the strong-signal flag.

- [ ] **Step 5: Run aggregation tests**

Run: `& $PY -m pytest tests/test_seepage_channel_aggregation.py tests/test_seepage_channel_validation.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit aggregation**

```powershell
git add src/atem3d/seepage_channel_validation.py tools/run_seepage_channel_benchmark.py tests/test_seepage_channel_aggregation.py
git commit -m "feat: compare seepage background and anomaly responses"
```

### Task 9: Scientific plots and manifest

**Files:**
- Create: `tools/plot_seepage_channel_benchmark.py`
- Create: `tests/test_seepage_channel_plots_manifest.py`

- [ ] **Step 1: Write failing plot/manifest tests**

```python
from pathlib import Path
from tools.plot_seepage_channel_benchmark import build_output_inventory


def test_inventory_is_not_self_referential(tmp_path: Path) -> None:
    (tmp_path / "benchmark_results.npz").write_bytes(b"data")
    (tmp_path / "benchmark_manifest.json").write_text("stale")
    inventory = build_output_inventory(tmp_path)
    assert set(inventory) == {"benchmark_results.npz"}
    assert inventory["benchmark_results.npz"]["sha256"]


def test_plot_source_contains_required_panels() -> None:
    source = Path("tools/plot_seepage_channel_benchmark.py").read_text(encoding="utf-8")
    for name in ("model_geometry", "background_response", "channel_response", "channel_delta", "convergence"):
        assert name in source
```

- [ ] **Step 2: Run plot tests and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_plots_manifest.py -v`

Expected: module import fails.

- [ ] **Step 3: Implement plots from canonical NPZ/JSON only**

Generate PNG and PDF versions of:

- 3D geometry and x-z/y-z sections with the wire, all five receivers and channel box.
- Three-algorithm background response curves.
- SimPEG/FEniCSx channel response curves.
- Signed and absolute channel delta curves.
- Background error and delta-error curves.
- Spatial/time convergence panels.

Use log time, symlog or separate sign markers for signed responses, explicit SI units, receiver labels and no interpolation across sign changes.

- [ ] **Step 4: Build manifest after all other outputs**

`build_output_inventory` hashes every formal input/output except `benchmark_manifest.json` itself. Store byte size and SHA-256, and include Python/empymod/SimPEG/discretize/FEniCSx/PETSc versions plus the resolved model contract.

- [ ] **Step 5: Run plot tests and a synthetic-data smoke**

Run: `& $PY -m pytest tests/test_seepage_channel_plots_manifest.py -v`

Expected: all tests pass; synthetic smoke creates every required PNG/PDF without warnings about invalid log values.

- [ ] **Step 6: Commit plots and manifest**

```powershell
git add tools/plot_seepage_channel_benchmark.py tests/test_seepage_channel_plots_manifest.py
git commit -m "feat: plot seepage benchmark comparisons"
```

### Task 10: End-to-end orchestrator and controlled formal runs

**Files:**
- Modify: `tools/run_seepage_channel_benchmark.py`
- Create: `tests/test_seepage_channel_orchestrator.py`

- [ ] **Step 1: Write a failing dry-run command test**

```python
from tools.run_seepage_channel_benchmark import build_run_plan


def test_run_plan_contains_no_mirrored_fenicsx_jobs() -> None:
    plan = build_run_plan(output_root="output/seepage_channel_100m_5rx")
    assert [job.name for job in plan] == [
        "empymod_background", "simpeg_background", "simpeg_channel",
        "fenicsx_background", "fenicsx_channel", "aggregate", "plot", "manifest",
    ]
    assert all("mirror" not in " ".join(job.command).lower() for job in plan)
```

- [ ] **Step 2: Run orchestrator test and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_orchestrator.py -v`

Expected: `build_run_plan` is missing.

- [ ] **Step 3: Implement resumable job orchestration**

Each job has a name, command, required inputs and expected outputs. Before reusing an output, validate its shape/provenance/hash contract. Write `run_events.jsonl` with start/end time, exit code, elapsed time and peak-memory evidence. A failed command stops downstream jobs and leaves completed outputs intact.

- [ ] **Step 4: Run all unit tests before formal solving**

Run:

```powershell
& $PY -m pytest tests/test_seepage_channel_model.py tests/test_config_conductivity_boxes.py tests/test_seepage_channel_fenicsx_contract.py tests/test_seepage_channel_fenicsx_mesh_material.py tests/test_seepage_channel_fenicsx_receivers.py tests/test_seepage_channel_fenicsx_command.py tests/test_seepage_channel_validation.py tests/test_seepage_channel_aggregation.py tests/test_seepage_channel_plots_manifest.py tests/test_seepage_channel_orchestrator.py -v
```

Expected: zero failures.

- [ ] **Step 5: Run smoke cases**

Use one early time and a coarse mesh for SimPEG/FEniCSx while retaining all five direct receivers and the channel. Verify both background and channel cases create finite `(5,1,3)` data and non-empty channel cell audits.

- [ ] **Step 6: Run formal background/channel and convergence cases**

Run the orchestrator for the approved 31-time model, then run:

- SimPEG refined channel mesh and finer internal time steps.
- FEniCSx refined channel/receiver mesh and finer internal time steps.

Do not mark a case complete from process exit alone; load every output and run the 465-value, provenance, material-volume and convergence checks.

- [ ] **Step 7: Commit the orchestrator**

```powershell
git add tools/run_seepage_channel_benchmark.py tests/test_seepage_channel_orchestrator.py
git commit -m "feat: orchestrate seepage benchmark runs"
```

### Task 11: Chinese Word report with render verification

**Files:**
- Create: `tools/build_seepage_channel_word_report.py`
- Create: `tests/test_seepage_channel_report.py`
- Generate: `output/doc/100m长导线渗流通道三算法正演对比报告.docx`

- [ ] **Step 1: Write failing report contract tests**

```python
from pathlib import Path
from docx import Document
from tools.build_seepage_channel_word_report import build_report


def test_report_requires_manifest_and_direct_receiver_proof(tmp_path: Path) -> None:
    try:
        build_report(tmp_path, tmp_path / "report.docx")
    except FileNotFoundError as exc:
        assert "benchmark_manifest.json" in str(exc)
    else:
        raise AssertionError("report must fail without formal manifest")


def test_built_report_contains_required_sections(formal_result_dir: Path, tmp_path: Path) -> None:
    path = build_report(formal_result_dir, tmp_path / "report.docx")
    text = "\n".join(p.text for p in Document(path).paragraphs)
    for heading in ("模型与坐标", "渗流通道参数", "FEniCSx 五点直接求解证明", "收敛性", "算法能力边界"):
        assert heading in text
```

- [ ] **Step 2: Run report tests and verify RED**

Run: `& $PY -m pytest tests/test_seepage_channel_report.py -v`

Expected: report module import fails.

- [ ] **Step 3: Implement report generation from manifest-listed files**

Use `python-docx` and the established report style helpers from `tools/build_four_way_100m_5rx_word_report.py`. Read only `benchmark_manifest.json`, `model_audit.json`, `benchmark_summary.json`, `convergence_summary.json` and the manifest-listed figures. Include:

- Physical z-down and solver-internal coordinate tables.
- Exact source, receivers, channel bounds/volume/conductivity and observation times.
- empymod, SimPEG and FEniCSx versions and solver parameters.
- Background, channel, delta, error and convergence figures.
- A five-row provenance table with `explicit_full_domain` for every FEniCSx receiver.
- The explicit statement that empymod is the 1D background reference and cannot represent the finite 3D cuboid.
- Any unmet numerical target as a visible limitation, not as omitted data.

- [ ] **Step 4: Run report unit tests**

Run: `& $PY -m pytest tests/test_seepage_channel_report.py -v`

Expected: all tests pass against formal or fixture artifacts.

- [ ] **Step 5: Generate and render the formal report**

Run:

```powershell
& $PY tools/build_seepage_channel_word_report.py --result-dir output/seepage_channel_100m_5rx --output "output/doc/100m长导线渗流通道三算法正演对比报告.docx"
& $PY "C:\Users\paidaxin\.codex\skills\doc\scripts\render_docx.py" "output/doc/100m长导线渗流通道三算法正演对比报告.docx" --output_dir tmp/docs/seepage_channel_report
```

Inspect every rendered page at 100% zoom. Fix clipped tables, orphan headings, unreadable legends, missing Chinese glyphs, incorrect units and distorted images; rerender after each correction.

- [ ] **Step 6: Commit report code and tests**

```powershell
git add tools/build_seepage_channel_word_report.py tests/test_seepage_channel_report.py
git commit -m "feat: report seepage benchmark results"
```

### Task 12: Final requirement audit and verification

**Files:**
- Modify if evidence requires fixes: files from Tasks 1-11
- Verify: `output/seepage_channel_100m_5rx/benchmark_manifest.json`
- Verify: `output/doc/100m长导线渗流通道三算法正演对比报告.docx`

- [ ] **Step 1: Run the complete automated test suite**

Run:

```powershell
& $PY -m pytest -v
```

Expected: zero failures; pre-existing environment skips are listed explicitly.

- [ ] **Step 2: Run the FEniCSx-focused suite inside WSL**

Run:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解 && /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_seepage_channel_fenicsx_contract.py tests/test_seepage_channel_fenicsx_mesh_material.py tests/test_seepage_channel_fenicsx_receivers.py tests/test_seepage_channel_fenicsx_command.py -v'
```

Expected: zero failures.

- [ ] **Step 3: Audit each explicit requirement against formal evidence**

Confirm from current files, not memory:

```text
[ ] Cuboid is parallel to the wire and exactly 60 x 10 x 10 m.
[ ] Physical center depth is 20 m and bounds are z=15..25 m.
[ ] SimPEG background and channel outputs both contain 465 finite values.
[ ] empymod output is background-only and contains 465 finite values.
[ ] FEniCSx background and channel outputs both contain 465 finite values.
[ ] All five FEniCSx receiver rows say explicit_full_domain.
[ ] No FEniCSx negative-y value is produced by a mirror function.
[ ] Spatial and time convergence summaries are present for both 3D solvers.
[ ] Background, channel and signed-delta comparisons are present.
[ ] Word report records geometry, materials, mesh, time and solver parameters.
[ ] Every Word page has been rendered and visually checked.
[ ] Manifest hashes every formal input/output and excludes itself.
```

- [ ] **Step 4: Verify the formal manifest hashes**

Run a verifier that recalculates SHA-256 and byte size for every manifest entry and exits nonzero on a mismatch. Expected: `verified N/N manifest entries` and exit code 0.

- [ ] **Step 5: Review the final diff without unrelated files**

Run:

```powershell
git status --short
git diff --check
git log --oneline --max-count=15
```

Confirm no unrelated pre-existing modification was staged or committed.

- [ ] **Step 6: Commit only evidence-driven final corrections**

If Task 12 reveals corrections, stage their exact source/test paths and commit:

```powershell
git commit -m "fix: close seepage benchmark verification gaps"
```

If no corrections are needed, do not create an empty commit.
