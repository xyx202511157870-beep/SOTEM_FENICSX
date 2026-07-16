# FEniCSx Magnetic Receiver Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add physically consistent Faraday-loop `dBzdt`, tetrahedral-quadrature Biot–Savart `Hz`, symmetry diagnostics, and convergence gates for the five-receiver full-domain seepage-channel benchmark.

**Architecture:** Keep the existing E-form full-domain time stepper and add receiver operators around it. Pure NumPy geometry, integration, and audit functions live in small modules; `dolfinx/sotem_pipeline.py` only adapts FEniCSx fields and selects the configured output. Every candidate magnetic quantity is generated from the same full-domain field, and no receiver value is mirrored or forced to zero.

**Tech Stack:** Python 3.11, NumPy, SciPy, FEniCSx/DOLFINx, Gmsh, meshio, pytest, matplotlib, python-docx.

---

## Execution setup

The current worktree contains unrelated user changes. Start implementation in an isolated worktree based on the commit containing this plan:

```powershell
git worktree add ..\fenicsx-magnetic-receiver-stability -b codex/fenicsx-magnetic-receiver-stability HEAD
Set-Location ..\fenicsx-magnetic-receiver-stability
git status --short
```

Expected: the new worktree is on `codex/fenicsx-magnetic-receiver-stability` and `git status --short` is empty.

Use the Windows scientific environment for pure Python tests:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest <test-path> -q
```

Use the existing WSL FEniCSx environment for DOLFINx integration tests and forward runs:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest <test-path> -q
```

### Task 1: Add pure symmetry and convergence metrics

**Files:**
- Create: `src/atem3d/magnetic_symmetry_audit.py`
- Create: `tests/test_magnetic_symmetry_audit.py`

- [ ] **Step 1: Write failing tests for odd/even receiver parity**

```python
import numpy as np

from atem3d.magnetic_symmetry_audit import audit_receiver_symmetry


def test_exact_odd_magnetic_and_even_electric_fields_have_zero_residual():
    times = np.array([1.0e-5, 2.0e-5])
    values = np.zeros((5, 2, 3), dtype=float)
    values[:, :, 0] = np.array([[1], [2], [3], [2], [1]])
    values[:, :, 1] = np.array([[-4], [-2], [0], [2], [4]])
    values[:, :, 2] = np.array([[-8], [-3], [0], [3], [8]])

    audit = audit_receiver_symmetry(
        values,
        components=("Ex", "dBzdt", "Hz"),
        times=times,
    )

    assert audit["Ex"]["parity"] == "even"
    assert audit["dBzdt"]["parity"] == "odd"
    assert audit["dBzdt"]["rx3_zero_ratio"] == 0.0
    assert audit["dBzdt"]["pair_24_residual"] == 0.0
    assert audit["Hz"]["pair_15_residual"] == 0.0


def test_zero_denominator_returns_absolute_residual_without_infinite_ratio():
    values = np.zeros((5, 3, 3), dtype=float)
    values[2, :, 1] = 2.5e-12
    audit = audit_receiver_symmetry(values, ("Ex", "dBzdt", "Hz"))
    assert audit["dBzdt"]["rx3_abs_max"] == 2.5e-12
    assert audit["dBzdt"]["rx3_zero_ratio"] is None
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_symmetry_audit.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'atem3d.magnetic_symmetry_audit'`.

- [ ] **Step 3: Implement the minimal audit API**

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _normalised_residual(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float | None, float]:
    abs_numerator = float(np.max(np.abs(numerator)))
    scale = float(np.max(np.abs(denominator)))
    return (None if scale == 0.0 else abs_numerator / scale), abs_numerator


def audit_receiver_symmetry(
    values: np.ndarray,
    components: Sequence[str],
    times: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    data = np.asarray(values, dtype=float)
    names = tuple(str(name) for name in components)
    if data.ndim != 3 or data.shape[0] != 5 or data.shape[2] != len(names):
        raise ValueError("values must have shape (5, n_times, n_components)")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    if times is not None and np.asarray(times).shape != (data.shape[1],):
        raise ValueError("times must contain one value per time sample")

    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        field = data[:, :, index]
        odd = name in {"dBzdt", "Hz"}
        pair_24 = field[1] + field[3] if odd else field[1] - field[3]
        pair_15 = field[0] + field[4] if odd else field[0] - field[4]
        flank_24 = np.maximum(np.abs(field[1]), np.abs(field[3]))
        flank_15 = np.maximum(np.abs(field[0]), np.abs(field[4]))
        zero_ratio, zero_abs = _normalised_residual(field[2], flank_24)
        residual_24, absolute_24 = _normalised_residual(pair_24, flank_24)
        residual_15, absolute_15 = _normalised_residual(pair_15, flank_15)
        result[name] = {
            "parity": "odd" if odd else "even",
            "rx3_zero_ratio": zero_ratio,
            "rx3_abs_max": zero_abs,
            "pair_24_residual": residual_24,
            "pair_24_abs_max": absolute_24,
            "pair_15_residual": residual_15,
            "pair_15_abs_max": absolute_15,
        }
    return result
```

- [ ] **Step 4: Add tests for background/channel/delta and acceptance gates**

Add these tests for `audit_model_triplet(background, channel, components)` and `evaluate_magnetic_gates(...)`:

```python
def test_triplet_audits_signed_channel_minus_background():
    background = np.zeros((5, 2, 3))
    channel = background.copy()
    channel[:, :, 1] = np.array([[-2], [-1], [0], [1], [2]])
    audit = audit_model_triplet(background, channel, ("Ex", "dBzdt", "Hz"))
    assert audit["delta"]["dBzdt"]["rx3_zero_ratio"] == 0.0
    assert audit["delta"]["dBzdt"]["pair_24_residual"] == 0.0


def test_missing_scale_fails_magnetic_gate():
    metrics = {
        "dBzdt": {"rx3_zero_ratio": None, "pair_24_residual": 0.0, "pair_15_residual": 0.0},
        "Hz": {"rx3_zero_ratio": 0.0, "pair_24_residual": 0.0, "pair_15_residual": 0.0},
    }
    result = evaluate_magnetic_gates(metrics, threshold=0.01)
    assert result["passed"] is False
    assert "dBzdt.rx3_zero_ratio" in result["failures"]
```

The remaining assertions must prove that:

- delta is calculated as `channel - background`;
- `None` ratios fail the gate instead of being treated as zero;
- `rx3_zero_ratio <= 0.01` and both pair residuals `<= 0.01` pass;
- `Ex` is audited as even but is not subjected to the magnetic zero gate.

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_symmetry_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the audit module**

```powershell
git add src/atem3d/magnetic_symmetry_audit.py tests/test_magnetic_symmetry_audit.py
git commit -m "feat: add magnetic receiver symmetry audit"
```

### Task 2: Add Faraday-loop geometry and pure integration

**Files:**
- Create: `dolfinx/magnetic_receiver_operators.py`
- Create: `tests/test_magnetic_receiver_operators.py`

- [ ] **Step 1: Write failing tests for loop geometry**

```python
import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path("dolfinx/magnetic_receiver_operators.py")
    spec = importlib.util.spec_from_file_location("magnetic_receiver_operators", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_horizontal_loop_points_and_tangents_are_pairwise_symmetric():
    module = load_module()
    rule = module.horizontal_loop_rule((1.0, 2.0, 3.0), radius=2.0, point_count=32)
    np.testing.assert_allclose(rule.points[:16, :2] + rule.points[16:, :2], [[2.0, 4.0]], atol=1e-14)
    np.testing.assert_allclose(rule.tangents[:16] + rule.tangents[16:], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(rule.line_weights), 4.0 * np.pi)


def test_constant_electric_field_has_zero_closed_loop_integral():
    module = load_module()
    rule = module.horizontal_loop_rule((0.0, 0.0, 0.1), radius=2.0, point_count=32)
    electric = np.repeat([[3.0, -4.0, 0.0]], 32, axis=0)
    value = module.faraday_loop_dbdt(electric, rule)
    assert abs(value) < 1.0e-14


def test_linear_field_matches_known_vertical_curl():
    module = load_module()
    curl_z = 7.5
    rule = module.horizontal_loop_rule((0.0, 0.0, 0.1), radius=2.0, point_count=64)
    x, y = rule.points[:, 0], rule.points[:, 1]
    electric = np.column_stack((-0.5 * curl_z * y, 0.5 * curl_z * x, np.zeros_like(x)))
    np.testing.assert_allclose(module.faraday_loop_dbdt(electric, rule), -curl_z, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run the tests and verify the missing-file failure**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_receiver_operators.py -q
```

Expected: loading `dolfinx/magnetic_receiver_operators.py` fails because the file does not exist.

- [ ] **Step 3: Implement the loop rule and Faraday integral**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizontalLoopRule:
    center: np.ndarray
    radius: float
    points: np.ndarray
    tangents: np.ndarray
    line_weights: np.ndarray
    area: float


def horizontal_loop_rule(center, *, radius: float, point_count: int) -> HorizontalLoopRule:
    center_array = np.asarray(center, dtype=float).reshape(3)
    radius = float(radius)
    point_count = int(point_count)
    if radius <= 0.0:
        raise ValueError("faraday loop radius must be positive")
    if point_count < 8 or point_count % 4:
        raise ValueError("faraday loop point count must be a multiple of 4 and at least 8")
    theta = 2.0 * np.pi * np.arange(point_count, dtype=float) / point_count
    offsets = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(point_count)))
    tangents = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(point_count)))
    return HorizontalLoopRule(
        center=center_array,
        radius=radius,
        points=center_array + offsets,
        tangents=tangents,
        line_weights=np.full(point_count, 2.0 * np.pi * radius / point_count),
        area=np.pi * radius**2,
    )


def faraday_loop_dbdt(electric_values, rule: HorizontalLoopRule) -> float:
    electric = np.asarray(electric_values, dtype=float)
    if electric.shape != rule.points.shape or not np.all(np.isfinite(electric)):
        raise ValueError("electric values must be finite and match the loop points")
    circulation = np.sum(rule.line_weights * np.einsum("ij,ij->i", electric, rule.tangents))
    return -float(circulation) / rule.area
```

- [ ] **Step 4: Add validation tests for radius and point count**

Test `radius <= 0`, point counts `4`, `10`, and non-finite electric samples. Each must raise `ValueError` with the corresponding explicit message.

- [ ] **Step 5: Run the pure operator tests**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_receiver_operators.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the pure Faraday operator**

```powershell
git add dolfinx/magnetic_receiver_operators.py tests/test_magnetic_receiver_operators.py
git commit -m "feat: add Faraday loop receiver operator"
```

### Task 3: Add stable Biot–Savart volume integration

**Files:**
- Modify: `dolfinx/magnetic_receiver_operators.py`
- Modify: `tests/test_magnetic_receiver_operators.py`

- [ ] **Step 1: Write failing tests for tetrahedral quadrature and compensated summation**

```python
def test_tetra4_rule_integrates_constant_and_linear_coordinates():
    module = load_module()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    points, weights = module.tetra4_rule(vertices)
    np.testing.assert_allclose(weights.sum(), 1.0 / 6.0)
    np.testing.assert_allclose(np.sum(weights[:, None] * points, axis=0), [1 / 24, 1 / 24, 1 / 24])


def test_neumaier_vector_sum_retains_small_residual_after_cancellation():
    module = load_module()
    values = np.array([[1.0e16, 0, 0], [1.0, 0, 0], [-1.0e16, 0, 0]])
    np.testing.assert_allclose(module.neumaier_vector_sum(values), [1.0, 0.0, 0.0])


def test_biot_volume_integral_has_expected_sign_for_x_current_above_receiver():
    module = load_module()
    points = np.array([[0.0, 1.0, 0.0]])
    currents = np.array([[1.0, 0.0, 0.0]])
    result, audit = module.biot_savart_volume_h(
        receiver=(0.0, 0.0, 0.0),
        points=points,
        current_density=currents,
        weights=np.array([2.0]),
    )
    assert result[2] < 0.0
    assert audit["sample_count"] == 1
    assert audit["cancellation_ratio"][2] == 1.0
```

- [ ] **Step 2: Run the new tests and verify missing attributes**

Expected: tests fail because `tetra4_rule`, `neumaier_vector_sum`, and `biot_savart_volume_h` are absent.

- [ ] **Step 3: Implement tetra4, stable summation, and Biot integration**

```python
def tetra4_rule(vertices):
    xyz = np.asarray(vertices, dtype=float)
    if xyz.shape != (4, 3):
        raise ValueError("tetrahedron vertices must have shape (4, 3)")
    determinant = np.linalg.det(np.column_stack((xyz[1] - xyz[0], xyz[2] - xyz[0], xyz[3] - xyz[0])))
    volume = abs(float(determinant)) / 6.0
    alpha, beta = 0.5854101966249685, 0.1381966011250105
    barycentric = np.array([
        [alpha, beta, beta, beta], [beta, alpha, beta, beta],
        [beta, beta, alpha, beta], [beta, beta, beta, alpha],
    ])
    return barycentric @ xyz, np.full(4, volume / 4.0)


def neumaier_vector_sum(values):
    total = np.zeros(3, dtype=float)
    correction = np.zeros(3, dtype=float)
    for value in np.asarray(values, dtype=float).reshape(-1, 3):
        updated = total + value
        correction += np.where(
            np.abs(total) >= np.abs(value),
            (total - updated) + value,
            (value - updated) + total,
        )
        total = updated
    return total + correction


def biot_savart_volume_h(*, receiver, points, current_density, weights):
    receiver = np.asarray(receiver, dtype=float).reshape(3)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    current = np.asarray(current_density, dtype=float).reshape(-1, 3)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if current.shape != points.shape or weights.shape != (points.shape[0],):
        raise ValueError("Biot samples, currents, and weights must have matching lengths")
    displacement = receiver[None, :] - points
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("receiver coincides with a Biot integration point")
    contributions = weights[:, None] * np.cross(current, displacement) / (
        4.0 * np.pi * distance[:, None] ** 3
    )
    result = neumaier_vector_sum(contributions)
    absolute_sum = np.sum(np.abs(contributions), axis=0)
    ratio = np.divide(np.abs(result), absolute_sum, out=np.ones(3), where=absolute_sum > 0.0)
    return result, {
        "sample_count": int(points.shape[0]),
        "absolute_contribution_sum": absolute_sum.tolist(),
        "cancellation_ratio": ratio.tolist(),
    }
```

- [ ] **Step 4: Run the operator tests**

Expected: all pure tests pass.

- [ ] **Step 5: Commit stable Biot integration**

```powershell
git add dolfinx/magnetic_receiver_operators.py tests/test_magnetic_receiver_operators.py
git commit -m "feat: add stable tetrahedral Biot integration"
```

### Task 4: Adapt FEniCSx fields to the new receiver operators

**Files:**
- Modify: `dolfinx/magnetic_receiver_operators.py`
- Create: `tests/test_dolfinx_magnetic_receiver_adapters.py`

- [ ] **Step 1: Write a failing test for fail-closed loop point location**

Use lightweight fakes so the test runs without DOLFINx:

```python
def test_evaluate_faraday_loop_rejects_any_unlocated_point():
    module = load_module()

    class Field:
        def eval(self, points, cells):
            return np.zeros((len(cells), 3))

    def locator(_mesh, point):
        return [] if point[0] > 0.0 else [0]

    with pytest.raises(RuntimeError, match="Faraday loop point"):
        module.evaluate_faraday_loop_field(
            Field(), object(), center=(0, 0, 0.1), radius=2.0,
            point_count=32, find_cells=locator,
        )
```

- [ ] **Step 2: Implement `evaluate_faraday_loop_field`**

The adapter must locate every loop point, choose the smallest deterministic candidate cell index when a point lies on a shared facet, call `E.eval`, and return both `dBzdt` and this audit dictionary:

```python
{
    "method": "faraday_loop",
    "radius_m": 2.0,
    "point_count": 32,
    "located_point_count": 32,
    "missing_point_indices": [],
}
```

It must raise before field evaluation if any point is missing.

- [ ] **Step 3: Write and implement a tetra4 DOLFINx adapter test**

Create a unit-cube DOLFINx mesh, interpolate a constant x-directed current density, evaluate with the existing `_cell_biot_quadrature_points_weights` data contract, and assert finite `H` with the expected sign. The adapter signature is:

```python
evaluate_biot_current_field(
    current_density,
    *,
    receiver,
    points,
    cells,
    weights,
) -> tuple[np.ndarray, dict[str, object]]
```

The focused integration test body is:

```python
msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
spaces = sp.build_function_spaces(msh, sp.PipelineConfig())
current = fem.Function(spaces["V"])
current.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
points, cells, weights = sp._cell_biot_quadrature_points_weights(msh)
h, audit = module.evaluate_biot_current_field(
    current,
    receiver=(0.5, 1.5, 0.5),
    points=points,
    cells=cells,
    weights=weights,
)
assert h[2] > 0.0
assert audit["sample_count"] == len(points)
```

- [ ] **Step 4: Run pure tests on Windows and DOLFINx tests in WSL**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_receiver_operators.py tests/test_dolfinx_magnetic_receiver_adapters.py -q
```

Expected on Windows: pure tests pass; DOLFINx test is skipped by `pytest.importorskip`.

Run in WSL:

```bash
/home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest tests/test_dolfinx_magnetic_receiver_adapters.py -q
```

Expected: all DOLFINx adapter tests pass.

- [ ] **Step 5: Commit the adapters**

```powershell
git add dolfinx/magnetic_receiver_operators.py tests/test_dolfinx_magnetic_receiver_adapters.py
git commit -m "feat: adapt FEniCSx fields to stable magnetic receivers"
```

### Task 5: Add configuration and CLI contracts without changing defaults

**Files:**
- Modify: `dolfinx/sotem_pipeline.py`
- Modify: `tests/test_dolfinx_model_consistency.py`
- Create: `tests/test_magnetic_receiver_cli.py`

- [ ] **Step 1: Write failing configuration validation tests**

```python
def test_faraday_loop_and_tetra4_configuration_is_accepted():
    sp = load_pipeline_module()
    config = sp.PipelineConfig(
        magnetic_receiver_mode="biot_current",
        magnetic_dbdt_mode="faraday_loop",
        biot_current_integration="tetra4",
        faraday_loop_radius=2.0,
        faraday_loop_quadrature_points=32,
        magnetic_diagnostic_methods=(
            "curl", "biot_rate", "faraday_loop", "biot_center", "biot_tetra4"
        ),
    )
    diagnostics = sp.validate_model_consistency(config)
    assert diagnostics["magnetic_dbdt_mode"] == "faraday_loop"
    assert diagnostics["biot_current_integration"] == "tetra4"


@pytest.mark.parametrize("points", [4, 10, 30])
def test_faraday_loop_rejects_invalid_point_count(points):
    sp = load_pipeline_module()
    with pytest.raises(ValueError, match="multiple of 4"):
        sp.validate_model_consistency(sp.PipelineConfig(
            magnetic_dbdt_mode="faraday_loop",
            faraday_loop_quadrature_points=points,
        ))
```

- [ ] **Step 2: Run the tests and verify constructor/validation failures**

Expected: `PipelineConfig` rejects unknown fields or validation rejects `faraday_loop`.

- [ ] **Step 3: Add fields and parser choices**

Add these dataclass fields:

```python
biot_current_integration: str = "cell_center"
faraday_loop_radius: float = 2.0
faraday_loop_quadrature_points: int = 32
magnetic_diagnostic_methods: tuple[str, ...] = ()
```

Extend model consistency checks:

```python
if magnetic_dbdt_mode not in {"curl", "biot_rate", "ampere_rate", "faraday_loop"}:
    raise ValueError("magnetic_dbdt_mode must be 'curl', 'biot_rate', 'ampere_rate', or 'faraday_loop'")
if config.biot_current_integration not in {"cell_center", "tetra4"}:
    raise ValueError("biot_current_integration must be 'cell_center' or 'tetra4'")
if config.faraday_loop_radius <= 0.0:
    raise ValueError("faraday_loop_radius must be positive")
if config.faraday_loop_quadrature_points < 8 or config.faraday_loop_quadrature_points % 4:
    raise ValueError("faraday_loop_quadrature_points must be a multiple of 4 and at least 8")
allowed_diagnostics = {"curl", "biot_rate", "faraday_loop", "biot_center", "biot_tetra4"}
unknown_diagnostics = set(config.magnetic_diagnostic_methods) - allowed_diagnostics
if unknown_diagnostics:
    raise ValueError(f"unknown magnetic diagnostic methods: {sorted(unknown_diagnostics)}")
```

Add CLI arguments and pass them into `PipelineConfig`:

```python
parser.add_argument("--biot-current-integration", choices=["cell_center", "tetra4"], default="cell_center")
parser.add_argument("--faraday-loop-radius", type=float, default=2.0)
parser.add_argument("--faraday-loop-quadrature-points", type=int, default=32)
parser.add_argument(
    "--magnetic-diagnostic-methods",
    default="",
    help="Comma-separated diagnostic methods evaluated from the same E trajectory",
)
```

Implement `_parse_magnetic_diagnostic_methods(value: str) -> tuple[str, ...]` to de-duplicate names while preserving input order. An empty string becomes `()`; whitespace-only entries fail. Pass its return value into the `PipelineConfig(...)` construction inside `main`.

- [ ] **Step 4: Add CLI regression assertions**

Verify defaults and the dedicated parser helper with these assertions:

```python
defaults = sp.PipelineConfig()
assert defaults.magnetic_dbdt_mode == "curl"
assert defaults.biot_current_integration == "cell_center"
assert defaults.magnetic_diagnostic_methods == ()

parsed = sp._parse_magnetic_diagnostic_methods(
    "curl,biot_rate,faraday_loop,biot_center,biot_tetra4"
)
assert parsed == (
    "curl", "biot_rate", "faraday_loop", "biot_center", "biot_tetra4"
)
```

- [ ] **Step 5: Run focused configuration tests**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_dolfinx_model_consistency.py tests/test_magnetic_receiver_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the configuration contract**

```powershell
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py tests/test_magnetic_receiver_cli.py
git commit -m "feat: configure stable magnetic receiver modes"
```

### Task 6: Evaluate all magnetic methods from the same five-receiver field

**Files:**
- Modify: `dolfinx/sotem_pipeline.py`
- Modify: `dolfinx/seepage_channel_full_domain.py`
- Modify: `tests/test_dolfinx_biot_receiver.py`
- Modify: `tests/test_seepage_channel_fenicsx_receivers.py`

- [ ] **Step 1: Write a failing selector test**

Add a pure selector to `sotem_pipeline.py` and test it before implementation:

```python
def test_selected_magnetic_outputs_do_not_modify_electric_component():
    sp = _load_pipeline_module()
    methods = {
        "Ex": 3.0,
        "dBzdt_curl": 1.0,
        "dBzdt_biot_rate": 2.0,
        "dBzdt_faraday_loop": 4.0,
        "Hz_biot_center": 5.0,
        "Hz_biot_tetra4": 6.0,
    }
    selected = sp._select_magnetic_receiver_outputs(
        methods,
        magnetic_dbdt_mode="faraday_loop",
        biot_current_integration="tetra4",
    )
    assert selected == {"Ex": 3.0, "dBzdt": 4.0, "Hz": 6.0}
```

- [ ] **Step 2: Implement the selector with explicit key checks**

Map `curl`, `biot_rate`, and `faraday_loop` to their named diagnostic keys, and `cell_center`/`tetra4` to the two `Hz` keys. Raise `KeyError` naming the missing diagnostic instead of returning NaN.

- [ ] **Step 3: Add a failing five-call diagnostic test**

Extend the full-domain receiver test with a `magnetic_evaluator` fake and assert that each of the five receiver coordinates is passed independently and every record contains:

```python
{
    "dBzdt_curl",
    "dBzdt_biot_rate",
    "dBzdt_faraday_loop",
    "Hz_biot_center",
    "Hz_biot_tetra4",
    "provenance",
}
```

- [ ] **Step 4: Implement one explicit receiver-method evaluator**

Add this function to `sotem_pipeline.py`:

```python
def evaluate_magnetic_receiver_methods(
    E,
    dbdt,
    msh,
    materials,
    receiver_config,
    *,
    source_current: float,
    config: PipelineConfig,
    h_new=None,
    h_old=None,
    dt: float | None = None,
) -> dict[str, float | dict[str, object]]:
    curl_record = evaluate_receivers(E, dbdt, msh, receiver_config)
    faraday_value, faraday_audit = evaluate_faraday_loop_field(
        E,
        msh,
        center=receiver_config.receiver,
        radius=config.faraday_loop_radius,
        point_count=config.faraday_loop_quadrature_points,
        find_cells=_find_cells_for_point,
    )
    center_h = _biot_savart_total_h_at_receiver(
        E, msh, materials, receiver_config, source_current,
        integration="cell_center",
    )
    tetra_h, tetra_audit = _biot_savart_total_h_at_receiver(
        E, msh, materials, receiver_config, source_current,
        integration="tetra4",
        return_audit=True,
    )
    result = {
        "dBzdt_curl": float(curl_record["dBzdt"]),
        "dBzdt_faraday_loop": float(faraday_value),
        "Hz_biot_center": float(center_h[2]),
        "Hz_biot_tetra4": float(tetra_h[2]),
        "faraday_audit": faraday_audit,
        "biot_tetra4_audit": tetra_audit,
    }
    if h_new is not None and h_old is not None and dt is not None:
        result["dBzdt_biot_rate"] = float(
            _biot_receiver_dbdt_from_h(h_new, h_old, dt=dt)[2]
        )
    return result
```

Extend `_biot_savart_total_h_at_receiver` with the explicit `integration` and `return_audit` keywords while keeping its historical call behavior unchanged. For `tetra4`, evaluate `E` and any Debye memory functions at the four quadrature points of each owning cell, index the cellwise conductivity with the owning-cell array, form `J`, call `biot_savart_volume_h`, then add the finite-wire contribution. The `cell_center` branch must continue using the current implementation.

- [ ] **Step 5: Make Biot history depend on outputs and requested diagnostics**

Change the history predicate to:

```python
def _biot_h_required_for_step(
    magnetic_dbdt_mode: str,
    *,
    is_output: bool,
    diagnostic_methods: tuple[str, ...] = (),
) -> bool:
    return (
        bool(is_output)
        or str(magnetic_dbdt_mode).strip().lower() == "biot_rate"
        or "biot_rate" in diagnostic_methods
    )
```

When `biot_rate` is requested only as a diagnostic, initialize and update `H_old_receiver` exactly as for the formal `biot_rate` mode, but keep formal `dBzdt` selected by `magnetic_dbdt_mode`. Add a regression test proving that `faraday_loop` plus diagnostic `biot_rate` requests Biot H on non-output internal steps.

- [ ] **Step 6: Integrate diagnostics into the output-time path**

At each formal output, calculate `dbdt = compute_dbdt(E_new, spaces)` once, then for each receiver:

```python
methods = evaluate_magnetic_receiver_methods(
    E_new,
    dbdt,
    msh,
    materials,
    receiver_config,
    source_current=_source_current(float(t), config),
    config=config,
    h_new=H_new_receiver[index] if H_new_receiver is not None else None,
    h_old=H_old_receiver[index] if H_old_receiver is not None else None,
    dt=dt,
)
selected = _select_magnetic_receiver_outputs(
    {"Ex": receiver_record["Ex"], **methods},
    magnetic_dbdt_mode=config.magnetic_dbdt_mode,
    biot_current_integration=config.biot_current_integration,
)
receiver_record.update(methods)
receiver_record.update(selected)
```

Do not calculate a second electric solve, and do not populate any receiver from its opposite-y partner.

- [ ] **Step 7: Preserve the formal array and extend diagnostic CSV output**

Keep `records_to_array(records, ("Ex", "dBzdt", "Hz"))` unchanged. Add `write_magnetic_receiver_diagnostics_csv` with one row per receiver and observation time and explicit method/audit columns.

- [ ] **Step 8: Run focused tests**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_dolfinx_biot_receiver.py tests/test_seepage_channel_fenicsx_receivers.py -q
```

Expected: all pure tests pass. Run the same files in WSL to exercise DOLFINx paths.

- [ ] **Step 9: Commit five-receiver integration**

```powershell
git add dolfinx/sotem_pipeline.py dolfinx/seepage_channel_full_domain.py tests/test_dolfinx_biot_receiver.py tests/test_seepage_channel_fenicsx_receivers.py
git commit -m "feat: record five-receiver magnetic diagnostics"
```

### Task 7: Add full-domain mesh symmetry auditing and mirrored-topology construction

**Files:**
- Create: `dolfinx/symmetric_full_domain_mesh.py`
- Create: `tools/build_symmetric_fenicsx_mesh.py`
- Create: `tests/test_symmetric_full_domain_mesh.py`
- Modify: `dolfinx/sotem_pipeline.py`

- [ ] **Step 1: Write failing pure topology tests**

Build a two-tetra positive-y fixture with plane nodes at `y=0`. Test that `mirror_half_topology(...)`:

- reuses plane nodes;
- adds one negative-y node for every positive-y node;
- duplicates tetrahedra and cell tags;
- reverses one mirrored vertex pair so signed volumes stay positive;
- removes `y=0` half-domain boundary triangles from the external-facet output;
- duplicates all other physical facet tags;
- retains source-wire line elements on `y=0` exactly once with their physical tag.

Use this public signature:

```python
mirror_half_topology(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    tetra_tags: np.ndarray,
    triangles: np.ndarray,
    triangle_tags: np.ndarray,
    lines: np.ndarray,
    line_tags: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, np.ndarray]
```

- [ ] **Step 2: Implement the minimal topology mirror**

Classify nodes with `abs(y) <= tolerance` as plane nodes. Mirror only nodes with `y > tolerance`. For each mirrored tetrahedron, map node indices and swap local vertices 1 and 2. Mirror off-plane line elements, retain plane line elements once, and preserve all line physical tags. Reject any input node with `y < -tolerance` because the input must be a positive-half mesh.

- [ ] **Step 3: Add geometry/tag audits**

Implement `audit_y_reflection(points, tetrahedra, tetra_tags, tolerance)` returning exact node-pair fraction, exact centroid-pair fraction, maximum reflection distance, tag mismatch count, original/mirrored volume, and pass/fail. Tests must prove that a missing mirrored tetrahedron or mismatched tag fails.

- [ ] **Step 4: Add half-domain mesh generation mode**

Refactor only the y bounds in `generate_verification_mesh` behind a new internal helper:

```python
def _mesh_y_bounds(config):
    if config.mesh_symmetry_mode == "positive_half_source":
        return 0.0, float(config.y_extent)
    return -float(config.y_extent), float(config.y_extent)
```

Add `mesh_symmetry_mode` choices `ordinary` and `positive_half_source`, defaulting to `ordinary`. The positive-half file is an intermediate mesh only and must never be passed to the solver.

- [ ] **Step 5: Write the conversion tool**

`tools/build_symmetric_fenicsx_mesh.py` must:

1. run Gmsh in `positive_half_source` mode;
2. read points/cells/physical tags with meshio;
3. call `mirror_half_topology`;
4. write a complete full-domain `.msh`;
5. run `audit_y_reflection`;
6. verify that the source physical line is present exactly once on `y=0`;
7. refuse to emit `PASS` unless node, tetrahedron, material-tag, facet-tag, and source-line checks all pass.

The generated metadata must include `solver_domain="full"` and `field_mirroring=false`.

The conversion entry point must follow this fail-closed structure:

```python
def build_symmetric_mesh(half_path: Path, output_path: Path, audit_path: Path) -> Path:
    mesh = meshio.read(half_path)
    mirrored = mirror_half_topology_from_meshio(mesh)
    audit = audit_y_reflection(
        mirrored["points"], mirrored["tetrahedra"], mirrored["tetra_tags"], 1.0e-10
    )
    if not audit["passed"]:
        raise RuntimeError(f"symmetric full-domain mesh audit failed: {audit}")
    if mirrored["source_line_count"] != 1:
        raise RuntimeError("symmetric full-domain mesh must contain one source physical line")
    write_meshio_full_domain(output_path, mirrored)
    audit_path.write_text(json.dumps({
        **audit, "solver_domain": "full", "field_mirroring": False
    }, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
```

- [ ] **Step 6: Run pure topology tests and a small Gmsh smoke test**

Run the pure tests on Windows. Run a reduced `x_extent=20`, `y_extent=20`, `air_height=10`, `earth_depth=20` build in WSL and load it through `dolfinx.io.gmshio.read_from_msh`. Expected: both positive and negative y tetrahedra exist, all cell tags load, and the symmetry audit passes.

- [ ] **Step 7: Commit symmetric full-domain mesh support**

```powershell
git add dolfinx/symmetric_full_domain_mesh.py tools/build_symmetric_fenicsx_mesh.py tests/test_symmetric_full_domain_mesh.py dolfinx/sotem_pipeline.py
git commit -m "feat: build auditable symmetric full-domain meshes"
```

### Task 8: Add the magnetic receiver audit runner and artifacts

**Files:**
- Create: `tools/run_fenicsx_magnetic_receiver_audit.py`
- Create: `tests/test_magnetic_receiver_audit_runner.py`

- [ ] **Step 1: Write failing tests for method-table aggregation**

Use synthetic `(5, 4, 3)` arrays and diagnostic rows. Assert that `build_method_summary(...)` creates separate background, channel, and delta metrics for:

- `curl`;
- `biot_rate`;
- `faraday_loop_16`, `faraday_loop_32`, `faraday_loop_64`;
- `biot_center`;
- `biot_tetra4`.

Assert that relative error is `None` for Rx3 magnetic values below the configured signal floor.

- [ ] **Step 2: Implement deterministic readers and summary writers**

The runner must read only explicit input paths, validate times/receivers/components, call `audit_model_triplet`, and write:

- `magnetic_receiver_methods.csv`;
- `magnetic_symmetry_metrics.json`;
- `magnetic_convergence_summary.json`;
- `mesh_symmetry_audit.json`;
- `run_manifest.json` with SHA-256 and byte counts.

Use a single validated payload shape and stable JSON conversion:

```python
def load_method_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: np.asarray(stored[name]) for name in stored.files}
    validate_method_payload(payload)
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
```

- [ ] **Step 3: Add gate selection logic**

Implement `select_formal_method(summary)` so a method passes only when:

```python
rx3_zero_ratio <= 0.01
pair_24_residual <= 0.01
pair_15_residual <= 0.01
strong_signal_error_increase_percentage_points <= 2.0
ex_median_change <= 0.005
```

Return a structured failure list; never choose the lowest Rx3 residual alone.

```python
def select_formal_method(summary):
    accepted = []
    rejected = {}
    for name, metrics in summary.items():
        failures = magnetic_method_failures(metrics)
        if failures:
            rejected[name] = failures
        else:
            accepted.append(name)
    if not accepted:
        return {"passed": False, "selected": None, "rejected": rejected}
    ranking = sorted(accepted, key=lambda name: (
        summary[name]["strong_signal_median_error"],
        summary[name]["rx3_zero_ratio"],
        name,
    ))
    return {"passed": True, "selected": ranking[0], "rejected": rejected}
```

- [ ] **Step 4: Add CLI dry-run and fail-closed tests**

Test missing files, shape mismatch, non-finite values, absent provenance, and an empty method set using:

```python
@pytest.mark.parametrize("case", ["missing", "shape", "nan", "provenance", "empty"])
def test_invalid_audit_inputs_never_write_passing_manifest(tmp_path, case):
    args = build_invalid_case(tmp_path, case)
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        run_audit(args)
    manifest = tmp_path / "run_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8"))["passed"] is False
```

- [ ] **Step 5: Run runner tests and commit**

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_receiver_audit_runner.py -q
git add tools/run_fenicsx_magnetic_receiver_audit.py tests/test_magnetic_receiver_audit_runner.py
git commit -m "feat: audit magnetic receiver stability"
```

### Task 9: Add plots and Word report sections

**Files:**
- Modify: `tools/plot_seepage_channel_benchmark.py`
- Modify: `tools/build_seepage_channel_word_report.py`
- Modify: `tests/test_seepage_channel_plots_manifest.py`
- Modify: `tests/test_seepage_channel_word_report.py`

- [ ] **Step 1: Write failing plot inventory assertions**

Require these output stems:

```python
for stem in (
    "magnetic_receiver_comparison",
    "rx3_absolute_residual",
    "magnetic_symmetry_convergence",
):
    assert stem in source
```

Require Rx3 magnetic panels to use absolute residual and a horizontal acceptance line at `R0=0.01`; forbid ordinary percent-relative-error labeling for Rx3.

- [ ] **Step 2: Implement three focused figures**

- `magnetic_receiver_comparison.png`: method overlays for all nonzero receivers.
- `rx3_absolute_residual.png`: signed/absolute Rx3 values with the numerical floor.
- `magnetic_symmetry_convergence.png`: `R0`, `Rodd24`, and `Rodd15` versus mesh/time/operator resolution.

All figures must retain original signed data and annotate the selected method.

Implement and call these exact entry points:

```python
plot_magnetic_receiver_comparison(method_table, output_dir / "magnetic_receiver_comparison.png")
plot_rx3_absolute_residual(method_table, metrics, output_dir / "rx3_absolute_residual.png")
plot_magnetic_symmetry_convergence(convergence, output_dir / "magnetic_symmetry_convergence.png")
```

Each function must create its own `Figure`, call `tight_layout()`, save at `dpi=200`, and close the figure in a `finally` block.

- [ ] **Step 3: Write failing Word content tests**

Require the rendered document text to contain:

- `Faraday 有限线圈`;
- `四面体四点 Biot-Savart`;
- `Rx3 对称理论零点`;
- `explicit_full_domain`;
- `不使用单侧求解后对称镜像`;
- the loop radius, point count, Biot integration mode, symmetry metrics, and pass/fail result.

- [ ] **Step 4: Extend the report builder**

Add sections for root cause, receiver operators, experiment matrix, symmetry audit, convergence, limitations, and hashes. Read values only from the new JSON/CSV artifacts. Do not hard-code a passing conclusion.

- [ ] **Step 5: Run document tests and render the report**

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_seepage_channel_plots_manifest.py tests/test_seepage_channel_word_report.py -q
```

After a report exists, use the bundled `render_docx.py` workflow to render every page to PNG and inspect for clipped tables, missing Chinese glyphs, unreadable plots, and bad page breaks.

- [ ] **Step 6: Commit plots and reporting**

```powershell
git add tools/plot_seepage_channel_benchmark.py tools/build_seepage_channel_word_report.py tests/test_seepage_channel_plots_manifest.py tests/test_seepage_channel_word_report.py
git commit -m "docs: report magnetic receiver stability"
```

### Task 10: Run staged diagnostics and choose the formal method

**Files:**
- Create: `tools/run_fenicsx_magnetic_background_short.sh`
- Create: `tools/run_fenicsx_magnetic_channel_full.sh`
- Modify: `tests/test_seepage_channel_fenicsx_command.py`
- Output only: `output/seepage_channel_100m_5rx_60x1x1/magnetic_receiver_stability/`

- [ ] **Step 1: Add command-contract tests before scripts**

Require both scripts to contain all five explicit receiver locations, `--magnetic-dbdt-mode faraday_loop`, `--biot-current-integration tetra4`, `--faraday-loop-radius 2`, `--faraday-loop-quadrature-points 32`, and `--magnetic-diagnostic-methods curl,biot_rate,faraday_loop,biot_center,biot_tetra4`. Forbid `y10_baseline` and any receiver-copy operation in solver commands. Permit the word `mirror` only in the separate full-domain mesh-builder invocation and metadata audit, never in receiver value generation.

- [ ] **Step 2: Implement the short background command**

Clone the approved 60 × 1 × 1 m FEniCSx parameters, use a distinct work directory, and add `--stop-after-outputs 10`. Preserve the current source, conductivity, time stepping, KSP, full-domain receiver set, and checkpoint settings.

- [ ] **Step 3: Run the current-mesh short diagnostic**

Run one E-field trajectory that writes all magnetic methods. Verify:

- 5 receivers × 10 times;
- all loop points found;
- all outputs finite;
- diagnostics contain every method;
- no pass/fail claim before the audit runner finishes.

- [ ] **Step 4: Run one-variable convergence cases**

Run these cases separately:

1. receiver mesh `2.5 m`, internal fraction `0.05`, ordinary mesh;
2. receiver mesh `1.25 m`, internal fraction `0.05`, ordinary mesh;
3. receiver mesh `0.625 m`, internal fraction `0.05`, ordinary mesh;
4. receiver mesh `1.25 m`, internal fraction `0.025`, ordinary mesh;
5. receiver mesh `1.25 m`, internal fraction `0.05`, symmetric full-domain mesh.

Do not change more than the named variable between adjacent comparisons.

- [ ] **Step 5: Run the audit and record the method decision**

Generate all CSV/JSON/PNG artifacts. If no method passes every gate, stop here, preserve outputs, and return to root-cause investigation. Do not run the expensive channel case.

- [ ] **Step 6: Commit only scripts and tests, never generated results**

```powershell
git add tools/run_fenicsx_magnetic_background_short.sh tools/run_fenicsx_magnetic_channel_full.sh tests/test_seepage_channel_fenicsx_command.py
git commit -m "test: add staged magnetic receiver runs"
```

### Task 11: Run the selected full benchmark and final verification

**Files:**
- Verify: `tools/build_seepage_channel_word_report.py`
- Output only: `output/seepage_channel_100m_5rx_60x1x1/magnetic_receiver_stability/`

- [ ] **Step 1: Run full background and channel models**

Use the selected receiver operator, 31 formal times, identical locked mesh hashes for background/channel, and the approved 60 × 1 × 1 m material audit. Preserve the old formal results.

- [ ] **Step 2: Verify formal data contracts**

Confirm for both runs:

- formal array shape `(5, 31, 3)`;
- 465 finite values;
- five `explicit_full_domain` provenance entries;
- identical receivers, times, components, source, mesh hash, and solver settings;
- channel discrete volume remains within the existing volume-audit tolerance.

- [ ] **Step 3: Run the complete test suite relevant to the change**

Run on Windows:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest tests/test_magnetic_symmetry_audit.py tests/test_magnetic_receiver_operators.py tests/test_dolfinx_magnetic_receiver_adapters.py tests/test_magnetic_receiver_cli.py tests/test_dolfinx_biot_receiver.py tests/test_seepage_channel_fenicsx_receivers.py tests/test_symmetric_full_domain_mesh.py tests/test_magnetic_receiver_audit_runner.py tests/test_seepage_channel_plots_manifest.py tests/test_seepage_channel_word_report.py tests/test_seepage_channel_fenicsx_command.py -q
```

Run DOLFINx-dependent tests in WSL. Expected: all tests pass; Windows-only missing DOLFINx paths are skipped, not failed.

- [ ] **Step 4: Recompute metrics, figures, manifest, and Word report**

Generate final artifacts from the completed full runs. Render the Word report page by page. Confirm that the written conclusion matches the JSON gates and that Rx3 is discussed using absolute residual and zero-point ratio.

- [ ] **Step 5: Perform final verification before claiming success**

Check:

```powershell
git diff --check
git status --short
```

Record test output, run hashes, formal method, failed alternatives, runtime, and every acceptance metric in the handoff. Generated result files remain uncommitted unless the user explicitly requests versioning them.

- [ ] **Step 6: Commit any data-driven report correction separately**

Only if the completed data requires a wording or table correction:

```powershell
git add tools/build_seepage_channel_word_report.py
git commit -m "docs: finalize magnetic receiver validation report"
```

Do not include unrelated workspace changes or generated output directories in this commit.
