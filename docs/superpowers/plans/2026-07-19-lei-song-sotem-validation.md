# Lei 2023 + Song 2025 SOTEM Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 Lei 2023 无 IP 与 Song 2025 IP/no-IP 两级、非循环的 SOTEM 验证套件，运行 empymod、SimPEG/Discretize 和 FEniCSx，并用可复现证据定位和修复 FEniCSx 的物理错误。

**Architecture:** 以版本控制内的规范 YAML 为唯一物理模型入口，通过小型适配器分别生成 empymod 精确层状参考、SimPEG/Discretize-Debye 配对计算和 FEniCSx CLI 参数。现有 `dolfinx/sotem_pipeline.py` 继续承担网格、边元、Debye、接收器和运行产物；新增的验收层只负责规范模型、单位、IP 差分、零交叉、状态机和 manifest，不复制求解器。

**Tech Stack:** Python 3.11、NumPy/SciPy、empymod、SimPEG/Discretize、FEniCSx/DOLFINx 0.8、PETSc、Gmsh、PyYAML、pytest、WSL2 Conda。

---

## 文件边界

新建文件：

- `benchmarks/sotem/lei2023_noip.yaml`：Lei 规范物理模型和时间道。
- `benchmarks/sotem/song2025_layered_pair.yaml`：Song IP/no-IP 配对模型和 Cole-Cole 参数。
- `src/atem3d/sotem_benchmark.py`：加载、校验规范模型并生成精确时间道和收敛层级。
- `src/atem3d/sotem_observables.py`：内部 `Ex/Ey/Hz/dBzdt` 到带单位规范表的单向转换。
- `src/atem3d/sotem_metrics.py`：1% 峰值下限、IP 差分和零交叉指标。
- `src/atem3d/sotem_simpeg_adapter.py`：规范模型到现有 SimPEG/Discretize-Debye 求解器的适配。
- `src/atem3d/sotem_gate.py`：两级验收状态机和非循环参考守卫。
- `src/atem3d/sotem_validation_cli.py`：生成参考、运行 SimPEG、评估配对结果和汇总状态。
- `dolfinx/run_sotem_benchmark.py`：把规范模型翻译为现有 FEniCSx 管线参数并调用 `main()`。
- `tests/test_sotem_benchmark.py`、`tests/test_sotem_observables.py`、`tests/test_sotem_metrics.py`、`tests/test_sotem_simpeg_adapter.py`、`tests/test_sotem_gate.py`、`tests/test_sotem_validation_cli.py`、`tests/test_run_sotem_benchmark.py`：对应窄职责测试。

修改文件：

- `dolfinx/sotem_pipeline.py`：显式观测时间、理想阶跃、精确 Cole-Cole empymod 参考和拟合门。
- `src/atem3d/polarization_effect.py`：复用新 IP 差分指标并输出零交叉。
- `pyproject.toml`：注册 `atem3d-sotem-validate`。
- `tests/test_dolfinx_model_consistency.py`、`tests/test_dolfinx_biot_receiver.py`、`tests/test_polarization_effect.py`：现有行为的回归测试。

不修改 `corrected_model_runner` 的旧失败产物，不覆盖任何现有 `current_task_runs` 目录，不把 3D 极化体纳入本计划。

### Task 1: 固定两个规范模型

**Files:**
- Create: `benchmarks/sotem/lei2023_noip.yaml`
- Create: `benchmarks/sotem/song2025_layered_pair.yaml`
- Create: `src/atem3d/sotem_benchmark.py`
- Test: `tests/test_sotem_benchmark.py`

- [ ] **Step 1: 写规范模型失败测试**

```python
from pathlib import Path
import pytest

from atem3d.sotem_benchmark import load_benchmark_case


ROOT = Path(__file__).resolve().parents[1]


def test_lei_case_matches_approved_design():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, 800.0, 0.1)
    assert case.current_a == 1.0
    assert case.rho_air_ohm_m == 1.0e8
    assert case.observation_times.size == 41
    assert case.observation_times[[0, -1]].tolist() == pytest.approx([1.0e-5, 1.0e-1])


def test_song_pair_changes_only_polarization():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, -500.0, 0.1)
    assert case.current_a == 10.0
    assert case.rho_air_ohm_m == 1.0e6
    assert case.observation_times.size == 51
    assert case.polarization == {"top_m": 0.0, "bottom_m": 300.0, "rho0_ohm_m": 100.0,
                                 "m": 0.3, "tau_s": 1.0, "c": 0.3}
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_benchmark.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'atem3d.sotem_benchmark'`.

- [ ] **Step 3: 写入两个 YAML**

```yaml
# benchmarks/sotem/lei2023_noip.yaml
case_id: lei2023_noip
source_start_down_m: [-500.0, 0.0, 0.1]
source_end_down_m: [500.0, 0.0, 0.1]
receiver_down_m: [0.0, 800.0, 0.1]
current_a: 1.0
rho_air_ohm_m: 1.0e8
rho_earth_ohm_m: 100.0
waveform: ideal_step_off
time_min_s: 1.0e-5
time_max_s: 1.0e-1
intervals_per_decade: 10
required_components: [Ex, dBzdt]
diagnostic_components: [Ey, Hz]
```

```yaml
# benchmarks/sotem/song2025_layered_pair.yaml
case_id: song2025_layered_pair
source_start_down_m: [-500.0, 0.0, 0.1]
source_end_down_m: [500.0, 0.0, 0.1]
receiver_down_m: [0.0, -500.0, 0.1]
current_a: 10.0
rho_air_ohm_m: 1.0e6
rho_earth_ohm_m: 100.0
waveform: ideal_step_off
time_min_s: 1.0e-5
time_max_s: 1.0
intervals_per_decade: 10
required_components: [Ex, Hz, dBzdt]
diagnostic_components: [Ey]
polarization:
  top_m: 0.0
  bottom_m: 300.0
  rho0_ohm_m: 100.0
  m: 0.3
  tau_s: 1.0
  c: 0.3
prony:
  n_terms: 16
  f_min_hz: 1.0e-3
  f_max_hz: 1.0e4
  n_frequency: 81
  max_relative_l2: 0.01
```

- [ ] **Step 4: 实现只读加载器和校验**

```python
# src/atem3d/sotem_benchmark.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atem3d.yaml_io import safe_load_yaml


@dataclass(frozen=True)
class SOTEMBenchmarkCase:
    case_id: str
    source_start_down: tuple[float, float, float]
    source_end_down: tuple[float, float, float]
    receiver_down: tuple[float, float, float]
    current_a: float
    rho_air_ohm_m: float
    rho_earth_ohm_m: float
    waveform: str
    time_min_s: float
    time_max_s: float
    intervals_per_decade: int
    required_components: tuple[str, ...]
    diagnostic_components: tuple[str, ...]
    polarization: dict[str, float] | None
    prony: dict[str, float | int] | None

    @property
    def observation_times(self) -> np.ndarray:
        decades = np.log10(self.time_max_s) - np.log10(self.time_min_s)
        count = int(round(decades * self.intervals_per_decade)) + 1
        return np.logspace(np.log10(self.time_min_s), np.log10(self.time_max_s), count)

    def point_up(self, point_down: tuple[float, float, float]) -> tuple[float, float, float]:
        return (point_down[0], point_down[1], -point_down[2])


def _point(raw: Any, name: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in raw)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite values")
    return values


def load_benchmark_case(path: str | Path) -> SOTEMBenchmarkCase:
    raw = safe_load_yaml(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark root must be a mapping")
    case = SOTEMBenchmarkCase(
        case_id=str(raw["case_id"]),
        source_start_down=_point(raw["source_start_down_m"], "source_start_down_m"),
        source_end_down=_point(raw["source_end_down_m"], "source_end_down_m"),
        receiver_down=_point(raw["receiver_down_m"], "receiver_down_m"),
        current_a=float(raw["current_a"]),
        rho_air_ohm_m=float(raw["rho_air_ohm_m"]),
        rho_earth_ohm_m=float(raw["rho_earth_ohm_m"]),
        waveform=str(raw["waveform"]),
        time_min_s=float(raw["time_min_s"]),
        time_max_s=float(raw["time_max_s"]),
        intervals_per_decade=int(raw["intervals_per_decade"]),
        required_components=tuple(str(x) for x in raw["required_components"]),
        diagnostic_components=tuple(str(x) for x in raw["diagnostic_components"]),
        polarization=dict(raw["polarization"]) if raw.get("polarization") else None,
        prony=dict(raw["prony"]) if raw.get("prony") else None,
    )
    if case.waveform != "ideal_step_off":
        raise ValueError("the validation suite requires ideal_step_off")
    if not (case.current_a > 0 and case.rho_air_ohm_m > 0 and case.rho_earth_ohm_m > 0):
        raise ValueError("current and resistivities must be positive")
    if not (0 < case.time_min_s < case.time_max_s and case.intervals_per_decade > 0):
        raise ValueError("invalid observation-time specification")
    return case
```

- [ ] **Step 5: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_benchmark.py -v`

Expected: PASS.

```bash
git add benchmarks/sotem src/atem3d/sotem_benchmark.py tests/test_sotem_benchmark.py
git commit -m "feat: define Lei and Song SOTEM benchmark cases"
```

### Task 2: 支持理想阶跃与显式输出时间

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:52-180,405-500,888-949,8140-8440`
- Test: `tests/test_dolfinx_model_consistency.py`

- [ ] **Step 1: 写理想阶跃和显式时间失败测试**

```python
def test_model_consistency_accepts_ideal_step_off():
    sp = _load_pipeline_module()
    result = sp.validate_model_consistency(sp.PipelineConfig(ramp_off_time=0.0))
    assert result["ramp_off_time"] == 0.0


def test_explicit_observation_times_override_growth_grid():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=(1.0e-5, 1.0e-4, 1.0e-3))
    assert sp.generate_time_array(config).tolist() == pytest.approx(config.observation_times)


def test_ideal_step_off_schedule_has_no_synthetic_ramp_steps():
    sp = _load_pipeline_module()
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])
    schedule = sp._forward_observation_schedule(
        times, sp.PipelineConfig(ramp_off_time=0.0, time_origin="after_ramp")
    )
    assert schedule["step_times"].tolist() == pytest.approx(times)
    assert schedule["output_step_indices"] == [0, 1, 2]
```

- [ ] **Step 2: 运行并确认当前代码拒绝 `ramp_off_time=0`**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_model_consistency.py -k "ideal_step_off or explicit_observation" -v`

Expected: FAIL because `ramp_off_time` is required to be positive and `observation_times` is undefined.

- [ ] **Step 3: 实现非负关断时间、显式时间和零斜坡调度**

Add to `PipelineConfig`:

```python
observation_times: tuple[float, ...] = ()
```

Change validation so `ramp_off_time` is removed from the positive-only list and checked explicitly:

```python
ramp_off_time = float(config.ramp_off_time)
if not math.isfinite(ramp_off_time) or ramp_off_time < 0.0:
    raise ValueError("ramp_off_time must be finite and nonnegative")
diagnostics["ramp_off_time"] = ramp_off_time
```

Change `generate_time_array` and `_forward_observation_schedule`:

```python
def generate_time_array(config: PipelineConfig):
    import numpy as np
    if config.observation_times:
        values = np.asarray(config.observation_times, dtype=float)
        if values.ndim != 1 or values.size == 0 or np.any(values <= 0.0) or np.any(np.diff(values) <= 0.0):
            raise ValueError("observation_times must be finite, positive, and strictly increasing")
        return values
    # retain the existing geometric-growth implementation below


# at the start of the after_ramp branch
if float(config.ramp_off_time) == 0.0:
    return {
        "step_times": observation_times.copy(),
        "output_internal_times": observation_times.copy(),
        "return_times": observation_times.copy(),
        "output_step_indices": list(range(observation_times.size)),
    }
```

Add CLI `--observation-times` using `_parse_float_csv` and propagate it to `PipelineConfig`. Record it in `_resolved_config_yaml`.

- [ ] **Step 4: 更新旧的拒绝零关断测试并运行相关测试**

Remove `("ramp_off_time", 0.0)` from the nonpositive-parameter test and add a separate negative case.

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_model_consistency.py -v`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: support ideal step-off benchmark schedules"
```

### Task 3: 分离精确 Cole-Cole 参考与 Debye 近似

**Files:**
- Modify: `dolfinx/sotem_pipeline.py:52-180,3621-3685,5426-5615,8140-8440`
- Test: `tests/test_dolfinx_biot_receiver.py`

- [ ] **Step 1: 写精确参考和拟合门失败测试**

```python
def test_song_prony_fit_passes_one_percent_material_gate():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        cole_rho0=100.0, cole_m=0.3, cole_tau=1.0, cole_c=0.3,
        cole_n_terms=16, cole_f_min=1.0e-3, cole_f_max=1.0e4,
        cole_n_freq=81, cole_fit_tolerance=0.01,
    )
    fit = sp.fit_cole_cole_to_debye(config)
    assert fit.relative_l2 <= 0.01
    assert fit.sigma_infinity == pytest.approx(1.0 / 70.0)
    assert sum(term.delta_sigma for term in fit.terms) == pytest.approx(1.0 / 70.0 - 0.01)


def test_exact_empymod_material_uses_cole_cole_not_debye_fit(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(300.0,), layer_resistivities=(100.0, 100.0),
        cole_layer_top=0.0, cole_layer_bottom=300.0,
        cole_rho0=100.0, cole_m=0.3, cole_tau=1.0, cole_c=0.3,
    )
    depth, material = sp._exact_cole_cole_empymod_material(config)
    frequencies = np.array([1.0e-3, 1.0, 1.0e4])
    eta_h, eta_v = material["func_eta"](None, {"freq": frequencies})
    expected = sp.cole_cole_complex_conductivity(frequencies, 100.0, 0.3, 1.0, 0.3)
    np.testing.assert_allclose(eta_h[:, 1], expected)
    np.testing.assert_allclose(eta_v[:, 1], expected)
```

- [ ] **Step 2: 运行并确认失败**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_biot_receiver.py -k "prony_fit_passes or exact_empymod" -v`

Expected: FAIL because `cole_fit_tolerance` and `_exact_cole_cole_empymod_material` do not exist.

- [ ] **Step 3: 增加拟合容差并让失败闭合**

Add `cole_fit_tolerance: float = 0.01` to `PipelineConfig`, CLI `--cole-fit-tolerance`, config construction, resolved YAML, and validation. At the end of `fit_cole_cole_to_debye`:

```python
if rel_l2 > float(config.cole_fit_tolerance):
    raise ValueError(
        f"Cole-Cole Debye fit relative L2 {rel_l2:.6e} exceeds "
        f"tolerance {float(config.cole_fit_tolerance):.6e}"
    )
```

- [ ] **Step 4: 实现精确 empymod 材料钩子**

```python
def _exact_cole_cole_empymod_material(config: PipelineConfig):
    import numpy as np

    depth, base_res = _empymod_depth_res(config)
    indices = _empymod_polarizable_layer_indices(depth, base_res, config)
    if not indices:
        raise RuntimeError("no empymod layer overlaps the Cole-Cole interval")
    sigma = 1.0 / np.asarray(base_res, dtype=float)

    def func_eta(_model, context):
        freq = np.asarray(context["freq"], dtype=float)
        eta = np.tile(sigma, (freq.size, 1)).astype(complex)
        exact = cole_cole_complex_conductivity(
            freq, config.cole_rho0, config.cole_m, config.cole_tau, config.cole_c
        )
        for index in indices:
            eta[:, index] = exact
        return eta, eta

    return depth, {"res": list(base_res), "func_eta": func_eta}
```

Add `mode="cole-cole-exact"` to `get_empymod_reference`; keep the current fitted mode as `cole-cole-debye` for diagnostics. Change production IP postprocessing to use `cole-cole-exact`.

- [ ] **Step 5: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_dolfinx_biot_receiver.py tests/test_dolfinx_model_consistency.py -v`

Expected: PASS.

```bash
git add dolfinx/sotem_pipeline.py tests/test_dolfinx_biot_receiver.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: add exact Cole-Cole empymod reference"
```

### Task 4: 规范化响应名称和单位

**Files:**
- Create: `src/atem3d/sotem_observables.py`
- Test: `tests/test_sotem_observables.py`

- [ ] **Step 1: 写单位和对称性失败测试**

```python
import numpy as np
from scipy.constants import mu_0

from atem3d.sotem_observables import canonical_response


def test_canonical_response_writes_hz_bz_and_dbdt_without_aliasing():
    table = canonical_response(
        np.array([1.0e-5, 1.0e-4]),
        np.array([[2.0, 1.0e-4, 3.0, 4.0], [1.0, 5.0e-5, 2.0, 3.0]]),
        ["Ex", "Ey", "Hz", "dBzdt"],
    )
    assert table.columns == ("Ex_V_per_m", "Ey_V_per_m", "Hz_A_per_m", "Bz_T", "dBzdt_T_per_s")
    np.testing.assert_allclose(table.values[:, 3], mu_0 * table.values[:, 2])
    assert table.ey_to_ex_peak_ratio == 5.0e-5
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_observables.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现规范响应表**

```python
# src/atem3d/sotem_observables.py
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from scipy.constants import mu_0


@dataclass(frozen=True)
class CanonicalResponse:
    times: np.ndarray
    values: np.ndarray
    columns: tuple[str, ...]

    @property
    def ey_to_ex_peak_ratio(self) -> float:
        ex = float(np.max(np.abs(self.values[:, self.columns.index("Ex_V_per_m")])))
        ey = float(np.max(np.abs(self.values[:, self.columns.index("Ey_V_per_m")])))
        return ey / ex if ex else float("inf")

    def write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_obs_s", *self.columns])
            for time, row in zip(self.times, self.values):
                writer.writerow([float(time), *[float(value) for value in row]])


def canonical_response(times, values, components) -> CanonicalResponse:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    names = [str(name) for name in components]
    required = ["Ex", "Ey", "Hz", "dBzdt"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError("missing canonical components: " + ",".join(missing))
    columns = ("Ex_V_per_m", "Ey_V_per_m", "Hz_A_per_m", "Bz_T", "dBzdt_T_per_s")
    ex, ey, hz, dbdt = [values[:, names.index(name)] for name in required]
    return CanonicalResponse(times, np.column_stack([ex, ey, hz, mu_0 * hz, dbdt]), columns)
```

- [ ] **Step 4: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_observables.py -v`

Expected: PASS.

```bash
git add src/atem3d/sotem_observables.py tests/test_sotem_observables.py
git commit -m "feat: add canonical SOTEM observable units"
```

### Task 5: 实现 1% 峰值误差、IP 差分和零交叉

**Files:**
- Create: `src/atem3d/sotem_metrics.py`
- Modify: `src/atem3d/polarization_effect.py`
- Test: `tests/test_sotem_metrics.py`
- Test: `tests/test_polarization_effect.py`

- [ ] **Step 1: 写指标失败测试**

```python
import numpy as np
import pytest

from atem3d.sotem_metrics import compare_signed_response, linear_zero_crossings


def test_linear_zero_crossings_preserve_topology_and_interpolate_time():
    times = np.array([1.0, 2.0, 4.0])
    values = np.array([2.0, -2.0, -1.0])
    assert linear_zero_crossings(times, values).tolist() == pytest.approx([1.5])


def test_signed_response_uses_one_percent_peak_floor():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [0.0]]),
        np.array([[1.0], [0.005]]),
        ["Ex"], threshold=0.10,
    )
    assert result["floor_by_component"]["Ex"] == pytest.approx(0.01)
    assert result["max_robust_error_by_component"]["Ex"] == pytest.approx(0.5)
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_metrics.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现指标**

```python
# src/atem3d/sotem_metrics.py
import numpy as np

from atem3d.metrics import robust_component_errors


def linear_zero_crossings(times, values) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    crossings = []
    for t0, t1, y0, y1 in zip(times[:-1], times[1:], values[:-1], values[1:]):
        if y0 != 0.0 and y1 != 0.0 and np.signbit(y0) != np.signbit(y1):
            crossings.append(float(t0 - y0 * (t1 - t0) / (y1 - y0)))
    return np.asarray(crossings, dtype=float)


def compare_signed_response(times, prediction, reference, components, *, threshold: float) -> dict:
    reference = np.asarray(reference, dtype=float)
    floors = {
        str(name): 0.01 * float(np.max(np.abs(reference[:, index])))
        for index, name in enumerate(components)
    }
    if any(value <= 0.0 for value in floors.values()):
        raise ValueError("each reference component must have a nonzero peak")
    rows, summary = robust_component_errors(
        times, prediction, reference, components,
        threshold=threshold, floor_overrides=floors,
    )
    zero = {}
    for index, name in enumerate(components):
        pred_zero = linear_zero_crossings(times, np.asarray(prediction)[:, index])
        ref_zero = linear_zero_crossings(times, reference[:, index])
        relative = np.abs(pred_zero - ref_zero) / np.abs(ref_zero) if pred_zero.size == ref_zero.size else np.array([])
        zero[str(name)] = {
            "prediction": pred_zero.tolist(), "reference": ref_zero.tolist(),
            "count_match": bool(pred_zero.size == ref_zero.size),
            "max_relative_time_error": float(np.max(relative)) if relative.size else (0.0 if pred_zero.size == ref_zero.size else float("inf")),
        }
    return {
        "rows": rows, "summary": summary, "floor_by_component": floors,
        "max_robust_error_by_component": {str(name): summary[f"max_error_{name}"] for name in components},
        "zero_crossings": zero,
    }
```

- [ ] **Step 4: 让 `polarization_effect.py` 使用 10% 门和新零交叉结果**

Keep old artifact filenames. Change default threshold to `0.10`, call `compare_signed_response` on `effect_pred/effect_ref`, and add `zero_crossings` plus `definition="ip_minus_noip"` to `polarization_effect_summary.json`.

- [ ] **Step 5: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_metrics.py tests/test_polarization_effect.py -v`

Expected: PASS.

```bash
git add src/atem3d/sotem_metrics.py src/atem3d/polarization_effect.py tests/test_sotem_metrics.py tests/test_polarization_effect.py
git commit -m "feat: add signed IP effect acceptance metrics"
```

### Task 6: 建立两级状态机和循环参考守卫

**Files:**
- Create: `src/atem3d/sotem_gate.py`
- Test: `tests/test_sotem_gate.py`

- [ ] **Step 1: 写状态机失败测试**

```python
import pytest
from atem3d.sotem_gate import summarize_sotem_gates


def test_empymod_plus_zero_secondary_is_plumbing_only():
    summary = summarize_sotem_gates({
        "lei_simpeg": True, "lei_fenicsx": True,
        "song_noip_simpeg": True, "song_noip_fenicsx": True,
        "song_ip_simpeg": True, "song_ip_fenicsx": True,
        "song_delta_simpeg": True, "song_delta_fenicsx": True,
        "material_gate": True,
        "reference_provenance": "empymod_plus_zero_secondary",
    })
    assert summary["state"] == "plumbing_pass"
    assert summary["ip_internally_validated"] is False


def test_all_independent_gates_produce_ip_internal_validation():
    gates = {name: True for name in (
        "lei_simpeg", "lei_fenicsx", "song_noip_simpeg", "song_noip_fenicsx",
        "song_ip_simpeg", "song_ip_fenicsx", "song_delta_simpeg",
        "song_delta_fenicsx", "material_gate")}
    gates["reference_provenance"] = "empymod_exact_layered"
    assert summarize_sotem_gates(gates)["state"] == "ip_internally_validated"
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_gate.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现 fail-closed 状态机**

```python
# src/atem3d/sotem_gate.py
LEI = ("lei_simpeg", "lei_fenicsx")
SONG_NOIP = ("song_noip_simpeg", "song_noip_fenicsx")
SONG_IP = ("song_ip_simpeg", "song_ip_fenicsx", "song_delta_simpeg", "song_delta_fenicsx", "material_gate")
INDEPENDENT_REFERENCES = {"empymod_noip_layered", "empymod_exact_layered"}


def summarize_sotem_gates(gates: dict) -> dict:
    provenance = str(gates.get("reference_provenance", ""))
    independent = provenance in INDEPENDENT_REFERENCES
    lei_pass = all(bool(gates.get(name, False)) for name in LEI)
    noip_pass = lei_pass and all(bool(gates.get(name, False)) for name in SONG_NOIP)
    ip_pass = noip_pass and all(bool(gates.get(name, False)) for name in SONG_IP) and independent
    if ip_pass:
        state = "ip_internally_validated"
    elif noip_pass and independent:
        state = "noip_internally_validated"
    elif provenance == "empymod_plus_zero_secondary":
        state = "plumbing_pass"
    else:
        state = "failed_with_reproducible_evidence"
    return {
        "state": state,
        "noip_internally_validated": state in {"noip_internally_validated", "ip_internally_validated"},
        "ip_internally_validated": state == "ip_internally_validated",
        "reference_independent": independent,
        "gates": dict(gates),
    }
```

- [ ] **Step 4: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_gate.py -v`

Expected: PASS.

```bash
git add src/atem3d/sotem_gate.py tests/test_sotem_gate.py
git commit -m "feat: add fail-closed SOTEM gate state machine"
```

### Task 7: 适配现有 SimPEG/Discretize-Debye 求解器

**Files:**
- Create: `src/atem3d/sotem_simpeg_adapter.py`
- Test: `tests/test_sotem_simpeg_adapter.py`

- [ ] **Step 1: 写同网格退化和时间子步测试**

```python
import numpy as np
from atem3d.sotem_simpeg_adapter import build_internal_time_steps, paired_model_dicts


def test_internal_steps_hit_every_output_and_apply_substeps():
    outputs = np.array([1.0e-5, 1.0e-4, 1.0e-3])
    steps, indices = build_internal_time_steps(outputs, substeps=4)
    np.testing.assert_allclose(np.cumsum(steps)[indices], outputs)
    assert len(steps) == 12


def test_song_noip_and_ip_share_mesh_and_dc_conductivity(song_case):
    noip, ip = paired_model_dicts(song_case, spatial_level="S0", boundary_level="B0", substeps=1)
    assert noip["mesh"] == ip["mesh"]
    assert noip["model"]["sigma_infinity"] == 0.01
    assert ip["model"]["layers"][0]["rho0"] == 100.0
    assert ip["model"]["layers"][0]["chargeability"] == 0.3
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_simpeg_adapter.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现输出对齐的内部时间步**

```python
def build_internal_time_steps(outputs, *, substeps: int):
    outputs = np.asarray(outputs, dtype=float)
    if substeps <= 0:
        raise ValueError("substeps must be positive")
    nodes = [0.0]
    output_indices = []
    for output in outputs:
        segment = np.linspace(nodes[-1], float(output), substeps + 1)[1:]
        nodes.extend(float(value) for value in segment)
        output_indices.append(len(nodes) - 2)
    return np.diff(np.asarray(nodes)), np.asarray(output_indices, dtype=int)
```

Implement deterministic graded TensorMesh builders for S0/S1/S2 (`40/20/10 m` source cells and `20/10/5 m` receiver cells) and B0/B1/B2 (`25/50/100 km`). Use identical mesh dictionaries for Song no-IP/IP. Build receivers `[Ex, Ey, Hz, dBzdt]`, `StepOffWaveform(off_time=0)`, and the existing `build_simulation(...).run_data_only()`.

The adapter output must be:

```python
{
    "times": case.observation_times,
    "data": result.data[1:][output_indices],
    "components": ["Ex", "Ey", "Hz", "dBzdt"],
    "mesh_stats": {"n_cells": simulation.mesh.n_cells, "n_edges": simulation.mesh.n_edges},
    "solver_id": "atem3d_simpeg_discretize_debye",
}
```

- [ ] **Step 4: 加入最小 3x3x3 no-IP parity 回归**

Reuse the construction in `tests/test_simpeg_parity.py`; assert the adapter's no-IP field path remains equal to upstream SimPEG on the tiny mesh before any Song-scale run.

- [ ] **Step 5: 运行测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_simpeg_adapter.py tests/test_simpeg_parity.py -v`

Expected: PASS.

```bash
git add src/atem3d/sotem_simpeg_adapter.py tests/test_sotem_simpeg_adapter.py tests/test_simpeg_parity.py
git commit -m "feat: add paired SimPEG Debye benchmark adapter"
```

### Task 8: 建立套件 CLI、manifest 和唯一 run-id

**Files:**
- Create: `src/atem3d/sotem_validation_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_sotem_validation_cli.py`

- [ ] **Step 1: 写 CLI 失败测试**

```python
import json
from atem3d.sotem_validation_cli import main


def test_prepare_writes_unique_manifest(tmp_path):
    assert main(["prepare", "--case", "benchmarks/sotem/lei2023_noip.yaml",
                 "--solver", "empymod", "--output-root", str(tmp_path)]) == 0
    manifests = list(tmp_path.glob("lei2023_noip/*/manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["case_id"] == "lei2023_noip"
    assert payload["solver_id"] == "empymod"
    assert payload["status"] == "prepared"
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_validation_cli.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现 `prepare/reference/simpeg/effect/finalize` 子命令**

Use `argparse` subparsers. `prepare` creates `<output-root>/<case-id>/<UTC timestamp>-<8-char uuid>/manifest.json` with case path, case-file SHA256, Git commit, solver id, level, Python/library versions and `status=prepared`. `reference` calls `get_empymod_reference` through the pipeline module with `noip` or `cole-cole-exact`. `simpeg` calls Task 7. `effect` calls `write_polarization_effect_artifacts`. `finalize` calls Task 6 and writes `final_gate_summary.json`.

Register:

```toml
atem3d-sotem-validate = "atem3d.sotem_validation_cli:main"
```

Every subcommand must refuse an existing non-empty run directory unless `--resume` is supplied; `--resume` must verify case hash and solver id before writing.

- [ ] **Step 4: 运行 CLI 测试并提交**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_validation_cli.py -v`

Expected: PASS.

```bash
git add src/atem3d/sotem_validation_cli.py tests/test_sotem_validation_cli.py pyproject.toml
git commit -m "feat: add reproducible SOTEM validation CLI"
```

### Task 9: 把规范模型接到 FEniCSx 管线

**Files:**
- Create: `dolfinx/run_sotem_benchmark.py`
- Modify: `dolfinx/sotem_pipeline.py:8062-8440`
- Test: `tests/test_run_sotem_benchmark.py`

- [ ] **Step 1: 写真实参数传播失败测试**

```python
def test_song_ip_command_reaches_real_pipeline(monkeypatch, tmp_path):
    runner = _load_runner_module()
    captured = {}

    def fake_pipeline_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(runner, "PIPELINE_MAIN", fake_pipeline_main)
    code = runner.main([
        "--case", "benchmarks/sotem/song2025_layered_pair.yaml",
        "--variant", "ip", "--level", "S0T0B0", "--workdir", str(tmp_path),
        "--check-env-only", "--no-install",
    ])
    assert code == 0
    argv = captured["argv"]
    assert _arg(argv, "--source-current") == "10.0"
    assert _arg(argv, "--ramp-off-time") == "0.0"
    assert _arg(argv, "--rho-air") == "1000000.0"
    assert _arg(argv, "--cole-layer-bottom") == "300.0"
    assert _arg(argv, "--cole-m") == "0.3"
    assert len(_arg(argv, "--observation-times").split(",")) == 51
```

- [ ] **Step 2: 运行并确认 runner 不存在**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_run_sotem_benchmark.py -v`

Expected: FAIL because `dolfinx/run_sotem_benchmark.py` does not exist.

- [ ] **Step 3: 实现仅做参数翻译的 runner**

`build_pipeline_argv(case, variant, level, workdir)` must emit explicit flags for every approved physical parameter, `--time-method theta --time-theta 1`, `--initial-dc-mode fem`, `--magnetic-receiver-mode faraday_integrated`, `--magnetic-dbdt-mode curl`, explicit observations, S/T/B level mesh settings, `--error-min-time 0`, `--empymod-srcpts 9`, and `--reference-audit-srcpts 17`.

For Song no-IP emit `--polarization none --layer-depths 300 --layer-resistivities 100,100`. For Song IP additionally emit `--polarization cole-cole --cole-rho0 100 --cole-m 0.3 --cole-tau 1 --cole-c 0.3 --cole-n-terms 16 --cole-f-min 0.001 --cole-f-max 10000 --cole-n-freq 81 --cole-fit-tolerance 0.01 --cole-layer-top 0 --cole-layer-bottom 300`.

The file ends with:

```python
def main(argv=None):
    args = _parser().parse_args(argv)
    case = load_benchmark_case(args.case)
    pipeline_argv = build_pipeline_argv(case, args.variant, args.level, args.workdir)
    if args.check_env_only:
        pipeline_argv.extend(["--check-env-only"])
    if args.no_install:
        pipeline_argv.extend(["--no-install"])
    return PIPELINE_MAIN(pipeline_argv)
```

- [ ] **Step 4: 让 IP 后处理选择精确参考**

Change the pipeline's production `ref_mode` selection from `cole-cole` to `cole-cole-exact`. Preserve `cole-cole-debye` only as an explicit diagnostic mode in tests.

- [ ] **Step 5: 运行命令合同和环境检查**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_run_sotem_benchmark.py tests/test_dolfinx_model_consistency.py -v
wsl -d Ubuntu -- bash -lc "source /home/paidaxin/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx && cd '/mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解/.worktrees/lei2023-sotem-benchmark' && python dolfinx/run_sotem_benchmark.py --case benchmarks/sotem/song2025_layered_pair.yaml --variant ip --level S0T0B0 --workdir /tmp/song-ip-contract --check-env-only --no-install"
```

Expected: pytest PASS; WSL exits 0 and reports DOLFINx/PETSc versions without starting a mesh solve.

- [ ] **Step 6: 提交**

```bash
git add dolfinx/run_sotem_benchmark.py dolfinx/sotem_pipeline.py tests/test_run_sotem_benchmark.py tests/test_dolfinx_model_consistency.py
git commit -m "feat: connect benchmark cases to FEniCSx pipeline"
```

### Task 10: 运行低成本参考和建立首个真实失败基准

**Files:**
- Generated only: `generated/validation/...`
- Test evidence: existing tracked tests plus generated manifest/checksums

- [ ] **Step 1: 运行完整单元测试基线**

Run: `D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest`

Expected: all collected tests pass; record exact count in the run manifest.

- [ ] **Step 2: 生成三个 empymod 参考并检查有限源收敛**

Run the CLI for Lei no-IP, Song no-IP and Song IP with `srcpts=5,9,17`. Require the 9-to-17 change to be `<=0.5%` on `Ex/Hz/dBzdt`; otherwise stop and increase source quadrature before any FEniCSx comparison.

- [ ] **Step 3: 运行 SimPEG/Discretize S0T0B0 smoke**

Run no-IP and IP paired Song cases and Lei no-IP. Require finite signed values, exact output time axes, `Ey/Ex<=0.001`, nonzero `delta_IP`, and matching no-IP/IP mesh hashes. Do not apply the final 5% gate until S/T/B convergence is complete.

- [ ] **Step 4: 运行 FEniCSx source-only preflight**

Run each case with `--source-only`. Require source length/current balance, nonempty polarizable cell markers for Song IP, material-fit `<=1%`, receiver inside the mesh, and memory estimate below `30.4 GB` (`32 GB x 0.95`).

- [ ] **Step 5: 运行 5 时间道 FEniCSx smoke 并保留失败**

Use fixed times `1e-5,1e-4,1e-3,1e-2,1e-1` for Lei no-IP and Song no-IP. Write a unique run-id even when the physical gate fails. Expected current status is allowed to be `failed_with_reproducible_evidence`; zero errors from an empymod-primary shortcut are forbidden.

- [ ] **Step 6: 提交小型失败摘要，不提交大型场文件**

Copy only manifest、配置、误差摘要和诊断 reason codes into `docs/validation/lei-song-first-failure.md`; generated CSV/NPZ/mesh remain ignored.

```bash
git add docs/validation/lei-song-first-failure.md
git commit -m "docs: record first independent SOTEM benchmark failure"
```

### Task 11: 按证据修复 FEniCSx 无 IP 根因

**Files:**
- Modify only the function selected by the failed invariant in `dolfinx/sotem_pipeline.py`
- Test the matching existing test file listed below
- Update: `docs/validation/lei-song-first-failure.md`

- [ ] **Step 1: 使用 systematic-debugging 对失败分类**

Run the same mesh/time configuration and classify without changing thresholds:

| Evidence | First code boundary | Regression test |
|---|---|---|
| source balance/orientation fails | `build_source`, `_build_manual_line_source` | `tests/test_source_consistency.py` |
| DC field sign/amplitude fails | `_solve_initial_dc_field`, `_analytic_halfspace_dc_electric_field` | `tests/test_dolfinx_analytic_dc.py` |
| `Ex` fails but source/DC pass | `evaluate_receivers`, receiver point selection | `tests/test_dolfinx_biot_receiver.py` |
| `dBzdt` alone fails | `compute_dbdt`, `_empymod_rec_mapping` | `tests/test_dolfinx_biot_receiver.py` |
| early time changes under T0/T1/T2 | `_forward_observation_schedule`, BE assembly | `tests/test_dolfinx_model_consistency.py` |
| late time changes under B0/B1/B2 | boundary/sponge configuration | `tests/test_dolfinx_sponge_config.py` |

- [ ] **Step 2: 写一个只复现已观察根因的失败测试**

The test must use the exact sign, coefficient, coordinate or schedule value from the failed artifact. Run only that node and verify RED before editing production code.

- [ ] **Step 3: 做最小修复并验证 GREEN**

Change only the classified boundary. Re-run the new node, its full test file, then the 5-channel FEniCSx smoke. Do not alter the model, metric or gate.

- [ ] **Step 4: 对每个额外根因重复 RED/GREEN 并分别提交**

Use one commit per proven root cause:

```bash
git add dolfinx/sotem_pipeline.py tests/<matching-test-file>.py docs/validation/lei-song-first-failure.md
git commit -m "fix: correct <proven FEniCSx invariant>"
```

- [ ] **Step 5: 达到无 IP 继续条件**

Proceed only when the 5-channel Lei and Song no-IP smokes both pass signed `Ex/dBzdt` at `<=5%`, source/DC invariants pass, and no empirical rescaling exists. Otherwise keep `failed_with_reproducible_evidence` and continue the same diagnostic loop.

### Task 12: 运行 Song IP 并修复 Debye 根因

**Files:**
- Modify only diagnosed Debye functions in `dolfinx/sotem_pipeline.py:3621-3810,3938-3946,4762-4779`
- Test: `tests/test_dolfinx_total_field_ip.py`
- Test: `tests/test_debye_update.py`
- Update: `docs/validation/song2025-ip-debug.md`

- [ ] **Step 1: 验证 `M=0` 完整场退化**

Run Song no-IP and Song IP configuration with `cole_m=0` on the same mesh/time grid. Require all signed components to agree to solver tolerance and memory history current to be zero. Write a failing integration test if not.

- [ ] **Step 2: 运行真实 `M=0.3` 的 S0T0B0 IP smoke**

Compare IP total field and `delta_IP` against exact Cole-Cole empymod and the SimPEG/Discretize-Debye result. Preserve zero-crossing topology and signed curves.

- [ ] **Step 3: 按 IP 证据分类并写 RED 测试**

| Evidence | First code boundary | Test |
|---|---|---|
| material gate fails | `fit_cole_cole_to_debye` | `tests/test_dolfinx_biot_receiver.py` |
| first IP step wrong, later shape similar | `_initialise_debye_memories_to_field` | `tests/test_dolfinx_total_field_ip.py` |
| decay time wrong | `_debye_backward_euler_coefficients`, `_update_debye_memories` | `tests/test_debye_update.py` |
| amplitude/sign wrong at all times | `_matrix_for_effective_conductivity`, `_assemble_history_rhs` | `tests/test_dolfinx_total_field_ip.py` |
| only magnetic quantities wrong | Faraday/Biot receiver path | `tests/test_dolfinx_biot_receiver.py` |

- [ ] **Step 4: 最小修复、运行 GREEN 并单独提交**

Run the new node, both IP test files, then the IP smoke. Keep `cole_m`, time axis and thresholds fixed.

- [ ] **Step 5: 达到 IP 继续条件**

Require Song IP total `Ex/Hz/dBzdt<=5%`, `delta_IP<=10%`, zero-crossing count match and time error `<=5%` on S0T0B0 before expensive convergence.

### Task 13: 完成 S/T/B 收敛和最终报告

**Files:**
- Generated: `generated/validation/...`
- Create: `docs/validation/lei-song-final-report.md`
- Modify only if a convergence-specific regression is proven

- [ ] **Step 1: 运行 T0/T1/T2 单轴时间收敛**

Keep S1/B1 fixed and run substeps `1/2/4`. Require T1-to-T2 strong-channel change `<=2%` for Lei no-IP、Song no-IP、Song IP and `delta_IP`.

- [ ] **Step 2: 运行 S0/S1/S2 单轴空间收敛**

Keep T2/B1 fixed. Record cells、DOFs、peak memory and runtime. Require S1-to-S2 change `<=2%`.

- [ ] **Step 3: 运行 B0/B1/B2 单轴边界收敛**

Keep S1/T2 fixed and use `25/50/100 km`. Require B1-to-B2 change `<=2%`; do not fit sponge/Robin parameters to the reference.

- [ ] **Step 4: 运行最终组合并生成状态**

Run the selected converged S/T/B combination for all cases. Generate canonical CSV、signed plots、errors、zero crossings、checksums and `final_gate_summary.json`.

- [ ] **Step 5: 新鲜验证完整测试和报告一致性**

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest
git diff --check
git status --short
```

Expected: tests exit 0; no whitespace errors; only intended report/test/code files changed.

- [ ] **Step 6: 写科学结论并提交**

`docs/validation/lei-song-final-report.md` must distinguish calculated outputs、internal numerical validation and engineering validation. If any gate fails, state `failed_with_reproducible_evidence`; do not write “validated”.

```bash
git add docs/validation/lei-song-final-report.md src tests dolfinx benchmarks pyproject.toml
git commit -m "docs: report Lei and Song SOTEM validation results"
```

### Task 14: 完成分支交付

**Files:**
- No new implementation files

- [ ] **Step 1: 使用 verification-before-completion 复核声明**

Confirm the exact final pytest count、Git status、run manifest hashes and gate state from fresh commands. Do not infer success from earlier output.

- [ ] **Step 2: 使用 requesting-code-review 检查规格覆盖和科学表述**

Review against `docs/superpowers/specs/2026-07-19-lei2023-sotem-benchmark-design.md`. Block completion for missing time channels、missing convergence axis、circular reference、deleted failures or unsupported validation language.

- [ ] **Step 3: 使用 finishing-a-development-branch 向用户提供合并/PR/保留选项**

Do not push or open a PR unless the user selects that outcome.
