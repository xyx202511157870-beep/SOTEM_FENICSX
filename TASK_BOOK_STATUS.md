# 任务书当前状态

本文档记录当前仓库对任务书的可复现状态。结论按“已经可作为证据”和“仍是诊断/缺口”区分，避免把 smoke 或自收敛结果误写成最终物理验收。

## 当前权威模型

- 源：`(-500, 200, -0.1) -> (500, 200, -0.1)`
- 源长：`1000 m`
- 电流：`10 A`
- 线性关断时间：`1e-5 s`
- 接收点：`(0, -300, -0.1)`
- 平行偏移距：`500 m`
- 验证时间：关断结束后 `1e-5 s <= t_obs <= 1 s`
- 内部时间映射：`t_internal = t_off + t_obs`
- 强制三分量：当前最终验收路径使用 `Ex`, `Ey`, `dBzdt`

## 已完成且可复现

| 任务书要求 | 当前状态 | 权威证据 |
| --- | --- | --- |
| 保留旧 total-field 求解器 | 已保留为 legacy/experimental 路径 | `dolfinx/legacy_total_field_baseline.py`, `dolfinx/sotem_pipeline.py` |
| Waveform 接口与完整关断时间轴 | 已实现 | `src/atem3d/waveforms.py`, `tests/test_waveform.py`, `tests/test_time_grid.py` |
| 区间平均 `dI/dt` | 已纳入波形/时间轴测试链 | `tests/test_source_consistency.py`, `src/atem3d/waveforms.py` |
| Debye/Prony 材料模型 | 已实现并测试 | `src/atem3d/materials/prony.py`, `src/atem3d/materials/cole_cole.py`, `tests/test_prony.py`, `tests/test_cole_debye.py` |
| PrimaryProvider 抽象 | 已实现 Zero/Cached/Empymod 入口 | `src/atem3d/primary/`, `tests/test_primary_provider.py` |
| primary-secondary 正演接口 | 已实现可复用 forward path | `src/atem3d/solvers/primary_secondary_forward.py`, `src/atem3d/corrected_model_runner.py` |
| zero-contrast 验证 | 已实现并测试 | `tests/test_secondary_zero_contrast.py`, `tests/test_dolfinx_primary_secondary_forward_smoke.py` |
| 三分量验证产物 | 已实现 | `src/atem3d/validation_3comp.py`, `tests/test_dolfinx_validation_artifacts.py` |
| robust relative error 与 5% 线 | 已实现 | `src/atem3d/metrics.py`, `src/atem3d/validation_3comp.py`, `tests/test_error_metric_floor.py` |
| no-IP/IP 联合最终验收门 | 已实现 | `src/atem3d/final_acceptance.py`, `tests/test_final_acceptance.py` |
| 模型示意图 | 已实现并接入 corrected runner | `src/atem3d/model_schematic.py`, `tests/test_model_schematic.py` |
| 极化效应 IP-noIP 产物 | 已实现 | `src/atem3d/polarization_effect.py`, `tests/test_polarization_effect.py` |
| 运行时间记录 | 已记录到 diagnostics | `src/atem3d/corrected_model_runner.py` |
| WSL FEniCSx 后端 | 已确认可用，需激活 conda `fenicsx` | `python -m atem3d.cli dolfinx-backend-check` |

## 当前最终验收结论

同质背景、零二次场 corrected-model 基线已经在 WSL `fenicsx` 环境完成 no-IP/IP 全时窗验收：

```text
FINAL_ACCEPTANCE_PASSED=true
passed_cases=noip,ip
noip max_error_Ex=0, max_error_Ey=0, max_error_dBzdt=0
ip   max_error_Ex=0, max_error_Ey=0, max_error_dBzdt=0
```

这条结论证明：

- 源/接收器坐标、时间窗、关断后观测时间定义一致；
- empymod/1D primary reference 链路可用；
- no-IP/IP 三分量 artifact、误差、示意图、极化效应和最终报告链路可用；
- corrected-model 在零对比情况下满足 `total = primary`。

这条结论不证明：

- 非零三维异常体/渗漏通道的物理响应已与独立 3D 参考解吻合；
- 复杂地形非零二次场已达到 5% 物理精度。

## 诊断能力已完成但不能作为最终物理验收

| 能力 | 当前状态 | 限制 |
| --- | --- | --- |
| Gmsh 小地形网格 | 已有 WSL smoke | 小尺度诊断网格，不是论文/工程尺度 |
| 渗漏通道 cell marking | 已有 preflight 和 transient smoke | 只证明材料标记和 DOLFINx path 可运行 |
| no-IP/IP terrain smoke CLI | 已实现 `corrected-terrain-smoke-run --case noip/ip/both` | 使用 constant primary 与 `self_convergence` |
| 非零二次场收敛诊断 | 已记录 secondary-effect 非零性 | `dolfinx_refined`/`self_convergence` 不允许通过最终验收门 |
| 论文模型 metadata/曲线产物框架 | 已提取 Song et al. 2025 模型参数；Fig. 2/3/7/8/12/15 的 PDF 页码、单位、面板和图注已写入 digitization target；能从数字化 CSV 写出 `paper_response_overlay.png`, `paper_relative_error_curves.png`, `runtime_diagnostics.json` | 真实曲线数值仍需数字化或获得表格数据 |

## WSL 复现实用命令

所有真实 DOLFINx/FEniCSx 命令必须在 WSL conda 环境里运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx
cd "/mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解"
export PYTHONPATH="$PWD/src"
```

后端检查：

```bash
python -m atem3d.cli dolfinx-backend-check --output /tmp/atem3d_backend_status.json
cat /tmp/atem3d_backend_status.json
```

同质 corrected-model 最终验收：

```bash
python -m atem3d.cli corrected-model-spec dolfinx/runs/latest_corrected_model --output dolfinx/runs/latest_corrected_model/spec.json
python -u -m atem3d.cli corrected-model-run dolfinx/runs/latest_corrected_model/spec.json --case both --output-root dolfinx/runs/latest_corrected_model
python -u -m atem3d.cli acceptance-report dolfinx/runs/latest_corrected_model/acceptance.yaml
```

地形/渗漏诊断 smoke：

```bash
python -u -m atem3d.cli corrected-terrain-smoke-run dolfinx/runs/latest_terrain_smoke_cli --case both --spec-output dolfinx/runs/latest_terrain_smoke_cli/spec.json
```

每次 WSL 使用完，在 PowerShell 执行：

```powershell
wsl --shutdown
wsl -l -v
```

确认状态包含：

```text
Ubuntu    Stopped
```

## 剩余关键缺口

1. 复杂地形/渗漏通道的非零三维物理参考还没有最终独立证据。1D empymod 不能作为非零 3D anomaly 的物理验收参考。
2. Song et al. 2025 论文曲线仍需数字化或表格数据，之后才能生成论文响应 overlay 和误差曲线。
3. 工程尺度地形/异常体模型需要在 32 GB 内存约束下设计网格、边界和分层加密策略，当前只保留 memory-safe smoke。
4. 若最终选择 `Hz` 而非 `dBzdt` 验收，还需要把磁场恢复误差和 Faraday-integrated 路径进一步收敛。
