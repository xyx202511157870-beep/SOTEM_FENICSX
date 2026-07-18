# 渗流通道三算法正演项目

本仓库当前只维护 SimPEG、FEniCSx 和 empymod 三条开源求解路径，用于比较接地长导线下方长方体渗流通道的时域电磁响应。

## 模型约定

- 地表为 `z=0`，地下 `z>0`，空气 `z<0`；几何图按 Z 轴向下绘制。
- 发射源为 100 m 接地导线，渗流通道位于导线正下方约 20 m。
- FEniCSx 使用完整三维全域，不使用单侧计算后镜像。
- 当前接收器集合不包含导线上的不稳定观测点。

## 目录

- `src/atem3d/`：SimPEG 主链、empymod 背景参考和共用验证代码。
- `dolfinx/`：FEniCSx 求解器、完整渗流通道几何与接收器算子。
- `examples/`：当前渗流通道 SimPEG 配置和 empymod 无 IP 参考配置。
- `tools/`：三算法运行、汇总、绘图和 Word 报告工具。
- `tests/`：保留主链的自动化验证。
- `docs/current_status.md`：当前科学状态和晚期 `Ex` 已知问题。

## 典型入口

```powershell
python tools/run_seepage_channel_benchmark.py --help
python tools/run_seepage_verification_matrix.py --help
python tools/aggregate_seepage_verification.py --help
python tools/plot_verified_seepage_report.py --help
python tools/build_verified_seepage_word_report.py --help
```

FEniCSx 在 WSL2 Conda 环境 `/home/paidaxin/miniconda3/envs/fenicsx` 中运行。主入口为 `dolfinx/sotem_pipeline.py`，仓库中的 `tools/run_fenicsx_seepage_*.sh` 固化当前背景与异常体参数。

## 验证状态

均匀半空间和早期强信号结果已形成内部数值交叉检查，但异常体晚期 `Ex` 尚未通过空间收敛验证。请先阅读 `docs/current_status.md`；在晚期空间收敛问题解决前，不应把当前 Word 报告作为完整科学验证结论。

## 测试

```powershell
python -m pytest -q
```

大型网格、HDF5、NPZ、图像、Word 和本地运行目录均不进入 Git。需要复现报告时，应由保留的配置和工具重新生成。
