# SOTEM_FENICSX：三维时域电性源 TEM-IP 正演

本仓库提供基于 FEniCSx/DOLFINx 的三维时域电性源瞬变电磁法
（TEM）与激发极化（IP）正演代码。程序面向接地有限长导线源、
复杂三维电导率结构、Cole–Cole 极化介质以及电场和磁场多分量接收。

本仓库只保留算法源码、基准参数、运行示例和自包含测试，不包含历史
计算输出、网格文件、Word报告、论文图片或本机运行证据包。

## 1. 项目定位

主要研究内容包括：

- 三维接地电性源时域电磁正演；
- 二阶 Nédélec \(H(\mathrm{curl})\) 边元离散；
- 关断前DC初始场和关断后瞬变推进；
- Cole–Cole复电阻率与复电导率参数化；
- 广义Debye/Prony记忆变量逼近；
- \(E_x,E_y,E_z\)、\(H_x,H_y,H_z\)、\(B_x,B_y,B_z\)和
  \(dB_x/dt,dB_y/dt,dB_z/dt\)接收；
- 有限面积三轴线圈法向、面积求积与姿态接口；
- FEniCSx、empymod和SimPEG之间的公开基准对比；
- 5 μs线性关断、分段线性关断和实测发射波形验证；
- 网格、时间步、边界、源积分和接收算子的一致性检查。

代码适用于算法研究和合成模型验证。自动化测试仅证明当前代码在规定
环境中的内部一致性，不能替代网格/时间步收敛、独立软件对比和现场
工程验证。

## 2. 数学模型

准静态Maxwell方程的电场形式可写为：

\[
\nabla\times\left(\mu^{-1}\nabla\times\mathbf E\right)
+\frac{\partial\mathbf J_c}{\partial t}
=-\frac{\partial\mathbf J_s}{\partial t},
\]

其中：

- \(\mathbf E\)：电场；
- \(\mathbf J_s\)：人工发射源电流；
- \(\mathbf J_c\)：介质传导和极化电流；
- \(\mu\)：磁导率。

无极化介质中：

\[
\mathbf J_c=\sigma\mathbf E.
\]

极化介质采用复电阻率Cole–Cole约定：

\[
\rho^*(\omega)=
\rho_0\left[
1-m\left(
1-\frac{1}{1+(i\omega\tau)^c}
\right)
\right],
\qquad
\sigma^*(\omega)=\frac{1}{\rho^*(\omega)}.
\]

这里：

- \(\rho_0\)为直流电阻率；
- \(m\)为充电率；
- \(\tau\)为时间常数；
- \(c\)为频率指数；
- \(\omega=2\pi f\)。

时域计算使用若干Debye项逼近复电导率：

\[
\sigma^*(\omega)\approx
\sigma_\infty-
\sum_k\frac{\Delta\sigma_k}{1+i\omega\tau_k}.
\]

每个Debye项对应一个局部记忆变量，从而避免保存全部历史电场。

## 3. 坐标、符号和单位

除非模型配置另有说明，程序使用：

- \(x,y,z\)：右手坐标系；
- FEniCSx模型通常采用\(z\)向上；
- 电流方向由源端点A指向B；
- 电场单位：V/m；
- 磁场强度单位：A/m；
- 磁通密度单位：T；
- 磁通密度时间导数单位：T/s；
- 电导率单位：S/m；
- 电阻率单位：\(\Omega\cdot\mathrm m\)；
- 时间单位：s。

与empymod或SimPEG比较时，必须同时统一：

1. 坐标正方向；
2. 发射电流方向；
3. 接收器方向；
4. \(B\)、\(H\)和\(dB/dt\)的物理量定义；
5. step-off时间原点；
6. Cole–Cole参数约定；
7. 总场、异常场和IP增量的定义。

## 4. 代码结构

```text
src/atem3d/      通用TEM-IP算法、材料、源、接收器和验证工具
dolfinx/         FEniCSx三维时域有限元求解器及运行入口
sotem_ip/        轻量Cole–Cole、Debye和参考正演模块
benchmarks/      公开文献基准模型与参数来源
examples/        YAML配置和最小Python示例
tests/           自包含单元测试与内部一致性测试
```

核心入口：

- `dolfinx/sotem_pipeline.py`：三维FEniCSx求解管线；
- `dolfinx/run_sotem_benchmark.py`：公开SOTEM基准配置转换与运行；
- `src/atem3d/cli.py`：通用命令行入口；
- `src/atem3d/hj.py`：H/J形式及相关时间推进；
- `src/atem3d/simulation.py`：正演调度和状态管理；
- `src/atem3d/materials/`：Cole–Cole与Prony/Debye材料；
- `src/atem3d/sources.py`：有限长接地源；
- `src/atem3d/receivers.py`：电场、磁场和线圈接收；
- `src/atem3d/receiver_groups.py`：三分量接收器组合与数据重排；
- `src/atem3d/magnetic_recovery.py`：磁场恢复与一致性检查；
- `src/atem3d/empymod_magnetic6.py`：empymod磁场六通道参考与误差评估；
- `src/atem3d/empymod_waveform.py`：有限关断和实测发射波形卷积；
- `src/atem3d/empymod_magnetic6_cli.py`：磁场六通道正式验证入口。

## 5. 运行环境

推荐使用WSL2。当前实际验证环境为：

```text
Python 3.10.20
DOLFINx 0.8.0
petsc4py 3.21.6
mpi4py 4.1.1
Gmsh 4.15.2
empymod 2.5.4
SimPEG 0.24.0
```

本机FEniCSx Conda环境：

```bash
conda activate fenicsx
```

若当前终端尚未初始化 Conda，请先按本机 Conda 安装位置执行其
`etc/profile.d/conda.sh`。其他机器可以使用自己的FEniCSx环境，但应记录Python、DOLFINx、PETSc、
MPI、Gmsh、empymod和SimPEG的确切版本。

## 6. 安装

进入仓库后执行：

```bash
python -m pip install -e .
```

检查模块和命令行入口：

```bash
python -c "import atem3d; print(atem3d.__file__)"
atem3d-run --help
atem3d-initial-field-diagnostics --help
atem3d-sotem-validate --help
atem3d-validate-empymod-magnetic6 --help
```

DOLFINx、PETSc和MPI通常应由Conda或系统FEniCSx环境提供，不建议仅依靠
普通Windows Python安装这些组件。

## 7. 最小检查

只检查运行环境和基准配置，不启动完整三维正演：

```bash
python dolfinx/run_sotem_benchmark.py \
  --case benchmarks/sotem/zhou2020_grounded_wire.yaml \
  --variant noip \
  --level S0T0B0 \
  --workdir generated/zhou2020-env-check \
  --check-env-only \
  --no-install
```

正式计算示例：

```bash
bash benchmarks/sotem/run_zhou2020_fenicsx.sh noip S0T0B0
bash benchmarks/sotem/run_zhou2020_fenicsx.sh ip S0T0B0
```

完整三维计算可能需要较长时间和大量内存。运行前应先检查模型范围、
局部网格、外边界、时间步、线性求解器和可用内存。

## 8. 基准模型

仓库保留以下配置：

- `benchmarks/sotem/zhou2020_grounded_wire.yaml`；
- `benchmarks/sotem/song2025_layered_pair.yaml`；
- `benchmarks/sotem/lei2023_noip.yaml`；
- `examples/empymod_validation_magnetic6.yaml`；
- `examples/empymod_validation_magnetic6_5us.yaml`。

`zhou2020_parameter_provenance.json`记录参数来源和约定。文献没有明确给出
的参数不得伪装成“原文精确参数”，必须标记为推断、数字化读取或本项目
约定。

## 9. 输出和误差

典型输出包括：

- 电场：\(E_x,E_y,E_z\)；
- 磁场强度：\(H_x,H_y,H_z\)；
- 磁通密度：\(B_x,B_y,B_z\)；
- 线圈场量：\(dB_x/dt,dB_y/dt,dB_z/dt\)；
- 无IP、IP总场及其差值；
- 网格、时间步、边界、波形和运行元数据。

磁场六通道正式合同固定为：

```text
Hx, Hy, Hz, dBxdt, dBydt, dBzdt
```

其中：

```text
Hx, Hy, Hz              A/m
dBxdt, dBydt, dBzdt     T/s
```

普通点相对误差为：

\[
\varepsilon_i=
\frac{|d_i^{\mathrm{FEniCSx}}-d_i^{\mathrm{ref}}|}
{|d_i^{\mathrm{ref}}|}\times100\%.
\]

参考值接近零时，该指标会被假性放大。因此还应同时报告：

- 归一化均方根误差；
- 最大绝对误差；
- 分量峰值归一化误差；
- 带分量幅值下限的相对误差；
- 主有效时间道误差；
- 过零时刻；
- IP增量误差；
- 网格和时间步收敛误差。

不得通过删除过零点、平滑曲线或放宽既定门槛制造“通过”结论。

## 10. 测试

运行全部保留测试：

```bash
python -m pytest -q
```

运行重点模块测试：

```bash
python -m pytest -q \
  tests/test_hj.py \
  tests/test_materials_cole_cole.py \
  tests/test_sources.py \
  tests/test_receivers.py \
  tests/test_average_receivers.py \
  tests/test_receiver_groups.py \
  tests/test_magnetic_recovery.py \
  tests/test_empymod_magnetic6.py \
  tests/test_empymod_waveform.py \
  tests/test_magnetic6_production_policy.py
```

测试不会自动执行大型三维生产网格正演。大型正演应单独记录网格、时间步、
边界、求解时间、峰值内存、软件版本和输出哈希。

## 11. 已知限制

- 三维FEniCSx生产计算需要单独安装DOLFINx/PETSc/MPI；
- 即使固定使用二阶边元，粗网格、过近外边界或过大的时间步仍会改变早期响应和过零时刻；
- 有限项Debye拟合不等于精确Cole–Cole，强IP磁响应可能放大拟合误差；
- Biot–Savart、Faraday积分和curl接收器具有不同离散误差；
- 人工外边界必须通过域尺寸或吸收层收敛检查；
- empymod层状基准不能单独证明三维坝体渗流模型已经通过工程验证；
- `magnetic6_numerical.npz`当前是点位场量合同，不等于真实线圈输出电压；
- 自动化测试和软件对比不能替代现场工程验证。

## 12. 许可和引用

本仓库当前采用“保留所有权利”的许可说明，不自动授予复制、再发布或
商业使用权。详见`LICENSE`。

如在论文、报告或项目中使用本算法，请至少说明：

- 使用的仓库提交ID；
- FEniCSx/DOLFINx、PETSc、MPI、empymod和SimPEG版本；
- 模型配置文件；
- 网格和时间步；
- 实际或假定的发射电流波形；
- Cole–Cole/Debye参数约定；
- 接收器物理量、方向和单位；
- 独立验证方法。

## 13. 二阶 Nédélec 生产策略

DOLFINx生产正演固定使用二阶`N1curl(2)`边元。
`PipelineConfig.nedelec_order`的默认值为2，生产配置若传入其他阶次会立即
报错。公开基准包装器也显式传递`--nedelec-order=2`。一阶结果仅允许保留
在明确标注的legacy/reference文件中，不得作为正式坝体弱异常结论。

## 14. 磁场六通道生产输出

DOLFINx生产管线通过以下参数启用统一磁场六通道合同：

```text
--magnetic-output-contract=magnetic6
```

输出顺序严格固定为：

```text
Hx, Hy, Hz, dBxdt, dBydt, dBzdt
```

### 14.1 E形式

E形式的正式六通道模式要求：

```text
--formulation=e
--magnetic-receiver-mode=biot_current
--magnetic-dbdt-mode=curl
--magnetic-output-contract=magnetic6
```

其物理路径为：

- \(H_x,H_y,H_z\)：由介质电流和外加有限导线电流的总Biot–Savart场恢复；
- \(dB_x/dt,dB_y/dt,dB_z/dt\)：由\(-\nabla\times\mathbf E\)独立计算。

H与\(dB/dt\)采用不同恢复路径，可用于检验方向、单位和离散闭合性。

### 14.2 H形式

H形式直接使用求解得到的\(H_x,H_y,H_z\)，并依据离散时间导数获得
\(dB_x/dt,dB_y/dt,dB_z/dt\)。启用方式为：

```text
--formulation=h
--magnetic-output-contract=magnetic6
```

### 14.3 生产运行示例

```bash
python dolfinx/sotem_pipeline.py \
  --workdir=output/dolfinx_magnetic6 \
  --formulation=e \
  --magnetic-output-contract=magnetic6 \
  --magnetic-receiver-mode=biot_current \
  --magnetic-dbdt-mode=curl \
  --ramp-off-time=5e-6 \
  --time-origin=after_ramp \
  --source-current=10.0 \
  --nedelec-order=2
```

运行完成或部分完成时，程序写出：

```text
magnetic6_numerical.npz
```

主要字段包括：

```text
times
data
components
units
receiver_locations
coordinate_system
time_origin
ramp_off_time
source_current
nedelec_order
formulation
magnetic_receiver_mode
magnetic_dbdt_mode
magnetic_output_contract
```

标准数据形状为：

```text
times: (n_times,)
data:  (n_times, n_locations, 6)
```

单测点生产管线通常写为：

```text
data: (n_times, 1, 6)
```

## 15. empymod磁场六通道验证

empymod只用于均匀半空间或一维层状模型，验证对象固定为：

```text
Hx, Hy, Hz              [A/m]
dBxdt, dBydt, dBzdt     [T/s]
```

验证重点包括：

- 有限长接地源方向和积分精度；
- `z_up`与empymod深度向下坐标之间的轴矢量符号；
- \(H\)、\(B\)和\(dB/dt\)单位；
- 理想阶跃和有限关断的时间原点；
- 六个分量的误差和弱分量幅值下限；
- empymod源积分点数收敛。

### 15.1 理想阶跃关断

```bash
atem3d-validate-empymod-magnetic6 \
  examples/empymod_validation_magnetic6.yaml \
  --numerical output/magnetic6_numerical.npz \
  --depths 0 \
  --resistivities 1e8,100 \
  --srcpts 9 \
  --srcpts-audit 17 \
  --ideal-step-off \
  --dbdt-reference auto \
  --audit-tolerance 0.01 \
  --comparison-tolerance 0.05 \
  --output-dir output/empymod_magnetic6_step
```

### 15.2 5 μs线性关断

```bash
atem3d-validate-empymod-magnetic6 \
  examples/empymod_validation_magnetic6_5us.yaml \
  --numerical output/dolfinx_magnetic6/magnetic6_numerical.npz \
  --depths 0 \
  --resistivities 1e8,100 \
  --srcpts 9 \
  --srcpts-audit 17 \
  --waveform-quadrature-order 8 \
  --dbdt-reference auto \
  --audit-tolerance 0.01 \
  --comparison-tolerance 0.05 \
  --output-dir output/empymod_magnetic6_5us
```

也可用命令行指定线性关断时间：

```text
--ramp-off-time 5e-6
```

有限关断响应通过延迟理想阶跃响应的叠加计算：

\[
\mathbf d_{\mathrm{wave}}(t)=
\int_{-T_{\mathrm{off}}}^{0}
\left[-\frac{\mathrm d i(\tau)}{\mathrm d\tau}\right]
\mathbf d_{\mathrm{step}}(t-\tau)\,\mathrm d\tau.
\]

其中\(t=0\)定义为电流完全关断的时刻。每个分段采用Gauss–Legendre求积，
并检查求积权重是否满足总电流变化守恒。

### 15.3 实测波形CSV

通过：

```text
--waveform-csv path/to/transmitter_current.csv
```

读取实测或标定波形。推荐格式为：

```csv
time_s,current_scale
0.0,1.0
1.0e-6,0.92
2.0e-6,0.70
3.0e-6,0.42
4.0e-6,0.15
5.0e-6,0.0
```

程序将最后一个波形时刻平移为\(t=0\)，以保证数值结果与empymod参考使用
相同的“关断结束后时间”定义。

### 15.4 dB/dt参考路径

项目兼容`empymod>=2.5.4,<3`。

- empymod 2.5.4及以上可使用`mrec=True, signal=0`的H冲激响应，并按
  \(dB/dt=-\mu_0H_{\mathrm{impulse}}\)构造参考；
- empymod 2.6及以上还可使用`mrec="b", signal=-1`直接获得原生
  \(dB/dt\)；
- `--dbdt-reference auto`会根据版本选择主路径；
- 原生路径可用时，程序默认对两条路径进行交叉审计。

### 15.5 验证产物

主要输出包括：

```text
magnetic6_comparison.csv
magnetic6_error_summary.json
empymod_dbdt_crosscheck.json
empymod_waveform_reference.json
empymod_magnetic6_reference.npz
magnetic6_H3_comparison.png
magnetic6_dBdt3_comparison.png
empymod_source_quadrature_audit.json
```

详细说明见：

```text
docs/empymod_magnetic6_validation.md
```

## 16. 与无人机三分量感应线圈的边界

`magnetic6_numerical.npz`与empymod六通道基准比较的是点位场量：

```text
H [A/m]
dB/dt [T/s]
```

真实感应线圈直接输出电压：

\[
u_i(t)=-N_i\frac{\mathrm d}{\mathrm dt}
\int_{A_i}\mathbf B(\mathbf r,t)\cdot\mathbf n_i\,\mathrm dA.
\]

因此，点位六通道验证通过后，工程数据链仍需单独处理：

- 三轴线圈有效面积和匝数；
- 三个线圈法向和非正交矩阵；
- 无人机姿态与坐标旋转；
- 有限面积积分；
- 前置放大器、滤波器和接收机复传递函数；
- 由线圈电压恢复\(dB/dt\)和\(H\)的处理链；
- 航高、吊舱摆动、同步误差和环境噪声。

点位层状基准通过，不代表真实无人机线圈系统已经完成绝对标定，也不代表
三维坝体中的百分之一弱异常已经达到现场可检测水平。
