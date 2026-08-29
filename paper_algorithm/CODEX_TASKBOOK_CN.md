# Codex 算法论文总任务书

## 1. 项目目标

在分支 `paper/algorithm-forward-matrix` 上完成一篇可审阅的算法论文初稿，并形成可重复运行的正演、后处理、图件、表格和证据索引。

论文题目暂定为：

> A validated second-order edge-element framework for three-dimensional grounded-source transient electromagnetic modelling with induced polarization, realistic transmitter waveforms and finite-area receivers

中文题目暂定为：

> 考虑激发极化、真实发射波形与有限面积接收器的三维接地电性源瞬变电磁二阶边元正演方法

论文以三分量 `dB/dt` 为主要磁观测量，以三分量 `H` 为独立磁场恢复和数值闭合审计量，以 `Ex`、`Ey` 为既有 SOTEM 基准对照量。

---

## 2. 最终交付物

Codex 完成以下交付物：

```text
paper_algorithm/
├── CODEX_TASKBOOK_CN.md
├── CODEX_MASTER_PROMPT_CN.md
├── README_CN.md
├── run_algorithm_paper.sh
├── run_ip_debye_sweep.py
├── run_3d_channel_demo.py
├── state/
│   ├── task_state.json
│   ├── environment.json
│   ├── hardware.md
│   ├── commit.txt
│   └── run_manifest.csv
├── configs/
│   ├── paper_cases.yaml
│   ├── figure_manifest.yaml
│   └── acceptance_thresholds.yaml
├── postprocess/
│   ├── collect_case_results.py
│   ├── build_error_tables.py
│   ├── build_runtime_tables.py
│   ├── build_convergence_figures.py
│   ├── build_magnetic_figures.py
│   ├── build_waveform_figures.py
│   ├── build_receiver_figures.py
│   ├── build_ip_figures.py
│   ├── build_3d_demo_figures.py
│   ├── build_closure_error.py
│   ├── build_model_schematics.py
│   └── verify_manuscript_numbers.py
├── manuscript/
│   ├── draft_v0_1_cn.md
│   ├── abstract_cn.md
│   ├── abstract_en.md
│   ├── figure_captions_cn.md
│   ├── table_captions_cn.md
│   ├── references.bib
│   ├── results_traceability.csv
│   └── figures/
├── tables/
│   ├── table_model_parameters.csv
│   ├── table_validation_errors.csv
│   ├── table_convergence.csv
│   ├── table_waveform_effects.csv
│   ├── table_receiver_effects.csv
│   ├── table_ip_debye.csv
│   └── table_runtime_memory.csv
└── qa/
    ├── qa_report.md
    ├── unit_consistency.csv
    ├── figure_audit.csv
    ├── parameter_audit.csv
    └── manuscript_number_audit.csv

generated/paper_algorithm/
├── runner.log
├── env_check/
├── preflight_lei/
├── v1_lei_noip_*/
├── v2_zhou_noip_*/
├── v3_magnetic6_*/
├── v4_receiver_*/
├── v5_ip_debye_sweep/
└── v6_3d_channel_demo/
```

所有图件同时输出 `PDF` 与 `PNG`。正文优先使用矢量 `PDF`，审阅包使用 `PNG`。

---

## 3. 执行管理规范

### 3.1 分支与提交

1. 切换到 `paper/algorithm-forward-matrix`。
2. 记录当前提交 SHA 到 `paper_algorithm/state/commit.txt`。
3. 每个任务阶段完成一次独立提交。
4. 提交信息采用：

```text
paper-algorithm: T00 freeze environment
paper-algorithm: T01 map equations to code
paper-algorithm: T02 run homogeneous benchmark
...
paper-algorithm: T12 write Chinese draft v0.1
paper-algorithm: T13 complete QA package
```

### 3.2 状态文件

创建 `paper_algorithm/state/task_state.json`，格式为：

```json
{
  "project": "algorithm-paper-v0.1",
  "branch": "paper/algorithm-forward-matrix",
  "commit": "<sha>",
  "tasks": {
    "T00": {"status": "pending", "artifacts": []},
    "T01": {"status": "pending", "artifacts": []}
  }
}
```

每个任务状态采用：

```text
pending -> running -> completed -> reviewed
```

每次更新同时记录开始时间、结束时间、命令、输出目录和提交 SHA。

### 3.3 运行记录

每个生产正演目录保存：

```text
run.log
verification_report.txt 或 validation_summary.json
resolved_config.json 或等价配置记录
环境版本
模型参数
网格参数
时间步参数
接收器定义
源定义
求解器参数
峰值内存
运行时间
输出文件哈希
Git commit SHA
```

`paper_algorithm/state/run_manifest.csv` 每行对应一个正演，字段固定为：

```text
case_id,stage,command,workdir,commit,status,start_time,end_time,
mesh_cells,dofs,time_steps,wall_seconds,max_rss_gb,output_hash
```

---

# 4. 严格任务流程

## T00 环境冻结与基础测试

### 任务

1. 激活 `fenicsx` 环境。
2. 执行 editable 安装。
3. 记录 Python、DOLFINx、PETSc、MPI、Gmsh、empymod、SimPEG、NumPy、SciPy 和 Matplotlib 版本。
4. 记录 CPU、内存、操作系统、MPI 进程数和 PETSc 配置。
5. 执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh env
```

6. 执行全量自包含测试：

```bash
python -m pytest -q
```

7. 将环境和测试结果写入：

```text
paper_algorithm/state/environment.json
paper_algorithm/state/hardware.md
paper_algorithm/qa/pytest.log
```

### 完成条件

- 核心模块可导入。
- 重点测试和全量测试均有完整日志。
- 环境信息和提交 SHA 已冻结。

---

## T01 方程—代码映射与论文方法初稿

### 任务

1. 阅读并映射以下模块：

```text
dolfinx/sotem_pipeline.py
src/atem3d/sources.py
src/atem3d/materials/cole_cole.py
src/atem3d/materials/prony.py
src/atem3d/receivers.py
src/atem3d/receiver_groups.py
src/atem3d/magnetic_recovery.py
src/atem3d/empymod_magnetic6.py
src/atem3d/empymod_waveform.py
src/atem3d/metrics.py
```

2. 创建 `paper_algorithm/manuscript/method_code_map.md`，逐项写明：

```text
论文方程编号
物理含义
离散形式
代码模块
函数或类
输入
输出
验证算例
```

3. 完成论文第 2 章和第 3 章初稿：

```text
2 Governing equations and constitutive relations
2.1 Quasi-static Maxwell equations
2.2 Ohmic and Cole–Cole media
2.3 Debye/Prony approximation
2.4 Initial conditions and source waveforms

3 Numerical formulation
3.1 Second-order Nédélec discretization
3.2 Finite grounded-wire source integration
3.3 DC initial-field solution
3.4 Implicit time stepping and memory update
3.5 Independent H and dB/dt recovery
3.6 Point and finite-area receiver operators
3.7 Linear solvers, parallelization and reproducibility
```

4. 生成 Fig. 1 算法流程图：

```text
model + mesh
    -> DC initial field
    -> source waveform
    -> edge-element time stepping
    -> Debye memory update
    -> E field
    -> curl(E) dB/dt
    -> current recovery
    -> Biot-Savart H
    -> receiver operators
    -> validation and error metrics
```

5. 生成 Fig. 2 方程—代码结构图。

### 完成条件

- 每个核心方程均对应代码实现位置。
- 第 2、3 章形成完整连续文本。
- Fig. 1 和 Fig. 2 具备可编辑源文件。

---

## T02 源、网格和内存预检

### 任务

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh preflight
```

提取并汇总：

- 源长度与方向；
- 源线段覆盖率；
- 端点电流平衡；
- 边界消元后的源残差；
- 局部网格质量；
- 最小单元质量；
- 最大长宽比；
- 预计自由度；
- 预计内存；
- 接收点局部网格。

生成 Supplementary Fig. S1：源积分与局部网格。

### 完成条件

- 源预检报告完整。
- 网格质量和内存预算进入 `run_manifest.csv`。

---

## T03 V1 均匀半空间无 IP 基准

### 模型参数

| 参数 | 数值 |
|---|---:|
| 源起点 | `(-500, 0, -0.1) m` |
| 源终点 | `(500, 0, -0.1) m` |
| 电流 | `1 A` |
| 接收点 | `(0, 800, -0.1) m` |
| 空气电阻率 | `1e8 ohm m` |
| 半空间电阻率 | `100 ohm m` |
| 时间 | `1e-5–1e-1 s`，41 个对数点 |
| 分量 | `Ex, Ey, Hz, dBzdt` |
| 参考 | empymod |

### 正演矩阵

依次运行：

```text
S0T0B0
S1T0B0
S2T0B0
S2T1B0
S2T2B0
S2T2B1
S2T2B2
```

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh convergence
```

### 后处理

1. 收集所有等级的响应与误差。
2. 计算：

```text
relative L2
relative Linf
absolute Linf
peak-normalized error
robust relative error
source quadrature audit
zero-crossing time
```

3. 分离空间、时间和边界收敛序列。
4. 生成：

- Fig. 3：V1 四分量与 empymod 对比；
- Fig. 4：空间—时间—边界收敛；
- Table 1：V1 参数；
- Table 2：各等级误差、自由度、时间和内存。

### 预设门槛

```text
参考有限源积分审计 <= 0.5%
主要有效时间窗峰值归一化误差 <= 5%
空间、时间和边界加密序列均形成误差演化记录
```

门槛外条目进入诊断任务，Codex 定位到源积分、网格、时间步、边界或接收算子，并形成独立诊断提交。

---

## T04 V2 薄导层无 IP 基准

### 模型参数

| 参数 | 数值 |
|---|---:|
| 源 | `(-500,0,-0.1) -> (500,0,-0.1) m` |
| 电流 | `10 A` |
| 接收点 | `(0,1000,-0.1) m` |
| 第 1 层 | `0–500 m, 100 ohm m` |
| 第 2 层 | `500–520 m, 10 ohm m` |
| 第 3 层 | `>520 m, 200 ohm m` |
| 时间 | `1e-4–3 s`，101 个对数点 |
| 分量 | `Ex, Ey, Hz, dBzdt` |

### 运行

运行 `S0T0B0` 作为 pilot，运行 `S2T2B2` 作为正式结果。

### 后处理

1. 计算全时间窗误差。
2. 标记早期、中期和晚期时间窗。
3. 提取响应极值和过零时间。
4. 生成：

- Fig. 5：薄导层四分量及误差；
- Table 3：分时间窗误差；
- Supplementary Fig. S2：晚期边界敏感性。

### 完成条件

- 正式等级结果与 empymod 对比完成。
- 晚期误差来源形成量化说明。

---

## T05 V3 三分量 dB/dt 与辅助 H 恢复验证

### 模型参数

| 参数 | 数值 |
|---|---:|
| 源起点 | `(-20,-7,-0.1) m` |
| 源终点 | `(20,7,-0.1) m` |
| 源长 | `42.37924020083418 m` |
| 电流 | `1 A` |
| 接收点 | `(13,31,-0.2) m` |
| 平行偏移 | `24.96505352588116 m` |
| 半空间 | `100 ohm m` |
| 空气 | `1e8 ohm m` |
| 时间 | `1e-5–1e-2 s`，31 个对数点 |
| 主分量 | `dBxdt, dBydt, dBzdt` |
| 审计分量 | `Hx, Hy, Hz` |

### 运行

```bash
bash paper_algorithm/run_algorithm_paper.sh magnetic6
```

### 后处理

1. 比较三分量 `dB/dt` 与 empymod。
2. 比较三分量 `H` 与 empymod。
3. 计算 Maxwell 数值闭合：

\[
(dB_j/dt)_H^n=\mu_0\frac{H_j^n-H_j^{n-1}}{t_n-t_{n-1}},
\]

\[
(dB_j/dt)_E^n=-(\nabla\times E^n)_j.
\]

4. 定义：

\[
\varepsilon_{closure,j}=
\frac{\|(dB_j/dt)_H-(dB_j/dt)_E\|_2}
{\max(\|(dB_j/dt)_E\|_2,d_{floor,j})}.
\]

5. 创建 `paper_algorithm/postprocess/build_closure_error.py`。
6. 生成：

- Fig. 6：三分量 `dB/dt` 严格验证；
- Fig. 7：三分量 H 恢复与闭合误差；
- Table 4：六项观测量误差和闭合误差；
- Supplementary Fig. S3：Biot–Savart 求积阶数 4、6、8、10 收敛。

### 完成条件

- 三分量 `dB/dt` 和三分量 H 均完成独立参考比较。
- 闭合误差按分量和时间窗汇总。

---

## T06 V4 真实关断波形

### 波形组

```text
W0: ideal step-off, t_off = 0 us
W1: linear ramp-off, t_off = 5 us
W2: linear ramp-off, t_off = 20 us
```

沿用 V3 几何和介质。

### 运行

```bash
bash paper_algorithm/run_algorithm_paper.sh waveform
```

### 后处理

1. 绘制三种电流波形。
2. 比较三分量 `dB/dt`。
3. 计算：

\[
\delta_{ramp,j}(t)=
\frac{d_{ramp,j}(t)-d_{step,j}(t)}
{\max_t|d_{step,j}(t)|}.
\]

4. 对每个分量提取 1%、5% 和 10% 差异对应的最早稳定时间。
5. 生成：

- Fig. 8：波形和三分量响应；
- Fig. 9：关断效应时—分量热图；
- Table 5：关断时间影响指标。

### 完成条件

- 三种波形共享同一时间原点和单位。
- 波形求积阶数和权重守恒记录进入结果索引。

---

## T07 V5 有限面积接收器

### 模型参数

沿用 V3 的 5 us 线性关断模型。

### 接收器组

```text
R0: point
R1: disk_average, radius = 0.5 m
R2: disk_average, radius = 1.0 m
R3: disk_average, radius = 2.0 m
R4: disk_average, radius = 4.0 m
```

### 运行

```bash
bash paper_algorithm/run_algorithm_paper.sh receiver
```

### 后处理

1. 提取点接收和圆盘平均响应。
2. 计算：

\[
\delta_{coil,j}(t)=
\frac{d_{disk,j}(t)-d_{point,j}(t)}
{\max_t|d_{point,j}(t)|}.
\]

3. 计算圆盘半径与最大面积平均差异的关系。
4. 生成：

- Fig. 10：点接收器与有限圆盘响应；
- Fig. 11：半径—时间—分量差异图；
- Table 6：有限面积接收器效应。

### 完成条件

- 每个半径均有完整响应、差值和运行记录。
- 正文形成点观测近似适用范围的定量描述。

---

## T08 V6 Cole–Cole/IP 与 Debye 项数扫描

### 模型参数

| 参数 | 数值 |
|---|---:|
| 源 | `(-500,0,-0.1) -> (500,0,-0.1) m` |
| 电流 | `10 A` |
| 接收点 | `(0,1000,-0.1) m` |
| 层状模型 | `100 / 10 / 200 ohm m` |
| 界面 | `500 m, 520 m` |
| 极化层 | `500–520 m` |
| rho0 | `10 ohm m` |
| m | `0.1` |
| tau | `1 s` |
| c | `0.3` |
| 时间 | `1e-4–3 s`，101 点 |

### Debye 项数

```text
K = 8, 12, 16, 20
```

### 运行

```bash
bash paper_algorithm/run_algorithm_paper.sh ip
```

正式等级采用：

```text
IP_LEVEL=S1T1B1
```

### 后处理

1. 绘制 exact Cole–Cole 与各 K 值 Debye 拟合曲线。
2. 计算频率域复电导率拟合误差。
3. 绘制无 IP、IP 总场和 IP 增量：

\[
\Delta d_{IP}=d_{IP}-d_{noIP}.
\]

4. 计算各 K 值的时域误差、过零时间、运行时间和峰值内存。
5. 生成：

- Fig. 12：Cole–Cole–Debye 频率域拟合；
- Fig. 13：IP 总场与 exact 参考；
- Fig. 14：IP 增量与 K 收敛；
- Fig. 15：精度—时间—内存关系；
- Table 7：Debye 拟合和时域误差；
- Table 8：计算成本。

### 预设门槛

```text
频率域拟合最大相对误差 <= 1%
正式 K 值在主要时间窗形成稳定时域响应
K 增长对应的精度、内存和时间形成完整量化关系
```

### 完成条件

- 选出一个正式 K 值并写入论文方法参数。
- 选择依据同时包含精度、内存和运行时间。

---

## T09 V7 三维曲折导电体展示

### 模型参数

| 参数 | 数值 |
|---|---:|
| 计算域 | `x: -600–600 m` |
|  | `y: -350–350 m` |
|  | `z: -350–0 m` |
| 背景电阻率 | `100 ohm m` |
| 通道电阻率 | `10 ohm m` |
| 通道半径 | `40 m` |
| 源 | `(-500,200,-0.1) -> (500,200,-0.1) m` |
| 电流 | `10 A` |
| 关断时间 | `5 us` |
| 接收点 | `(-150,-300,-0.1)` |
|  | `(0,-300,-0.1)` |
|  | `(150,-300,-0.1)` |
| 时间 | `1e-5–1e-1 s`，31 点 |
| 分量 | `Ex, Ey, dBzdt` |

通道控制点：

```text
(-350,-100,-60)
(-120,-30,-120)
(80,60,-180)
(260,-40,-100)
```

### 网格

Pilot：

```text
coarse = 20 x 12 x 6
fine   = 30 x 18 x 9
```

Full：

```text
coarse = 40 x 24 x 12
fine   = 60 x 36 x 18
```

### 运行

```bash
DEMO3D_PROFILE=pilot bash paper_algorithm/run_algorithm_paper.sh demo3d
DEMO3D_PROFILE=full FORCE=1 bash paper_algorithm/run_algorithm_paper.sh demo3d
```

### 后处理

1. 绘制三维模型、源、通道和接收点。
2. 绘制 coarse 和 fine 网格切片。
3. 计算总场和异常场：

\[
\Delta d=d_{3D}-d_{background}.
\]

4. 计算 coarse–fine 差异：

\[
\varepsilon_{cf}=
\frac{\|d_{coarse}-d_{fine}\|_2}{\|d_{fine}\|_2}.
\]

5. 计算异常—数值差异比：

\[
R_{anomaly}=
\frac{\|d_{fine}-d_{background}\|_2}
{\|d_{coarse}-d_{fine}\|_2}.
\]

6. 生成：

- Fig. 16：三维曲折导电体和观测系统；
- Fig. 17：粗细网格切片；
- Fig. 18：三个接收点的总场和异常场；
- Fig. 19：coarse–fine 收敛和异常—数值差异比；
- Table 9：三维模型参数和数值误差。

### 完成条件

- 三维模型图、总场、异常场和粗细网格对比齐全。
- 三维展示结论均关联数值收敛证据。

---

## T10 统一结果收集和图表生成

### 任务

1. 创建 `collect_case_results.py`，遍历 `generated/paper_algorithm/`。
2. 建立统一结果数据表：

```text
case_id
model_family
source_geometry
receiver_geometry
waveform
material
mesh_level
time_level
boundary_level
components
error_metrics
runtime
memory
artifact_paths
commit
```

3. 创建所有表格 CSV。
4. 创建 `figure_manifest.yaml`，每张图记录：

```text
figure_id
source_cases
source_files
script
panels
x_axis
y_axis
units
caption_file
output_pdf
output_png
```

5. 图件规范：

```text
单栏宽 85 mm
双栏宽 178 mm
正文 PDF 为矢量格式
PNG 为 600 dpi
字体统一
坐标轴包含量和单位
对数轴明确标记
分量顺序统一
时间原点统一为关断结束后
```

6. 生成 `paper_algorithm/manuscript/figures/` 全部正文图和补充图。

### 完成条件

- 每张图可由单一脚本从原始正演结果重建。
- 每张图均进入图件清单和审计表。

---

## T11 论文表格与结果追溯

### 任务

1. 创建参数表、误差表、收敛表、波形表、接收器表、IP表和计算成本表。
2. 创建 `results_traceability.csv`，字段为：

```text
manuscript_section
sentence_id
reported_value
unit
case_id
source_file
source_column
processing_script
figure_or_table
commit
```

3. 所有正文定量结论登记到追溯表。
4. 生成 `parameter_audit.csv`，比对：

```text
正文参数
表格参数
运行配置参数
输出元数据参数
```

### 完成条件

- 正文每个数值均可定位到原始输出。
- 模型参数在配置、表格和正文之间一致。

---

## T12 完成中文初稿 v0.1

### 文件

```text
paper_algorithm/manuscript/draft_v0_1_cn.md
```

### 初稿结构

```text
Title
Abstract
Keywords
1 Introduction
2 Governing equations and constitutive relations
3 Numerical formulation
4 Validation models and numerical settings
5 Results
  5.1 Homogeneous half-space benchmark
  5.2 Thin conductive layer benchmark
  5.3 Three-component dB/dt and independent H recovery
  5.4 Realistic transmitter waveform effects
  5.5 Finite-area receiver effects
  5.6 Cole–Cole/IP and Debye-term convergence
  5.7 Three-dimensional tortuous conductive-body demonstration
6 Discussion
  6.1 Accuracy and dominant numerical error sources
  6.2 Value of independent magnetic recovery
  6.3 Waveform and receiver modelling implications
  6.4 Computational cost and reproducibility
7 Conclusions
Code and data availability
Author contributions
Acknowledgements
References
Figure captions
Table captions
```

### 字数与内容

- 中文正文：约 12,000–18,000 字。
- 摘要：600–900 字。
- 英文摘要：300–450 词。
- 结论：4–6 条定量结论。
- 每个结果小节依次写：模型、指标、主要数值、图表引用、物理或数值解释。
- Discussion 将结果上升到数值方法层面。
- 每张正文图在正文中至少被引用一次。
- 每个表格在正文中至少被引用一次。
- 参考文献写入 `references.bib`，包含 DOI、期刊、卷页或文章号。

### 初稿生成顺序

1. 写第 2、3、4 章。
2. 根据结果表写第 5 章。
3. 根据第 5 章写第 6、7 章。
4. 写 Introduction。
5. 写 Abstract、Keywords 和题目。
6. 写图题、表题、代码与数据可得性声明。
7. 运行数值追溯审计。

### 完成条件

- 初稿形成完整论文叙事。
- 所有定量结论进入追溯表。
- 所有图表编号连续。
- 所有方程符号在首次出现处定义。

---

## T13 初稿质量审计

### 任务

1. 执行全量测试。
2. 检查：

```text
坐标系
源电流方向
时间原点
H/B/dBdt 定义
单位
接收器法向
Cole–Cole 参数约定
Debye 项数
网格等级
边界等级
误差 floor
图表数字
正文数字
```

3. 创建：

```text
paper_algorithm/qa/qa_report.md
paper_algorithm/qa/unit_consistency.csv
paper_algorithm/qa/figure_audit.csv
paper_algorithm/qa/parameter_audit.csv
paper_algorithm/qa/manuscript_number_audit.csv
```

4. 生成审阅包：

```text
paper_algorithm/review_package_v0_1/
├── draft_v0_1_cn.md
├── abstract_en.md
├── figures/
├── tables/
├── qa_report.md
├── run_manifest.csv
└── results_traceability.csv
```

5. 提交最终初稿版本：

```text
paper-algorithm: T13 complete draft v0.1 review package
```

### 完成条件

- QA 报告覆盖代码、结果、图件、表格和正文。
- 审阅包可独立阅读。
- 初稿版本号、提交 SHA 和运行结果相互对应。

---

# 5. Codex 每轮工作格式

每轮开始输出：

```text
当前任务：Txx
输入文件：...
计划修改：...
计划运行：...
预期产物：...
```

每轮结束输出：

```text
任务状态：completed / diagnostic
修改文件：...
运行命令：...
结果目录：...
关键指标：...
生成图表：...
提交 SHA：...
下一任务：...
```

每轮同步更新 `task_state.json` 和 `run_manifest.csv`。

---

# 6. 推荐执行命令顺序

```bash
conda activate fenicsx
git checkout paper/algorithm-forward-matrix
python -m pip install -e .

bash paper_algorithm/run_algorithm_paper.sh env
bash paper_algorithm/run_algorithm_paper.sh preflight
bash paper_algorithm/run_algorithm_paper.sh benchmark-pilot
bash paper_algorithm/run_algorithm_paper.sh magnetic6
bash paper_algorithm/run_algorithm_paper.sh convergence
bash paper_algorithm/run_algorithm_paper.sh waveform
bash paper_algorithm/run_algorithm_paper.sh receiver
bash paper_algorithm/run_algorithm_paper.sh ip
DEMO3D_PROFILE=pilot bash paper_algorithm/run_algorithm_paper.sh demo3d
DEMO3D_PROFILE=full FORCE=1 bash paper_algorithm/run_algorithm_paper.sh demo3d
```

随后执行统一后处理、表格、图件、初稿和 QA 阶段。

---

# 7. 初稿完成判据

初稿 v0.1 同时具备：

1. 完整第 1–7 章；
2. 19 张以内正文候选图，经编辑后选择 10–12 张进入正文；
3. 9 张以内正文候选表，经编辑后选择 5–7 张进入正文；
4. 均匀半空间、薄导层、三分量磁观测、真实波形、有限面积接收器、IP 和三维模型七类证据；
5. 每个主要数值对应原始结果路径；
6. 每个生产结果对应提交 SHA、配置、日志、运行时间和内存；
7. 完整 QA 报告和审阅包。
