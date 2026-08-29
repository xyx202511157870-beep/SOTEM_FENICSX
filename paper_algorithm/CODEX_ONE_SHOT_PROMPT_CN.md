# Codex 单次执行总提示词：SOTEM_FENICSX 算法论文 v0.1

你正在 GitHub 仓库：

```text
xyx202511157870-beep/SOTEM_FENICSX
```

目标分支：

```text
paper/algorithm-forward-matrix
```

## 总目标

从环境冻结开始，完成一篇可审阅的中文算法论文初稿 v0.1，并完成全部正演、复杂模型绘制、结果图、表格、运行记录、结果追溯和 QA 审阅包。

论文暂定英文题目：

> A validated second-order edge-element framework for three-dimensional grounded-source transient electromagnetic modelling with induced polarization, realistic transmitter waveforms and finite-area receivers

中文题目：

> 考虑激发极化、真实发射波形与有限面积接收器的三维接地电性源瞬变电磁二阶边元正演方法

论文以：

```text
dBxdt, dBydt, dBzdt
```

作为主要磁观测量；以：

```text
Hx, Hy, Hz
```

作为独立磁场恢复与 Maxwell 闭合审计量；以：

```text
Ex, Ey
```

作为 SOTEM 文献基准对照量。

正式空间离散使用二阶：

```text
N1curl(2)
```

正式时间推进使用后向 Euler。均匀半空间与层状介质使用 empymod 独立参考；三维异常使用 DOLFINx coarse–fine 收敛作为数值证据。

---

# 一、执行规则

1. 先读取：

```text
paper_algorithm/CODEX_TASKBOOK_CN.md
paper_algorithm/CODEX_MASTER_PROMPT_CN.md
paper_algorithm/README_CN.md
paper_algorithm/CODEX_V8_COMPLEX_DAM_TASK_CN.md
paper_algorithm/configs/v8_complex_dam.yaml
paper_algorithm/manuscript/draft_v0_1_cn.md
paper_algorithm/run_algorithm_paper.sh
README.md
```

2. 切换到目标分支并记录当前提交：

```bash
git fetch origin
git checkout paper/algorithm-forward-matrix
git rev-parse HEAD
```

3. 创建并持续更新：

```text
paper_algorithm/state/task_state.json
paper_algorithm/state/run_manifest.csv
paper_algorithm/state/environment.json
paper_algorithm/state/hardware.md
paper_algorithm/state/commit.txt
```

4. 按 T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08 → T09 → T09B → T10 → T11 → T12 → T13 的顺序执行。

5. 每个阶段依次完成：

```text
读取配置
→ 检查已有实现
→ 编写或修改代码
→ 单元测试
→ 运行 pilot
→ 检查日志和数值指标
→ 运行正式模型
→ 后处理
→ 生成图表
→ 更新论文
→ 更新状态与运行清单
→ Git提交
```

6. 遇到失败、超内存、数值不收敛、符号异常或误差超门槛时，定位源积分、网格、边界、时间步、材料、接收器、坐标或求解器问题，修复后重新运行，并保存诊断日志。

7. 每个正演目录保存：

```text
run.log
resolved_config.json 或等价配置
Git commit SHA
软件版本
源定义
接收器定义
材料参数
网格参数
时间步参数
PETSc参数
运行时间
峰值内存
单元数
自由度
输出文件哈希
```

8. 每个正文定量结果必须登记到：

```text
paper_algorithm/manuscript/results_traceability.csv
```

9. 每张图由独立 Python 后处理脚本从原始输出重建，同时保存 PDF 和 PNG。

10. 最终中文初稿中的所有数值占位符均由真实正演与后处理结果替换。

11. 每个任务独立提交，提交信息使用：

```text
paper-algorithm: T00 freeze environment
paper-algorithm: T01 map equations to code
paper-algorithm: T02 complete source and mesh preflight
paper-algorithm: T03 validate homogeneous halfspace
paper-algorithm: T04 validate layered model
paper-algorithm: T05 validate three-component magnetic observables
paper-algorithm: T06 quantify transmitter waveform effects
paper-algorithm: T07 quantify finite-area receiver effects
paper-algorithm: T08 validate Cole-Cole Debye modelling
paper-algorithm: T09 complete generic 3D demonstration
paper-algorithm: T09B complete complex dam demonstration
paper-algorithm: T10 build figures and tables
paper-algorithm: T11 complete result traceability
paper-algorithm: T12 write Chinese draft v0.1
paper-algorithm: T13 complete QA review package
```

---

# 二、最终交付物

形成以下目录和文件：

```text
paper_algorithm/
├── CODEX_ONE_SHOT_PROMPT_CN.md
├── CODEX_TASKBOOK_CN.md
├── CODEX_V8_COMPLEX_DAM_TASK_CN.md
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
│   ├── acceptance_thresholds.yaml
│   └── v8_complex_dam.yaml
├── postprocess/
│   ├── collect_case_results.py
│   ├── build_error_tables.py
│   ├── build_runtime_tables.py
│   ├── build_convergence_figures.py
│   ├── build_magnetic_figures.py
│   ├── build_closure_error.py
│   ├── build_waveform_figures.py
│   ├── build_receiver_figures.py
│   ├── build_ip_figures.py
│   ├── build_3d_demo_figures.py
│   ├── build_complex_dam_figures.py
│   ├── build_model_schematics.py
│   └── verify_manuscript_numbers.py
├── complex_dam/
│   ├── build_complex_dam_mesh.py
│   ├── validate_complex_dam_geometry.py
│   ├── run_complex_dam_forward.py
│   ├── postprocess_complex_dam.py
│   ├── plot_complex_dam_model.py
│   ├── plot_complex_dam_mesh.py
│   ├── plot_complex_dam_fields.py
│   ├── plot_complex_dam_responses.py
│   ├── plot_complex_dam_convergence.py
│   └── README_CN.md
├── manuscript/
│   ├── draft_v0_1_cn.md
│   ├── abstract_cn.md
│   ├── abstract_en.md
│   ├── figure_captions_cn.md
│   ├── table_captions_cn.md
│   ├── references.bib
│   ├── method_code_map.md
│   ├── results_traceability.csv
│   └── figures/
├── tables/
│   ├── table_model_parameters.csv
│   ├── table_validation_errors.csv
│   ├── table_convergence.csv
│   ├── table_waveform_effects.csv
│   ├── table_receiver_effects.csv
│   ├── table_ip_debye.csv
│   ├── table_3d_demo.csv
│   ├── table_complex_dam.csv
│   └── table_runtime_memory.csv
├── tests/
│   ├── test_grounded_polyline_source.py
│   ├── test_complex_dam_geometry.py
│   ├── test_complex_dam_material_tags.py
│   └── test_complex_dam_receiver_array.py
├── qa/
│   ├── qa_report.md
│   ├── unit_consistency.csv
│   ├── figure_audit.csv
│   ├── parameter_audit.csv
│   └── manuscript_number_audit.csv
└── review_package_v0_1/

generated/paper_algorithm/
├── runner.log
├── env_check/
├── preflight_lei/
├── v1_lei_noip_*/
├── v2_zhou_noip_*/
├── v3_magnetic6_*/
├── v4_receiver_*/
├── v5_ip_debye_sweep/
├── v6_3d_channel_demo/
└── v8_complex_dam/
```

---

# 三、启动命令

执行：

```bash
conda activate fenicsx
git checkout paper/algorithm-forward-matrix
python -m pip install -e .
```

设置：

```bash
export OUTPUT_ROOT="$PWD/generated/paper_algorithm"
export MEMORY_LIMIT_GB=32
```

根据机器内存调整 `MEMORY_LIMIT_GB`。

---

# 四、T00 环境冻结与测试

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh env
python -m pytest -q | tee paper_algorithm/qa/pytest.log
```

记录：

```text
Python
DOLFINx
PETSc
petsc4py
MPI
mpi4py
Gmsh
empymod
SimPEG
NumPy
SciPy
Matplotlib
操作系统
CPU
物理核心与逻辑核心
内存
MPI进程数
Git SHA
```

生成环境 JSON、硬件说明、测试日志和 T00 状态记录。

---

# 五、T01 方程—代码映射与方法初稿

阅读：

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

创建 `method_code_map.md`，字段为：

```text
方程编号
物理意义
离散形式
代码模块
函数或类
输入
输出
验证算例
```

完成论文：

```text
2 控制方程与本构关系
2.1 准静态 Maxwell 方程
2.2 欧姆与 Cole–Cole 介质
2.3 Debye/Prony 近似
2.4 初始条件与发射波形

3 数值方法
3.1 二阶 Nédélec 边元
3.2 有限接地导线源线积分
3.3 直流初场
3.4 隐式时间推进与记忆变量更新
3.5 dB/dt 与 H 的独立恢复
3.6 点接收与有限面积接收
3.7 线性求解器、并行和可复现性
```

生成：

```text
Fig. 1 算法总流程图
Fig. 2 方程—代码结构图
```

Fig. 1 流程：

```text
模型与网格
→ DC初场
→ 发射波形
→ 二阶边元时间推进
→ Debye记忆变量
→ E场
→ curl(E)得到dB/dt
→ 电流恢复
→ Biot–Savart得到H
→ 点/有限面积接收算子
→ 独立参考与误差统计
```

---

# 六、T02 源、网格和内存预检

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh preflight
```

提取：

```text
源长度
源方向
线段单元覆盖率
端点电流平衡
边界消元后源残差
局部网格质量
最小单元质量
最大长宽比
预计单元数
预计自由度
预计峰值内存
接收点局部网格
```

生成：

```text
Supplementary Fig. S1 源路径、局部网格与源积分审计
```

---

# 七、T03 V1 均匀半空间无 IP 基准

模型参数：

```text
源起点      (-500, 0, -0.1) m
源终点      (500, 0, -0.1) m
电流        1 A
接收点      (0, 800, -0.1) m
空气        1e8 ohm m
半空间      100 ohm m
时间        1e-5–1e-1 s
时间道      41个对数点
分量        Ex, Ey, Hz, dBzdt
参考        empymod
```

运行收敛矩阵：

```text
S0T0B0
S1T0B0
S2T0B0
S2T1B0
S2T2B0
S2T2B1
S2T2B2
```

等级定义：

```text
Level 0: 源网格40 m，接收网格20 m，时间子步1，外边界25 km
Level 1: 源网格20 m，接收网格10 m，时间子步2，外边界50 km
Level 2: 源网格10 m，接收网格5 m，时间子步4，外边界100 km
```

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh convergence
```

计算：

```text
relative L2
relative Linf
absolute Linf
peak-normalized error
robust relative error
source quadrature audit
zero-crossing time
wall time
peak memory
DOFs
```

生成：

```text
Fig. 3 V1四分量与empymod比较
Fig. 4 空间、时间、边界收敛
Table 1 V1模型参数
Table 2 各等级误差、自由度、运行时间和内存
```

验收目标：

```text
有限源参考积分审计 <= 0.5%
主要有效时间窗峰值归一化误差 <= 5%
空间、时间和边界加密形成量化误差序列
```

---

# 八、T04 V2 薄导层无 IP 基准

模型参数：

```text
源          (-500,0,-0.1) → (500,0,-0.1) m
电流        10 A
接收点      (0,1000,-0.1) m
第1层       0–500 m，100 ohm m
第2层       500–520 m，10 ohm m
第3层       >520 m，200 ohm m
时间        1e-4–3 s
时间道      101个对数点
分量        Ex, Ey, Hz, dBzdt
参考        empymod
```

运行：

```text
S0T0B0 pilot
S2T2B2 formal
```

提取：

```text
早期、中期、晚期误差
响应极值时间
过零时间
晚期边界误差
```

生成：

```text
Fig. 5 薄导层响应与误差
Table 3 分时间窗误差
Supplementary Fig. S2 晚期边界敏感性
```

---

# 九、T05 V3 三分量 dB/dt 与辅助 H 验证

模型参数：

```text
源起点      (-20,-7,-0.1) m
源终点      (20,7,-0.1) m
源长        42.37924020083418 m
电流        1 A
接收点      (13,31,-0.2) m
平行偏移    24.96505352588116 m
空气        1e8 ohm m
半空间      100 ohm m
时间        1e-5–1e-2 s
时间道      31个对数点
主分量      dBxdt, dBydt, dBzdt
审计分量    Hx, Hy, Hz
```

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh magnetic6
```

创建：

```text
paper_algorithm/postprocess/build_closure_error.py
```

计算：

```text
(dBj/dt)_H^n = mu0 * (Hj^n - Hj^(n-1)) / (tn - t(n-1))
(dBj/dt)_E^n = -(curl E^n)_j
```

闭合误差：

```text
epsilon_closure,j = ||(dBj/dt)_H - (dBj/dt)_E||2 /
                    max(||(dBj/dt)_E||2, floor_j)
```

生成：

```text
Fig. 6 三分量dB/dt与empymod
Fig. 7 H恢复与Maxwell闭合误差
Table 4 dB/dt、H和闭合误差
Supplementary Fig. S3 Biot–Savart求积阶数收敛
```

验收目标：

```text
主要有效时间窗dB/dt峰值归一化误差 <= 5%
H恢复在有效幅值区保持符号和数量级一致
闭合误差形成随网格和时间加密的稳定序列
```

---

# 十、T06 V4 真实关断波形

沿用 V3 几何，计算：

```text
W0 理想阶跃：0 μs
W1 线性关断：5 μs
W2 线性关断：20 μs
```

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh waveform
```

计算：

```text
delta_ramp,j(t) = [d_ramp,j(t)-d_step,j(t)] / max_t |d_step,j(t)|
```

提取：

```text
各分量最大波形差异
差异降至10%的时间
差异降至5%的时间
差异降至1%的时间
```

生成：

```text
Fig. 8 三种发射波形与三分量响应
Fig. 9 关断时间—观测时间—分量差异图
Table 5 波形效应指标
```

---

# 十一、T07 V5 有限面积接收器

采用 V3 几何和 5 μs 关断。

接收器：

```text
R0 point
R1 disk radius 0.5 m
R2 disk radius 1.0 m
R3 disk radius 2.0 m
R4 disk radius 4.0 m
```

执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh receiver
```

计算：

```text
delta_coil,j(t) = [d_disk,j(t)-d_point,j(t)] / max_t |d_point,j(t)|
```

生成：

```text
Fig. 10 点接收器与有限圆盘接收器
Fig. 11 半径—时间—分量差异图
Table 6 有限面积效应
```

---

# 十二、T08 V6 Cole–Cole/IP 验证

模型参数：

```text
源          (-500,0,0) → (500,0,0) m
电流        10 A
接收点      (0,1000,0) m
第1层       0–500 m，100 ohm m
极化层      500–520 m，rho0=10 ohm m
第3层       >520 m，200 ohm m
m           0.1
tau         1 s
c           0.3
时间        1e-4–3 s
时间道      101个对数点
Debye项数   K=8,12,16,20
```

执行：

```bash
IP_LEVEL=S1T1B1 IP_TERMS=8,12,16,20 \
  bash paper_algorithm/run_algorithm_paper.sh ip
```

计算：

```text
频域Cole–Cole与Debye拟合误差
IP总场误差
IP增量：delta_d_IP = d_IP - d_noIP
K值之间的时域差异
运行时间
峰值内存
```

生成：

```text
Fig. 12 exact Cole–Cole与Debye频域拟合
Fig. 13 IP总场与参考
Fig. 14 IP增量与K值收敛
Fig. 15 精度—运行时间—内存
Table 7 Debye拟合与时域误差
Table 8 计算成本
```

验收目标：

```text
选定频带最大拟合误差 <= 1%
正式K值由频域误差、时域误差和计算成本共同确定
```

---

# 十三、T09 V7 通用三维曲折导电体

模型参数：

```text
计算域      x=-600–600 m
            y=-350–350 m
            z=-350–0 m
背景        100 ohm m
通道        10 ohm m
通道半径    40 m
源          (-500,200,-0.1) → (500,200,-0.1) m
电流        10 A
关断        5 μs线性关断
时间        1e-5–1e-1 s，31个对数点
分量        Ex, Ey, dBzdt
接收点      (-150,-300,-0.1)
            (0,-300,-0.1)
            (150,-300,-0.1)
```

通道控制点：

```text
(-350,-100,-60)
(-120,-30,-120)
(80,60,-180)
(260,-40,-100)
```

Pilot：

```text
coarse 20×12×6
fine   30×18×9
```

Full：

```text
coarse 40×24×12
fine   60×36×18
```

执行：

```bash
DEMO3D_PROFILE=pilot \
  bash paper_algorithm/run_algorithm_paper.sh demo3d

DEMO3D_PROFILE=full FORCE=1 \
  bash paper_algorithm/run_algorithm_paper.sh demo3d
```

计算：

```text
delta_d = d_3D - d_background
error_cf = ||d_coarse-d_fine||2 / ||d_fine||2
R_anomaly = ||d_fine-d_background||2 / ||d_coarse-d_fine||2
```

生成：

```text
Fig. 16 三维曲折导电体、源和接收点
Fig. 17 coarse/fine网格
Fig. 18 三个接收点总场与异常场
Fig. 19 coarse–fine差异与异常—数值差异比
Table 9 三维模型、误差和计算成本
```

---

# 十四、T09B V8 完整复杂坝体模型

复杂坝体参数的唯一来源：

```text
paper_algorithm/configs/v8_complex_dam.yaml
```

## 14.1 坐标系

```text
x：坝轴方向
y：上游至下游方向
z：高程，向上为正
```

## 14.2 计算域

```text
x = -2000–2000 m
y = -2500–2500 m
z = -1500–1000 m
```

## 14.3 几何公式

谷地/山体高程：

```text
ground(x)=0                           , |x|<=50
ground(x)=(|x|-50)/1.5              , 50<|x|<275
ground(x)=150                        , |x|>=275
```

上游坡面：

```text
y_up(z) = -356 + 3.5 z
```

下游坡面：

```text
y_down(z) = 306 - 3 z
```

坝基：

```text
x=-500–500 m
y=-850–800 m
z=-250–0 m
```

左山体 x-z 多边形：

```text
(-500,0)
(-500,150)
(-275,150)
(-50,0)
```

右山体 x-z 多边形：

```text
(50,0)
(275,150)
(500,150)
(500,0)
```

坝体 x 截面：

```text
-200,-150,-100,-50,50,100,150,200 m
```

库水 x 截面：

```text
-177.5,-150,-100,-50,50,100,150,177.5 m
```

覆盖层 x 截面：

```text
-125,-100,-50,50,100,125 m
```

采用 Gmsh OpenCASCADE 建立：

```text
外部计算域
坝基
左右山体
多截面loft坝体
多截面loft库水
多截面loft下游覆盖层
曲折通道
空气
```

对全部材料体执行 BooleanFragments，创建稳定 physical tags：

```text
AIR
FOUNDATION_MOUNTAIN
DAM
RESERVOIR_WATER
DOWNSTREAM_COVER
LEAKAGE_CHANNEL
```

## 14.4 曲折通道

中心线：

```text
P1 = (30,-128.5,65) m
P2 = (30,0,40) m
P3 = (30,276,10) m
```

半径：

```text
5 m
```

用相邻圆柱和折点球体融合，保留为独立 physical volume。背景和通道模型使用同一网格及同一 physical tag，仅切换材料。

## 14.5 折线接地源

A 电极：

```text
(0,-240,50) m
```

B 电极：

```text
(0,330,5) m
```

外部电缆顶点按 B→A 电流方向：

```text
(0,330,5)
(-50,330,20)
(-200,330,120)
(-275,330,175)
(-350,330,190)
(-350,-240,190)
(-275,-240,175)
(-200,-240,120)
(-50,-240,120)
(0,-240,95)
(0,-240,50)
```

实现：

```python
GroundedPolylineSource(vertices, current, waveform)
```

完成测试：

```text
两顶点折线与直线源一致
多段折线等于各段代数和
顶点顺序反转后整体变号
MPI分区下每段原子区间唯一归属且覆盖率为1
```

输出源审计：

```text
总长度
每段长度
单元覆盖率
端点平衡
源投影残差
边界消元后残差
```

## 14.6 材料

```text
空气                    1e8 ohm m
坝基与山体              500 ohm m
坝体                    100 ohm m
库水                    20 ohm m
下游覆盖层              50 ohm m
活动渗漏通道            10 ohm m
所有材料                mu_r=1
```

背景模型中通道 physical volume 继承宿主材料。

## 14.7 发射波形

```text
电流          10 A
线性关断      5 μs
关断内步数    10
时间原点      关断结束
时间          1e-6–1e-1 s
时间道        51个对数点
```

## 14.8 接收阵列

三条测线：

```text
R0 z=100.3 m
R1 z=105.0 m
R2 z=110.0 m
```

每条测线：

```text
x=-150:5:150 m
y=0 m
61个接收点
```

接收类型：

```text
point
disk_average，radius=1 m，法向x/y/z
```

主要输出：

```text
dBxdt, dBydt, dBzdt
```

选定接收点审计：

```text
Hx, Hy, Hz
```

## 14.9 网格

Pilot：

```text
通道附近        2.5 m
源路径附近      5 m
接收线附近      5 m
坝体和覆盖层    15 m
山体            30 m
远区地下        120 m
空气            150 m
N1curl(2)
```

Full：

```text
通道附近        1.25 m
源路径附近      2.5 m
接收线附近      2.5 m
坝体和覆盖层    8 m
山体            20 m
远区地下        80 m
空气            100 m
N1curl(2)
```

生成网格审计：

```text
四面体数量
二阶边元自由度
各材料单元数
质量分位数
最大长宽比
通道直径单元层数
源和接收附近网格尺寸
内存预估
```

## 14.10 正演案例

运行：

```text
D0_background_pilot
D1_channel_r5_pilot
D0_background_full
D1_channel_r5_full
```

同一级别的背景和通道案例使用完全相同网格。

差分：

```text
delta_d_j(x,t) = d_channel,j(x,t) - d_background,j(x,t)
```

Pilot–Full：

```text
error_cf_total = ||d_pilot-d_full||2 / ||d_full||2
error_cf_anomaly = ||delta_d_pilot-delta_d_full||2 /
                   max(||delta_d_full||2, floor)
R_A_N = ||delta_d_full||2 / ||d_pilot-d_full||2
```

## 14.11 V8 输出图表

生成：

```text
Fig. 20a 三维材料体
Fig. 20b 上下游剖面
Fig. 20c 沿坝轴剖面
Fig. 20d 通道入口、中段和出口局部放大

Fig. 21a 全域网格
Fig. 21b 坝体剖面网格
Fig. 21c 通道附近网格
Fig. 21d 源与接收线附近网格

Fig. 22 早期、峰值和晚期三分量场快照
Fig. 23 三条接收高度的总场响应
Fig. 24 通道差分异常x-t热图
Fig. 25 Pilot–Full差异与R_A_N

Table 10 复杂坝体几何、材料、源和接收参数
Table 11 Pilot/Full网格与计算成本
Table 12 总场、异常场、航高和有限面积指标
Supplementary Fig. S8 折线源积分审计
```

代表时间自动选择：

```text
异常首次达到峰值10%的时间
异常峰值时间
异常衰减至峰值10%的时间
```

## 14.12 V8 实现入口

创建：

```text
paper_algorithm/complex_dam/build_complex_dam_mesh.py
paper_algorithm/complex_dam/validate_complex_dam_geometry.py
paper_algorithm/complex_dam/run_complex_dam_forward.py
paper_algorithm/complex_dam/postprocess_complex_dam.py
paper_algorithm/complex_dam/plot_complex_dam_model.py
paper_algorithm/complex_dam/plot_complex_dam_mesh.py
paper_algorithm/complex_dam/plot_complex_dam_fields.py
paper_algorithm/complex_dam/plot_complex_dam_responses.py
paper_algorithm/complex_dam/plot_complex_dam_convergence.py
```

在 `paper_algorithm/run_algorithm_paper.sh` 增加：

```text
complex-dam-pilot
complex-dam-full
complex-dam
```

并将 `complex-dam` 纳入 `all`。

建议命令：

```bash
bash paper_algorithm/run_algorithm_paper.sh complex-dam-pilot

MEMORY_LIMIT_GB=64 FORCE=1 \
  bash paper_algorithm/run_algorithm_paper.sh complex-dam-full
```

---

# 十五、T10 统一后处理、图和表

创建：

```text
collect_case_results.py
build_error_tables.py
build_runtime_tables.py
build_convergence_figures.py
build_magnetic_figures.py
build_closure_error.py
build_waveform_figures.py
build_receiver_figures.py
build_ip_figures.py
build_3d_demo_figures.py
build_complex_dam_figures.py
build_model_schematics.py
```

所有图：

```text
PDF矢量图
PNG 600 dpi
单栏宽85 mm
双栏宽178 mm
```

统一：

```text
时间 s
电场 V/m
H A/m
dB/dt T/s
z向上
时间零点为关断结束
分量顺序x,y,z
统一字体、字号、线型、图例和对数时间轴
```

先生成全部 Fig. 1–25 候选图，再选择 10–12 张进入正文，其余进入补充材料。

---

# 十六、T11 结果追溯

创建：

```text
paper_algorithm/manuscript/results_traceability.csv
```

字段：

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

每个正文数值、表格单元格和图注定量结论均建立追溯关系。

---

# 十七、T12 中文论文初稿 v0.1

编辑：

```text
paper_algorithm/manuscript/draft_v0_1_cn.md
```

写作顺序：

```text
第2章 控制方程与本构
第3章 数值方法
第4章 验证模型与计算设置
第5章 结果
第6章 讨论
第7章 结论
第1章 引言
中文摘要
英文摘要
```

章节结构：

```text
1 引言
2 控制方程与本构关系
  2.1 准静态Maxwell方程
  2.2 欧姆与Cole–Cole介质
  2.3 Debye/Prony近似
  2.4 初始条件与发射波形
3 数值方法
  3.1 二阶Nédélec边元
  3.2 有限接地线源
  3.3 DC初场
  3.4 时间推进与记忆变量
  3.5 三分量dB/dt
  3.6 独立H恢复与闭合
  3.7 点和有限面积接收器
  3.8 求解器与并行
4 验证模型与计算设置
  4.1 均匀半空间
  4.2 薄导层
  4.3 非对称三分量磁观测
  4.4 真实关断波形
  4.5 有限面积接收器
  4.6 Cole–Cole极化层
  4.7 通用三维曲折导电体
  4.8 复杂坝体模型
5 结果
  5.1 均匀半空间精度与收敛
  5.2 薄导层和晚期边界
  5.3 三分量dB/dt与H闭合
  5.4 关断波形效应
  5.5 有限面积接收效应
  5.6 IP与Debye项数
  5.7 通用三维异常
  5.8 复杂坝体复杂几何展示
6 讨论
  6.1 误差来源与验证链
  6.2 波形和接收器物理意义
  6.3 IP精度—成本权衡
  6.4 复杂几何和弱异常的数值要求
  6.5 可复现性与工程扩展
7 结论
```

篇幅：

```text
中文正文 12,000–18,000字
中文摘要 600–900字
英文摘要 300–450词
结论 4–6条
正文图 10–12张
正文表 5–7张
```

每个结果小节必须包含：

```text
模型
指标
定量结果
图表引用
物理或数值解释
```

同时创建：

```text
abstract_cn.md
abstract_en.md
figure_captions_cn.md
table_captions_cn.md
references.bib
```

---

# 十八、T13 QA 和审阅包

执行全量测试并审计：

```text
坐标系
源方向
时间原点
H/B/dBdt定义
单位
接收器法向
有限面积归一化
Cole–Cole参数约定
Debye项数
网格等级
外边界等级
模型参数
图表数字
正文数字
结果追溯
```

创建：

```text
paper_algorithm/qa/qa_report.md
paper_algorithm/qa/unit_consistency.csv
paper_algorithm/qa/figure_audit.csv
paper_algorithm/qa/parameter_audit.csv
paper_algorithm/qa/manuscript_number_audit.csv
```

创建：

```text
paper_algorithm/review_package_v0_1/
```

其中包含：

```text
draft_v0_1_cn.md
abstract_cn.md
abstract_en.md
figures/
tables/
qa_report.md
run_manifest.csv
results_traceability.csv
环境与硬件记录
主要运行命令
```

最终提交：

```text
paper-algorithm: T13 complete draft v0.1 review package
```

---

# 十九、Codex 每个阶段的回复格式

开始阶段时输出：

```text
当前任务：Txx
计划读取：...
计划修改：...
计划运行：...
预期产物：...
```

完成阶段时输出：

```text
任务状态：completed / diagnostic
修改文件：...
运行命令：...
关键数值指标：...
图表产物：...
结果目录：...
提交SHA：...
下一任务：...
```

---

# 二十、现在开始

从 T00 开始，持续执行至：

```text
paper_algorithm/review_package_v0_1/
```

全部完成。