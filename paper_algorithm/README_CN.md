# 算法论文正演方案：SOTEM_FENICSX

## 0. 论文定位

**建议英文题目**

> A validated second-order edge-element framework for three-dimensional grounded-source transient electromagnetic modelling with induced polarization, realistic transmitter waveforms and finite-area receivers

**建议中文题目**

> 考虑激发极化、真实发射波形与有限面积接收器的三维接地电性源瞬变电磁二阶边元正演方法

论文的核心贡献应限定为：

1. 基于 FEniCSx/DOLFINx 的二阶 Nédélec \(H(\mathrm{curl})\) 时域有限元实现；
2. 有限长接地导线源、关断前直流初场和关断后瞬变推进的一致处理；
3. Cole–Cole 复电导率的广义 Debye/Prony 时域记忆变量实现；
4. 理想阶跃、有限线性关断和表格化实测波形的统一处理；
5. \(H_x,H_y,H_z\) 与 \(dB_x/dt,dB_y/dt,dB_z/dt\) 的独立恢复，以及点接收器和有限面积线圈接收器；
6. empymod、文献基准和网格—时间—边界三级收敛组成的可复现验证链。

**不要宣称**“首次提出二阶边元”或“首次实现三维 TEM-IP”，除非最终完成系统查新。论文应强调“统一实现、严格验证和工程观测量建模”。

坝体曲折导电通道只作为三维展示模型，不承担算法正确性的最终证明。

---

# 1. 论文各章内容

## 1 Introduction

建议写成 6 个自然段。

### 第 1 段：应用背景

说明接地电性源 TEM/SOTEM/SATEM 的特点：

- 发射源为有限长接地导线；
- 近源三维场和源几何效应明显；
- 半航空接收需要磁场三分量或感应线圈电压；
- 矿产、煤矿、水文和工程探测中常遇到复杂三维电性结构。

本段不写算法细节。

### 第 2 段：现有数值方法

概括频率—时间变换、有限差分、有限体积和有限元方法。指出：

- 频率—时间变换需要大量频点和稳定变换；
- 低阶离散在复杂几何、近源和弱分量上可能需要大量网格；
- 三维 grounded-wire TEM 的源积分、直流初场和磁场恢复容易出现不一致；
- 大量论文只给出总场曲线，未报告数值误差预算。

### 第 3 段：IP 和真实波形问题

说明：

- 极化介质使电流具有记忆性；
- 直接保存完整电场历史不可行；
- 理想阶跃会高估早期响应；
- 有限关断、时间原点和 Cole–Cole 参数约定必须统一。

### 第 4 段：接收器问题

说明真实无人机线圈测量的是有限面积磁通的时间导数，不是理想点位 \(H\) 或 \(B\)。指出：

- 线圈面积平均；
- 三轴法向；
- 姿态变化；
- \(H\)、\(B\)、\(dB/dt\) 的单位和符号；
- 近零弱分量的普通相对误差会失真。

### 第 5 段：研究缺口

将缺口压缩为一句话：

> 目前仍需要一个能够同时处理有限接地线源、直流初场、三维电性/IP介质、真实关断波形和有限面积磁接收器，并通过独立参考解和系统收敛试验验证的统一时域有限元框架。

### 第 6 段：本文贡献

按以下顺序列 4 点即可：

1. 推导二阶 Nédélec 时域离散及 Debye 记忆变量消元；
2. 构建电荷守恒有限线源和直流初场；
3. 建立两条独立磁场恢复路径和有限面积接收算子；
4. 通过无 IP、有限关断、IP、数值收敛和复杂三维模型进行验证。

---

## 2 Governing equations and constitutive model

### 2.1 Quasi-static Maxwell equations

采用电场形式：

\[
\nabla\times(\mu^{-1}\nabla\times\mathbf E)
+\frac{\partial\mathbf J_c}{\partial t}
=-\frac{\partial\mathbf J_s}{\partial t}.
\]

定义：

- \(\mathbf J_s\)：人工接地导线源；
- \(\mathbf J_c\)：传导和极化电流；
- \(\mathbf E\)：电场；
- \(\mu\)：磁导率。

说明忽略位移电流的适用频段。

### 2.2 Ohmic material

无 IP 情况：

\[
\mathbf J_c=\sigma\mathbf E.
\]

### 2.3 Cole–Cole and generalized Debye representation

写出所采用的复电阻率约定：

\[
\rho^*(\omega)=
\rho_0\left[
1-m\left(
1-\frac{1}{1+(i\omega\tau)^c}
\right)
\right].
\]

然后写成复电导率 Debye 近似：

\[
\sigma^*(\omega)\approx
\sigma_\infty-
\sum_{k=1}^{K}
\frac{\Delta\sigma_k}{1+i\omega\tau_k}.
\]

每个记忆变量满足：

\[
\tau_k\frac{\partial\boldsymbol\chi_k}{\partial t}
+\boldsymbol\chi_k=\mathbf E,
\]

\[
\mathbf J_c=
\sigma_\infty\mathbf E-
\sum_k\Delta\sigma_k\boldsymbol\chi_k.
\]

### 2.4 Initial state and transmitter waveform

长时间供电后的初始条件：

\[
\boldsymbol\chi_k(0)=\mathbf E_0.
\]

说明 \(\mathbf E_0\) 来自直流初场求解。

任意发射波形写成：

\[
I(t)=I_0 w(t),
\]

并明确论文所有观测时间以“关断结束时刻”为零点。

---

## 3 Numerical method

## 3.1 Spatial discretization

说明使用二阶第一族 Nédélec 边元：

\[
\mathbf E_h(\mathbf r,t)
=\sum_i e_i(t)\mathbf N_i(\mathbf r).
\]

给出离散矩阵：

\[
\mathbf K_{ij}=
\int_\Omega
(\nabla\times\mathbf N_i)\cdot
\mu^{-1}(\nabla\times\mathbf N_j)\,d\Omega,
\]

\[
\mathbf M_{\sigma,ij}=
\int_\Omega
\sigma\mathbf N_i\cdot\mathbf N_j\,d\Omega.
\]

强调二阶边元为正式生产配置，一阶只允许作为补充诊断。

## 3.2 Grounded-wire source

说明有限导线沿真实路径积分，不使用源中点电偶极近似。写成：

\[
\mathbf q_i=
\int_{\Gamma_s}\mathbf N_i\cdot d\mathbf l.
\]

必须说明：

- 源端点；
- 电流正方向；
- 坐标正方向；
- 线段穿越单元的确定性分配；
- 端点电荷守恒检查。

## 3.3 DC initial field

给出关断前稳态电场求解，并解释为什么 step-off TEM 不能从零场开始。

## 3.4 Time stepping and memory elimination

正式结果使用后向 Euler：

\[
\frac{\mathbf E^{n+1}-\mathbf E^n}{\Delta t}.
\]

推导 Debye 记忆变量的更新：

\[
\boldsymbol\chi_k^{n+1}
=
\alpha_k\boldsymbol\chi_k^n+
\beta_k\mathbf E^{n+1},
\]

\[
\alpha_k=\frac{\tau_k}{\tau_k+\Delta t},
\quad
\beta_k=\frac{\Delta t}{\tau_k+\Delta t}.
\]

然后得到每步线性系统。矩阵表达必须与代码中的符号约定一致。

## 3.5 Magnetic recovery

分开写两条路径：

1. \(H_x,H_y,H_z\)：由外加电流和介质电流的 Biot–Savart 积分恢复；
2. \(d\mathbf B/dt=-\nabla\times\mathbf E\)：由电场旋度独立获得。

强调二者不是同一离散路径，可相互审计。

## 3.6 Receiver operators

点接收器：

\[
d_p(t)=\mathbf a\cdot\mathbf f(\mathbf r_p,t).
\]

有限圆盘线圈：

\[
\bar d(t)=
\frac{1}{A}
\int_A
\mathbf f(\mathbf r,t)\cdot\mathbf n\,dA.
\]

真实线圈电压：

\[
u(t)=
-NA\,
\overline{\frac{\partial\mathbf B}{\partial t}\cdot\mathbf n}.
\]

正文可报告面积平均场量，线圈匝数和面积只作线性换算。

## 3.7 Linear solver and reproducibility

报告：

- DOLFINx、PETSc、MPI、Gmsh、empymod 和 SimPEG 版本；
- KSP 类型；
- 相对/绝对容差；
- 最大迭代次数；
- 网格单元数、自由度、峰值内存和运行时间；
- Git 提交号；
- 每个算例的配置文件和输出哈希。

---

# 4 Verification design

验证必须遵循以下顺序：

\[
\text{均匀半空间}
\rightarrow
\text{层状无IP}
\rightarrow
\text{有限关断}
\rightarrow
\text{层状IP}
\rightarrow
\text{网格/时间/边界收敛}
\rightarrow
\text{三维复杂模型}.
\]

## 4.1 Error metrics

至少报告：

\[
\varepsilon_{L_2}
=
\frac{\|\mathbf d_h-\mathbf d_{\rm ref}\|_2}
{\|\mathbf d_{\rm ref}\|_2},
\]

\[
\varepsilon_{\infty}
=
\frac{\|\mathbf d_h-\mathbf d_{\rm ref}\|_\infty}
{\|\mathbf d_{\rm ref}\|_\infty},
\]

最大绝对误差，以及带幅值下限的稳健相对误差：

\[
\varepsilon_i^{\rm robust}
=
\frac{|d_i-d_i^{\rm ref}|}
{\max(|d_i^{\rm ref}|,d_{\rm floor})}.
\]

过零附近不能只用普通相对误差。

## 4.2 Acceptance gates

建议预先锁定：

- 主要有效时间窗内，强分量峰值归一化误差 \(\le 5\%\)；
- 参考有限线源积分从 9 点到 17 点的差异 \(\le 0.5\%\)；
- 正负号一致；
- 网格、时间和边界逐级变化均趋于稳定；
- Debye 拟合误差不超过 1%，且时域响应相对 exact Cole–Cole 参考误差可解释；
- 不删除过零点，不平滑后再判断通过。

---

# 5 正演模型矩阵

## V0 环境与源预检

用途：不做时间推进，检查环境、网格、源积分、边界消元和内存估算。

使用 `lei2023_noip.yaml`，级别 `S0T0B0`，运行 `--source-only`。

## V1 均匀半空间无 IP 严格验证

| 参数 | 值 |
|---|---|
| 发射源 | \((-500,0,-0.1)\rightarrow(500,0,-0.1)\) m |
| 源长 | 1000 m |
| 电流 | 1 A |
| 接收点 | \((0,800,-0.1)\) m |
| 空气电阻率 | \(10^8\ \Omega\,m\) |
| 地层电阻率 | \(100\ \Omega\,m\) |
| 时间 | \(10^{-5}\)–\(10^{-1}\) s，41 个对数点 |
| 分量 | \(E_x,E_y,H_z,dB_z/dt\) |
| 参考 | empymod 有限长 bipole |

采用七个逐级算例：

1. `S0T0B0`
2. `S1T0B0`
3. `S2T0B0`
4. `S2T1B0`
5. `S2T2B0`
6. `S2T2B1`
7. `S2T2B2`

这样分别判断空间、时间和边界误差，不需要运行全部 27 个组合。

级别对应：

| 等级 | 源局部网格 | 接收局部网格 | 每输出区间子步 | 外边界尺度 |
|---|---:|---:|---:|---:|
| 0 | 40 m | 20 m | 1 | 25 km |
| 1 | 20 m | 10 m | 2 | 50 km |
| 2 | 10 m | 5 m | 4 | 100 km |

## V2 三层无 IP 模型

| 参数 | 值 |
|---|---|
| 源 | \((-500,0,-0.1)\rightarrow(500,0,-0.1)\) m |
| 电流 | 10 A |
| 接收点 | \((0,1000,-0.1)\) m |
| 层 1 | 0–500 m，100 \(\Omega\,m\) |
| 层 2 | 500–520 m，10 \(\Omega\,m\) |
| 层 3 | >520 m，200 \(\Omega\,m\) |
| 时间 | \(10^{-4}\)–3 s，101 个对数点 |
| 分量 | \(E_x,E_y,H_z,dB_z/dt\) |

用途：验证薄导电层、晚期扩散和过零附近误差。

## V3 六磁分量非对称基准

| 参数 | 值 |
|---|---|
| 源 | \((-20,-7,-0.1)\rightarrow(20,7,-0.1)\) m |
| 源长 | 42.3792402 m |
| 电流 | 1 A |
| 接收点 | \((13,31,-0.2)\) m |
| 源—接收垂直偏移 | 24.9650535 m |
| 空气 | \(10^8\ \Omega\,m\) |
| 半空间 | \(100\ \Omega\,m\) |
| 时间 | \(10^{-5}\)–\(10^{-2}\) s，31 点 |
| 输出 | \(H_x,H_y,H_z,dB_x/dt,dB_y/dt,dB_z/dt\) |

非对称几何避免理论零分量。分别运行：

- 理想阶跃；
- 5 μs 线性关断；
- 20 μs 线性关断。

理想阶跃和 5 μs 算例使用仓库已有 magnetic6 empymod 验证配置。

## V4 有限面积接收器

沿用 V3 几何。主接收器仍使用 point，以保持 empymod 严格验证；同时输出 disk-average 诊断。

圆盘半径：

\[
r_c=0.5,\ 1,\ 2,\ 4\ {\rm m}.
\]

比较：

\[
\delta_{\rm coil}(t)=
\frac{d_{\rm disk}(t)-d_{\rm point}(t)}
{\max_t|d_{\rm point}(t)|}.
\]

用途：量化点接收近似何时失效。

## V5 层状 IP 严格验证

采用 Zhou 2020 基准：

| 参数 | 值 |
|---|---|
| 源 | 1000 m，10 A |
| 接收偏移 | 1000 m |
| 电性层 | 100 / 10 / 200 \(\Omega\,m\) |
| 界面 | 500 m、520 m |
| 极化层 | 500–520 m |
| \(\rho_0\) | 10 \(\Omega\,m\) |
| \(m\) | 0.1 |
| \(\tau\) | 1 s |
| \(c\) | 0.3 |
| 时间 | \(10^{-4}\)–3 s，101 点 |

Debye 项数：

\[
K=8,\ 12,\ 16,\ 20.
\]

正式论文以 \(K=16\) 或首个满足拟合和时域误差门槛的最小项数作为推荐配置。

需要同时运行 no-IP，计算 IP 增量：

\[
\Delta d_{\rm IP}=d_{\rm IP}-d_{\rm noIP}.
\]

## V6 三维曲折导电通道展示

这不是严格参考解，而是 coarse/fine 自收敛展示。

| 参数 | 值 |
|---|---|
| 背景 | 100 \(\Omega\,m\) |
| 通道 | 10 \(\Omega\,m\) |
| 通道半径 | 40 m |
| 通道控制点 | \((-350,-100,-60)\), \((-120,-30,-120)\), \((80,60,-180)\), \((260,-40,-100)\) m |
| 源 | \((-500,200,-0.1)\rightarrow(500,200,-0.1)\) m，10 A |
| 接收 y | -300 m |
| 接收 x | -150、0、150 m |
| 关断 | 5 μs |
| 时间 | \(10^{-5}\)–\(10^{-1}\) s，31 点 |
| 分量 | \(E_x,E_y,dB_z/dt\) |

正式网格：

- coarse：\(40\times24\times12\) 六面体分割尺度；
- refined：\(60\times36\times18\)。

该模型只证明三维异常、曲折几何和 primary-secondary 管线能够运行；不得把它写成现场探测能力验证。

---

# 6 Results 章节安排

## 6.1 Half-space accuracy

展示 V1：

- 四个分量的 FEM 与 empymod 曲线；
- 强分量误差；
- 弱分量带 floor 的误差；
- 正负号和过零时刻。

## 6.2 Spatial, temporal and boundary convergence

展示七级 V1 结果：

- 空间收敛；
- 时间收敛；
- 边界收敛；
- 运行时间和内存变化。

不要把三类误差混在一张单曲线里。

## 6.3 Layered-earth response

展示 V2：

- 薄导层响应；
- 晚期误差；
- \(H_z\) 与 \(dB_z/dt\) 的一致性。

## 6.4 Six-component magnetic validation

展示 V3：

- 三个 \(H\) 分量；
- 三个 \(dB/dt\) 分量；
- 独立恢复路径；
- 非对称弱分量误差。

## 6.5 Transmitter waveform effects

展示 V3 的 0、5、20 μs：

- 发射电流波形；
- 早期响应；
- 相对理想阶跃偏差；
- 时间原点敏感性。

## 6.6 Finite-area receiver effects

展示 V4：

- point 与不同半径 disk-average；
- 不同分量的面积平滑比例；
- 何种偏移距/时间段必须考虑线圈面积。

## 6.7 IP verification

展示 V5：

- Cole–Cole 与 Debye 频域拟合；
- \(K=8,12,16,20\)；
- IP 总场；
- IP 增量；
- 过零时刻；
- 计算内存随 Debye 项数增加的变化。

## 6.8 Three-dimensional demonstration

展示 V6：

- 三维通道和网格；
- 三个接收点响应；
- coarse/fine 差异；
- 电流或电场扩散快照。

---

# 7 论文图件

建议正文 10 张图。

| 图 | 内容 |
|---|---|
| Fig. 1 | 算法流程图：DC 初场—关断波形—时域推进—磁场恢复—接收算子 |
| Fig. 2 | V1/V3 源—接收几何、局部网格和外边界 |
| Fig. 3 | V1 四分量 FEM–empymod 对比及误差 |
| Fig. 4 | 网格、时间、边界三组收敛图 |
| Fig. 5 | V2 三层模型曲线、薄导层响应和晚期误差 |
| Fig. 6 | V3 六磁分量验证 |
| Fig. 7 | 0、5、20 μs 波形及响应差异 |
| Fig. 8 | 点接收器与有限圆盘接收器 |
| Fig. 9 | Cole–Cole–Debye 拟合、IP总场与IP增量 |
| Fig. 10 | 三维曲折通道、场快照和 coarse/fine 响应 |

补充材料：

- S1：源积分收敛；
- S2：Biot–Savart 求积阶数；
- S3：线性求解器迭代；
- S4：完整误差表；
- S5：软件版本、提交号和全部配置。

---

# 8 论文表格

## Table 1

算法与物理量定义：

- 方程；
- 空间离散；
- 时间离散；
- 源；
- IP；
- 磁恢复；
- 接收器。

## Table 2

全部验证模型及参数。

## Table 3

V1/V2/V3/V5 的误差指标。

## Table 4

网格、自由度、时间步数、运行时间和峰值内存。

---

# 9 推荐运行顺序

1. `env`：环境检查；
2. `preflight`：网格/源/内存预检；
3. `benchmark-pilot`：粗级别检查符号和单位；
4. `magnetic6`：六磁分量严格验证；
5. `convergence`：七级无 IP 收敛；
6. `waveform`：0、5、20 μs；
7. `receiver`：有限面积接收器；
8. `ip`：Debye 项数扫频；
9. `demo3d`：三维曲折导电通道；
10. 所有图确认后，再跑最高精度补算。

运行入口：

```bash
conda activate fenicsx
bash paper_algorithm/run_algorithm_paper.sh env
bash paper_algorithm/run_algorithm_paper.sh preflight
bash paper_algorithm/run_algorithm_paper.sh benchmark-pilot
bash paper_algorithm/run_algorithm_paper.sh magnetic6
bash paper_algorithm/run_algorithm_paper.sh convergence
bash paper_algorithm/run_algorithm_paper.sh waveform
bash paper_algorithm/run_algorithm_paper.sh receiver
bash paper_algorithm/run_algorithm_paper.sh ip
bash paper_algorithm/run_algorithm_paper.sh demo3d
```

默认输出：

```text
generated/paper_algorithm/
```

可通过环境变量覆盖：

```bash
OUTPUT_ROOT=/data/paper_algorithm \
CONDA_ENV_NAME=fenicsx \
MEMORY_LIMIT_GB=64 \
bash paper_algorithm/run_algorithm_paper.sh all
```

---

# 10 稿件中的限制必须明确写出

1. 坝体或通道模型是电性等效体，不是 Darcy 渗流耦合；
2. 线源不等于有限宽铜带接触电极；
3. 点位场和面积平均场不等于完整接收机电压链；
4. empymod 只能验证均匀/层状背景，不能作为三维异常参考；
5. 三维展示采用数值自收敛，不替代现场验证；
6. 不能用删除过零点、平滑或提高目标尺度制造“验证通过”。
