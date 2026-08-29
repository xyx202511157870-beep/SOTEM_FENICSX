# 考虑激发极化、真实发射波形与有限面积接收器的三维接地电性源瞬变电磁二阶边元正演方法

> 版本：v0.1 scaffold  
> 分支：paper/algorithm-forward-matrix  
> 结果占位符由 T10–T12 自动替换并登记到 `results_traceability.csv`。

## 摘要

接地电性源瞬变电磁法在复杂三维结构探测中具有较强的源—目标耦合能力，其数值模拟需要统一处理有限长接地导线源、关断前直流初场、关断后电磁扩散、激发极化记忆效应、真实发射波形以及有限面积磁接收器。本文基于 FEniCSx/DOLFINx 构建三维时域二阶 Nédélec 边元正演框架。电场采用二阶 \(H(\mathrm{curl})\) 边元离散，有限接地导线源通过线积分投影进入离散系统；长时间供电条件下的直流初场作为 step-off 瞬变初值。Cole–Cole 复电导率采用广义 Debye/Prony 展开，并通过局部记忆变量实现时域推进。三分量磁通密度时间导数由 \(-\nabla\times\mathbf E\) 计算，三分量磁场由外加源电流与地下传导/极化电流的 Biot–Savart 积分独立恢复。接收算子同时支持点接收器和有限圆盘平均接收器，发射端支持理想阶跃、有限线性关断和表格化波形。

算法通过均匀半空间、薄导层、非对称三分量磁观测以及 Cole–Cole 极化层模型进行验证，并系统评价空间网格、时间步、外边界、源积分、Debye 项数、关断时间和接收器面积的影响。均匀半空间主要分量的峰值归一化误差为 {{V1_MAIN_PEAK_ERROR}}，有限源参考积分审计误差为 {{V1_SRCPTS_ERROR}}；三分量 \(d\mathbf B/dt\) 与独立参考的相对误差为 {{V3_DBDT_ERROR_RANGE}}，由磁场时间差分与电场旋度构造的闭合误差为 {{V3_CLOSURE_ERROR_RANGE}}。当关断时间由理想阶跃增加至 5 μs 和 20 μs 时，早期响应差异分别达到 {{WAVEFORM_5US_MAX}} 和 {{WAVEFORM_20US_MAX}}。有限圆盘半径由 0.5 m 增至 4 m 时，面积平均效应由 {{COIL_R05_MAX}} 增至 {{COIL_R4_MAX}}。Cole–Cole 模型采用 {{SELECTED_DEBYE_K}} 个 Debye 项时，在目标频率范围内的最大拟合误差为 {{DEBYE_FIT_MAX}}，相应时域主要分量误差为 {{DEBYE_TIME_ERROR}}。三维曲折导电体算例显示，算法能够稳定恢复空间变化响应，粗细网格差异为 {{DEMO3D_CF_ERROR}}，异常—数值差异比为 {{DEMO3D_ANOMALY_RATIO}}。

结果表明，该框架能够在统一物理量、坐标、时间原点和接收器定义下模拟三维 grounded-source TEM 与 IP 响应，并通过独立参考、数值闭合和多层级收敛形成可追溯的验证链。该实现为复杂地形、三维导电/极化异常以及半航空感应线圈观测系统的正演研究提供了可复现计算基础。

**关键词：** 电性源瞬变电磁；二阶 Nédélec 边元；激发极化；Cole–Cole；真实关断波形；有限面积接收器；FEniCSx

---

# 1 引言

接地电性源瞬变电磁法通过有限长接地导线向地下供入时变电流，并在地面或空中记录电场与磁场瞬变响应。该类方法兼具较强的近源信号、灵活的源—接收几何和多分量观测能力，已形成 SOTEM、LOTEM、SATEM 等多种观测形式。复杂地形、真实导线几何、三维电性异常以及接收器移动平台使其电磁场具有显著三维特征，因此高精度三维正演是观测系统设计、响应解释和反演成像的基础。

接地电性源 TEM 正演可采用频率域计算结合频时转换，也可直接在时域中离散 Maxwell 方程。频率域路线适合层状介质和成熟积分算法，直接时域路线则能够自然处理任意发射波形、非均匀时间步以及局部记忆变量。复杂几何通常采用有限差分、有限体积或有限元离散，其中 \(H(\mathrm{curl})\) 边元能够保持切向电场连续性并适应非结构网格。接地线源、空气—地面界面、关断初值和近源局部网格共同决定数值精度，源投影与接收算子的符号、单位和坐标约定也直接影响跨软件验证。

含极化介质中的电流具有时间记忆。Cole–Cole 模型在频域中可通过复电阻率或复电导率描述，时域直接卷积会引入完整历史存储。广义 Debye 或 Prony 展开可将频率相关电导率表示为多个一阶松弛过程，每个过程由局部记忆变量推进，从而将历史卷积转化为有限状态更新。该策略的精度取决于拟合频带、Debye 项数、时间离散和初始极化状态。

真实发射系统具有有限关断时间。早期瞬变同时受到关断波形、时间原点和接收器带宽影响。半航空感应线圈测量的是有限面积磁通的时间导数，点位 \(d\mathbf B/dt\) 仅对应理想化接收量。有限面积平均、线圈法向和平台姿态会改变强空间梯度区域的响应，因此正演框架需要将离散场投影到真实接收泛函。

本文构建一个统一的三维时域二阶边元框架，将有限接地线源、直流初场、欧姆与 Cole–Cole/Prony 介质、真实关断波形、三分量 \(d\mathbf B/dt\)、独立磁场恢复和有限面积接收器纳入同一计算链。验证体系由独立层状参考、源积分审计、空间—时间—边界收敛、磁场闭合检验以及三维 coarse–fine 对比组成。

本文的主要工作包括：（1）构建基于二阶 Nédélec 边元的 grounded-source TEM 时域离散；（2）实现有限线源、电荷守恒源投影和关断前直流初场；（3）采用 Debye/Prony 记忆变量处理 Cole–Cole 极化介质；（4）统一实现真实关断波形、三分量 \(d\mathbf B/dt\)、独立三分量磁场和有限面积接收算子；（5）建立覆盖层状介质、波形、接收器、IP 和三维异常的可追溯验证流程。

---

# 2 控制方程与本构关系

## 2.1 准静态 Maxwell 方程

在目标频带内采用准静态近似，电场形式写为

\[
\nabla\times\left(\mu^{-1}\nabla\times\mathbf E\right)
+\frac{\partial\mathbf J_c}{\partial t}
=-\frac{\partial\mathbf J_s}{\partial t},
\tag{1}
\]

其中，\(\mathbf E\) 为电场，\(\mathbf J_s\) 为人工源电流密度，\(\mathbf J_c\) 为介质传导与极化电流密度，\(\mu\) 为磁导率。

对于欧姆介质，

\[
\mathbf J_c=\sigma\mathbf E.
\tag{2}
\]

## 2.2 Cole–Cole 复电阻率

本文采用

\[
\rho^*(\omega)=
\rho_0\left[
1-m\left(
1-\frac{1}{1+(i\omega\tau)^c}
\right)
\right],
\tag{3}
\]

\[
\sigma^*(\omega)=\frac{1}{\rho^*(\omega)},
\tag{4}
\]

其中，\(\rho_0\) 为低频电阻率，\(m\) 为充电率，\(\tau\) 为时间常数，\(c\) 为频率指数。

## 2.3 Debye/Prony 近似

复电导率近似为

\[
\sigma^*(\omega)\approx
\sigma_\infty-
\sum_{k=1}^{K}
\frac{\Delta\sigma_k}{1+i\omega\tau_k}.
\tag{5}
\]

对应时域记忆变量满足

\[
\tau_k\frac{\partial\boldsymbol\chi_k}{\partial t}
+\boldsymbol\chi_k=\mathbf E,
\tag{6}
\]

\[
\mathbf J_c=
\sigma_\infty\mathbf E-
\sum_{k=1}^{K}\Delta\sigma_k\boldsymbol\chi_k.
\tag{7}
\]

采用后向 Euler 离散，

\[
\boldsymbol\chi_k^{n+1}
=\alpha_k\boldsymbol\chi_k^n+
\beta_k\mathbf E^{n+1},
\tag{8}
\]

\[
\alpha_k=\frac{\tau_k}{\tau_k+\Delta t_n},
\qquad
\beta_k=\frac{\Delta t_n}{\tau_k+\Delta t_n}.
\tag{9}
\]

## 2.4 发射波形与初始条件

发射电流表示为

\[
I(t)=I_0w(t).
\tag{10}
\]

长时间供电形成直流初始电场 \(\mathbf E_0\)。对于 step-off 初值，记忆变量取

\[
\boldsymbol\chi_k(0)=\mathbf E_0.
\tag{11}
\]

本文将电流完全关断时刻定义为观测时间零点，并分别考虑理想阶跃、有限线性关断和表格化发射波形。

---

# 3 数值离散

## 3.1 二阶 Nédélec 边元

电场近似为

\[
\mathbf E_h(\mathbf r,t)=
\sum_{i=1}^{N_e}e_i(t)\mathbf N_i(\mathbf r),
\tag{12}
\]

其中 \(\mathbf N_i\) 为二阶 Nédélec 基函数。旋度刚度矩阵和电导质量矩阵分别为

\[
K_{ij}=\int_\Omega
(\nabla\times\mathbf N_i)\cdot
\mu^{-1}(\nabla\times\mathbf N_j)\,d\Omega,
\tag{13}
\]

\[
M_{\sigma,ij}=\int_\Omega
\sigma\mathbf N_i\cdot\mathbf N_j\,d\Omega.
\tag{14}
\]

## 3.2 有限接地线源

有限导线沿路径 \(\Gamma_s\) 进入离散源向量：

\[
q_i=I(t)\int_{\Gamma_s}\mathbf N_i\cdot d\mathbf l.
\tag{15}
\]

程序计算线段与四面体的精确交区间，统一处理单元边界、分区边界和源端点，并对离散源执行端点电流平衡与边界消元后残差审计。

## 3.3 直流初场

关断前电位满足

\[
\nabla\cdot(\sigma\nabla\phi_0)
=-\nabla\cdot\mathbf J_s,
\tag{16}
\]

\[
\mathbf E_0=-\nabla\phi_0.
\tag{17}
\]

直流初场传递至时域电场和 Debye 记忆变量。

## 3.4 隐式时间推进

将式（8）代入式（1），形成每个时间步的有效电导矩阵和历史电流右端项。本文正式计算采用后向 Euler，时间节点覆盖关断过程和关断后对数观测时间。

## 3.5 磁观测量恢复

主要磁观测量由

\[
\frac{\partial\mathbf B}{\partial t}
=-\nabla\times\mathbf E
\tag{18}
\]

获得。辅助磁场由总电流的 Biot–Savart 积分恢复：

\[
\mathbf H(\mathbf r,t)=
\frac{1}{4\pi}
\int
\mathbf J(\mathbf r',t)\times
\frac{\mathbf r-\mathbf r'}{|\mathbf r-\mathbf r'|^3}
\,dV'.
\tag{19}
\]

利用

\[
\left(\frac{\partial\mathbf B}{\partial t}\right)_H^n
=\mu_0\frac{\mathbf H^n-\mathbf H^{n-1}}
{t_n-t_{n-1}}
\tag{20}
\]

与式（18）构造独立闭合误差。

## 3.6 接收算子

点接收器写为

\[
d_p(t)=\mathbf a\cdot\mathbf f(\mathbf r_p,t).
\tag{21}
\]

有限圆盘接收器写为

\[
d_A(t)=\frac{1}{A}\int_A
\mathbf f(\mathbf r,t)\cdot\mathbf n\,dA.
\tag{22}
\]

其中 \(\mathbf n\) 为接收器法向。真实感应线圈电压可由

\[
u(t)=-N\frac{d}{dt}\int_A\mathbf B\cdot\mathbf n\,dA
\tag{23}
\]

进一步换算。

## 3.7 求解器与可复现性

本文记录 DOLFINx、PETSc、MPI、Gmsh、empymod 和 SimPEG 版本，报告每个生产模型的网格单元数、自由度、时间步、KSP 参数、运行时间、峰值内存、Git 提交 SHA 和输出哈希。

---

# 4 验证模型与数值设置

## 4.1 均匀半空间模型 V1

源从 \((-500,0,-0.1)\) m 延伸至 \((500,0,-0.1)\) m，电流为 1 A；接收点位于 \((0,800,-0.1)\) m。空气和半空间电阻率分别为 \(10^8\) 和 100 \(\Omega\cdot\mathrm m\)。观测时间为 \(10^{-5}\)–\(10^{-1}\) s，共 41 个对数点。输出 \(E_x,E_y,H_z,dB_z/dt\)，参考解由 empymod 计算。

空间、时间和边界等级采用 S0–S2、T0–T2 和 B0–B2，形成七组逐级收敛序列。

## 4.2 薄导层模型 V2

层状地球由 0–500 m 的 100 \(\Omega\cdot\mathrm m\) 层、500–520 m 的 10 \(\Omega\cdot\mathrm m\) 薄导层和下伏 200 \(\Omega\cdot\mathrm m\) 半空间组成。源长 1000 m，电流 10 A，接收偏移 1000 m，观测时间为 \(10^{-4}\)–3 s。

## 4.3 非对称三分量磁观测模型 V3

源从 \((-20,-7,-0.1)\) m 延伸至 \((20,7,-0.1)\) m，接收点为 \((13,31,-0.2)\) m，半空间电阻率为 100 \(\Omega\cdot\mathrm m\)。输出三分量 \(d\mathbf B/dt\) 和三分量 \(\mathbf H\)，观测时间为 \(10^{-5}\)–\(10^{-2}\) s。

## 4.4 真实波形模型 V4

沿用 V3 几何，比较理想阶跃、5 μs 线性关断和 20 μs 线性关断。

## 4.5 有限面积接收器模型 V5

沿用 V3 的 5 μs 线性关断，比较点接收器与半径为 0.5、1、2 和 4 m 的圆盘平均接收器。

## 4.6 IP 模型 V6

极化薄层位于 500–520 m，\(\rho_0=10\ \Omega\cdot\mathrm m\)、\(m=0.1\)、\(\tau=1\) s、\(c=0.3\)。Debye 项数取 8、12、16 和 20，并与 exact Cole–Cole 频域材料参考比较。

## 4.7 三维曲折导电体模型 V7

背景电阻率为 100 \(\Omega\cdot\mathrm m\)，曲折管状导电体电阻率为 10 \(\Omega\cdot\mathrm m\)，半径 40 m。控制点为

\[
(-350,-100,-60),
(-120,-30,-120),
(80,60,-180),
(260,-40,-100)\ \mathrm m.
\]

源位于 \((-500,200,-0.1)\)–\((500,200,-0.1)\) m，三个接收点的横坐标分别为 -150、0 和 150 m，纵坐标为 -300 m。粗细网格分别采用 {{DEMO3D_COARSE_CELLS}} 和 {{DEMO3D_FINE_CELLS}}。

---

# 5 结果

## 5.1 均匀半空间基准

Fig. 3 比较了二阶边元结果与 empymod 参考。主要有效时间窗内，\(E_x\)、\(E_y\)、\(H_z\) 和 \(dB_z/dt\) 的峰值归一化误差分别为 {{V1_EX_ERROR}}、{{V1_EY_ERROR}}、{{V1_HZ_ERROR}} 和 {{V1_DBZ_ERROR}}。参考有限源积分从 9 点增加至 17 点时的最大变化为 {{V1_SRCPTS_ERROR}}。

Fig. 4 给出了空间、时间和外边界收敛序列。空间加密使综合误差从 {{V1_S0_ERROR}} 降至 {{V1_S2_ERROR}}；时间子步增加使误差从 {{V1_T0_ERROR}} 降至 {{V1_T2_ERROR}}；边界范围扩大使晚期误差从 {{V1_B0_ERROR}} 变化至 {{V1_B2_ERROR}}。相应自由度、运行时间和峰值内存见 Table 2。

## 5.2 薄导层基准

薄导层模型在中晚期形成明显扩散特征。正式 S2T2B2 结果中，四个分量的主要时间窗误差范围为 {{V2_ERROR_RANGE}}，\(dB_z/dt\) 的过零时间与参考差异为 {{V2_ZERO_CROSSING_DIFF}}。晚期边界敏感性见 Supplementary Fig. S2。

## 5.3 三分量 dB/dt 与独立 H 恢复

非对称几何使六项观测量均具有有效幅值。三分量 \(d\mathbf B/dt\) 的相对误差分别为 {{V3_DBX_ERROR}}、{{V3_DBY_ERROR}} 和 {{V3_DBZ_ERROR}}；三分量 \(\mathbf H\) 的相对误差分别为 {{V3_HX_ERROR}}、{{V3_HY_ERROR}} 和 {{V3_HZ_ERROR}}。由式（20）和式（18）构造的闭合误差范围为 {{V3_CLOSURE_ERROR_RANGE}}。

## 5.4 真实关断波形

5 μs 和 20 μs 关断主要影响早期时间窗。相对于理想阶跃，5 μs 关断的三分量最大峰值归一化差异为 {{WAVEFORM_5US_MAX}}，20 μs 为 {{WAVEFORM_20US_MAX}}。各分量进入 5% 差异范围的时间分别为 {{WAVEFORM_STABLE_TIMES}}。

## 5.5 有限面积接收器

圆盘平均效应随接收器半径增加。半径 0.5、1、2 和 4 m 时，相对于点接收器的最大差异分别为 {{COIL_R05_MAX}}、{{COIL_R1_MAX}}、{{COIL_R2_MAX}} 和 {{COIL_R4_MAX}}。不同分量对空间平均的敏感性见 Fig. 11。

## 5.6 Cole–Cole/IP 与 Debye 项数

Debye 项数从 8 增至 20 时，频率域最大拟合误差由 {{DEBYE_K8_FIT}} 变化至 {{DEBYE_K20_FIT}}。时域主要分量误差由 {{DEBYE_K8_TIME}} 变化至 {{DEBYE_K20_TIME}}。综合精度、运行时间和峰值内存，本文正式采用 \(K={{SELECTED_DEBYE_K}}\)。IP 增量的过零时间和峰值随 K 的变化见 Fig. 14。

## 5.7 三维曲折导电体

三维导电体在三个接收点产生不同幅度与形态的异常响应。正式 fine 网格相对于背景的异常范数为 {{DEMO3D_ANOMALY_NORM}}，coarse–fine 差异为 {{DEMO3D_CF_ERROR}}，异常—数值差异比为 {{DEMO3D_ANOMALY_RATIO}}。结果显示三维空间变化能够由现有正演框架稳定表达。

---

# 6 讨论

## 6.1 数值精度与主要误差来源

结合 V1 和 V2 的逐级收敛结果，分析空间离散、时间离散、外边界和有限源积分对不同分量及不同时间窗的相对贡献。将误差峰值与响应过零、弱分量和晚期扩散联系起来，并总结正式生产参数。

## 6.2 独立磁场恢复的数值价值

三分量 H 的独立恢复为源方向、总电流、Biot–Savart 求积和单位换算提供审计路径。H 时间差分与 \(-\nabla\times E\) 的闭合检验进一步约束电流恢复和旋度观测的一致性。

## 6.3 波形和接收器建模意义

根据 V4 和 V5 的定量结果，讨论理想阶跃和点接收近似的时间与空间适用范围。将关断时间与首个可靠观测时间联系起来，将圆盘半径与局部场空间变化尺度联系起来。

## 6.4 IP 精度与计算成本

根据 Debye 项数扫描，讨论频域材料拟合、时域响应、运行时间和内存之间的平衡，并给出正式项数和拟合频带选择依据。

## 6.5 三维扩展与可复现性

总结非结构网格、曲折三维异常、多接收位置和 coarse–fine 参考的扩展能力。结合运行清单、结果哈希和提交 SHA，说明整个计算链的可追溯性。

---

# 7 结论

1. 二阶 Nédélec 边元框架在均匀半空间和薄导层基准中达到 {{CONCLUSION_ERROR_LEVEL}} 的主要分量精度，并形成空间—时间—边界逐级收敛证据。
2. 三分量 \(d\mathbf B/dt\) 与独立参考保持一致，三分量 H 恢复及 Maxwell 闭合误差为 {{CONCLUSION_CLOSURE}}。
3. 真实关断波形对早期响应产生 {{CONCLUSION_WAVEFORM}} 的最大影响，有限面积接收器产生 {{CONCLUSION_RECEIVER}} 的最大空间平均差异。
4. 采用 \(K={{SELECTED_DEBYE_K}}\) 的 Debye/Prony 展开实现 Cole–Cole 时域材料，频率域拟合和时域响应达到 {{CONCLUSION_IP}} 的精度水平。
5. 三维曲折导电体算例的异常—数值差异比为 {{DEMO3D_ANOMALY_RATIO}}，验证了框架处理三维空间变化模型的能力。

---

# 代码与数据可得性

本文正演代码、基准配置、运行脚本、后处理脚本、图件清单、结果追溯表和版本信息保存在 `SOTEM_FENICSX` 论文分支及对应归档版本中。正式稿填写仓库版本号、提交 SHA 和归档标识。

# 作者贡献

按实际作者分工填写方法、软件、验证、分析、写作与项目管理贡献。

# 致谢

填写项目、计算资源和学术讨论支持。

# 参考文献

由 `references.bib` 生成，覆盖 grounded-wire TEM、二阶边元、Cole–Cole/Debye、真实发射波形、半航空接收和基准论文。
