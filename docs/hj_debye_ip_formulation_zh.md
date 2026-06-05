# H/J 直接时域 Debye-IP 耦合公式说明

本文记录当前 H/J 接地导线源直接时域实现所用的严格离散公式，以及尚未闭合的近源磁场 source-history/MMR 项。它是 `docs/formulation_and_validation.md` 和 `docs/source_history_mmr_derivation.md` 的中文速查版。

## 1. 连续方程

忽略位移电流的准静态一阶 Maxwell 方程写为

```text
curl H = J_c + s_f
mu dH/dt + curl E = 0
```

其中 `s_f` 是接地导线源投影到 H/J 面自由度上的 impressed current source。对 step-off 源，长通电初始状态为

```text
s_f(0-) = s_0
s_f(t>0) = 0
```

代码采用的 H/J 符号约定是

```text
s_0 = -source.initial_face_vector(mesh)
J_total = J_c + s_f = C h
J_c = C h - s_f
```

这里 `C` 是离散 edge-curl，`h` 在边上，`J_c` 和 `s_f` 在面上。

## 2. Debye 极化本构

导电电流采用 conductivity-form Debye 展开：

```text
J_c(t) = sigma_inf E(t) - sum_i delta_sigma_i y_i(t)
tau_i dy_i/dt + y_i = E
```

低频电导率为

```text
sigma_0 = sigma_inf - sum_i delta_sigma_i
```

必须满足 `sigma_0 > 0`。后退欧拉时间步 `dt_n = t_{n+1}-t_n` 给出

```text
alpha_i = tau_i / (tau_i + dt_n)
beta_i  = dt_n / (tau_i + dt_n)
y_i^{n+1} = alpha_i y_i^n + beta_i E^{n+1}
```

代回电流本构：

```text
J_c^{n+1}
  = sigma_eff E^{n+1} - sum_i alpha_i delta_sigma_i y_i^n

sigma_eff = sigma_inf - sum_i beta_i delta_sigma_i
```

这就是极化效应进入直接时域方程的地方：不是额外加一个经验场，而是通过 `sigma_eff` 和历史项 `sum alpha_i delta_sigma_i y_i^n` 同时进入每个时间步。

## 3. H/J 反本构

H/J 推进先由 Ampere 方程得到电流

```text
J_c^{n+1} = C h^{n+1} - s_f^{n+1}
```

因此需要把 Debye 本构反解为电场：

```text
E^{n+1} = rho_eff J_c^{n+1} + E_hist^n
rho_eff = 1 / sigma_eff
E_hist^n = rho_eff sum_i alpha_i delta_sigma_i y_i^n
```

在有限体积实现里，`sigma_inf` 和 `delta_sigma` 可以是 cell-centered 或 face-centered。cell-centered 层状模型在 H/J 中通过 face inner product 投影到面自由度，代码路径为：

```text
face_project_debye_model
_hj_inverse_constitutive_coefficients
_electric_from_hj_current
```

对应实现文件是 `src/atem3d/hj.py` 和 `src/atem3d/ip.py`。

## 4. H/J 磁场线性系统

Faraday 方程离散为

```text
M_mu (h^{n+1} - h^n) / dt_n + C^T M_u E^{n+1} = 0
```

把反本构和 `J_c = C h - s_f` 代入：

```text
[M_mu/dt_n + C^T M_rho_eff C] h^{n+1}
  = M_mu h^n/dt_n
    + C^T M_rho_eff s_f^{n+1}
    - C^T M_u E_hist^n
```

这就是 `hj_magnetic_system_matrix` 与 `hj_magnetic_rhs` 的核心公式。求得 `h^{n+1}` 后，再恢复

```text
J_c^{n+1} = C h^{n+1} - s_f^{n+1}
E^{n+1} = rho_eff J_c^{n+1} + E_hist^n
y_i^{n+1} = alpha_i y_i^n + beta_i E^{n+1}
```

所以 H/J 直接时域里 Debye 极化不是后处理耦合，而是已经进入主时间步矩阵和右端项。

## 5. Step-off 初值

接地导线源不能用零初值直接开始。长通电初始电流由 DC 连续性方程给出：

```text
D (j_0 + s_0) = 0
j_0 = -M_rho0^{-1} G phi
D M_rho0^{-1} G phi = D s_0
rho0 = 1 / sigma_0
```

电场为

```text
E_0 = M_rho0 j_0 / M_u
```

Debye 记忆变量长通电平衡为

```text
y_i^0 = E_0
```

初始磁场通过 MMR/vector-potential 方程求得，使其满足

```text
C h_0 = j_0 + s_0
```

代码里的 `test_hj_mmr_initial_magnetic_field_satisfies_stabilized_system` 检查了这个平衡；相反源符号会留下 `O(1)` 残差。

## 6. 边界策略

当前 EB formulation 有实验性 split-curl CPML，因为 EB 的 Faraday/Ampere 拉伸方程和 CPML 记忆变量已经接入时间步。

H/J formulation 目前没有 active CPML。原因是 H/J 使用反本构和 `C^T M_rho C` 磁场系统，CPML 不能只作为普通外壳电导率直接套进去；必须重新推导 split-curl H/J 系统、PML 记忆变量、Debye 反本构历史项之间的耦合。这个生产实现尚未完成。

因此当前 H/J 近源验证应使用：

```text
boundary: {kind: none}
```

配合足够大的计算域，或使用外部 sponge 做边界收敛候选。H/J 结果不能声称已经使用了严格 CPML。

## 7. 磁场接收与未闭合项

H/J 主求解给出边上的 `h`。近源 `Hz` 还可以通过 receiver-side MMR/Biot 从电流恢复：

```text
H_rec(t, r) = B_R [C h(t) - s_f(t)]
```

当前 `current_biot`、`face_basis_biot` 和 `face_basis_cell_biot` 都属于这个接收侧恢复路径。`face_basis_biot` 直接把 `C h - s_f` 用 face-basis Biot 积分恢复磁场；在 `sourcecell0p25/0p75` no-IP sub5 的代表时间门上，它把 aggregate `Hz` L2 从约 `0.296` 变为约 `0.336`，说明仅替换接收积分不能解决 no-IP 近源基线。实验证据表明，主 H/J Debye 耦合和 step-off 初值已经能工作，但近源 IP `Hz` 仍缺一个 source-history/MMR 修正项。

诊断形式写为

```text
H_sh(t, r) = sum_q a_q(t) B_R v_q(r)
```

其中 `v_q` 通常取接地源面向量的低阶纵向矩，例如

```text
v_0 = s_0
v_2 = xi^2 s_0
```

已验证的事实：

```text
1. time_series_source_moments replay 与 receiver-space replay 一致；
2. Ampere residual 投影不是缺失项；
3. 初始极化电流和 charge-conserving 初始极化电流不是缺失项；
4. high-minus-low DC 电流差不是缺失项；
5. H/J delta6 face-current 可被 degree-0 source-history 精确复现，但仍给出
   Hz L2 = 0.889/1.035/0.890，说明它不是 H/J 生产公式；
6. 局部 driven recovery 加入 `charge_conserving_mmr` 非零初始 `h` 状态，
   或把全局 `global_charge_conserving_mmr` 初态限制到局部 support 后，
   在 0.1-1 ms 采样窗的 trace L2 都仍约为 0.2154，与零初态结果相同；
   说明单纯一次性回流 MMR 初态会在快速局部磁扩散时间尺度内消失，
   不是缺失的生产项；
7. 将全局 MMR 稳态限制到局部 support 并作为持续
   `global_mmr_steady` edge RHS forcing 后，trace L2 仍约为 0.2154，
   首个采样门仍明显过大，说明稳态全局 MMR forcing 也不是缺失项。
   与 rise-decay `relaxation_difference` 时间核组合后也没有实质改善：
   compact amplitude L2 约 0.1736，略差于普通 source RHS 的 0.1717。
   真正的回流影响若存在，必须改变时间核，或作为动态边界/算子耦合进入。
```

因此当前未闭合的不是代码通道，而是非拟合的局部 FV/MMR 源恢复律：必须从接地源 step-off、Debye 记忆、回流电流路径、局部/非局部恢复算子推导 `a_q(t)` 或等价的局部算子。

## 8. empymod 验证要求

`empymod` 用作层状介质半解析参考。每个候选实现必须至少通过以下层级：

```text
1. 无 IP 均匀/层状基线；
2. 单 Debye IP，使用相同 sigma_inf、delta_sigma、tau；
3. Ex/Ey/Hz 三分量符号、单位、源强和接收方向校验；
4. 0.1-1 ms 或指定时窗的逐分量 relative L2；
5. 网格、时间步、边界收敛扫描。
```

对当前 H/J source-centered `current_biot` 工况，未修正 IP `Hz` 为

```text
0.588 / 0.650 / 0.589
```

若某个非拟合 source-history/MMR 候选不能稳定优于这个基线，并且不能解释其公式来源，就只能保留为 diagnostic-only。

当前新增的 `atem3d.clean_delta_decomposition_cli` 用来固定 IP 修正项的验收口径。它采用

```text
error = numerical - reference
ideal_clean_delta = noip_error - raw_ip_error
actual_correction = corrected_ip_numerical - raw_ip_numerical
correction_mismatch = actual_correction - ideal_clean_delta
```

因此

```text
corrected_ip_error = noip_error + correction_mismatch
```

若某个候选只降低总 IP L2，但 `correction_mismatch` 没有下降，就说明它主要在重放 no-IP 基线误差或其它公共误差，不是所需的 Debye-IP clean-delta 源历史项。对当前 sub5 `current_biot` 数据，把 no-IP source-cell law 单独加到 Debye-IP 结果后，三个 `Hz` 的 `actual_correction_relative_l2_to_ideal` 约为 `1.24/1.27/1.24`，所以它不能作为 IP clean-delta 生产律。

同一口径下，`[-7,8,0]` shell compact time-series replay 的 clean-delta mismatch 约为 `0.026/0.037/0.026`，trace-fitted amplitude 版本约为 `0.0169/0.0157/0.0171`，旧 one-mode driven recovery 约为 `0.063/0.072/0.063`。因此 IP 分支已经有接近 clean-delta 的诊断候选，但总 `Hz` 误差仍受 no-IP/current-Biot 基线 `0.214/0.239/0.214` 限制。后续必须并行解决两个问题：非拟合 IP clean-delta 源历史律，以及 no-IP H/J 近源磁场恢复基线。

运行时也强制这个不变量：`requires_ip: true` 的 source-history correction 在没有 Debye terms 或所有 `delta_sigma=0` 时都会被跳过；只有 no-IP 诊断基线 `source_diffusion_kernel_source_moments` 标记为 `requires_ip: false`，可以在纯欧姆模型中运行。`prescribed_source_moments` 和 `driven_recovery_source_moments` 现在也接受 `normalized_coefficients`，运行时按 `mu * sum(delta_sigma_source) * source_length^2` 转成绝对系数；这只是量纲复现接口，不能替代非拟合 source-history/MMR 律的推导。
