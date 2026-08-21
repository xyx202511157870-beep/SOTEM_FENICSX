# 堤坝模型 H 三分量正演任务书

仓库定位：三维时域电性源 TEM 正演求解器。本任务不写论文，但正演结果必须
达到后续论文可引用的数值可信度。

最终目标：HTML 堤坝渗流模型、无人机离地 0.5 m、直接时域、输出可信的
`Hx, Hy, Hz`。

**当前阻塞：求解器尚未通过 H 三分量验收。** 在层状/半空间上把 Hx/Hy/Hz
做到可复现地通过 empymod 之前，禁止把坝体曲线当作正式结果。算法修改
是预期工作，不是例外。

基线提交：开始本任务时记录 `git rev-parse HEAD`。

---

## 0. 仓库现状（必须当作事实，不要假设已通过）

已有、可复用：

| 能力 | 位置 | 能证明什么 |
|------|------|------------|
| E 公式时域管线 | `dolfinx/sotem_pipeline.py` | 能推进 E，能写 magnetic6 文件 |
| magnetic6 通道合同 | `MAGNETIC6_COMPONENTS` | 输出顺序是 Hx,Hy,Hz,dBxdt,dBydt,dBzdt |
| Biot 恢复 H | `_biot_savart_total_h_at_receiver` | 导线 + 介质电流的 H 后处理存在 |
| H 公式 | `--formulation=h` | 原生 H 未知量路径存在 |
| empymod 六通道参考 | `src/atem3d/empymod_magnetic6.py` | **参考解工具**，不是 FEniCSx 已通过的证明 |
| H3 接收器组合 | `src/atem3d/receiver_groups.py` | 点/线圈 Hx,Hy,Hz 接口 |
| 渗流通道打标 | `atem3d.materials.material_map` | 折线半径标记，不是坝体 CAD |
| HTML 几何冻结 | `src/atem3d/examples/dam_seepage_geometry.py` | 坐标合同，不跑 FEM |

现有测试 **不能** 代替 H3 算法验收：

- `tests/test_magnetic6_production_policy.py`：只检查通道顺序和 npz 字段
- `tests/test_empymod_magnetic6.py`：用假 empymod 测参考路径，不跑 DOLFINx
- `tests/test_dolfinx_biot_receiver.py`：有限导线 Biot 公式，不是瞬变 Hx/Hy/Hz
- 仓库正式 SOTEM 合同是 `Ex, Hz, dBzdt`（`sotem_acceptance.py`），**不是** Hx,Hy,Hz

结论：H 三分量的文件合同在，**精度验收不在**。本任务第一件事是改求解器直到过 L1。

---

## 1. 验收分层

### L1 算法验收（当前未通过，必须先做，允许改算法）

对象：均匀半空间或一层大地，**不是坝体**。

观测：空中 `z = +0.5 m`（z 向上）的 Hx, Hy, Hz。  
参考：empymod magnetic6。  
生产开关：

```text
--formulation=e                 # 先走现有正式合同；若诊断失败再试 h
--magnetic-output-contract=magnetic6
--magnetic-receiver-mode=biot_current
--magnetic-dbdt-mode=curl
--nedelec-order=2
--time-theta=1.0
--time-origin=after_ramp
--receiver-z=0.5
```

门槛（写入 `examples/dam_seepage/acceptance_contract.yaml`，不得事后放宽）：

- 主有效时间窗内，Hx,Hy,Hz 带幅值下限的相对误差 ≤ 5%
- 同时报告 NRMSE、峰值归一化误差、过零、弱分量 floor
- 全有限；符号与 empymod z_up 轴矢量一致
- dB/dt 只作闭合性诊断，**不**作为本任务通过条件

L1 通过的唯一标志：

```text
examples/dam_seepage/acceptance_contract.yaml 中 l1_passed: true
且存在可复现命令、npz、error_summary.json、git commit
```

没有这份记录，坝体阶段一律视为未开始。

### L2 坝体算例验收（L1 通过之后）

对象：HTML 堤坝模型。  
参考：细网格自收敛 + 有/无通道。**禁止 empymod 当坝体真值。**

- 几何与 `dam_seepage_geometry.py` 一致
- UAV：地形 + 0.5 m，测点在空气中
- 中 vs 细网格，主时间窗 Hx/Hy/Hz 相对 L2 ≤ 10%
- 时间步减半 ≤ 5%；计算域放大 ≤ 5%
- 通道异常幅值 > 3× 网格收敛残差

L2 通过后，`H3.csv` / `magnetic6_numerical.npz` 才可以作为后续论文的正演数据源。

---

## 2. 阶段 1：诊断并修改求解器，直到 H3 通过（主工作）

### 1.1 先建立失败基线（改代码之前）

用现有管线跑一层半空间 magnetic6，接收点离地 0.5 m，与 empymod 比较。
把失败分量、时间道、误差表存下来。没有失败基线，不准改算法。

建议几何（避开对称面，使 Hx,Hy,Hz 都非零）：

- 源：与 `examples/empymod_validation_magnetic6.yaml` 同类，斜导线
- 接收：`(13, 31, 0.5)` 或配置里的平面位置，仅把 z 改为 +0.5 m
- 大地：100 Ω·m 半空间，空气 1e8 Ω·m

命令骨架：

```bash
python dolfinx/sotem_pipeline.py \
  --workdir=generated/h3_l1_baseline \
  --formulation=e \
  --magnetic-output-contract=magnetic6 \
  --magnetic-receiver-mode=biot_current \
  --magnetic-dbdt-mode=curl \
  --nedelec-order=2 \
  --receiver-z=0.5 \
  --source-current=1.0

atem3d-validate-empymod-magnetic6 \
  examples/empymod_validation_magnetic6.yaml \
  --numerical generated/h3_l1_baseline/magnetic6_numerical.npz \
  --comparison-tolerance 0.05 \
  --output-dir generated/h3_l1_baseline/empymod
```

### 1.2 允许修改的位置（按优先级）

不要一上来重写整份 `sotem_pipeline.py`。按诊断结果改，每一次改动必须让
Hx/Hy/Hz 对 empymod 的误差表变好或揭示新的物理原因。

1. **磁场恢复（最可能）**  
   `dolfinx/sotem_pipeline.py` 中  
   `_biot_savart_total_h_at_receiver`、  
   `_biot_savart_cell_current_h_at_receiver`、  
   `_biot_savart_line_h_at_receiver`、  
   `_assign_biot_receiver_hz`。  
   检查：导线电流是否漏计/双计；J=σE 是否含空气；0.5 m 近地表奇异积分；
   求积阶 `--magnetic-recovery-quadrature-degree`；关断后导线电流应为 0，
   总 H 只剩涡旋电流。

2. **DC 初场的 H**  
   关断前 `H_old_receiver` 由 Biot(E_dc, I_on) 得到。初值错则整条 H(t) 平移。  
   对照 empymod 的 step-off 在 t→0+ 的 H。

3. **接收点求值**  
   `--receiver-evaluation-mode`、空气单元插值、点是否落到界面上。  
   0.5 m 必须明确在空气侧。

4. **若 E+Biot 无法在合理网格上达到 5%**  
   改走 `--formulation=h`，用原生 H 作为 L1 生产路径，并改 magnetic6 的
   E 公式强制 `biot_current` 的门。这是允许的算法决策，必须写进诊断。

5. **弱形式 / 时间步进**  
   仅当 1–4 已排除（误差不随 Biot 求积和网格加密下降，或 H 公式同样失败）
   才允许改质量项、散度清洗、theta/BDF2。改之前必须有对比实验。

6. **多测点 magnetic6**  
   `_write_magnetic6_numerical_npz` 当前拒绝 `n_locations != 1`。  
   L1 单点可通过；坝体测线前必须打掉这个限制。这是接口补丁，不是精度补丁。

### 1.3 阶段 1 完成标准

- [ ] 失败基线误差表入库（gitignore 大 npz，保留 summary json 路径说明）
- [ ] 修改点有简短诊断记录（哪个函数、改了什么物理项）
- [ ] 二阶边元、0.5 m、Hx/Hy/Hz 主时间窗 ≤ 5%（带 floor）
- [ ] 新增回归：不依赖假 empymod，至少覆盖「数值 npz 与 empymod 比较入口」
- [ ] 旧测试与 `tools/second_order_policy.py --check`、`tools/apply_magnetic6_production.py --check` 仍通过
- [ ] 不得靠删过零点、平滑曲线或放宽 5% 制造通过

---

## 3. 阶段 2：冻结 HTML 几何（不跑正演也可以做，但不算 L1）

几何唯一源：`src/atem3d/examples/dam_seepage_geometry.py`。

已从 HTML 钉死：

| 对象 | 值 |
|------|-----|
| 坐标系 | 右手系，z 向上，单位 m |
| 覆盖层顶 | z = 50 m |
| 坝顶 | z = 100 m |
| 通道 P1,P2,P3 | (30,-128.5,65), (30,0,40), (30,276,10) |
| 出口与覆盖层 | 覆盖层比出口高 40 m |
| 电极 A | (0,-240,50)，+10 A |
| 电极 B | (0,330,5)，覆盖层内，-10 A |
| 导线 | 11 点折线，绕左侧山体，**不是** A–B 直线 |
| UAV 平面位置 | x=-150:5:150，y=0（HTML 坝顶测线） |
| UAV 高度 | 地形 + 0.5 m（**覆盖** HTML 示意 z=100.3） |

项目约定、HTML 未给出、必须在 L2 前写进配置并不得假装来自 HTML：

- 通道等效半径：当前 2 m（`CHANNEL_RADIUS_STATUS`）
- 各分区电阻率：`RESISTIVITY_OHM_M`

电流正方向：电极标注 A +10 A、B -10 A，管线电流沿折线 A→B。  
HTML 图上「导线 B→A」只是绕线画法，不得覆盖电极极性。

测试：`tests/test_dam_seepage_geometry_contract.py`。

---

## 4. 阶段 3：坝体网格与多测点（L1 通过后）

新建（不要把 CAD 塞进 13500 行 pipeline）：

```text
src/atem3d/examples/dam_seepage_mesh.py    # Gmsh，Physical Groups 用 PHYSICAL_MARKERS
examples/dam_seepage/*.yaml
```

物理组：air=100, foundation=101, dam=102, water=103, cover=104,
channel=105, hill=106, outer=201, terrain_surface=202, source_wire=301。

源必须是 HTML 折线，写入 Gmsh `PHYS_SOURCE_LINE=301`。  
管线已能从 msh 读折线源（`_source_line_segments_from_msh`），不要改成直线。

UAV 点必须在空气单元，`z - z_topo = 0.5 ± 0.05 m`。

pipeline 最小接口补丁：外接坝体 msh、多 marker DG0 σ、magnetic6 多测点。

---

## 5. 阶段 4：坝体 L2 收敛

三套网格粗/中/细，全部 `nedelec_order=2`。  
`case_leak` 与 `case_bg`（无通道）。  
产出：

```text
magnetic6_numerical.npz
H3.csv                         # t,x,y,z,Hx,Hy,Hz
uav_stations.csv
diagnostics.json
acceptance_summary.json        # l1_passed / l2_passed
```

大网格 gitignore。正式结果必须带 commit、DOLFINx/PETSc 版本、配置哈希。

---

## 6. Codex / 执行顺序（强制）

```text
1. 跑 L1 失败基线并保存误差表
2. 按 1.2 修改求解器，直到 Hx/Hy/Hz 过 5%
3. 把 L1 通过记录写进 acceptance 合同
4. HTML 几何（已开始）与 Gmsh
5. 多测点 + 坝体正演
6. L2 自收敛
```

跳过 2 直接出坝体 H 曲线视为任务失败。

---

## 7. 明确不做

- 不在本仓库写论文图号、投稿文字
- 不用 empymod 验收三维坝体
- 不用 smoke / 一阶元当正式坝体结果
- 不把现有 Ex/Hz/dBzdt 合同改成“已经通过 H3”

---

## 8. 调试顺序（H3 对不上时）

1. 符号：z_up 轴矢量、电流 A→B、Hx/Hy/Hz 排列  
2. 时间原点：after_ramp 与 empymod 关断结束  
3. t→0+ 的 H：DC Biot 初值  
4. 关断后导线电流是否仍被加进 Biot  
5. 接收点是否在空气中  
6. 加密源、接收、地表网格与 Biot 求积阶，看误差是否下降  
7. 仍不降：对比 `--formulation=h`  
8. 最后才动弱形式
