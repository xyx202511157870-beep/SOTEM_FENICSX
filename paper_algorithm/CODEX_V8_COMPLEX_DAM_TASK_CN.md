# V8 复杂坝体最终三维算例任务书

## 1. 任务定位

在完成均匀半空间、层状介质、真实波形、有限面积接收器、IP 和通用三维曲折导电体验证后，建立完整复杂坝体模型，作为算法处理复杂地形、多材料、折线接地源、曲折细通道和接收阵列的最终三维展示算例。

模型参数的唯一来源为：

```text
paper_algorithm/configs/v8_complex_dam.yaml
```

Codex 应将本任务插入总任务书的 T09 之后、统一后处理之前，任务编号记为 `T09B`。

---

## 2. 最终产物

创建：

```text
paper_algorithm/complex_dam/
├── build_complex_dam_mesh.py
├── validate_complex_dam_geometry.py
├── run_complex_dam_forward.py
├── postprocess_complex_dam.py
├── plot_complex_dam_model.py
├── plot_complex_dam_mesh.py
├── plot_complex_dam_fields.py
├── plot_complex_dam_responses.py
├── plot_complex_dam_convergence.py
└── README_CN.md

paper_algorithm/tests/
├── test_grounded_polyline_source.py
├── test_complex_dam_geometry.py
├── test_complex_dam_material_tags.py
└── test_complex_dam_receiver_array.py

generated/paper_algorithm/v8_complex_dam/
├── geometry/
├── mesh_pilot/
├── mesh_full/
├── source_audit/
├── D0_background_pilot/
├── D1_channel_r5_pilot/
├── D0_background_full/
├── D1_channel_r5_full/
├── postprocess/
└── figures/
```

在论文中新增：

```text
4.X Complex dam model and computational setup
5.X Complex-geometry demonstration
6.X Numerical implications for complex engineering models
```

---

# 3. T09B-01 复杂坝体 CAD

## 3.1 读取配置

读取 `paper_algorithm/configs/v8_complex_dam.yaml`，建立右手坐标系：

```text
x：坝轴方向
y：上游至下游方向
z：高程，向上为正
```

## 3.2 创建材料体

使用 Gmsh OpenCASCADE 构建：

1. 外部计算域；
2. 坝基水平基岩体；
3. 左、右山体；
4. 沿 x 方向多截面 loft 的坝体；
5. 沿 x 方向多截面 loft 的库水；
6. 沿 x 方向多截面 loft 的下游覆盖层；
7. 沿三控制点中心线构建的半径 5 m 曲折通道体；
8. 空气体。

坝体、库水和覆盖层截面严格使用 YAML 中的 `ground(x)`、`y_up(z)` 和 `y_down(z)` 公式。

## 3.3 布尔分割

对全部材料体执行统一 `BooleanFragments`，生成互不重叠的体。为以下材料创建稳定 physical tags：

```text
AIR
FOUNDATION_MOUNTAIN
DAM
RESERVOIR_WATER
DOWNSTREAM_COVER
LEAKAGE_CHANNEL
```

通道体在背景模型和有通道模型中使用同一网格及同一 physical tag，仅切换材料参数。

## 3.4 几何审计

输出：

```text
cad_volume_and_physical_tag_report.json
complex_dam_geometry.brep
complex_dam_geometry.geo_unrolled
complex_dam_geometry_preview.x3d 或 vtk
```

报告每个 physical volume 的：

```text
体积
包围盒
质心
实体数量
与其他材料重叠体积
```

验证：

- 每个材料体积为正；
- 材料体之间无体积重叠；
- 通道入口位于坝体内；
- 通道中段位于坝体内；
- 通道出口位于坝体—覆盖层界面邻域；
- B 电极位于覆盖层内部；
- 所有接收点位于空气中；
- 计算域完整包围全部实体。

生成：

```text
Fig. 20a 三维材料体
Fig. 20b 上下游剖面
Fig. 20c 沿坝轴剖面
Fig. 20d 通道入口、中段和出口局部放大
```

---

# 4. T09B-02 折线接地源实现与验证

## 4.1 新增源类

在 `src/atem3d/sources.py` 中新增：

```python
GroundedPolylineSource(vertices, current, waveform, ...)
```

源顶点按 YAML 中 B→A 的真实电缆电流方向排列。离散源向量为各线段积分之和：

\[
\mathbf q=I\sum_{s=1}^{N_s}
\int_{\Gamma_s}\mathbf N\cdot d\mathbf l.
\]

共享折点只作为相邻线段公共端点，不产生重复点源项。

## 4.2 配置与输出

为 DOLFINx 管线增加：

```text
--source-polyline-yaml
--source-polyline-current
--source-polyline-direction
```

输出：

```text
source_polyline_vertices.csv
source_polyline_segment_report.csv
source_polyline_audit.json
```

## 4.3 单元测试

完成四项测试：

1. 两顶点折线与现有直线源结果一致；
2. 多段折线源等于各独立线段源向量的代数和；
3. 顶点顺序反转后源向量整体变号；
4. MPI 分区下每个线段原子区间唯一归属且总覆盖率为 1。

记录：

```text
线段总长度
每段长度
单元覆盖率
端点电流平衡
原始与电荷守恒修正残差
边界消元后残差
```

生成 Supplementary Fig. S8：折线源路径、局部网格与分段积分审计。

---

# 5. T09B-03 多材料网格

## 5.1 Pilot 网格

使用 YAML 中 `mesh_profiles.pilot`：

```text
通道附近：2.5 m
源路径附近：5 m
接收线附近：5 m
坝体和覆盖层：15 m
山体：30 m
远区地下：120 m
空气：150 m
二阶 Nédélec
```

## 5.2 Full 网格

使用 YAML 中 `mesh_profiles.full`：

```text
通道附近：1.25 m
源路径附近：2.5 m
接收线附近：2.5 m
坝体和覆盖层：8 m
山体：20 m
远区地下：80 m
空气：100 m
二阶 Nédélec
```

## 5.3 网格检查

输出：

```text
mesh_quality_report.json
mesh_statistics.csv
material_cell_counts.csv
channel_cross_section_resolution.csv
```

报告：

- 四面体数量；
- 二阶边元自由度；
- 每个材料的单元数；
- 最小和分位数质量；
- 最大长宽比；
- 通道直径上的单元层数；
- 源和接收器周围局部网格尺寸；
- 内存预估。

生成：

```text
Fig. 21a 全域网格
Fig. 21b 坝体剖面网格
Fig. 21c 通道附近网格
Fig. 21d 源与接收线附近网格
```

---

# 6. T09B-04 正演矩阵

## 6.1 统一设置

```text
电流：10 A
关断：5 μs线性关断
关断内步数：10
时间原点：关断结束
输出时间：1e-6–1e-1 s，51个对数点
正式空间离散：N1curl(2)
正式时间推进：后向Euler
主要输出：dBxdt,dBydt,dBzdt
审计输出：选定接收点的Hx,Hy,Hz
```

## 6.2 接收阵列

一次正演同时计算三条测线：

```text
R0：z=100.3 m
R1：z=105.0 m
R2：z=110.0 m
```

每条测线：

```text
x=-150:5:150 m
y=0 m
61个接收点
```

同时计算：

```text
point receiver
disk-average receiver，半径1 m，法向x/y/z
```

## 6.3 正演案例

### D0_background_pilot

- Pilot 网格；
- 通道 physical volume 保留；
- 通道材料设置为所在宿主材料；
- 保存背景响应。

### D1_channel_r5_pilot

- 使用与 D0 完全相同的 Pilot 网格文件；
- 通道电阻率设为 10 Ω·m；
- 保存总响应和异常响应。

### D0_background_full

- Full 网格；
- 通道材料设置为宿主材料。

### D1_channel_r5_full

- 使用与 D0 full 完全相同的网格文件；
- 通道电阻率设为 10 Ω·m。

## 6.4 运行脚本

创建：

```text
paper_algorithm/run_complex_dam_demo.py
```

支持：

```bash
python paper_algorithm/run_complex_dam_demo.py \
  --config paper_algorithm/configs/v8_complex_dam.yaml \
  --profile pilot \
  --case all

python paper_algorithm/run_complex_dam_demo.py \
  --config paper_algorithm/configs/v8_complex_dam.yaml \
  --profile full \
  --case all \
  --resume
```

每个正演保存：

```text
resolved_config.json
mesh_hash.txt
source_hash.txt
receiver_hash.txt
forward_checkpoint.npz
response.npz
run.log
runtime_memory.json
```

---

# 7. T09B-05 后处理指标

定义：

\[
\Delta d_j(x,t)=
d_{j,channel}(x,t)-d_{j,background}(x,t).
\]

计算：

### 7.1 总场误差

\[
\varepsilon_{cf,j}^{total}=
\frac{\|d_{j,pilot}-d_{j,full}\|_2}
{\|d_{j,full}\|_2}.
\]

### 7.2 异常场误差

\[
\varepsilon_{cf,j}^{anom}=
\frac{\|\Delta d_{j,pilot}-\Delta d_{j,full}\|_2}
{\max(\|\Delta d_{j,full}\|_2,d_{floor})}.
\]

### 7.3 异常—数值差异比

\[
R_{A/N,j}=
\frac{\|\Delta d_{j,full}\|_2}
{\|d_{j,pilot}-d_{j,full}\|_2}.
\]

### 7.4 点—有限面积差异

\[
\delta_{coil,j}=
\frac{\|d_{j,disk}-d_{j,point}\|_2}
{\max(\|d_{j,point}\|_2,d_{floor})}.
\]

### 7.5 航高效应

比较 R0、R1 和 R2 的：

```text
峰值异常
异常峰值位置
峰值时间
时空L2范数
```

### 7.6 空间定位指标

对每个时间道提取：

```text
异常峰值x坐标
异常质心x坐标
半峰宽
通道投影位置误差
```

全部写入：

```text
complex_dam_metrics.csv
complex_dam_convergence.csv
complex_dam_height_effect.csv
complex_dam_receiver_effect.csv
```

---

# 8. T09B-06 图件和表格

生成：

## 正文候选图

```text
Fig. 20 复杂坝体CAD、材料体和曲折通道
Fig. 21 Pilot/Full网格及局部加密
Fig. 22 三个代表时间的三分量dB/dt场快照
Fig. 23 三条航高测线的总场响应
Fig. 24 曲折通道异常的x-t热图
Fig. 25 Pilot–Full收敛及异常—数值差异比
```

代表时间从实际扩散过程自动选择：

```text
早期：通道异常首次超过峰值的10%
中期：异常达到峰值
晚期：异常衰减至峰值的10%
```

## 正文候选表

```text
Table 10 复杂坝体几何、材料、源和接收参数
Table 11 Pilot/Full网格与计算成本
Table 12 总场、异常场、航高和有限面积指标
```

## 补充材料

```text
Fig. S8 折线源积分审计
Fig. S9 全材料physical tags和体积
Fig. S10 每个分量完整x-t热图
Fig. S11 H恢复选点审计
Table S5 全接收点和时间道指标
```

所有图输出 PDF 和 600 dpi PNG，保存绘图数据 CSV/NPZ 和绘图脚本。

---

# 9. T09B-07 论文初稿写作

在初稿中新增以下内容。

## 第4章模型设置

写明：

- 坐标定义；
- 山谷地形；
- 坝体多截面 loft；
- 库水和覆盖层；
- 曲折5 m通道；
- B→A折线回流电缆；
- 三条坝顶接收线；
- 材料参数；
- Pilot/Full网格；
- 相同网格背景差分策略。

## 第5章结果

按以下顺序写：

1. CAD与网格质量；
2. 折线源积分和电流守恒；
3. 三维场扩散快照；
4. 三分量坝顶响应；
5. 曲折通道差分异常；
6. 接收高度与有限面积效应；
7. Pilot–Full异常收敛；
8. 运行时间和内存。

## 第6章讨论

讨论：

- 多材料复杂几何对源场和扩散场的影响；
- 复杂背景下总场和差分场的不同数值要求；
- 细通道响应与网格误差的量级关系；
- 折线回流电缆对磁场计算的贡献；
- 点接收器和有限面积接收器的差异；
- 航高对空间分辨率的影响。

每个定量结果写入 `results_traceability.csv`。

---

# 10. 完成条件

T09B 完成时，必须同时具备：

1. 可重复生成的 Gmsh CAD 和 Pilot/Full 网格；
2. 稳定 physical tags 和材料体积报告；
3. 折线源类、配置入口和单元测试；
4. 背景与通道使用相同网格；
5. Pilot 与 Full 正演结果；
6. 三分量 `dB/dt` 响应和选点 `H` 审计；
7. 总场与异常场 coarse–fine 指标；
8. Fig. 20–25 和 Table 10–12；
9. 论文第4–6章对应段落；
10. 正演日志、配置、哈希、运行时间、内存和结果追溯记录。

完成提交信息：

```text
paper-algorithm: T09B complete complex dam demonstration
```
