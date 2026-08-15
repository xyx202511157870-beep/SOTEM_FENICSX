# empymod 磁场六分量验证

## 1. 固定数据合同

正式通道顺序固定为：

```text
Hx, Hy, Hz, dBxdt, dBydt, dBzdt
```

单位固定为：

```text
Hx, Hy, Hz              A/m
dBxdt, dBydt, dBzdt     T/s
```

该验证只用于均匀半空间或一维层状介质。empymod 不能作为复杂三维坝体的
参考解。正确顺序是：

```text
层状基准认证二阶有限元
        ->
复杂三维坝体网格、边界和时间收敛
        ->
坝体渗漏异常研究
```

## 2. dB/dt 参考路径

### empymod 2.5.4+

对理想阶跃关断，使用磁场 H 的冲激响应：

```python
mrec=True
signal=0
```

并根据

```text
dB/dt = -mu0 * H_impulse
```

获得 `dB/dt [T/s]`。

### empymod 2.6+

可以使用：

```python
mrec="b"
signal=-1
```

直接获得开关断响应的 `dB/dt [T/s]`。

命令行默认：

```text
--dbdt-reference auto
```

其行为为：

```text
empymod >= 2.6    native_b 为主参考，并与 impulse_h 交叉检查
empymod <  2.6    impulse_h 为主参考，记录 native_b 不可用
```

两条路径不一致通常表示：时间信号、坐标方向、磁场轴矢量符号、H/B 单位或
Fourier/Hankel 数值参数存在问题。

## 3. 有限关断波形

数值观测时间统一以**关断结束时刻**为零点。分段线性关断采用：

```text
y(t) = integral[-di(tau)/d tau * y_stepoff(t-tau) d tau]
```

其中 `tau <= 0`。程序使用每段 Gauss-Legendre 求积，将实际关断响应表示为
延迟理想阶跃响应的线性叠加。

### 5 us 线性关断

可直接从示例 YAML 读取：

```bash
atem3d-validate-empymod-magnetic6 \
  examples/empymod_validation_magnetic6_5us.yaml \
  --numerical output/magnetic6_numerical.npz \
  --depths 0 \
  --resistivities 1e8,100 \
  --waveform-quadrature-order 8 \
  --srcpts 9 \
  --srcpts-audit 17 \
  --output-dir output/empymod_magnetic6_5us
```

也可以覆盖 YAML 波形：

```bash
--ramp-off-time 5e-6
```

或使用任意分段线性波形：

```bash
--waveform-csv transmitter_turnoff.csv
```

CSV 格式为：

```text
time_s,current_scale
0.0,1.0
3.0e-6,0.4
5.0e-6,0.0
```

程序会把最后一个时间节点平移到关断结束时刻零点。若需要恢复理想阶跃基准，
使用：

```bash
--ideal-step-off
```

输出中的 `empymod_waveform_reference.json` 保存波形节点、求积阶数、权重守恒
误差和 empymod 实际求值时间数。

## 4. DOLFINx 六通道生产输出

DOLFINx 仍默认使用旧验证通道。正式启用六通道时设置：

```text
--magnetic-output-contract=magnetic6
```

E 形式必须同时使用：

```text
--formulation=e
--magnetic-receiver-mode=biot_current
--magnetic-dbdt-mode=curl
```

其物理分工为：

```text
H 三分量       总导电电流 + 外加有限导线电流的 Biot-Savart 场
dB/dt 三分量  -curl(E)，与 H 恢复路径相互独立
```

H 形式直接使用求解得到的 H 三分量，并由时间差分得到 dB/dt。生产运行固定为
二阶 `N1curl(2)`。

完成或部分完成的六通道运行会写出：

```text
magnetic6_numerical.npz
```

关键字段为：

```text
times                       (n_times,)
data                        (n_times, 1, 6)
components                  固定六分量顺序
units                       A/m, A/m, A/m, T/s, T/s, T/s
receiver_locations          (1, 3)
coordinate_system           z_up
time_origin                 after_ramp 或 ramp_start
ramp_off_time
source_current
nedelec_order               2
formulation
magnetic_receiver_mode
magnetic_dbdt_mode
magnetic_output_contract    magnetic6
```

当前自包含 DOLFINx 管线配置一个接收点；多测点数组应使用数组正演管线输出同一
六通道合同。

## 5. 数值文件格式

验证 CLI 推荐直接读取上述 `magnetic6_numerical.npz`。也支持单测点 CSV：

```text
time_obs,Hx,Hy,Hz,dBxdt,dBydt,dBzdt
```

通用 NPZ 的 `data` 可以是：

```text
(n_times, 6)
(n_times, n_locations, 6)
```

若保存 `components`，程序会自动重排到固定顺序。

## 6. 输出文件

```text
magnetic6_comparison.csv
magnetic6_error_summary.json
empymod_dbdt_crosscheck.json
empymod_waveform_reference.json
empymod_magnetic6_reference.npz
magnetic6_H3_comparison.png
magnetic6_dBdt3_comparison.png
empymod_source_quadrature_audit.json   # 设置 --srcpts-audit 时
```

## 7. 几何与误差要求

六分量基准应采用非对称源—接收几何，避免某个分量因理论对称性接近零。
正式验收至少应同时检查：

- 分量峰值归一化误差；
- 带幅值下限的相对误差；
- 正负号一致性；
- `srcpts` 有限长源积分收敛；
- 波形求积阶数收敛；
- 二阶有限元网格、外边界和时间步收敛。

## 8. 与无人机三分量线圈的关系

该基准比较点位场量：

```text
H [A/m]
dB/dt [T/s]
```

实际感应线圈直接输出：

```text
u = -N d/dt integral(B dot n) dA
```

因此，层状点位验证通过后，还需要独立验证线圈有效面积、三轴法向、姿态旋转、
前放和接收机复传递函数、有限面积积分，以及由线圈电压恢复 dB/dt 和 H 的
处理链。
