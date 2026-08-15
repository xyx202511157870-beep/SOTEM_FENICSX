# empymod 磁场六分量验证

## 1. 验证对象

正式通道顺序固定为：

```text
Hx, Hy, Hz, dBxdt, dBydt, dBzdt
```

单位固定为：

```text
Hx, Hy, Hz              A/m
dBxdt, dBydt, dBzdt     T/s
```

该验证只用于均匀半空间或一维层状介质。empymod 不能作为复杂三维坝体模型
的参考解；正确流程是先在层状基准中认证二阶有限元的源、坐标、分量、符号、
时间波形和单位，再进入三维坝体模型。

## 2. dB/dt 参考路径

### 兼容路径：empymod 2.5.4+

对理想阶跃关断，使用磁场 H 的冲激响应：

```python
mrec=True
signal=0
```

并根据

```text
dB/dt = -mu0 * H_impulse
```

获得 `dB/dt [T/s]`。该路径兼容项目当前的 Python 3.10 环境。

### 原生路径：empymod 2.6+

empymod 2.6 及以上版本可以使用：

```python
mrec="b"
signal=-1
```

直接获得开关断响应的 `dB/dt [T/s]`。

命令行默认采用：

```text
--dbdt-reference auto
```

其行为为：

```text
empymod >= 2.6    native_b 为主参考，并与 impulse_h 交叉检查
empymod <  2.6    impulse_h 为主参考，记录原生交叉检查不可用
```

显式设置：

```text
--dbdt-reference native_b
```

时，若 empymod 版本低于 2.6，程序会直接报错。

两条路径不一致通常意味着以下问题之一：

- 开关断/脉冲信号约定错误；
- `z_up` 与 empymod 深度正方向转换错误；
- 磁场轴矢量的水平分量符号错误；
- H、B、dB/dt 单位混淆；
- Fourier/Hankel 数值参数不充分。

## 3. 数值数据格式

### CSV（单测点）

表头必须包含：

```text
time_obs,Hx,Hy,Hz,dBxdt,dBydt,dBzdt
```

`time_s` 也可以代替 `time_obs`。

### NPZ（单测点或多测点）

必须包含：

```text
times
data
```

`data` 形状为：

```text
(n_times, 6)
```

或：

```text
(n_times, n_locations, 6)
```

建议同时保存：

```text
components
receiver_locations
```

若给出 `components`，程序会自动重排到固定六分量顺序。

## 4. 命令示例

```bash
atem3d-validate-empymod-magnetic6 \
  examples/empymod_validation_magnetic6.yaml \
  --numerical output/magnetic6_numerical.npz \
  --depths 0 \
  --resistivities 1e8,100 \
  --srcpts 9 \
  --srcpts-audit 17 \
  --dbdt-reference auto \
  --audit-tolerance 0.01 \
  --comparison-tolerance 0.05 \
  --output-dir output/empymod_magnetic6
```

在 empymod 2.6+ 环境中，要求两条 dB/dt 路径必须同时存在并通过时，可增加：

```bash
--require-audit-pass
```

含 Debye/IP 层时，可改用：

```bash
--use-config-ip
```

此时不再传 `--resistivities`，程序依据 YAML 中各层的 `sigma_infinity` 和
`debye_terms` 建立 empymod 复电导率模型。

## 5. 输出文件

```text
magnetic6_comparison.csv
magnetic6_error_summary.json
empymod_dbdt_crosscheck.json
empymod_magnetic6_reference.npz
magnetic6_H3_comparison.png
magnetic6_dBdt3_comparison.png
empymod_source_quadrature_audit.json   # 仅设置 --srcpts-audit 时
```

`empymod_magnetic6_reference.npz` 还会保存：

```text
primary_dbdt_reference
empymod_version
dbdt_native
dbdt_impulse
```

不可用的参考路径保存为空数组。

## 6. 几何要求

六分量验证应采用非对称源—接收几何。若接收点位于源中垂面、源延长线或其他
对称面，某些分量理论上接近零，此时普通相对误差会失去意义。验证程序采用
分量峰值的一定比例作为误差下限，但正式基准仍应主动选择使六个通道均具有
可辨信号的几何。

## 7. 当前波形边界

该正式入口针对理想阶跃关断：

```text
signal = -1
```

你实际系统若采用 5 us 线性关断，不能直接把理想阶跃 empymod 曲线与完整
关断过程逐点比较。应先完成实际发射电流波形卷积，或者只比较已经统一到
“关断后时间原点”的二次场响应。实际波形验证应作为下一阶段独立模块。

## 8. 与无人机三分量线圈的关系

该基准比较的是点位场量：

```text
H [A/m]
dB/dt [T/s]
```

真实感应线圈直接输出电压：

```text
u = -N d/dt ∫ B·n dA
```

因此，empymod 点位验证通过后，还需要独立验证：

- 三轴线圈有效面积；
- 线圈法向与姿态旋转；
- 前放和接收机复传递函数；
- 有限面积积分；
- 由线圈电压恢复 dB/dt 和 H 的处理链。
