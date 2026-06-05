# 短偏移距电性源瞬变电磁法

这个仓库保留可复用的 SOTEM/IP 计算工具代码，删除了 COMSOL 大模型、运行输出、临时调试图、缓存文件和本地论文 PDF。

## 保留内容

- `sotem_ip/`: 核心 Python 包。
- `examples/`: 小型可复现实例。
- `tests/`: 轻量单元测试。

## 主要能力

- Cole-Cole 复电阻率/复电导率计算。
- Cole-Cole 到 Debye/Prony 记忆项拟合。
- 有限长电性源测线几何一致性检查。
- 可选 empymod 参考响应接口。
- 响应误差和 IP 百分比效应后处理。

## 安装

```bash
python -m pip install -e .[dev,plot,empymod]
```

如果只需要基础数值函数：

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest -q
```

## 说明

当前目录里如果还看到 `*.pdf`，说明该文件被本机 PDF 阅读器占用，Windows 暂时无法删除。`.gitignore` 已经排除 PDF，不会进入 Git 提交。

