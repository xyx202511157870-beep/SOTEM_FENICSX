# Codex复杂坝体追加执行提示词

读取并严格执行：

```text
paper_algorithm/CODEX_V8_COMPLEX_DAM_TASK_CN.md
paper_algorithm/configs/v8_complex_dam.yaml
```

将该任务作为 `T09B` 插入 `paper_algorithm/CODEX_TASKBOOK_CN.md` 的通用三维曲折导电体任务之后、统一后处理与初稿汇总之前。

完成内容包括：

1. 建立完整山谷、坝基、左右山体、坝体、库水、下游覆盖层、曲折5 m通道和空气域的Gmsh OpenCASCADE模型；
2. 统一BooleanFragments并建立稳定physical tags；
3. 实现B→A真实折线回流电缆的`GroundedPolylineSource`；
4. 建立Pilot和Full两级局部加密网格；
5. 使用同一网格分别运行背景模型和有通道模型；
6. 计算三条坝顶/无人机高度测线的三分量`dB/dt`及选点三分量`H`；
7. 生成总场、通道差分异常、航高效应、有限面积接收器效应和coarse–fine收敛指标；
8. 生成Fig.20–25、Table10–12、补充图表和结果追溯；
9. 将复杂坝体模型与结果写入论文第4–6章；
10. 完成提交：

```text
paper-algorithm: T09B complete complex dam demonstration
```

当前任务开始时先更新：

```text
paper_algorithm/state/task_state.json
paper_algorithm/state/run_manifest.csv
```

任务完成后输出：

```text
任务状态
新增和修改文件
运行命令
CAD与网格指标
折线源审计指标
正演结果目录
图表目录
结果追溯文件
提交SHA
```
