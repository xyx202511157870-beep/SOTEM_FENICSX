# Codex 总执行提示词

你正在仓库 `xyx202511157870-beep/SOTEM_FENICSX` 中工作。

## 总目标

在分支 `paper/algorithm-forward-matrix` 上，严格执行 `paper_algorithm/CODEX_TASKBOOK_CN.md` 的 T00–T13，完成算法论文中文初稿 v0.1、全部正演、结果图、表格、结果追溯和 QA 审阅包。

论文暂定题目：

> A validated second-order edge-element framework for three-dimensional grounded-source transient electromagnetic modelling with induced polarization, realistic transmitter waveforms and finite-area receivers

论文采用：

- `dBxdt, dBydt, dBzdt` 作为主要磁观测量；
- `Hx, Hy, Hz` 作为独立恢复与 Maxwell 闭合审计量；
- `Ex, Ey` 作为 SOTEM 文献基准对照量；
- 二阶 `N1curl(2)` 作为正式空间离散；
- 后向 Euler 作为正式时间推进；
- empymod 作为均匀半空间和层状介质独立参考；
- coarse–fine DOLFINx 对比作为三维异常数值收敛证据。

## 工作方式

1. 读取：

```text
paper_algorithm/CODEX_TASKBOOK_CN.md
paper_algorithm/README_CN.md
paper_algorithm/run_algorithm_paper.sh
README.md
```

2. 创建并持续更新：

```text
paper_algorithm/state/task_state.json
paper_algorithm/state/run_manifest.csv
```

3. 从 T00 开始，按顺序执行到 T13。
4. 每个任务先输出任务计划，再修改文件、运行命令、分析结果、生成产物、更新状态并提交。
5. 每个任务形成独立 Git commit。
6. 每个正演保存运行命令、配置、日志、提交 SHA、运行时间、峰值内存、自由度和输出哈希。
7. 每个图件由独立 Python 脚本从正演原始输出重建。
8. 每个正文数值登记到 `results_traceability.csv`。
9. 每个模型参数同时进入配置文件、表格和正文参数审计。
10. 门槛外项目进入诊断队列；完成源积分、网格、时间步、边界、波形、接收算子或求解器定位后，重新运行对应任务并更新证据。

## 首轮执行

依次执行：

```bash
conda activate fenicsx
git checkout paper/algorithm-forward-matrix
python -m pip install -e .

bash paper_algorithm/run_algorithm_paper.sh env
bash paper_algorithm/run_algorithm_paper.sh preflight
bash paper_algorithm/run_algorithm_paper.sh benchmark-pilot
```

完成 T00–T04 的基础目录、状态文件、方法—代码映射、第 2–4 章初稿和 pilot 结果报告。

## 第二轮执行

依次执行：

```bash
bash paper_algorithm/run_algorithm_paper.sh magnetic6
bash paper_algorithm/run_algorithm_paper.sh convergence
bash paper_algorithm/run_algorithm_paper.sh waveform
bash paper_algorithm/run_algorithm_paper.sh receiver
```

创建以下后处理模块：

```text
paper_algorithm/postprocess/collect_case_results.py
paper_algorithm/postprocess/build_error_tables.py
paper_algorithm/postprocess/build_convergence_figures.py
paper_algorithm/postprocess/build_magnetic_figures.py
paper_algorithm/postprocess/build_closure_error.py
paper_algorithm/postprocess/build_waveform_figures.py
paper_algorithm/postprocess/build_receiver_figures.py
```

完成 Fig. 1–11 和 Table 1–6 的候选版本。

## 第三轮执行

依次执行：

```bash
IP_LEVEL=S1T1B1 IP_TERMS=8,12,16,20 \
  bash paper_algorithm/run_algorithm_paper.sh ip

DEMO3D_PROFILE=pilot \
  bash paper_algorithm/run_algorithm_paper.sh demo3d

DEMO3D_PROFILE=full FORCE=1 \
  bash paper_algorithm/run_algorithm_paper.sh demo3d
```

创建：

```text
paper_algorithm/postprocess/build_ip_figures.py
paper_algorithm/postprocess/build_runtime_tables.py
paper_algorithm/postprocess/build_3d_demo_figures.py
paper_algorithm/postprocess/build_model_schematics.py
```

完成 Fig. 12–19 和 Table 7–9 的候选版本。

## 第四轮执行

1. 汇总全部正演结果。
2. 生成最终正文候选图 10–12 张和正文候选表 5–7 张。
3. 生成补充图和补充表。
4. 创建：

```text
paper_algorithm/manuscript/draft_v0_1_cn.md
paper_algorithm/manuscript/abstract_cn.md
paper_algorithm/manuscript/abstract_en.md
paper_algorithm/manuscript/figure_captions_cn.md
paper_algorithm/manuscript/table_captions_cn.md
paper_algorithm/manuscript/references.bib
paper_algorithm/manuscript/results_traceability.csv
```

5. 初稿按以下顺序写作：

```text
第 2 章 -> 第 3 章 -> 第 4 章 -> 第 5 章 -> 第 6 章 -> 第 7 章 -> 第 1 章 -> 摘要
```

6. 初稿正文长度设置为 12,000–18,000 字，摘要 600–900 字，英文摘要 300–450 词。
7. 每个结果小节写明模型、指标、定量结果、图表引用和解释。
8. 每个定量结果写入结果追溯表。

## 第五轮执行

1. 执行全量测试。
2. 完成单位、坐标、源方向、时间原点、H/B/dBdt、Cole–Cole 参数、图件、表格和正文数字审计。
3. 创建：

```text
paper_algorithm/qa/qa_report.md
paper_algorithm/qa/unit_consistency.csv
paper_algorithm/qa/figure_audit.csv
paper_algorithm/qa/parameter_audit.csv
paper_algorithm/qa/manuscript_number_audit.csv
```

4. 创建：

```text
paper_algorithm/review_package_v0_1/
```

5. 将初稿、图、表、QA、运行清单和结果追溯表复制到审阅包。
6. 提交：

```text
paper-algorithm: T13 complete draft v0.1 review package
```

## 每轮回复格式

开始时：

```text
当前任务：Txx
计划修改：...
计划运行：...
预期产物：...
```

结束时：

```text
任务状态：completed / diagnostic
修改文件：...
运行命令：...
关键指标：...
图表产物：...
结果目录：...
提交 SHA：...
下一任务：...
```

现在从 T00 开始执行，并持续推进到 `review_package_v0_1` 完成。