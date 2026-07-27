# SOTEM FEniCSx 中文算法仓库发布 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将最新SOTEM FEniCSx算法整理为仅包含源码、配置、示例和自包含测试的中文说明仓库，并安全更新到远端`main`。

**Architecture:** 保留现有Git历史，在独立发布分支上用允许清单构造干净快照。数值算法保持不变，只修正Python安装约束、移除依赖本机输出的报告测试、重写中文README，并为关键物理与数值模块增加中文注释。

**Tech Stack:** Python 3.10、FEniCSx/DOLFINx、PETSc、MPI、Gmsh、NumPy、SciPy、SimPEG、empymod、pytest、Git。

---

## 文件结构

最终保留：

```text
.gitignore
LICENSE
README.md
pyproject.toml
benchmarks/
dolfinx/
examples/
sotem_ip/
src/atem3d/
tests/
```

最终删除：

```text
assets/
docs/
output/
scripts/
IMPLEMENTATION_REPORT.md
TASK_BOOK_STATUS.md
deep-research-report.md
dolfinx/current_task_runs/
dolfinx/current_acceptance_summary.md
dolfinx/sponge_boundary_validation_notes.md
tests/test_audit_zhou2020_reference_stability.py
tests/test_plot_zhou2020_strict_validation.py
tests/test_zhou2020_validation_report.py
```

## Task 1：用测试锁定公开仓库文件边界

**Files:**

- Create: `tests/test_public_repository_layout.py`
- Delete: `assets/`
- Delete: `docs/`
- Delete: `output/`
- Delete: `scripts/`
- Delete: `IMPLEMENTATION_REPORT.md`
- Delete: `TASK_BOOK_STATUS.md`
- Delete: `deep-research-report.md`
- Delete: `dolfinx/current_task_runs/`
- Delete: `dolfinx/current_acceptance_summary.md`
- Delete: `dolfinx/sponge_boundary_validation_notes.md`
- Delete: `tests/test_audit_zhou2020_reference_stability.py`
- Delete: `tests/test_plot_zhou2020_strict_validation.py`
- Delete: `tests/test_zhou2020_validation_report.py`

- [ ] **Step 1：编写失败的公开布局测试**

测试内容：

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_has_only_approved_top_level_entries():
    approved = {
        ".git",
        ".gitignore",
        ".pytest_cache",
        ".dolfinx.writer.lock",
        "LICENSE",
        "README.md",
        "benchmarks",
        "dolfinx",
        "examples",
        "pyproject.toml",
        "sotem_ip",
        "src",
        "tests",
    }
    present = {path.name for path in ROOT.iterdir()}
    unexpected = sorted(present - approved)
    assert unexpected == []


def test_public_repository_contains_no_tracked_runtime_artifacts():
    forbidden = {
        "current_task_runs",
        "output",
        "outputs",
        "generated",
        "assets",
        "scripts",
    }
    found = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name in forbidden
    )
    assert found == []
```

- [ ] **Step 2：运行测试并确认失败**

Run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_public_repository_layout.py
```

Expected: FAIL，并列出`assets`、`docs`、`output`、`scripts`等当前非发布内容。

- [ ] **Step 3：删除明确排除的已跟踪文件**

Run:

```bash
git rm -r assets docs output scripts dolfinx/current_task_runs
git rm IMPLEMENTATION_REPORT.md TASK_BOOK_STATUS.md deep-research-report.md
git rm dolfinx/current_acceptance_summary.md
git rm dolfinx/sponge_boundary_validation_notes.md
git rm tests/test_audit_zhou2020_reference_stability.py
git rm tests/test_plot_zhou2020_strict_validation.py
git rm tests/test_zhou2020_validation_report.py
```

删除前用`git ls-files`核对这些目标全部位于当前工作树内。

- [ ] **Step 4：清理工作树中被忽略的运行产物**

只清理本发布工作树内由测试或失败安装生成的：

```text
.pytest_cache/
src/atem3d.egg-info/
__pycache__/
```

不删除主工作区或其他工作树中的任何文件。

- [ ] **Step 5：运行公开布局测试并确认通过**

Run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_public_repository_layout.py
```

Expected: PASS。

- [ ] **Step 6：提交仓库清理**

```bash
git add tests/test_public_repository_layout.py
git commit -m "清理：仅保留SOTEM FEniCSx算法与自包含测试"
```

## Task 2：修正Python安装约束

**Files:**

- Modify: `pyproject.toml`
- Create: `tests/test_package_metadata.py`

- [ ] **Step 1：编写Python 3.10兼容性测试**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_supports_validated_python_310():
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert metadata["project"]["requires-python"] == ">=3.10"
```

- [ ] **Step 2：运行测试并确认失败**

Run:

```bash
python -m pytest -q tests/test_package_metadata.py
```

Expected: FAIL，当前值为`>=3.11`。

- [ ] **Step 3：修改项目元数据**

将：

```toml
requires-python = ">=3.11"
```

改为：

```toml
requires-python = ">=3.10"
```

并将项目描述改为中文：

```toml
description = "基于FEniCSx的接地电性源三维时域TEM-IP正演与验证工具。"
```

不加入`python-docx`，因为Word报告生成器已经删除。

- [ ] **Step 4：运行元数据测试并确认通过**

Run:

```bash
python -m pytest -q tests/test_package_metadata.py
```

Expected: PASS。

- [ ] **Step 5：验证可编辑安装**

Run:

```bash
python -m pip install -e .
python -c "import atem3d; print(atem3d.__file__)"
python -m atem3d.initial_field_diagnostics_cli --help
```

Expected: 三条命令退出码均为0，模块路径位于当前发布工作树。

- [ ] **Step 6：提交安装修复**

```bash
git add pyproject.toml tests/test_package_metadata.py
git commit -m "修复：支持已验证的Python 3.10 FEniCSx环境"
```

## Task 3：重写中文README并补充许可说明

**Files:**

- Replace: `README.md`
- Create: `LICENSE`
- Test: `tests/test_public_repository_layout.py`

- [ ] **Step 1：扩展文档存在性测试**

在`tests/test_public_repository_layout.py`加入：

```python
def test_public_documents_are_chinese_and_do_not_overclaim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "三维时域" in readme
    assert "Cole–Cole" in readme
    assert "不能替代现场工程验证" in readme
    assert "保留所有权利" in license_text
```

- [ ] **Step 2：运行测试并确认失败**

Run:

```bash
python -m pytest -q tests/test_public_repository_layout.py
```

Expected: FAIL，因为`LICENSE`尚不存在，README也不满足中文发布要求。

- [ ] **Step 3：重写README**

README必须包含：

```text
项目定位
算法结构
坐标与单位
Cole–Cole和Debye约定
DC初始场
FEniCSx/WSL安装
最小运行命令
公开基准配置
输出量
测试命令
已知限制
引用与许可
```

明确写出：

```text
自动化测试仅证明当前代码在规定环境中的内部一致性，
不能替代网格/时间步收敛、独立软件对比和现场工程验证。
```

- [ ] **Step 4：添加保守许可文件**

在用户没有明确授权开源许可证的情况下，采用不自动授予再分发权利的中文声明：

```text
Copyright (c) 2026 SOTEM_FENICSX contributors
保留所有权利。

本仓库仅供作者授权的科研、教学和内部验证使用。
未经版权所有者书面许可，不得复制、再发布或用于商业用途。
第三方依赖仍分别遵循其原始许可证。
```

- [ ] **Step 5：运行文档测试**

Run:

```bash
python -m pytest -q tests/test_public_repository_layout.py
```

Expected: PASS。

- [ ] **Step 6：提交中文文档**

```bash
git add README.md LICENSE tests/test_public_repository_layout.py
git commit -m "文档：添加中文安装运行与算法说明"
```

## Task 4：为核心算法加入中文注释

**Files:**

- Modify: `dolfinx/sotem_pipeline.py`
- Modify: `src/atem3d/simulation.py`
- Modify: `src/atem3d/hj.py`
- Modify: `src/atem3d/materials/cole_cole.py`
- Modify: `src/atem3d/materials/prony.py`
- Modify: `src/atem3d/primary/dc.py`
- Modify: `src/atem3d/sources.py`
- Modify: `src/atem3d/receivers.py`
- Modify: `src/atem3d/magnetic_recovery.py`
- Modify: `sotem_ip/cole_cole.py`
- Modify: `sotem_ip/debye.py`
- Modify: `sotem_ip/forward.py`
- Create: `tests/test_chinese_core_comments.py`

- [ ] **Step 1：编写中文注释覆盖测试**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = [
    "dolfinx/sotem_pipeline.py",
    "src/atem3d/simulation.py",
    "src/atem3d/hj.py",
    "src/atem3d/materials/cole_cole.py",
    "src/atem3d/materials/prony.py",
    "src/atem3d/primary/dc.py",
    "src/atem3d/sources.py",
    "src/atem3d/receivers.py",
    "src/atem3d/magnetic_recovery.py",
    "sotem_ip/cole_cole.py",
    "sotem_ip/debye.py",
    "sotem_ip/forward.py",
]


def test_core_algorithm_files_contain_chinese_explanations():
    missing = []
    for relative in CORE_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if not any("\u4e00" <= char <= "\u9fff" for char in text):
            missing.append(relative)
    assert missing == []
```

- [ ] **Step 2：运行测试并确认当前未覆盖文件**

Run:

```bash
python -m pytest -q tests/test_chinese_core_comments.py
```

Expected: FAIL，并列出尚无中文说明的核心模块。

- [ ] **Step 3：添加中文模块说明和关键步骤注释**

注释必须解释以下内容而不改变代码：

```text
Nédélec边元及切向连续性
接地源DC初始场
Cole–Cole复电阻率到复电导率转换
广义Debye记忆变量
后向Euler/BDF2时间推进
Biot–Savart磁场接收
dB/dt接收的单位与符号
网格、边界和时间步质量门槛
FEniCSx、empymod、SimPEG的坐标对齐
```

不得修改函数签名、离散矩阵、参数默认值或验证门槛。

- [ ] **Step 4：运行中文注释测试和核心模块测试**

Run:

```bash
python -m pytest -q \
  tests/test_chinese_core_comments.py \
  tests/test_hj.py \
  tests/test_materials_cole_cole.py \
  tests/test_sources.py \
  tests/test_receivers.py \
  tests/test_magnetic_recovery.py
```

Expected: 所有实际存在且相关的测试通过。

- [ ] **Step 5：提交中文注释**

```bash
git add dolfinx/sotem_pipeline.py src/atem3d sotem_ip
git add tests/test_chinese_core_comments.py
git commit -m "文档：为核心TEM-IP算法补充中文注释"
```

## Task 5：执行最终验证与仓库审计

**Files:**

- Verify: whole repository

- [ ] **Step 1：安装当前快照**

Run:

```bash
python -m pip install -e .
```

Expected: exit 0。

- [ ] **Step 2：运行全部保留测试**

Run:

```bash
python -m pytest -q
```

Expected: 0 failures；跳过项必须在最终说明中记录。

- [ ] **Step 3：验证公开入口**

Run:

```bash
atem3d-run --help
atem3d-initial-field-diagnostics --help
atem3d-sotem-validate --help
atem3d-zhou2020-reference --help
```

Expected: 全部退出码为0。

- [ ] **Step 4：检查路径、密钥和生成物**

检查：

```text
C:\Users\
/home/paidaxin/
BEGIN OPENSSH PRIVATE KEY
ghp_
github_pat_
*.docx
*.pdf
*.png
*.csv（examples/benchmarks中明确属于输入的除外）
*.pyc
__pycache__
current_task_runs
```

发现命中时逐项判断；不能直接忽略绝对路径或密钥命中。

- [ ] **Step 5：检查文件范围和大文件**

Run:

```bash
git status -sb
git diff --check
git ls-files
```

并统计所有已跟踪文件大小。最终快照不得存在意外二进制产物或超大文件。

- [ ] **Step 6：核对未重新正演**

确认工作树没有新增：

```text
output/
generated/
*.msh
*.h5
*.npz
*.npy
```

## Task 6：提交并安全更新GitHub main

**Files:**

- Publish: `git@github.com:xyx202511157870-beep/SOTEM_FENICSX.git`

- [ ] **Step 1：记录远端更新前状态**

Run:

```bash
git ls-remote --symref origin HEAD
git ls-remote origin refs/heads/main
```

保存远端`main`提交ID。预期它仍是发布分支祖先。

- [ ] **Step 2：确认所有发布修改已经分别提交**

Run:

```bash
git status --short
```

Expected: 无输出。若最终审计发现问题，先修正对应的明确文件，重新运行相关测试，并把这些明确文件单独提交；禁止使用不经检查的`git add -A`。

- [ ] **Step 3：验证本地提交和工作树**

Run:

```bash
git status -sb
git log -5 --oneline --decorate
git diff origin/main...HEAD --stat
```

Expected: 工作树干净。

- [ ] **Step 4：防止覆盖并发远端更新**

重新执行：

```bash
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
```

Expected: exit 0。若失败则停止，不强制推送。

- [ ] **Step 5：推送发布快照**

Run:

```bash
git push origin HEAD:main
```

Expected: fast-forward成功。

- [ ] **Step 6：验证远端**

Run:

```bash
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

Expected: 两个提交ID完全一致。

- [ ] **Step 7：最终报告**

最终报告必须包含：

```text
远端仓库
发布提交ID
删除的主要内容
保留的算法目录
通过/跳过/失败测试数量
安装与CLI检查
没有重新运行大型三维正演
许可证为保留所有权利而非开源授权
```
