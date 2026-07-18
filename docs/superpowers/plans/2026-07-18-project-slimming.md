# Three-Solver Project Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the current branch to the latest SimPEG, FEniCSx, and empymod seepage-channel workflow while removing COMSOL, obsolete IP work, and generated transition artifacts.

**Architecture:** A repository-contract test defines the retained and forbidden surfaces. Deletions are applied in small groups, followed by import, focused-test, full-test, and Git-content gates. Scientific status is represented by a compact Markdown record rather than raw solver outputs.

**Tech Stack:** Python 3.11, pytest, SimPEG, FEniCSx/DOLFINx, empymod, Git, PowerShell/WSL2.

---

### Task 1: Freeze the slimming contract

**Files:**
- Create: `tests/test_repository_slimming.py`
- Create: `docs/current_status.md`

- [ ] **Step 1: Write the failing repository-contract test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_removed_surfaces_are_absent():
    forbidden = [
        "COMSOL",
        "sotem_ip",
        "dolfinx/current_task_runs",
        "src/atem3d/comsol_seepage_channel_3d.py",
        "src/atem3d/four_way_validation.py",
        "tools/run_comsol_seepage_channel_3d.py",
    ]
    assert [path for path in forbidden if (ROOT / path).exists()] == []


def test_three_solver_surfaces_and_known_issue_are_retained():
    required = [
        "src/atem3d/seepage_reference.py",
        "src/atem3d/empymod_compare.py",
        "dolfinx/seepage_channel_full_domain.py",
        "dolfinx/symmetric_full_domain_mesh.py",
        "tools/run_seepage_channel_benchmark.py",
        "examples/seepage_channel_100m_5rx_simpeg_channel.yaml",
    ]
    assert [path for path in required if not (ROOT / path).is_file()] == []
    status = (ROOT / "docs/current_status.md").read_text(encoding="utf-8")
    assert "晚期 Ex" in status
    assert "未通过空间收敛验证" in status
```

- [ ] **Step 2: Run the contract test and observe the expected failure**

Run: `python -m pytest tests/test_repository_slimming.py -q`

Expected: FAIL because COMSOL, `sotem_ip`, `current_task_runs`, and the status document still violate the desired contract.

- [ ] **Step 3: Add the truthful current-status document**

Document the retained geometry and coordinate convention, the three solver roles, the early-time evidence, the Rx4/Rx5 late-time `Ex` discrepancy, and the requirement for a future spatial-convergence repair before scientific acceptance.

- [ ] **Step 4: Re-run only the status assertion**

Run: `python -m pytest tests/test_repository_slimming.py::test_three_solver_surfaces_and_known_issue_are_retained -q`

Expected: PASS.

### Task 2: Remove COMSOL and obsolete IP surfaces

**Files:**
- Delete: `COMSOL/`
- Delete: `src/atem3d/comsol_seepage_channel_3d.py`
- Delete: `src/atem3d/four_way_validation.py`
- Delete: `tools/run_comsol_seepage_channel_3d.py`
- Delete: `tests/test_comsol_seepage_channel_3d.py`
- Delete: `sotem_ip/`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify/Delete: COMSOL/IP-only docs, examples, tests, and imports identified by `git grep`

- [ ] **Step 1: Record the failing forbidden-surface test from Task 1**

Run: `python -m pytest tests/test_repository_slimming.py::test_removed_surfaces_are_absent -q`

Expected: FAIL listing the existing forbidden paths.

- [ ] **Step 2: Delete COMSOL executable surfaces and remove imports**

Delete the listed files and use `git grep -In -e comsol -e COMSOL -- src tools tests pyproject.toml README.md` to remove remaining executable/documentation coupling. Historical mention is allowed only in the approved design and plan.

- [ ] **Step 3: Delete the standalone and unreachable IP-only surface**

Remove `sotem_ip`, IP-only YAML examples/tests, and package entry points not reachable from the no-IP seepage workflow. Keep shared electromagnetic primitives required by the retained solver and prove the boundary with imports and tests.

- [ ] **Step 4: Run the contract test**

Run: `python -m pytest tests/test_repository_slimming.py -q`

Expected: PASS.

### Task 3: Remove generated transition artifacts and historical clutter

**Files:**
- Delete: `dolfinx/current_task_runs/`
- Delete: obsolete files under `docs/superpowers/plans/` and `docs/superpowers/specs/`
- Delete: stale root reports and old progress-report tooling not used by the retained report
- Modify: `.gitignore`

- [ ] **Step 1: Remove all tracked transition outputs**

Delete `dolfinx/current_task_runs` and local ignored cache/temp directories. Do not delete untracked scientific evidence outside the current worktree until the compact status record is committed.

- [ ] **Step 2: Keep only current design/plan and user-facing scientific status**

Delete obsolete plans/specifications and duplicated reports; retain this plan, this design, `docs/current_status.md`, and the current report builder sources needed to regenerate Word output.

- [ ] **Step 3: Verify tracked content size and forbidden paths**

Run: `git ls-files | Select-String -Pattern 'current_task_runs|sotem_ip|(?i)comsol'`

Expected: only the approved design/plan history note may mention COMSOL; no algorithm or generated-output paths remain.

### Task 4: Verify the retained three-solver project

**Files:**
- Modify: any retained import/test file that exposes a dangling deleted dependency

- [ ] **Step 1: Compile retained Python files**

Run: `python -m compileall -q src dolfinx tools tests`

Expected: exit code 0.

- [ ] **Step 2: Run focused three-solver and reporting tests**

Run: `python -m pytest tests/test_repository_slimming.py tests/test_seepage_channel_model.py tests/test_seepage_channel_validation.py tests/test_seepage_channel_fenicsx_contract.py tests/test_seepage_channel_fenicsx_mesh_material.py tests/test_seepage_channel_fenicsx_receivers.py tests/test_seepage_verification.py tests/test_simpeg_parity.py tests/test_empymod_compare.py tests/test_empymod_validation.py tests/test_verified_seepage_plots.py tests/test_verified_seepage_word_report.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Run every retained test**

Run: `python -m pytest -q`

Expected: zero failures; environment-dependent tests may remain skipped for documented missing optional runtimes.

- [ ] **Step 4: Audit the final diff and repository surface**

Run: `git diff --check`, `git status --short`, and `git ls-tree -r -l HEAD` after staging.

Expected: no whitespace errors, only approved deletions/additions, and a substantial tracked-size reduction.

### Task 5: Commit, push, and remove the local COMSOL worktree

**Files:**
- Stage only the approved slimming changes.

- [ ] **Step 1: Commit the verified project snapshot**

Run: `git add -A` followed by `git commit -m "chore: slim project to three open solvers"`.

Expected: one intentional commit on `codex/seepage-channel-verification`.

- [ ] **Step 2: Push without rewriting history**

Run: `git push -u origin codex/seepage-channel-verification`.

Expected: branch is uploaded successfully; no force push.

- [ ] **Step 3: Remove the clean local COMSOL-only worktree and branch**

From the main checkout, first re-check the absolute target and clean status, then run `git worktree remove <verified-comsol-worktree-path>` and `git branch -D codex/comsol-uniform-halfspace-validation`.

Expected: the local COMSOL-only worktree and local branch are gone; the pushed slim branch remains intact.
