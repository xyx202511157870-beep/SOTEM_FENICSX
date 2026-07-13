# FEniCSx Algorithm Progress Word Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and visually verify a detailed Chinese Word report that traces the FEniCSx grounded-source TDEM algorithm from its initial implementation through the latest validation state, with every retained evidence-bearing debugging image mapped to a problem, change, result, and status.

**Architecture:** A read-only inventory layer scans Git history, project documents, result metadata, and raster images into deterministic manifests. A curated report source and figure-selection manifest keep technical claims and image choices reviewable outside the DOCX generator. A focused `python-docx` writer turns those inputs into a compact A4 report, after which the bundled renderer produces page images for full visual QA.

**Tech Stack:** PowerShell, bundled Python 3, `pytest`, `python-docx`, Pillow, `lxml`, Git, LibreOffice-backed `render_docx.py`.

---

## Working Constraints

- Work in the current `codex/layered-convergence` branch because the report depends on untracked and ignored run artifacts that would not appear in a new Git worktree.
- Treat the project tree as read-only evidence except for the report-specific files listed below.
- Never use `git add -A`; stage only paths created by this plan.
- Do not resume FEniCSx or COMSOL long solves during report generation.
- Use the bundled Python runtime:

```powershell
$py = 'C:\Users\paidaxin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

- Store temporary manifests, contact sheets, render pages, and the temporary PDF under `tmp/docs/fenicsx_progress_report/`.
- Store the only user-facing deliverable under `output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx`.

## File Structure

### Files to create

- `tools/fenicsx_progress_report/__init__.py`: stable public imports for the report tooling.
- `tools/fenicsx_progress_report/inventory.py`: image discovery, metadata extraction, hashing, deduplication, Git event extraction, and JSON/CSV serialization.
- `tools/fenicsx_progress_report/selection.py`: phase classification, selection validation, coverage audit, and Pillow contact-sheet generation.
- `tools/fenicsx_progress_report/source_audit.py`: required-section, required-claim, status, and figure-directive validation for the report source.
- `tools/fenicsx_progress_report/docx_writer.py`: compact A4 styles, source parsing, tables, figures, captions, headers, footers, and status labels.
- `tools/fenicsx_progress_report/build_report.py`: command-line orchestration and build summary generation.
- `tools/fenicsx_progress_report/figure_selection.json`: curated figure IDs, paths, captions, phase assignments, statuses, and duplicate rationale.
- `docs/fenicsx_algorithm_progress_report_source.md`: complete Chinese report source with figure directives.
- `tests/test_fenicsx_progress_inventory.py`: deterministic inventory and deduplication tests.
- `tests/test_fenicsx_progress_selection.py`: classification, contact-sheet, and coverage-audit tests.
- `tests/test_fenicsx_progress_source_audit.py`: claim-boundary and source-completeness tests.
- `tests/test_fenicsx_progress_docx_writer.py`: A4 layout, style, figure, and DOCX integrity tests.
- `tests/test_fenicsx_progress_build.py`: end-to-end build test using a minimal fixture report.
- `output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx`: final deliverable, created after all checks pass.

### Generated temporary files

- `tmp/docs/fenicsx_progress_report/inventory.json`
- `tmp/docs/fenicsx_progress_report/inventory.csv`
- `tmp/docs/fenicsx_progress_report/git_timeline.json`
- `tmp/docs/fenicsx_progress_report/contact_sheets/*.png`
- `tmp/docs/fenicsx_progress_report/build_summary.json`
- `tmp/docs/fenicsx_progress_report/rendered/page-*.png`
- `tmp/docs/fenicsx_progress_report/rendered/FEniCSx_算法建立与调试进展报告_2026-07-13.pdf`

## Task 1: Build the deterministic evidence inventory

**Files:**
- Create: `tools/fenicsx_progress_report/__init__.py`
- Create: `tools/fenicsx_progress_report/inventory.py`
- Create: `tests/test_fenicsx_progress_inventory.py`

- [ ] **Step 1: Write failing inventory tests**

Create two byte-identical PNG files, one distinct PNG, and one excluded file under `.git`. Assert that discovery excludes `.git`, records dimensions, assigns the same SHA-256 to duplicate files, and marks exactly one canonical record.

```python
from pathlib import Path

from PIL import Image

from tools.fenicsx_progress_report.inventory import (
    deduplicate_images,
    discover_images,
)


def test_discover_and_deduplicate_images(tmp_path: Path) -> None:
    Image.new("RGB", (40, 20), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    Image.new("RGB", (20, 40), "black").save(tmp_path / "c.png")
    excluded = tmp_path / ".git"
    excluded.mkdir()
    Image.new("RGB", (10, 10), "red").save(excluded / "ignored.png")

    records = discover_images(tmp_path)
    deduplicated = deduplicate_images(records)

    assert [record.path for record in records] == ["a.png", "b.png", "c.png"]
    assert records[0].width == 40
    assert records[0].height == 20
    assert records[0].sha256 == records[1].sha256
    assert sum(record.duplicate_of is not None for record in deduplicated) == 1
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
& $py -m pytest tests/test_fenicsx_progress_inventory.py -q
```

Expected: collection fails because `tools.fenicsx_progress_report.inventory` does not exist.

- [ ] **Step 3: Implement the inventory API**

Implement a frozen `ImageEvidence` dataclass with these fields:

```python
@dataclass(frozen=True)
class ImageEvidence:
    path: str
    suffix: str
    byte_size: int
    width: int
    height: int
    modified_utc: str
    sha256: str
    duplicate_of: str | None = None
```

Implement the following behavior; keep the small hashing helper private:

```python
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(
    root: Path,
    *,
    excluded_parts: tuple[str, ...] = (".git", ".worktrees", "__pycache__", "tmp"),
) -> list[ImageEvidence]:
    records: list[ImageEvidence] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in excluded_parts for part in relative.parts):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            width, height = image.size
        records.append(ImageEvidence(
            path=relative.as_posix(),
            suffix=path.suffix.lower(),
            byte_size=path.stat().st_size,
            width=width,
            height=height,
            modified_utc=datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            sha256=_sha256(path),
        ))
    return records

def deduplicate_images(records: Sequence[ImageEvidence]) -> list[ImageEvidence]:
    canonical_by_hash: dict[str, str] = {}
    result: list[ImageEvidence] = []
    for record in records:
        canonical = canonical_by_hash.setdefault(record.sha256, record.path)
        result.append(replace(
            record,
            duplicate_of=None if canonical == record.path else canonical,
        ))
    return result

def extract_git_timeline(root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "log", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%s"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [
        {"commit": commit, "date": date, "subject": subject}
        for commit, date, subject in (
            line.split("\t", 2) for line in completed.stdout.splitlines() if line
        )
    ]

def write_inventory(
    records: Sequence[ImageEvidence], json_path: Path, csv_path: Path
) -> None:
    rows = [asdict(record) for record in records]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"images": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
```

Use Pillow only to read dimensions, `hashlib.sha256()` for identity, relative POSIX paths for stable manifests, and `git log --date=iso-strict --pretty=format:%H%x09%ad%x09%s` for timeline extraction. Sort records by normalized relative path before serialization.

- [ ] **Step 4: Run inventory tests**

Run:

```powershell
& $py -m pytest tests/test_fenicsx_progress_inventory.py -q
```

Expected: all inventory tests pass.

- [ ] **Step 5: Generate the real project inventory**

Run:

```powershell
& $py -m tools.fenicsx_progress_report.inventory `
  --root . `
  --json tmp/docs/fenicsx_progress_report/inventory.json `
  --csv tmp/docs/fenicsx_progress_report/inventory.csv `
  --timeline tmp/docs/fenicsx_progress_report/git_timeline.json
```

Expected: the command reports the discovered count, canonical count, duplicate count, unreadable count, and Git event count without modifying any result directory.

- [ ] **Step 6: Commit the inventory layer**

```powershell
git add -- tools/fenicsx_progress_report/__init__.py tools/fenicsx_progress_report/inventory.py tests/test_fenicsx_progress_inventory.py
git commit -m "feat: inventory FEniCSx report evidence"
```

## Task 2: Classify phases, generate contact sheets, and audit selection coverage

**Files:**
- Create: `tools/fenicsx_progress_report/selection.py`
- Create: `tests/test_fenicsx_progress_selection.py`
- Create: `tools/fenicsx_progress_report/figure_selection.json`

- [ ] **Step 1: Write failing classification and coverage tests**

Test known path families and coverage rules:

```python
from tools.fenicsx_progress_report.selection import (
    audit_selection,
    classify_phase,
)


def test_classify_known_evidence_paths() -> None:
    assert classify_phase("COMSOL/exports/three_way_decay.png") == "uniform_three_way"
    assert classify_phase("output/publication_validation/convergence/layered_resistive_offset100/error.png") == "layered_convergence"
    assert classify_phase("dolfinx/dam_leakage_model/model.png") == "complex_models"


def test_audit_rejects_duplicate_and_unclassified_selection() -> None:
    inventory = [
        {"path": "a.png", "sha256": "same", "duplicate_of": None},
        {"path": "b.png", "sha256": "same", "duplicate_of": "a.png"},
    ]
    selection = [
        {"id": "f1", "path": "a.png", "phase": "source", "status": "诊断用途"},
        {"id": "f2", "path": "b.png", "phase": "", "status": ""},
    ]
    report = audit_selection(inventory, selection)
    assert report.duplicate_selections == ["b.png"]
    assert report.unclassified == ["f2"]
```

- [ ] **Step 2: Run the tests and verify failure**

```powershell
& $py -m pytest tests/test_fenicsx_progress_selection.py -q
```

Expected: import failure for the missing `selection` module.

- [ ] **Step 3: Implement phase rules and selection validation**

Define stable phase IDs:

```python
PHASE_IDS = (
    "legacy_simpeg",
    "formulation",
    "source_electrodes",
    "mesh_domain",
    "receiver_fields",
    "initial_primary_secondary",
    "time_turnoff",
    "cole_cole_debye",
    "uniform_three_way",
    "layered_convergence",
    "stage2_pending",
    "complex_models",
    "reproducibility",
)

ALLOWED_STATUSES = (
    "已验证",
    "通过",
    "未通过",
    "待完成",
    "诊断用途",
    "历史实现",
)
```

Implement `classify_phase(path)`, `load_selection(path)`, `audit_selection(inventory, selection)`, and `write_contact_sheets(records, root, output_dir, columns=4, thumb_size=(360, 240))`. Contact sheets must use Pillow, preserve aspect ratio, print the figure ID and shortened path beneath each thumbnail, and never overwrite source images.

- [ ] **Step 4: Run the selection tests**

```powershell
& $py -m pytest tests/test_fenicsx_progress_selection.py -q
```

Expected: all selection tests pass.

- [ ] **Step 5: Produce contact sheets for canonical candidates**

Run:

```powershell
& $py -m tools.fenicsx_progress_report.selection `
  --root . `
  --inventory tmp/docs/fenicsx_progress_report/inventory.json `
  --contact-sheet-dir tmp/docs/fenicsx_progress_report/contact_sheets
```

Expected: contact sheets are grouped by phase, every thumbnail includes its path, and the command prints an `unclassified` count for manual review.

- [ ] **Step 6: Visually inspect and curate the figure selection**

Inspect every contact sheet and selected full-resolution image. Populate `figure_selection.json` with this schema:

```json
{
  "figures": [
    {
      "id": "uniform_three_way_decay",
      "path": "COMSOL/exports/three_way_final_ideal_stepoff_1e-5_1e-2_dBzdt_decay.png",
      "phase": "uniform_three_way",
      "status": "已验证",
      "placement": "body",
      "caption": "均匀半空间 FEniCSx、COMSOL 与 empymod 瞬变衰减曲线。",
      "problem": "检查三种独立实现的时间道、符号和幅值是否一致。",
      "change": "统一模型参数、坐标、源接收几何和输出物理量。",
      "result": "记录对应误差文件中的中位数、RMS 和最大误差。",
      "source_run": "COMSOL/exports"
    }
  ]
}
```

Use inventory paths and run IDs verified from the workspace. Include every canonical image with independent debugging value either as `body` or `appendix`; record excluded canonical candidates and a concrete exclusion reason in the manifest's `excluded` array.

- [ ] **Step 7: Run the real coverage audit**

```powershell
& $py -m tools.fenicsx_progress_report.selection `
  --root . `
  --inventory tmp/docs/fenicsx_progress_report/inventory.json `
  --selection tools/fenicsx_progress_report/figure_selection.json `
  --audit-only
```

Expected: zero missing paths, zero duplicate selections, zero invalid statuses, zero selected images without a phase, and zero evidence-bearing canonical candidates lacking either inclusion or exclusion rationale.

- [ ] **Step 8: Commit the selection tooling and curated manifest**

```powershell
git add -- tools/fenicsx_progress_report/selection.py tools/fenicsx_progress_report/figure_selection.json tests/test_fenicsx_progress_selection.py
git commit -m "feat: curate FEniCSx report figures"
```

## Task 3: Author and audit the complete Chinese report source

**Files:**
- Create: `tools/fenicsx_progress_report/source_audit.py`
- Create: `tests/test_fenicsx_progress_source_audit.py`
- Create: `docs/fenicsx_algorithm_progress_report_source.md`

- [ ] **Step 1: Write failing source-audit tests**

```python
from tools.fenicsx_progress_report.source_audit import audit_source


def test_audit_requires_claim_boundaries_and_figure_ids() -> None:
    source = "# 报告\n均匀半空间中位误差约 2.75%。\n[[FIGURE:missing]]"
    selection = {"figures": [{"id": "known"}]}
    report = audit_source(source, selection)
    assert "missing_required_section" in report.codes
    assert "unknown_figure_id:missing" in report.codes


def test_audit_accepts_required_status_language() -> None:
    source = "\n".join([
        "# FEniCSx 算法建立与调试进展",
        "## 摘要与当前状态",
        "已验证；未通过；待完成；诊断用途；历史实现。",
        "地下为正，空气为负。",
        "dBz/dt 中位误差约 2.75%，RMS 约 3.13%，最大约 5.97%。",
        "第二阶段长时间求解尚未完成。",
        "[[FIGURE:known]]",
    ])
    report = audit_source(source, {"figures": [{"id": "known"}]})
    assert not report.unknown_figure_ids
```

- [ ] **Step 2: Run the tests and verify failure**

```powershell
& $py -m pytest tests/test_fenicsx_progress_source_audit.py -q
```

Expected: import failure for the missing source-audit module.

- [ ] **Step 3: Implement source auditing**

Implement `audit_source(source_text, selection)` with explicit checks for:

- all 16 required body chapters and five appendices;
- the coordinate convention;
- current authoritative FEniCSx implementation paths;
- homogeneous-halfspace `dBz/dt` values `2.75%`, `3.13%`, and `5.97%`;
- layered 12 km values `0.83%`, `2.76%`, and `9.22%`;
- explicit first-stage mesh pass, time-step fail, and domain fail statements;
- explicit stage-2 pending language;
- every `[[FIGURE:figure_id]]` directive resolving to an ID in the selection manifest;
- every selected `body` figure appearing in the source;
- no `TBD`, `TODO`, fabricated future curve, or universal 3D-validation claim.

- [ ] **Step 4: Write the full report source**

Author `docs/fenicsx_algorithm_progress_report_source.md` using the approved six-part debugging template. For each major phase, cite verified repository paths and Git milestones, then place figure directives exactly where the evidence is discussed:

```markdown
## 9. 时间离散与关断调试

### 9.1 遇到的问题

早期时间道相对参考解偏差较大，且偏差对输出时间步设置敏感。

### 9.2 原因判断

关断附近的内部积分分辨率不足，输出时间道同时承担了数值积分步长的角色。

[[FIGURE:time_turnoff_before_after]]

### 9.3 采取的修改与执行方式

保持用户请求的输出时间道不变，在相邻输出时刻之间插入内部子步，并记录内部步数与 KSP 迭代信息。

### 9.4 效果与阶段结论

早期响应稳定性提高；该图只证明时间离散改进，不单独构成空间收敛证据。
```

The source must include:

- a concise executive summary and current validation matrix;
- mathematical formulation and symbol definitions derived from current project documents;
- a detailed progression from legacy finite volume to the current FEniCSx primary-secondary formulation;
- source, electrode, mesh, receiver, initial-field, time-step, and solver debugging narratives;
- homogeneous FEniCSx/COMSOL/empymod comparison;
- layered convergence results with failures described as failures;
- stage-2 design and preflight evidence marked pending;
- complex-model scope boundaries;
- chronological Git appendix generated from real commits;
- parameter, command, artifact, and validation-status appendices.

- [ ] **Step 5: Run source audit and focused tests**

```powershell
& $py -m pytest tests/test_fenicsx_progress_source_audit.py -q
& $py -m tools.fenicsx_progress_report.source_audit `
  --source docs/fenicsx_algorithm_progress_report_source.md `
  --selection tools/fenicsx_progress_report/figure_selection.json
```

Expected: tests pass and the CLI reports zero missing sections, zero unknown figures, zero missing claim boundaries, and zero placeholder tokens.

- [ ] **Step 6: Commit the audited source**

```powershell
git add -- tools/fenicsx_progress_report/source_audit.py tests/test_fenicsx_progress_source_audit.py docs/fenicsx_algorithm_progress_report_source.md
git commit -m "docs: author FEniCSx algorithm progress archive"
```

## Task 4: Implement the compact A4 DOCX writer

**Files:**
- Create: `tools/fenicsx_progress_report/docx_writer.py`
- Create: `tests/test_fenicsx_progress_docx_writer.py`

- [ ] **Step 1: Write failing DOCX layout and integrity tests**

```python
from pathlib import Path

from docx import Document
from docx.shared import Mm
from PIL import Image

from tools.fenicsx_progress_report.docx_writer import build_docx


def test_build_docx_uses_compact_a4_and_embeds_figure(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (800, 500), "white").save(image_path)
    source = "# 测试报告\n## 摘要与当前状态\n正文。\n[[FIGURE:test_figure]]"
    selection = {
        "figures": [{
            "id": "test_figure",
            "path": str(image_path),
            "phase": "uniform_three_way",
            "status": "已验证",
            "placement": "body",
            "caption": "测试图。",
            "problem": "测试问题。",
            "change": "测试修改。",
            "result": "测试结果。",
            "source_run": "fixture",
        }]
    }
    output = tmp_path / "report.docx"

    build_docx(source, selection, project_root=tmp_path, output_path=output)

    document = Document(output)
    section = document.sections[0]
    assert abs(section.page_width - Mm(210)) < 1000
    assert abs(section.page_height - Mm(297)) < 1000
    assert abs(section.left_margin - Mm(17)) < 1000
    assert len(document.inline_shapes) == 1
    assert "测试图" in "\n".join(p.text for p in document.paragraphs)
```

- [ ] **Step 2: Run the tests and verify failure**

```powershell
& $py -m pytest tests/test_fenicsx_progress_docx_writer.py -q
```

Expected: import failure for the missing writer.

- [ ] **Step 3: Implement document styles and page primitives**

Implement these explicit tokens:

```python
PAGE_WIDTH_MM = 210
PAGE_HEIGHT_MM = 297
MARGIN_MM = 17
BODY_PT = 10
CAPTION_PT = 8.25
BODY_LINE_SPACING = 1.2
ACCENT = "287CA3"
TEXT = "203A49"
MUTED = "60747F"
```

Implement helpers for:

- Chinese and Latin font assignment on every style and run;
- compact Heading 1/2/3 spacing with keep-with-next;
- cover page, generated contents list, header, footer, and page number field;
- markdown-like headings, paragraphs, bullets, numbered lists, simple pipe tables, inline code, and bold text;
- `[[FIGURE:figure_id]]` directives;
- `[[PAGEBREAK]]` directives;
- figure sizing based on pixel dimensions and page width;
- captions kept with figures;
- four-up appendix tables that fall back to two-up or one-up when readability rules require it;
- status labels rendered as compact colored table cells, not floating shapes;
- alt text and image description metadata where supported by OOXML.

Do not implement a general Markdown engine. Support only the syntax used by the report source and reject unknown directives with a descriptive error.

- [ ] **Step 4: Add tests for tables, unknown directives, and figure fallback**

Add tests asserting that:

- pipe-table headers repeat across pages;
- `[[FIGURE:unknown]]` raises `UnknownFigureError`;
- portrait or dense images use a larger placement class than simple wide plots;
- no figure record can be embedded without caption, status, and source path.

- [ ] **Step 5: Run DOCX writer tests**

```powershell
& $py -m pytest tests/test_fenicsx_progress_docx_writer.py -q
```

Expected: all writer tests pass and fixture DOCX files reopen through `python-docx` without repair warnings.

- [ ] **Step 6: Commit the DOCX writer**

```powershell
git add -- tools/fenicsx_progress_report/docx_writer.py tests/test_fenicsx_progress_docx_writer.py
git commit -m "feat: write compact FEniCSx progress DOCX"
```

## Task 5: Add the end-to-end report builder

**Files:**
- Create: `tools/fenicsx_progress_report/build_report.py`
- Create: `tests/test_fenicsx_progress_build.py`

- [ ] **Step 1: Write a failing end-to-end build test**

The test creates a minimal source, selection, image, and inventory; invokes `main(argv)` with the explicit argument list shown below; then verifies the DOCX and build summary.

```python
def test_build_report_writes_docx_and_summary(tmp_path: Path) -> None:
    exit_code = main([
        "--root", str(tmp_path),
        "--source", str(tmp_path / "source.md"),
        "--selection", str(tmp_path / "selection.json"),
        "--inventory", str(tmp_path / "inventory.json"),
        "--output", str(tmp_path / "report.docx"),
        "--summary", str(tmp_path / "build_summary.json"),
    ])
    assert exit_code == 0
    assert (tmp_path / "report.docx").exists()
    summary = json.loads((tmp_path / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["missing_figures"] == []
    assert summary["unknown_directives"] == []
```

- [ ] **Step 2: Run the test and verify failure**

```powershell
& $py -m pytest tests/test_fenicsx_progress_build.py -q
```

Expected: import failure for the missing builder.

- [ ] **Step 3: Implement orchestration and fail-fast validation**

Implement CLI arguments from the test plus `--timeline`. The builder must:

1. load and validate inventory and selection;
2. audit the report source;
3. fail before DOCX creation on missing images, duplicate selections, invalid statuses, unknown directives, or claim-boundary failures;
4. create the DOCX in a temporary sibling path;
5. reopen the temporary DOCX with `python-docx`;
6. atomically replace the requested output only after reopen succeeds;
7. write `build_summary.json` with counts, figure IDs, source SHA-256, selection SHA-256, output SHA-256, and all warnings.

- [ ] **Step 4: Run the build test and the report-tool test suite**

```powershell
& $py -m pytest `
  tests/test_fenicsx_progress_inventory.py `
  tests/test_fenicsx_progress_selection.py `
  tests/test_fenicsx_progress_source_audit.py `
  tests/test_fenicsx_progress_docx_writer.py `
  tests/test_fenicsx_progress_build.py -q
```

Expected: all report-tool tests pass.

- [ ] **Step 5: Commit the builder**

```powershell
git add -- tools/fenicsx_progress_report/build_report.py tests/test_fenicsx_progress_build.py
git commit -m "feat: build FEniCSx progress report"
```

## Task 6: Build the full Word report

**Files:**
- Create: `output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx`
- Generate: `tmp/docs/fenicsx_progress_report/build_summary.json`

- [ ] **Step 1: Read the document-production skills and references**

Read these files before generation:

```text
C:\Users\paidaxin\.codex\skills\doc\SKILL.md
C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\SKILL.md
C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\references\design_presets.md
C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\references\header_templates.md
C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\tasks\create_edit.md
C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\tasks\verify_render.md
```

Expected: the build and QA steps conform to the current plugin instructions, even if a referenced command has changed since this plan was written.

- [ ] **Step 2: Re-run all evidence and source audits**

```powershell
& $py -m tools.fenicsx_progress_report.selection `
  --root . `
  --inventory tmp/docs/fenicsx_progress_report/inventory.json `
  --selection tools/fenicsx_progress_report/figure_selection.json `
  --audit-only

& $py -m tools.fenicsx_progress_report.source_audit `
  --source docs/fenicsx_algorithm_progress_report_source.md `
  --selection tools/fenicsx_progress_report/figure_selection.json
```

Expected: both audits pass with zero blocking findings.

- [ ] **Step 3: Build the complete DOCX**

```powershell
New-Item -ItemType Directory -Force -Path 'output/doc','tmp/docs/fenicsx_progress_report' | Out-Null

& $py -m tools.fenicsx_progress_report.build_report `
  --root . `
  --source docs/fenicsx_algorithm_progress_report_source.md `
  --selection tools/fenicsx_progress_report/figure_selection.json `
  --inventory tmp/docs/fenicsx_progress_report/inventory.json `
  --timeline tmp/docs/fenicsx_progress_report/git_timeline.json `
  --output 'output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx' `
  --summary tmp/docs/fenicsx_progress_report/build_summary.json
```

Expected: the builder reports zero missing figures and writes a DOCX that reopens successfully.

- [ ] **Step 4: Run programmatic document checks**

Use `python-docx` to print and verify:

- section count and A4 dimensions;
- approximately 17 mm margins;
- non-empty title, contents, all required chapters, and all appendices;
- embedded image count equals the builder summary;
- no empty caption paragraphs;
- no broken external image relationships;
- all status labels appear at least once when evidence exists for that state.

Expected: no programmatic check fails.

## Task 7: Render and inspect every page

**Files:**
- Generate: `tmp/docs/fenicsx_progress_report/rendered/page-*.png`
- Generate: `tmp/docs/fenicsx_progress_report/rendered/FEniCSx_算法建立与调试进展报告_2026-07-13.pdf`
- Modify as needed: `docs/fenicsx_algorithm_progress_report_source.md`
- Modify as needed: `tools/fenicsx_progress_report/figure_selection.json`
- Modify as needed: `tools/fenicsx_progress_report/docx_writer.py`

- [ ] **Step 1: Render DOCX to full-page PNGs and a temporary PDF**

```powershell
$renderer = 'C:\Users\paidaxin\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\render_docx.py'

& $py $renderer `
  'output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx' `
  --output_dir 'tmp/docs/fenicsx_progress_report/rendered' `
  --width 1600 `
  --height 2200 `
  --emit_pdf `
  --verbose
```

Expected: one PNG per page, a temporary PDF, and no LibreOffice conversion error.

- [ ] **Step 2: Build render contact sheets for navigation**

Use Pillow to create numbered contact sheets from the rendered pages without replacing individual full-resolution page PNGs.

Expected: every rendered page appears exactly once in a contact sheet.

- [ ] **Step 3: Inspect every page at full resolution**

Inspect each page PNG, not only contact sheets. Record page-level findings for:

- clipped text or plots;
- unreadable axis labels, legends, or error values;
- unexpected blank pages;
- isolated headings or captions;
- split figure and caption;
- crowded tables;
- excessive whitespace inconsistent with the approved compact layout;
- inconsistent Chinese fonts;
- incorrect figure numbering, status labels, or source paths.

Expected: every page receives a pass or a concrete revision note.

- [ ] **Step 4: Revise and re-render until all pages pass**

For each revision cycle:

```powershell
& $py -m pytest tests/test_fenicsx_progress_docx_writer.py tests/test_fenicsx_progress_build.py -q
& $py -m tools.fenicsx_progress_report.build_report `
  --root . `
  --source docs/fenicsx_algorithm_progress_report_source.md `
  --selection tools/fenicsx_progress_report/figure_selection.json `
  --inventory tmp/docs/fenicsx_progress_report/inventory.json `
  --timeline tmp/docs/fenicsx_progress_report/git_timeline.json `
  --output 'output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx' `
  --summary tmp/docs/fenicsx_progress_report/build_summary.json
& $py $renderer `
  'output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx' `
  --output_dir 'tmp/docs/fenicsx_progress_report/rendered' `
  --width 1600 `
  --height 2200 `
  --emit_pdf `
  --verbose
```

Expected: no regression in tests or evidence counts, and all previously noted page issues are resolved.

- [ ] **Step 5: Commit source or writer corrections**

Stage only changed report source, manifest, writer, and tests:

```powershell
git add -- docs/fenicsx_algorithm_progress_report_source.md tools/fenicsx_progress_report/figure_selection.json tools/fenicsx_progress_report/docx_writer.py tests/test_fenicsx_progress_docx_writer.py
git commit -m "fix: polish FEniCSx report rendering"
```

Skip this commit only if visual QA required no tracked-file changes.

## Task 8: Final verification and delivery

**Files:**
- Verify: `output/doc/FEniCSx_算法建立与调试进展报告_2026-07-13.docx`

- [ ] **Step 1: Run the complete report-tool test suite**

```powershell
& $py -m pytest `
  tests/test_fenicsx_progress_inventory.py `
  tests/test_fenicsx_progress_selection.py `
  tests/test_fenicsx_progress_source_audit.py `
  tests/test_fenicsx_progress_docx_writer.py `
  tests/test_fenicsx_progress_build.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify final hashes and build summary**

Confirm that the output SHA-256 matches `build_summary.json`, every selected figure exists, every selected body figure appears in the source, and the inventory/selection/source hashes match the inputs used for the final build.

Expected: zero mismatches.

- [ ] **Step 3: Verify the final output directory**

```powershell
Get-ChildItem -LiteralPath 'output/doc' -File | Select-Object Name,Length,LastWriteTime
```

Expected: the requested DOCX is present and no temporary PDF, page PNG, manifest, or lock file appears in `output/doc`.

- [ ] **Step 4: Inspect Git scope**

```powershell
git status --short
git log --oneline -8
```

Expected: report commits contain only the planned report tooling, tests, source, and design/plan documents; unrelated existing user changes remain untouched.

- [ ] **Step 5: Deliver the document**

Provide one standalone Markdown link to the final DOCX and no extra file links, matching the document-skill delivery requirement.
