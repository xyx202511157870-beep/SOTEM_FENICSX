from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_has_only_approved_top_level_entries():
    approved = {
        ".git",
        ".gitattributes",
        ".github",
        ".gitignore",
        ".numba_cache",
        ".pytest_cache",
        ".dolfinx.writer.lock",
        "LICENSE",
        "README.md",
        "benchmarks",
        "docs",
        "dolfinx",
        "examples",
        "pyproject.toml",
        "sotem_ip",
        "src",
        "tests",
        "tools",
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


def test_public_documents_are_chinese_and_do_not_overclaim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    task = (ROOT / "docs" / "dam_seepage_h3_forward_task.md").read_text(encoding="utf-8")
    assert "三维时域" in readme
    assert "Cole–Cole" in readme
    assert "不能替代现场工程验证" in readme
    assert "保留所有权利" in license_text
    assert "尚未通过 H 三分量验收" in task
    assert "禁止把坝体曲线当作正式结果" in task


def test_public_text_files_do_not_contain_personal_absolute_paths():
    text_suffixes = {".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".json"}
    windows_user_path = "C:" + "\\Users\\" + "paidaxin"
    linux_user_path = "/home/" + "paidaxin/"
    offending = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in {".git", ".pytest_cache", ".numba_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if windows_user_path in text or linux_user_path in text:
            offending.append(path.relative_to(ROOT).as_posix())

    assert offending == []


def test_shell_scripts_use_lf_line_endings():
    offending = []
    for path in ROOT.rglob("*.sh"):
        if b"\r\n" in path.read_bytes():
            offending.append(path.relative_to(ROOT).as_posix())

    assert offending == []
