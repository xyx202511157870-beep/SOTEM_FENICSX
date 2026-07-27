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


def test_public_documents_are_chinese_and_do_not_overclaim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "三维时域" in readme
    assert "Cole–Cole" in readme
    assert "不能替代现场工程验证" in readme
    assert "保留所有权利" in license_text
