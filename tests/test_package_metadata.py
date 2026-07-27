from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_supports_validated_python_310():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in pyproject


def test_editable_install_metadata_is_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.egg-info/" in gitignore
