from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_supports_validated_python_310():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in pyproject
