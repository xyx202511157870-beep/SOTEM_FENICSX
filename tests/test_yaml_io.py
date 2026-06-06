import builtins

from atem3d.yaml_io import safe_dump_yaml


def test_safe_dump_yaml_falls_back_when_pyyaml_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    text = safe_dump_yaml(
        {
            "acceptance": {
                "noip_summary_json": "outputs/noip/error_summary.json",
                "threshold": 0.05,
                "components": ["Ex", "Ey", "dBzdt"],
                "final_acceptance_passed": False,
            }
        }
    )

    assert not text.lstrip().startswith("{")
    assert "acceptance:" in text
    assert "  noip_summary_json: outputs/noip/error_summary.json" in text
    assert "  threshold: 0.05" in text
    assert "  final_acceptance_passed: false" in text
    assert "  components:" in text
    assert "  - Ex" in text
