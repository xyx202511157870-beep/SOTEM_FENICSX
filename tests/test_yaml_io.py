import builtins

from atem3d.yaml_io import safe_dump_yaml, safe_load_yaml


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


def test_safe_load_yaml_falls_back_when_pyyaml_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload = safe_load_yaml(
        """
acceptance:
  final_acceptance_passed: false
  threshold: 0.05
  components:
  - Ex
  - Ey
  - dBzdt
"""
    )

    assert payload == {
        "acceptance": {
            "final_acceptance_passed": False,
            "threshold": 0.05,
            "components": ["Ex", "Ey", "dBzdt"],
        }
    }


def test_safe_load_yaml_fallback_parses_task_book_inline_lists(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload = safe_load_yaml(
        """
source:
  start: [-500.0, 200.0, -0.1]
  end: [500.0, 200.0, -0.1]
validation:
  components: [Ex, Ey, dBzdt]
"""
    )

    assert payload["source"]["start"] == [-500.0, 200.0, -0.1]
    assert payload["source"]["end"] == [500.0, 200.0, -0.1]
    assert payload["validation"]["components"] == ["Ex", "Ey", "dBzdt"]


def test_safe_load_yaml_fallback_parses_nested_inline_lists_and_sequence_maps(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload = safe_load_yaml(
        """
time:
  time_steps: [[0.001, 3], 0.005]
receivers:
  - id: rx1
    location: [0.0, -300.0, -0.1]
    type: point
  - id: rx1_avg
    location: [0.0, -300.0, -0.1]
    type: volume_average
    radius: 2.0
"""
    )

    assert payload["time"]["time_steps"] == [[0.001, 3], 0.005]
    assert payload["receivers"] == [
        {"id": "rx1", "location": [0.0, -300.0, -0.1], "type": "point"},
        {
            "id": "rx1_avg",
            "location": [0.0, -300.0, -0.1],
            "type": "volume_average",
            "radius": 2.0,
        },
    ]
