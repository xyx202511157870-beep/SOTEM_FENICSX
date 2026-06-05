import json

from atem3d.benchmark_cli import load_benchmark_spec


def test_load_benchmark_spec_reads_cases_and_metadata(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        """
tolerance: 0.05
component_names: [Ex, Hz]
reference: large
time_min: 0.001
time_max: 0.01
cases:
  - name: small
    config: {time_steps: [0.1]}
  - name: large
    config: {time_steps: [0.1]}
""",
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.tolerance == 0.05
    assert spec.component_names == ["Ex", "Hz"]
    assert spec.reference == "large"
    assert spec.time_min == 0.001
    assert spec.time_max == 0.01
    assert [case.name for case in spec.cases] == ["small", "large"]


def test_load_benchmark_spec_merges_base_config_with_case_overrides(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        """
tolerance: 0.05
absolute_tolerance: 1.0e-8
component_names: [Ex]
data_only: true
base_config:
  mesh:
    hx: [1.0]
    hy: [1.0]
    hz: [1.0]
  boundary:
    kind: cpml
    thickness_cells: 2
  nested:
    keep: 1
    replace: 2
cases:
  - name: cpml
    overrides:
      boundary:
        sigma_max: 50.0
      nested:
        replace: 3
  - name: none
    overrides:
      boundary:
        kind: none
""",
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    assert spec.data_only is True
    assert spec.absolute_tolerance == 1.0e-8
    assert spec.cases[0].config["mesh"]["hx"] == [1.0]
    assert spec.cases[0].config["boundary"] == {
        "kind": "cpml",
        "thickness_cells": 2,
        "sigma_max": 50.0,
    }
    assert spec.cases[0].config["nested"] == {"keep": 1, "replace": 3}
    assert spec.cases[1].config["boundary"] == {"kind": "none", "thickness_cells": 2}
