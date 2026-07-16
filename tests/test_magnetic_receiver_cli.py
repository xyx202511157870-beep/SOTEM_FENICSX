from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


def load_pipeline_module():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_magnetic_cli",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_faraday_loop_and_tetra4_configuration_is_accepted() -> None:
    sp = load_pipeline_module()
    config = sp.PipelineConfig(
        magnetic_receiver_mode="biot_current",
        magnetic_dbdt_mode="faraday_loop",
        biot_current_integration="tetra4",
        faraday_loop_radius=2.0,
        faraday_loop_quadrature_points=32,
        magnetic_diagnostic_methods=(
            "curl",
            "biot_rate",
            "faraday_loop",
            "biot_center",
            "biot_tetra4",
        ),
    )

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["magnetic_dbdt_mode"] == "faraday_loop"
    assert diagnostics["biot_current_integration"] == "tetra4"
    assert diagnostics["magnetic_diagnostic_methods"] == [
        "curl",
        "biot_rate",
        "faraday_loop",
        "biot_center",
        "biot_tetra4",
    ]


def test_historical_magnetic_receiver_defaults_are_unchanged() -> None:
    sp = load_pipeline_module()

    config = sp.PipelineConfig()

    assert config.magnetic_dbdt_mode == "curl"
    assert config.biot_current_integration == "cell_center"
    assert config.magnetic_diagnostic_methods == ()
    assert config.faraday_loop_radius == 2.0
    assert config.faraday_loop_quadrature_points == 32


def test_parse_magnetic_diagnostic_methods_preserves_first_occurrence_order() -> None:
    sp = load_pipeline_module()

    parsed = sp._parse_magnetic_diagnostic_methods(
        "curl, biot_rate,faraday_loop,curl,biot_center,biot_tetra4"
    )

    assert parsed == (
        "curl",
        "biot_rate",
        "faraday_loop",
        "biot_center",
        "biot_tetra4",
    )
    assert sp._parse_magnetic_diagnostic_methods("") == ()


@pytest.mark.parametrize("points", [4, 10, 30])
def test_faraday_loop_rejects_invalid_point_count(points: int) -> None:
    sp = load_pipeline_module()

    with pytest.raises(ValueError, match="multiple of 4"):
        sp.validate_model_consistency(
            sp.PipelineConfig(
                magnetic_dbdt_mode="faraday_loop",
                faraday_loop_quadrature_points=points,
            )
        )


def test_magnetic_receiver_rejects_unknown_integration_and_diagnostic() -> None:
    sp = load_pipeline_module()

    with pytest.raises(ValueError, match="biot_current_integration"):
        sp.validate_model_consistency(
            sp.PipelineConfig(biot_current_integration="unknown")
        )
    with pytest.raises(ValueError, match="unknown magnetic diagnostic"):
        sp.validate_model_consistency(
            sp.PipelineConfig(magnetic_diagnostic_methods=("unknown",))
        )


def test_magnetic_diagnostic_parser_rejects_empty_entries() -> None:
    sp = load_pipeline_module()

    with pytest.raises(ValueError, match="empty"):
        sp._parse_magnetic_diagnostic_methods("curl,,biot_rate")


def test_resolved_yaml_records_magnetic_receiver_configuration() -> None:
    sp = load_pipeline_module()
    config = sp.PipelineConfig(
        magnetic_dbdt_mode="faraday_loop",
        biot_current_integration="tetra4",
        faraday_loop_radius=2.5,
        faraday_loop_quadrature_points=64,
        magnetic_diagnostic_methods=("curl", "faraday_loop", "biot_tetra4"),
    )

    resolved = yaml.safe_load(sp._resolved_config_yaml(config))

    assert resolved["magnetic_dbdt_mode"] == "faraday_loop"
    assert resolved["biot_current_integration"] == "tetra4"
    assert resolved["faraday_loop_radius"] == 2.5
    assert resolved["faraday_loop_quadrature_points"] == 64
    assert resolved["magnetic_diagnostic_methods"] == [
        "curl",
        "faraday_loop",
        "biot_tetra4",
    ]
