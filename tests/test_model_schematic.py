import json

from atem3d.cli import main
from atem3d.corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_leakage_channel_case_specs,
)
from atem3d.model_schematic import write_model_schematic


def test_write_model_schematic_records_corrected_geometry_metadata(tmp_path):
    specs = build_corrected_leakage_channel_case_specs(
        tmp_path,
        config=CorrectedModelValidationConfig(n_observation_times=3),
    )
    output = tmp_path / "model_schematic.png"

    info = write_model_schematic(specs["noip"], output)

    assert output.exists()
    assert output.stat().st_size > 1000
    assert info["source_length_m"] == 1000.0
    assert info["parallel_offset_m"] == 500.0
    assert info["domain_extent_m"] == [4000.0, 4000.0, 1100.0]
    assert info["leakage_point_count"] == 4


def test_model_schematic_cli_writes_png_from_corrected_spec(tmp_path):
    specs = build_corrected_leakage_channel_case_specs(
        tmp_path / "run",
        config=CorrectedModelValidationConfig(n_observation_times=3),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    output = tmp_path / "ip_model_schematic.png"

    exit_code = main(["model-schematic", str(spec_path), "--case", "ip", "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    assert output.stat().st_size > 1000
