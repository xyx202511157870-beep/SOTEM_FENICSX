import importlib.util
from pathlib import Path
import sys


def load_pipeline():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location("seepage_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_receiver_set_is_explicit_and_preserves_legacy_receiver() -> None:
    module = load_pipeline()
    cfg = module.PipelineConfig(
        receiver_locations=((0, -20, 0.1), (0, 20, 0.1))
    )
    assert module._resolved_receiver_locations(cfg) == (
        (0.0, -20.0, 0.1),
        (0.0, 20.0, 0.1),
    )
    legacy = module.PipelineConfig(receiver=(1, 2, 3))
    assert module._resolved_receiver_locations(legacy) == ((1.0, 2.0, 3.0),)


def test_channel_box_is_z_up_and_auditable() -> None:
    module = load_pipeline()
    cfg = module.PipelineConfig(
        conductivity_box_bounds=((-30, 30), (-5, 5), (-25, -15)),
        conductivity_box_sigma=1.0,
        conductivity_box_name="seepage_channel",
    )
    audit = module._conductivity_box_config_audit(cfg)
    assert audit["bounds"] == [
        [-30.0, 30.0],
        [-5.0, 5.0],
        [-25.0, -15.0],
    ]
    assert audit["theoretical_volume_m3"] == 6000.0


def test_cli_coordinate_parsers_are_deterministic() -> None:
    module = load_pipeline()
    assert module._parse_receiver_locations(
        ["0,-20,0.1", "0,20,0.1"]
    ) == ((0.0, -20.0, 0.1), (0.0, 20.0, 0.1))
    assert module._parse_conductivity_box_bounds(
        "-30,30;-5,5;-25,-15"
    ) == ((-30.0, 30.0), (-5.0, 5.0), (-25.0, -15.0))
