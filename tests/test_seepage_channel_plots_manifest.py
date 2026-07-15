from pathlib import Path

import json

from tools.plot_seepage_channel_benchmark import (
    build_output_inventory,
    load_geometry_contract,
)


def test_inventory_is_not_self_referential(tmp_path: Path) -> None:
    (tmp_path / "benchmark_results.npz").write_bytes(b"data")
    (tmp_path / "benchmark_manifest.json").write_text("stale", encoding="utf-8")
    inventory = build_output_inventory(tmp_path)
    assert set(inventory) == {"benchmark_results.npz"}
    assert inventory["benchmark_results.npz"]["sha256"]


def test_plot_source_contains_required_panels() -> None:
    source = Path("tools/plot_seepage_channel_benchmark.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "model_geometry",
        "background_response",
        "channel_response",
        "channel_delta",
        "convergence",
    ):
        assert name in source


def test_geometry_contract_is_loaded_from_result_audit(tmp_path: Path) -> None:
    audit = {
        "coordinate_convention": "z_down",
        "source_endpoints_m": [[-50.0, 0.0, 0.1], [50.0, 0.0, 0.1]],
        "receiver_locations_m": [[0.0, y, -0.1] for y in (-20, -10, 0, 10, 20)],
        "receiver_provenance": ["explicit_full_domain"] * 5,
        "channel": {
            "bounds_m": [[-30.0, 30.0], [-0.5, 0.5], [19.5, 20.5]],
            "size_m": [60.0, 1.0, 1.0],
            "conductivity_s_per_m": 1.0,
        },
        "empymod": {"background_only_1d": True},
    }
    (tmp_path / "model_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    contract = load_geometry_contract(tmp_path)
    assert contract["channel_bounds_m"] == audit["channel"]["bounds_m"]
