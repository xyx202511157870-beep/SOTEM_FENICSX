from pathlib import Path

from tools.plot_seepage_channel_benchmark import build_output_inventory


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
        "magnetic_receiver_comparison",
        "rx3_absolute_residual",
        "magnetic_symmetry_convergence",
    ):
        assert name in source
    assert "R0=0.01" in source
    assert "Rx3 absolute residual" in source
