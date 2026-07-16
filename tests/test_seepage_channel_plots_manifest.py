from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    for name in (
        "magnetic_receiver_comparison.png",
        "rx3_absolute_residual.png",
        "magnetic_symmetry_convergence.png",
    ):
        assert f"magnetic_root / \"{name}\"" in source


def test_report_geometry_draws_four_receivers_with_depth_down(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}

    def capture_figure(fig, output_root, stem) -> None:
        captured["fig"] = fig
        captured["output_root"] = output_root
        captured["stem"] = stem

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    plots.plot_model_geometry(tmp_path)

    fig = captured["fig"]
    try:
        axis_3d, axis_xz, axis_yz = fig.axes
        assert axis_3d.zaxis_inverted()
        assert axis_xz.yaxis_inverted()
        assert axis_yz.yaxis_inverted()
        assert len(axis_3d.collections[0]._offsets3d[0]) == 4
        assert len(axis_yz.collections[0].get_offsets()) == 4
        assert captured["stem"] == "model_geometry"
    finally:
        plt.close(fig)


def test_report_response_and_error_grids_omit_center_receiver(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    figures = []

    def capture_figure(fig, output_root, stem) -> None:
        figures.append((stem, fig))

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    times = np.asarray([1.0e-5, 1.0e-4])
    values = np.ones((5, 2, 3), dtype=float)
    plots._plot_response_grid(
        tmp_path,
        "response",
        times,
        {"FEniCSx": values},
        title="response",
    )
    plots._plot_error_grid(
        tmp_path,
        "error",
        times,
        {"FEniCSx": values * 1.1},
        values,
        title="error",
    )

    try:
        expected_labels = {"FEniCSx Rx1", "FEniCSx Rx2", "FEniCSx Rx4", "FEniCSx Rx5"}
        for _stem, fig in figures:
            labels = {
                line.get_label()
                for line in fig.axes[0].lines
                if not line.get_label().startswith("_")
            }
            assert labels == expected_labels
    finally:
        for _stem, fig in figures:
            plt.close(fig)
