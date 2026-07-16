import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.plot_seepage_channel_benchmark import (
    build_output_inventory,
    load_geometry_contract,
)


def test_relative_anomaly_percent_uses_pointwise_background_with_finite_floor() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    background = np.asarray([[[2.0], [0.0]]])
    delta = np.asarray([[[1.0], [0.25]]])

    result = plots.relative_anomaly_percent(delta, background)

    assert result[0, 0, 0] == 50.0
    assert np.isfinite(result).all()
    assert result[0, 1, 0] == 100.0 * 0.25 / (2.0e-12)


def test_representative_profile_indices_select_nearest_output_gates() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    times = np.asarray([1.0e-5, 1.0e-4, 3.162e-4, 1.0e-3, 1.0e-2])

    assert plots._representative_profile_indices(times) == (0, 2, 4)


def test_channel_diagnostic_plot_source_contains_new_formal_artifacts() -> None:
    import tools.plot_seepage_channel_benchmark as plots

    assert "channel_relative_anomaly" in plots.FIGURE_STEMS
    assert "channel_delta_signed" in plots.FIGURE_STEMS
    assert "channel_relative_anomaly_profiles" in plots.FIGURE_STEMS


def test_signed_anomaly_plot_preserves_negative_values(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}

    def capture_figure(fig, output_root, stem) -> None:
        captured["fig"] = fig

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    times = np.asarray([1.0e-5, 1.0e-4])
    values = np.ones((5, 2, 3), dtype=float)
    values[:, 1, :] = -0.5

    plots._plot_response_grid(
        tmp_path,
        "channel_delta_signed",
        times,
        {"FEniCSx": values},
        title="signed",
        magnitude_decay=False,
    )

    fig = captured["fig"]
    try:
        assert all(axis.get_yscale() == "symlog" for axis in fig.axes)
        assert any(
            np.any(np.asarray(line.get_ydata()) < 0.0)
            for line in fig.axes[0].lines
        )
    finally:
        plt.close(fig)


def test_relative_anomaly_profiles_exclude_rx3(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}

    def capture_figure(fig, output_root, stem) -> None:
        captured["fig"] = fig

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    times = np.asarray([1.0e-5, 3.162e-4, 1.0e-2])
    receiver_locations = np.column_stack(
        (np.zeros(5), [-20.0, -10.0, 0.0, 10.0, 20.0], np.zeros(5))
    )
    relative = np.ones((5, 3, 3), dtype=float)

    plots._plot_relative_anomaly_profiles(
        tmp_path,
        times,
        receiver_locations,
        {"FEniCSx": relative},
    )

    fig = captured["fig"]
    try:
        for axis in fig.axes:
            for line in axis.lines:
                np.testing.assert_array_equal(
                    line.get_xdata(), [-20.0, -10.0, 10.0, 20.0]
                )
    finally:
        plt.close(fig)


def test_relative_anomaly_decay_uses_log_percent_axes(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}

    def capture_figure(fig, output_root, stem) -> None:
        captured["fig"] = fig

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    times = np.asarray([1.0e-5, 1.0e-4, 1.0e-3])
    relative = np.full((5, 3, 3), 10.0, dtype=float)

    plots._plot_relative_anomaly_grid(
        tmp_path,
        times,
        {"FEniCSx": relative},
    )

    fig = captured["fig"]
    try:
        for axis in fig.axes:
            assert axis.get_xscale() == "log"
            assert axis.get_yscale() == "log"
            assert axis.get_ylabel() == "relative anomaly (%)"
    finally:
        plt.close(fig)


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
    (tmp_path / "model_audit.json").write_text(
        json.dumps(
            {
                "model_fingerprint": "a" * 64,
                "coordinate_convention": "z_down",
                "source_endpoints_m": [[-50.0, 0.0, 0.1], [50.0, 0.0, 0.1]],
                "receiver_locations_m": [
                    [0.0, -20.0, -0.1],
                    [0.0, -10.0, -0.1],
                    [0.0, 0.0, -0.1],
                    [0.0, 10.0, -0.1],
                    [0.0, 20.0, -0.1],
                ],
                "channel": {
                    "bounds_m": [[-30.0, 30.0], [-0.5, 0.5], [19.5, 20.5]],
                    "size_m": [60.0, 1.0, 1.0],
                    "conductivity_s_per_m": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    plots.plot_model_geometry(tmp_path)

    fig = captured["fig"]
    try:
        axis_3d, axis_xz, axis_yz = fig.axes
        assert axis_3d.zaxis_inverted()
        assert axis_xz.yaxis_inverted()
        assert axis_yz.yaxis_inverted()
        assert len(axis_3d.collections[0]._offsets3d[0]) == 4
        assert len(axis_yz.collections[0].get_offsets()) == 4
        channel_patch_xz = axis_xz.patches[0]
        channel_patch_yz = axis_yz.patches[0]
        assert channel_patch_xz.get_height() == 1.0
        assert channel_patch_yz.get_width() == 1.0
        assert channel_patch_yz.get_height() == 1.0
        assert captured["stem"] == "model_geometry"
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert not axis_3d.zaxis.label.get_window_extent(renderer).overlaps(
            axis_xz.yaxis.label.get_window_extent(renderer)
        )
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


def test_background_response_uses_absolute_log_decay(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.plot_seepage_channel_benchmark as plots

    captured = {}

    def capture_figure(fig, output_root, stem) -> None:
        captured["fig"] = fig

    monkeypatch.setattr(plots, "_save_figure", capture_figure)
    times = np.asarray([1.0e-5, 1.0e-4, 1.0e-3])
    signed_trace = np.asarray(
        [
            [1.0e-3, -1.0e-4, -1.0e-5],
            [1.0e-4, -1.0e-5, -1.0e-6],
            [1.0e-5, -1.0e-6, -1.0e-7],
        ]
    )
    values = np.repeat(signed_trace[None, :, :], 5, axis=0)
    original = values.copy()

    plots._plot_response_grid(
        tmp_path,
        "background_response",
        times,
        {"FEniCSx": values},
        title="background",
        magnitude_decay=True,
    )

    fig = captured["fig"]
    try:
        for axis in fig.axes:
            assert axis.get_yscale() == "log"
            assert axis.get_ylabel().startswith("|")
            for line in axis.lines:
                assert np.all(np.asarray(line.get_ydata()) > 0.0)
        np.testing.assert_array_equal(values, original)
    finally:
        plt.close(fig)


def test_all_formal_response_plots_enable_magnitude_decay() -> None:
    source = Path("tools/plot_seepage_channel_benchmark.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    configured_stems = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_plot_response_grid":
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            continue
        magnitude_keyword = next(
            (item for item in node.keywords if item.arg == "magnitude_decay"), None
        )
        if (
            magnitude_keyword is not None
            and isinstance(magnitude_keyword.value, ast.Constant)
            and magnitude_keyword.value.value is True
        ):
            configured_stems.add(node.args[1].value)

    assert configured_stems == {
        "background_response",
        "channel_response",
        "channel_delta",
    }


def test_geometry_contract_is_loaded_from_result_audit(tmp_path: Path) -> None:
    audit = {
        "model_fingerprint": "a" * 64,
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
    assert contract["model_fingerprint"] == "a" * 64
