#!/usr/bin/env python3
"""Plot and inventory the canonical seepage-channel benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_channel_model import MODEL  # noqa: E402
from atem3d.seepage_channel_validation import (  # noqa: E402
    COMPONENTS,
    ordinary_relative_error,
)


FIGURE_STEMS = (
    "model_geometry",
    "background_response",
    "channel_response",
    "channel_delta",
    "channel_relative_anomaly",
    "channel_delta_signed",
    "channel_relative_anomaly_profiles",
    "background_error",
    "channel_delta_error",
    "convergence",
)

REPORT_RECEIVER_INDICES = (0, 1, 3, 4)
PROFILE_TARGET_TIMES = (1.0e-5, 3.162e-4, 1.0e-2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_output_inventory(root: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(root)
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "benchmark_manifest.json":
            continue
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return inventory


def _save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_box_edges(axis, bounds, **kwargs) -> None:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    corners = [
        (x, y, z)
        for x in (xmin, xmax)
        for y in (ymin, ymax)
        for z in (zmin, zmax)
    ]
    for index, first in enumerate(corners):
        for second in corners[index + 1 :]:
            if sum(a != b for a, b in zip(first, second)) == 1:
                axis.plot(*zip(first, second), **kwargs)


def plot_model_geometry(output_root: Path) -> None:
    fig = plt.figure(figsize=(14, 4.2))
    axis_3d = fig.add_subplot(131, projection="3d")
    source = np.asarray(MODEL.source_endpoints)
    receivers = np.asarray(MODEL.receiver_locations)[list(REPORT_RECEIVER_INDICES)]
    axis_3d.plot(source[:, 0], source[:, 1], source[:, 2], color="crimson", lw=3, label="100 m wire")
    axis_3d.scatter(
        receivers[:, 0],
        receivers[:, 1],
        receivers[:, 2],
        color="navy",
        s=28,
        label="Rx1, Rx2, Rx4, Rx5",
    )
    _draw_box_edges(axis_3d, MODEL.channel.bounds, color="teal", lw=1.5)
    axis_3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="physical z-down (m)")
    axis_3d.invert_zaxis()
    axis_3d.legend(fontsize=7)
    axis_3d.set_title("Full 3D geometry")

    axis_xz = fig.add_subplot(132)
    axis_xz.plot(source[:, 0], source[:, 2], color="crimson", lw=3)
    (xmin, xmax), (_ymin, _ymax), (zmin, zmax) = MODEL.channel.bounds
    axis_xz.add_patch(plt.Rectangle((xmin, zmin), xmax - xmin, zmax - zmin, color="teal", alpha=0.3))
    axis_xz.axhline(0.0, color="black", lw=0.8)
    axis_xz.set(xlabel="x (m)", ylabel="physical z-down (m)", title="x-z section")
    axis_xz.invert_yaxis()

    axis_yz = fig.add_subplot(133)
    axis_yz.scatter(receivers[:, 1], receivers[:, 2], color="navy", s=28)
    (_xmin, _xmax), (ymin, ymax), (zmin, zmax) = MODEL.channel.bounds
    axis_yz.add_patch(plt.Rectangle((ymin, zmin), ymax - ymin, zmax - zmin, color="teal", alpha=0.3))
    axis_yz.axhline(0.0, color="black", lw=0.8)
    axis_yz.set(xlabel="y (m)", ylabel="physical z-down (m)", title="y-z section")
    axis_yz.invert_yaxis()
    _save_figure(fig, output_root, "model_geometry")


def _symlog_limit(values: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(value[np.isfinite(value)]).reshape(-1) for value in values])
    peak = float(np.max(finite)) if finite.size else 1.0
    return max(peak * 1.0e-6, np.finfo(float).tiny)


def relative_anomaly_percent(
    delta: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Return pointwise channel anomaly as a finite percentage of background."""

    delta_values = np.asarray(delta, dtype=float)
    background_values = np.asarray(background, dtype=float)
    if delta_values.shape != background_values.shape:
        raise ValueError("delta and background must have matching shapes")
    scale = np.max(np.abs(background_values), axis=1, keepdims=True)
    floor = np.maximum(scale * 1.0e-12, np.finfo(float).tiny)
    denominator = np.maximum(np.abs(background_values), floor)
    return 100.0 * np.abs(delta_values) / denominator


def _representative_profile_indices(times: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(times, dtype=float)
    return tuple(
        int(np.argmin(np.abs(np.log(values) - np.log(target))))
        for target in PROFILE_TARGET_TIMES
    )


def _plot_response_grid(
    output_root: Path,
    stem: str,
    times: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str,
    magnitude_decay: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPORT_RECEIVER_INDICES)))
    linthresh = _symlog_limit(list(series.values()))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for color_index, receiver_index in enumerate(REPORT_RECEIVER_INDICES):
                receiver_values = values[receiver_index, :, component_index]
                plotted_values = (
                    np.abs(receiver_values) if magnitude_decay else receiver_values
                )
                axis.plot(
                    times,
                    plotted_values,
                    color=colors[color_index],
                    ls={"empymod": ":", "SimPEG": "--", "FEniCSx": "-"}.get(method, "-"),
                    lw=1.1,
                    label=f"{method} Rx{receiver_index + 1}" if component_index == 0 else None,
                )
        axis.set_xscale("log")
        if magnitude_decay:
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=linthresh)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(component)
        axis.set_xlabel("time (s)")
        unit = {"Ex": "V/m", "dBzdt": "T/s", "Hz": "A/m"}[component]
        axis.set_ylabel(f"|{component}| ({unit})" if magnitude_decay else unit)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    _save_figure(fig, output_root, stem)


def _plot_relative_anomaly_grid(
    output_root: Path,
    times: np.ndarray,
    series: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPORT_RECEIVER_INDICES)))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for color_index, receiver_index in enumerate(REPORT_RECEIVER_INDICES):
                axis.plot(
                    times,
                    values[receiver_index, :, component_index],
                    color=colors[color_index],
                    ls={"SimPEG": "--", "FEniCSx": "-"}[method],
                    lw=1.1,
                    label=(
                        f"{method} Rx{receiver_index + 1}"
                        if component_index == 0
                        else None
                    ),
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("relative anomaly (%)")
        axis.set_title(component)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Channel relative anomaly: |channel - background| / |background|")
    _save_figure(fig, output_root, "channel_relative_anomaly")


def _plot_relative_anomaly_profiles(
    output_root: Path,
    times: np.ndarray,
    receiver_locations: np.ndarray,
    series: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    gate_indices = _representative_profile_indices(times)
    receiver_indices = list(REPORT_RECEIVER_INDICES)
    y_positions = np.asarray(receiver_locations)[receiver_indices, 1]
    gate_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gate_indices)))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for gate_color, gate_index in zip(gate_colors, gate_indices):
                axis.plot(
                    y_positions,
                    values[receiver_indices, gate_index, component_index],
                    color=gate_color,
                    ls={"SimPEG": "--", "FEniCSx": "-"}[method],
                    marker="o",
                    lw=1.1,
                    label=(
                        f"{method}, {times[gate_index]:.3g} s"
                        if component_index == 0
                        else None
                    ),
                )
        axis.set_xlabel("receiver y (m)")
        axis.set_ylabel("relative anomaly (%)")
        axis.set_title(component)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Channel relative-anomaly profiles at representative times")
    _save_figure(fig, output_root, "channel_relative_anomaly_profiles")


def _plot_error_grid(
    output_root: Path,
    stem: str,
    times: np.ndarray,
    series: dict[str, np.ndarray],
    reference: np.ndarray,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            errors = ordinary_relative_error(values, reference)
            for receiver_index in REPORT_RECEIVER_INDICES:
                axis.plot(times, errors[receiver_index, :, component_index], lw=1, label=f"{method} Rx{receiver_index + 1}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(xlabel="time (s)", ylabel="ordinary relative error", title=component)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    _save_figure(fig, output_root, stem)


def plot_convergence(output_root: Path) -> None:
    summary_path = output_root / "convergence_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(8, 4.2))
    names: list[str] = []
    medians: list[float] = []
    for name, result in summary.items():
        if name == "available" or not isinstance(result, dict):
            continue
        value = result.get("median_relative_change")
        if value is not None:
            names.append(name)
            medians.append(float(value))
    if medians:
        axis.bar(names, medians, color="slateblue")
        axis.axhline(0.05, color="crimson", ls="--", label="5% target")
        axis.set_yscale("log")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Convergence cases not yet available", ha="center", va="center", transform=axis.transAxes)
    axis.set(ylabel="median relative change", title="Spatial/time convergence")
    axis.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_root, "convergence")


def plot_magnetic_receiver_comparison(method_table, output_path: Path) -> None:
    """Overlay signed dBz/dt methods at the four nonzero-y receivers."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    try:
        times = np.asarray(method_table["times"], dtype=float)
        selected = str(method_table.get("selected", "not_selected"))
        methods = method_table["methods"]
        for axis, receiver_index in zip(axes.flat, (0, 1, 3, 4)):
            for name, values in sorted(methods.items()):
                array = np.asarray(values, dtype=float)
                axis.plot(times, array[receiver_index, :, 1], label=name)
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=1.0e-20)
            axis.set_title(f"Rx{receiver_index + 1}")
            axis.grid(True, which="both", alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Magnetic receiver methods; selected={selected}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def plot_rx3_absolute_residual(method_table, metrics, output_path: Path) -> None:
    """Plot signed and absolute Rx3 values without percent error at a zero point."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        times = np.asarray(method_table["times"], dtype=float)
        for name, values in sorted(method_table["methods"].items()):
            rx3 = np.asarray(values, dtype=float)[2, :, 1]
            axes[0].plot(times, rx3, label=name)
            axes[1].plot(times, np.abs(rx3), label=name)
        axes[0].set(title="Rx3 signed magnetic value", xscale="log", yscale="symlog")
        axes[1].set(title="Rx3 absolute residual", xscale="log", yscale="log")
        axes[1].axhline(
            float(metrics.get("absolute_floor", 1.0e-20)),
            color="crimson",
            linestyle="--",
            label="numerical floor",
        )
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def plot_magnetic_symmetry_convergence(convergence, output_path: Path) -> None:
    """Plot R0 and odd-pair residuals with the R0=0.01 acceptance line."""

    fig, axis = plt.subplots(figsize=(8, 4.5))
    try:
        names = list(convergence)
        x = np.arange(len(names))
        for key, label in (
            ("rx3_zero_ratio", "R0"),
            ("pair_24_residual", "Rodd24"),
            ("pair_15_residual", "Rodd15"),
        ):
            axis.plot(x, [convergence[name].get(key, np.nan) for name in names], marker="o", label=label)
        axis.axhline(0.01, color="crimson", linestyle="--", label="R0=0.01")
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("normalized absolute residual")
        axis.legend()
        axis.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def generate_plots(result_dir: str | Path) -> list[Path]:
    output_root = Path(result_dir)
    with np.load(output_root / "benchmark_results.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    times = arrays["times"]
    plot_model_geometry(output_root)
    _plot_response_grid(
        output_root,
        "background_response",
        times,
        {"empymod": arrays["empymod_background"], "SimPEG": arrays["simpeg_background"], "FEniCSx": arrays["fenicsx_background"]},
        title="Background response: three algorithms",
        magnitude_decay=True,
    )
    _plot_response_grid(
        output_root,
        "channel_response",
        times,
        {"SimPEG": arrays["simpeg_channel"], "FEniCSx": arrays["fenicsx_channel"]},
        title="Finite 3D seepage-channel response",
        magnitude_decay=True,
    )
    _plot_response_grid(
        output_root,
        "channel_delta",
        times,
        {"SimPEG": arrays["simpeg_delta"], "FEniCSx": arrays["fenicsx_delta"]},
        title="Absolute channel-minus-background anomaly decay",
        magnitude_decay=True,
    )
    relative_series = {
        "SimPEG": relative_anomaly_percent(
            arrays["simpeg_delta"], arrays["simpeg_background"]
        ),
        "FEniCSx": relative_anomaly_percent(
            arrays["fenicsx_delta"], arrays["fenicsx_background"]
        ),
    }
    _plot_relative_anomaly_grid(output_root, times, relative_series)
    _plot_response_grid(
        output_root,
        "channel_delta_signed",
        times,
        {"SimPEG": arrays["simpeg_delta"], "FEniCSx": arrays["fenicsx_delta"]},
        title="Signed channel-minus-background anomaly",
        magnitude_decay=False,
    )
    _plot_relative_anomaly_profiles(
        output_root,
        times,
        arrays["receiver_locations"],
        relative_series,
    )
    _plot_error_grid(
        output_root,
        "background_error",
        times,
        {"SimPEG": arrays["simpeg_background"], "FEniCSx": arrays["fenicsx_background"]},
        arrays["empymod_background"],
        title="Background error relative to empymod 1D",
    )
    _plot_error_grid(
        output_root,
        "channel_delta_error",
        times,
        {"SimPEG delta": arrays["simpeg_delta"]},
        arrays["fenicsx_delta"],
        title="Channel-delta error: SimPEG relative to FEniCSx",
    )
    plot_convergence(output_root)
    magnetic_root = output_root / "magnetic_receiver_stability"
    payload_path = magnetic_root / "magnetic_method_payload.npz"
    metrics_path = magnetic_root / "magnetic_symmetry_metrics.json"
    convergence_path = magnetic_root / "magnetic_convergence_summary.json"
    if payload_path.exists() and metrics_path.exists() and convergence_path.exists():
        with np.load(payload_path, allow_pickle=False) as payload:
            methods = {
                key.split("__", 1)[1]: np.asarray(payload[key])
                for key in payload.files
                if key.startswith("channel__")
            }
            method_table = {"times": np.asarray(payload["times"]), "methods": methods}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
        method_table["selected"] = convergence.get("selected")
        plot_magnetic_receiver_comparison(
            method_table, magnetic_root / "magnetic_receiver_comparison.png"
        )
        plot_rx3_absolute_residual(
            method_table, metrics, magnetic_root / "rx3_absolute_residual.png"
        )
        plot_magnetic_symmetry_convergence(
            metrics, magnetic_root / "magnetic_symmetry_convergence.png"
        )
    return [output_root / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_available_in_manifest_environment"


def write_manifest(result_dir: str | Path) -> Path:
    output_root = Path(result_dir)
    manifest = {
        "inventory": build_output_inventory(output_root),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "empymod": _package_version("empymod"),
            "simpeg": _package_version("simpeg"),
            "discretize": _package_version("discretize"),
            "fenicsx": _package_version("fenics-dolfinx"),
            "petsc4py": _package_version("petsc4py"),
        },
        "model_contract": {
            "coordinate_convention": MODEL.coordinate_convention,
            "source_endpoints_m": MODEL.source_endpoints,
            "receiver_locations_m": MODEL.receiver_locations,
            "channel_bounds_m": MODEL.channel.bounds,
            "channel_conductivity_s_per_m": MODEL.channel.conductivity,
            "times_s": MODEL.times.tolist(),
            "fenicsx_receiver_provenance": ["explicit_full_domain"] * 5,
            "empymod_background_only_1d": True,
        },
    }
    path = output_root / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.manifest_only:
        generate_plots(args.result_dir)
    manifest = write_manifest(args.result_dir)
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
