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

from atem3d.seepage_channel_validation import (  # noqa: E402
    COMPONENTS,
    ordinary_relative_error,
)


FIGURE_STEMS = (
    "model_geometry",
    "background_response",
    "channel_response",
    "channel_delta",
    "background_error",
    "channel_delta_error",
    "convergence",
)


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


def load_geometry_contract(result_dir: str | Path) -> dict[str, Any]:
    """Load the solved model geometry from its authoritative audit artifact."""

    audit = json.loads((Path(result_dir) / "model_audit.json").read_text(encoding="utf-8"))
    return {
        "variant": audit.get("variant", "baseline_60x10x10"),
        "coordinate_convention": audit["coordinate_convention"],
        "source_endpoints_m": audit["source_endpoints_m"],
        "receiver_locations_m": audit["receiver_locations_m"],
        "receiver_provenance": audit["receiver_provenance"],
        "channel_bounds_m": audit["channel"]["bounds_m"],
        "channel_size_m": audit["channel"]["size_m"],
        "channel_conductivity_s_per_m": audit["channel"]["conductivity_s_per_m"],
        "empymod_background_only_1d": audit["empymod"]["background_only_1d"],
    }


def plot_model_geometry(output_root: Path) -> None:
    contract = load_geometry_contract(output_root)
    fig = plt.figure(figsize=(14, 4.4))
    axis_3d = fig.add_subplot(131, projection="3d")
    source = np.asarray(contract["source_endpoints_m"], dtype=float)
    receivers = np.asarray(contract["receiver_locations_m"], dtype=float)
    channel_bounds = contract["channel_bounds_m"]
    axis_3d.plot(source[:, 0], source[:, 1], source[:, 2], color="crimson", lw=3, label="100 m wire")
    axis_3d.scatter(receivers[:, 0], receivers[:, 1], receivers[:, 2], color="navy", s=28, label="Rx1-Rx5")
    _draw_box_edges(axis_3d, channel_bounds, color="teal", lw=1.5)
    axis_3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z-down (m)")
    axis_3d.zaxis.labelpad = 5
    axis_3d.legend(fontsize=7)
    axis_3d.set_title("Full 3D geometry")

    axis_xz = fig.add_subplot(132)
    axis_xz.plot(source[:, 0], source[:, 2], color="crimson", lw=3)
    (xmin, xmax), (_ymin, _ymax), (zmin, zmax) = channel_bounds
    axis_xz.add_patch(plt.Rectangle((xmin, zmin), xmax - xmin, zmax - zmin, color="teal", alpha=0.3))
    axis_xz.axhline(0.0, color="black", lw=0.8)
    axis_xz.set(xlabel="x (m)", ylabel="z-down (m)", title="x-z section")
    axis_xz.invert_yaxis()

    axis_yz = fig.add_subplot(133)
    axis_yz.scatter(receivers[:, 1], receivers[:, 2], color="navy", s=28)
    (_xmin, _xmax), (ymin, ymax), (zmin, zmax) = channel_bounds
    axis_yz.add_patch(plt.Rectangle((ymin, zmin), ymax - ymin, zmax - zmin, color="teal", alpha=0.3))
    axis_yz.axhline(0.0, color="black", lw=0.8)
    axis_yz.set(xlabel="y (m)", ylabel="z-down (m)", title="y-z section")
    axis_yz.invert_yaxis()
    _save_figure(fig, output_root, "model_geometry")


def _symlog_limit(values: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(value[np.isfinite(value)]).reshape(-1) for value in values])
    peak = float(np.max(finite)) if finite.size else 1.0
    return max(peak * 1.0e-6, np.finfo(float).tiny)


def _plot_response_grid(
    output_root: Path,
    stem: str,
    times: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    linthresh = _symlog_limit(list(series.values()))
    for component_index, (axis, component) in enumerate(zip(axes, COMPONENTS)):
        for method, values in series.items():
            for receiver_index in range(5):
                axis.plot(
                    times,
                    values[receiver_index, :, component_index],
                    color=colors[receiver_index],
                    ls={"empymod": ":", "SimPEG": "--", "FEniCSx": "-"}.get(method, "-"),
                    lw=1.1,
                    label=f"{method} Rx{receiver_index + 1}" if component_index == 0 else None,
                )
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=linthresh)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(component)
        axis.set_xlabel("time (s)")
        axis.set_ylabel({"Ex": "V/m", "dBzdt": "T/s", "Hz": "A/m"}[component])
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    _save_figure(fig, output_root, stem)


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
            for receiver_index in range(5):
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
    )
    _plot_response_grid(
        output_root,
        "channel_response",
        times,
        {"SimPEG": arrays["simpeg_channel"], "FEniCSx": arrays["fenicsx_channel"]},
        title="Finite 3D seepage-channel response",
    )
    _plot_response_grid(
        output_root,
        "channel_delta",
        times,
        {"SimPEG": arrays["simpeg_delta"], "FEniCSx": arrays["fenicsx_delta"]},
        title="Signed channel minus background response",
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
    return [output_root / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_available_in_manifest_environment"


def write_manifest(result_dir: str | Path) -> Path:
    output_root = Path(result_dir)
    contract = load_geometry_contract(output_root)
    with np.load(output_root / "benchmark_results.npz", allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float).tolist()
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
            **contract,
            "times_s": times,
            "fenicsx_receiver_provenance": contract["receiver_provenance"],
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
