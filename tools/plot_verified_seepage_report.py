#!/usr/bin/env python3
"""Generate formal figures exclusively from passing seepage verification data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_verification import VerificationGateError  # noqa: E402


COMPONENTS = ("Ex", "dBzdt", "Hz")
UNITS = ("V/m", "T/s", "A/m")
REPORT_RECEIVER_INDICES = (0, 1, 3, 4)
FIGURE_NAMES = (
    "verified_model_geometry.png",
    "verified_background_decay.png",
    "verified_channel_decay.png",
    "verified_signed_anomaly.png",
    "verified_relative_anomaly.png",
    "verified_conductivity_sweep.png",
    "verified_volume_sweep.png",
    "verified_convergence.png",
    "verified_parity.png",
    "verified_two_solver_anomaly.png",
)


def require_verified_summary(result_dir: str | Path) -> dict[str, Any]:
    root = Path(result_dir)
    path = root / "verification_summary.json"
    if not path.is_file():
        raise VerificationGateError("verified plots require verification_summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("pass"):
        failed = ", ".join(summary.get("failed_gates", ["unknown"]))
        raise VerificationGateError(f"verified plots blocked by failed gates: {failed}")
    return summary


def _load(path: Path, fingerprint: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    stored = str(data.get("base_model_fingerprint", data["model_fingerprint"]).item())
    if stored != fingerprint:
        raise VerificationGateError(f"model fingerprint mismatch in {path}")
    if np.asarray(data["values"]).shape != (5, 31, 3):
        raise VerificationGateError(f"unexpected response shape in {path}")
    return data


def _case(root: Path, solver: str, case_id: str, fingerprint: str) -> dict[str, np.ndarray]:
    return _load(root / "verification_runs" / solver / case_id / "normalized.npz", fingerprint)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_box(axis: Any, bounds: list[list[float]]) -> None:
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
                axis.plot(*zip(first, second), color="#008b8b", lw=1.6)


def _plot_geometry(root: Path, fingerprint: str) -> None:
    audit = json.loads((root / "model_audit.json").read_text(encoding="utf-8"))
    if audit.get("model_fingerprint") != fingerprint:
        raise VerificationGateError("model_audit.json fingerprint mismatch")
    source = np.asarray(audit["source_endpoints_m"], dtype=float)
    receivers = np.asarray(audit["receiver_locations_m"], dtype=float)[
        list(REPORT_RECEIVER_INDICES)
    ]
    bounds = audit["channel"]["bounds_m"]
    fig = plt.figure(figsize=(13.5, 4.2))
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot(*source.T, color="crimson", lw=3, label="100 m wire")
    ax3d.scatter(*receivers.T, color="navy", s=28, label="Rx1, Rx2, Rx4, Rx5")
    _draw_box(ax3d, bounds)
    ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z-down (m)", title="Full 3D geometry")
    ax3d.invert_zaxis()
    ax3d.legend(fontsize=7)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax_xz = fig.add_subplot(132)
    ax_xz.plot(source[:, 0], source[:, 2], color="crimson", lw=3)
    ax_xz.add_patch(plt.Rectangle((xmin, zmin), xmax - xmin, zmax - zmin, color="#008b8b", alpha=0.3))
    ax_xz.axhline(0.0, color="black", lw=0.8)
    ax_xz.set(xlabel="x (m)", ylabel="z-down (m)", title="x-z section")
    ax_xz.invert_yaxis()

    ax_yz = fig.add_subplot(133)
    ax_yz.scatter(receivers[:, 1], receivers[:, 2], color="navy", s=28)
    ax_yz.add_patch(plt.Rectangle((ymin, zmin), ymax - ymin, zmax - zmin, color="#008b8b", alpha=0.3))
    ax_yz.axhline(0.0, color="black", lw=0.8)
    ax_yz.set(xlabel="y (m)", ylabel="z-down (m)", title="y-z section")
    ax_yz.invert_yaxis()
    _save(fig, root / FIGURE_NAMES[0])


def _positive(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(values, dtype=float))
    finite = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    floor = (float(np.max(finite)) * 1e-15) if finite.size else np.finfo(float).tiny
    return np.maximum(magnitude, floor)


def _plot_grid(
    path: Path,
    times: np.ndarray,
    series: Mapping[str, np.ndarray],
    *,
    title: str,
    mode: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPORT_RECEIVER_INDICES)))
    styles = {"empymod": ":", "SimPEG": "--", "FEniCSx": "-"}
    all_values = np.concatenate([np.ravel(np.abs(value)) for value in series.values()])
    linthresh = max(float(np.nanmax(all_values)) * 1e-7, np.finfo(float).tiny)
    for component_index, (axis, component, unit) in enumerate(zip(axes, COMPONENTS, UNITS)):
        for method, values in series.items():
            for color_index, receiver_index in enumerate(REPORT_RECEIVER_INDICES):
                curve = values[receiver_index, :, component_index]
                if mode in {"decay", "relative"}:
                    curve = _positive(curve)
                axis.plot(
                    times,
                    curve,
                    color=colors[color_index],
                    ls=styles.get(method, "-"),
                    lw=1.05,
                    label=f"{method} Rx{receiver_index + 1}" if component_index == 0 else None,
                )
        axis.set_xscale("log")
        if mode in {"decay", "relative"}:
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=linthresh)
        axis.set(xlabel="time (s)", title=component)
        axis.set_ylabel("relative anomaly (%)" if mode == "relative" else (f"|{component}| ({unit})" if mode == "decay" else unit))
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title)
    _save(fig, path)


def _relative(delta: np.ndarray, background: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(background), axis=1, keepdims=True)
    floor = np.maximum(scale * 1e-12, np.finfo(float).tiny)
    return 100.0 * np.abs(delta) / np.maximum(np.abs(background), floor)


def _plot_sweep(path: Path, summary: dict[str, Any], stem: str, xlabel: str) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    for solver, marker in (("simpeg", "o"), ("fenicsx", "s")):
        gate = summary["gates"][f"{stem}_trend_{solver}"]
        axis.plot(gate["control_values"], gate["energies"], marker=marker, label=solver.capitalize())
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel=xlabel, ylabel="normalized anomaly L2 energy")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    _save(fig, path)


def _plot_convergence(path: Path, summary: dict[str, Any]) -> None:
    labels: list[str] = []
    coarse: list[float] = []
    fine: list[float] = []
    for kind in ("spatial", "temporal"):
        for solver in ("simpeg", "fenicsx"):
            gate = summary["gates"][f"{kind}_convergence_{solver}"]
            labels.append(f"{kind}\n{solver}")
            coarse.append(gate["medium_coarse"]["median"])
            fine.append(gate["fine_medium"]["median"])
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    axis.bar(x - 0.18, np.asarray(coarse) * 100, 0.36, label="medium vs coarse")
    axis.bar(x + 0.18, np.asarray(fine) * 100, 0.36, label="fine vs medium")
    axis.set_xticks(x, labels)
    axis.set_ylabel("median relative change (%)")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    _save(fig, path)


def _plot_parity(path: Path, summary: dict[str, Any]) -> None:
    labels: list[str] = []
    pair: list[float] = []
    center: list[float] = []
    for solver in ("simpeg", "fenicsx"):
        components = summary["gates"][f"parity_{solver}"]["components"]
        for component in COMPONENTS:
            item = components[component]
            labels.append(f"{solver}\n{component}")
            pair.append(max(item["pair_15_residual"], item["pair_24_residual"]))
            center.append(item["center_ratio"])
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(9.3, 4.6))
    axis.semilogy(x, np.maximum(pair, 1e-16), "o-", label="max pair residual")
    axis.semilogy(x, np.maximum(center, 1e-16), "s--", label="center diagnostic")
    axis.axhline(0.05, color="crimson", ls=":", label="pair threshold")
    axis.set_xticks(x, labels)
    axis.set_ylabel("normalized residual")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    _save(fig, path)


def _plot_two_solver(path: Path, times: np.ndarray, deltas: Mapping[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12), sharex=True)
    styles = {"SimPEG": "--", "FEniCSx": "-"}
    for row, receiver_index in enumerate(REPORT_RECEIVER_INDICES):
        for column, (component, unit) in enumerate(zip(COMPONENTS, UNITS)):
            axis = axes[row, column]
            peak = max(float(np.max(np.abs(value[receiver_index, :, column]))) for value in deltas.values())
            for method, values in deltas.items():
                axis.plot(times, values[receiver_index, :, column], ls=styles[method], label=method)
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=max(peak * 1e-7, np.finfo(float).tiny))
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(f"Rx{receiver_index + 1} {component}")
            axis.set_ylabel(unit)
            if row == 3:
                axis.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Verified 3D channel anomaly: channel - matching background")
    _save(fig, path)


def generate_verified_plots(result_dir: str | Path) -> list[Path]:
    root = Path(result_dir)
    summary = require_verified_summary(root)
    fingerprint = str(summary["model_fingerprint"])
    _plot_geometry(root, fingerprint)

    backgrounds: dict[str, np.ndarray] = {}
    channels: dict[str, np.ndarray] = {}
    for solver, label in (("simpeg", "SimPEG"), ("fenicsx", "FEniCSx")):
        backgrounds[label] = _case(root, solver, f"{solver}-conductivity-background-reference", fingerprint)["values"]
        channels[label] = _case(root, solver, f"{solver}-conductivity-channel-sigma-1", fingerprint)["values"]
    empymod = _load(root / "verification_empymod_background.npz", fingerprint)
    backgrounds["empymod"] = empymod["values"]
    times = np.asarray(empymod["times"], dtype=float)

    _plot_grid(root / FIGURE_NAMES[1], times, backgrounds, title="Verified uniform-background response", mode="decay")
    _plot_grid(root / FIGURE_NAMES[2], times, channels, title="Verified finite 60 x 1 x 1 m channel total response", mode="decay")
    deltas = {name: channels[name] - backgrounds[name] for name in ("SimPEG", "FEniCSx")}
    _plot_grid(root / FIGURE_NAMES[3], times, deltas, title="Signed channel anomaly (channel - background)", mode="signed")
    relative = {name: _relative(deltas[name], backgrounds[name]) for name in deltas}
    _plot_grid(root / FIGURE_NAMES[4], times, relative, title="Relative channel anomaly", mode="relative")
    _plot_sweep(root / FIGURE_NAMES[5], summary, "conductivity", "channel conductivity (S/m)")
    _plot_sweep(root / FIGURE_NAMES[6], summary, "volume", "channel cross-section width (m)")
    _plot_convergence(root / FIGURE_NAMES[7], summary)
    _plot_parity(root / FIGURE_NAMES[8], summary)
    _plot_two_solver(root / FIGURE_NAMES[9], times, deltas)
    return [root / name for name in FIGURE_NAMES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in generate_verified_plots(args.result_dir):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
