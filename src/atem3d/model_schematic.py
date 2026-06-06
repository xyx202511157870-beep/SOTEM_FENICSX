"""Model schematic plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_model_schematic(case_spec: dict, output_path: str | Path) -> dict:
    """Write a plan-view/depth-view schematic PNG for a corrected-model spec."""

    output = Path(output_path)
    source_start = np.asarray(case_spec["source_start"], dtype=float)
    source_end = np.asarray(case_spec["source_end"], dtype=float)
    receiver = np.asarray(case_spec["receiver"], dtype=float)
    forward_cfg = dict(case_spec.get("dolfinx_forward", {}))
    domain_min = np.asarray(forward_cfg.get("domain_min", [np.nan, np.nan, np.nan]), dtype=float)
    domain_max = np.asarray(forward_cfg.get("domain_max", [np.nan, np.nan, np.nan]), dtype=float)
    leakage_cfg = dict(forward_cfg.get("leakage_channel", {}))
    leakage_points = np.asarray(leakage_cfg.get("points", []), dtype=float)
    if leakage_points.size:
        leakage_points = leakage_points.reshape((-1, 3))
    else:
        leakage_points = np.empty((0, 3), dtype=float)

    info = {
        "source_length_m": float(np.linalg.norm(source_end - source_start)),
        "parallel_offset_m": float(abs(receiver[1] - 0.5 * (source_start[1] + source_end[1]))),
        "domain_extent_m": [float(value) for value in (domain_max - domain_min)],
        "leakage_point_count": int(leakage_points.shape[0]),
    }

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    plan, depth = axes

    if np.all(np.isfinite(domain_min[:2])) and np.all(np.isfinite(domain_max[:2])):
        plan.add_patch(
            Rectangle(
                (domain_min[0], domain_min[1]),
                domain_max[0] - domain_min[0],
                domain_max[1] - domain_min[1],
                fill=False,
                linewidth=1.2,
                edgecolor="0.35",
                label="domain",
            )
        )
    plan.plot([source_start[0], source_end[0]], [source_start[1], source_end[1]], color="#1f77b4", lw=3, label="source")
    plan.scatter([receiver[0]], [receiver[1]], marker="v", s=70, color="#d62728", label="receiver", zorder=4)
    if leakage_points.size:
        plan.plot(
            leakage_points[:, 0],
            leakage_points[:, 1],
            color="#2ca02c",
            marker="o",
            lw=2,
            label="leakage channel",
        )
    plan.set_title("Plan view")
    plan.set_xlabel("x (m)")
    plan.set_ylabel("y (m)")
    plan.set_aspect("equal", adjustable="box")
    plan.grid(True, alpha=0.25)
    plan.legend(loc="best", fontsize=8)

    depth.plot([source_start[0], source_end[0]], [source_start[2], source_end[2]], color="#1f77b4", lw=3)
    depth.scatter([receiver[0]], [receiver[2]], marker="v", s=70, color="#d62728", zorder=4)
    if leakage_points.size:
        depth.plot(leakage_points[:, 0], leakage_points[:, 2], color="#2ca02c", marker="o", lw=2)
    if np.all(np.isfinite(domain_min[[0, 2]])) and np.all(np.isfinite(domain_max[[0, 2]])):
        depth.add_patch(
            Rectangle(
                (domain_min[0], domain_min[2]),
                domain_max[0] - domain_min[0],
                domain_max[2] - domain_min[2],
                fill=False,
                linewidth=1.2,
                edgecolor="0.35",
            )
        )
    depth.axhline(0.0, color="0.3", lw=1, ls="--")
    depth.set_title("Depth view")
    depth.set_xlabel("x (m)")
    depth.set_ylabel("z (m)")
    depth.grid(True, alpha=0.25)

    fig.suptitle(f"{case_spec.get('case_type', 'model')} corrected model schematic")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return info
