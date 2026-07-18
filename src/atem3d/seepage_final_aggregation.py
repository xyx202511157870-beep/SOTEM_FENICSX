"""Final fail-closed aggregation for the approved open-solver model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .seepage_matrix_aggregation import OPEN3D_REQUIRED_GATES, build_open3d_summary


FINAL_REQUIRED_GATES = OPEN3D_REQUIRED_GATES


def build_final_summary(
    output_root: str | Path, *, require_pass: bool = False
) -> dict[str, Any]:
    """Return the formal SimPEG/FEniCSx/empymod verification summary."""

    return build_open3d_summary(output_root, require_pass=require_pass)


__all__ = ["FINAL_REQUIRED_GATES", "build_final_summary"]

