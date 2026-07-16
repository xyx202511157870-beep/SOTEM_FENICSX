#!/usr/bin/env python3
"""Convert a positive-y Gmsh mesh into an audited full-domain mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "dolfinx"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from symmetric_full_domain_mesh import audit_y_reflection, mirror_half_topology

SOURCE_PHYSICAL_TAG = 301


def _cell_block(mesh: Any, cell_type: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    cells = []
    tags = []
    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    offset = 0
    for block in mesh.cells:
        if block.type != cell_type:
            continue
        data = np.asarray(block.data, dtype=np.int64).reshape(-1, width)
        cells.append(data)
        if physical is None:
            tags.append(np.zeros(data.shape[0], dtype=np.int32))
        else:
            tags.append(np.asarray(physical[offset : offset + data.shape[0]], dtype=np.int32))
        offset += data.shape[0]
    return (
        np.vstack(cells) if cells else np.empty((0, width), dtype=np.int64),
        np.concatenate(tags) if tags else np.empty(0, dtype=np.int32),
    )


def mirror_half_topology_from_meshio(mesh: Any) -> dict[str, Any]:
    tetrahedra, tetra_tags = _cell_block(mesh, "tetra", 4)
    triangles, triangle_tags = _cell_block(mesh, "triangle", 3)
    lines, line_tags = _cell_block(mesh, "line", 2)
    mirrored = mirror_half_topology(
        np.asarray(mesh.points[:, :3], dtype=float),
        tetrahedra,
        tetra_tags,
        triangles,
        triangle_tags,
        lines,
        line_tags,
    )
    plane_source = (
        mirrored["line_tags"] == SOURCE_PHYSICAL_TAG
    ) & np.all(
        np.abs(mirrored["points"][mirrored["lines"], 1]) <= 1.0e-10,
        axis=1,
    )
    mirrored["source_line_count"] = int(np.count_nonzero(plane_source))
    mirrored["field_data"] = dict(getattr(mesh, "field_data", {}))
    return mirrored


def write_meshio_full_domain(path: Path, mirrored: dict[str, Any]) -> Path:
    import meshio

    cell_blocks = []
    physical_blocks = []
    geometrical_blocks = []
    for cell_type, cells_key, tags_key in (
        ("tetra", "tetrahedra", "tetra_tags"),
        ("triangle", "triangles", "triangle_tags"),
        ("line", "lines", "line_tags"),
    ):
        cells = np.asarray(mirrored[cells_key])
        if cells.shape[0] == 0:
            continue
        tags = np.asarray(mirrored[tags_key], dtype=np.int32)
        cell_blocks.append((cell_type, cells))
        physical_blocks.append(tags)
        geometrical_blocks.append(tags.copy())
    output = meshio.Mesh(
        points=np.asarray(mirrored["points"], dtype=float),
        cells=cell_blocks,
        cell_data={
            "gmsh:physical": physical_blocks,
            "gmsh:geometrical": geometrical_blocks,
        },
        field_data=mirrored.get("field_data", {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(path, output, file_format="gmsh", binary=False)
    return path


def build_symmetric_mesh(half_path: Path, output_path: Path, audit_path: Path) -> Path:
    import meshio

    mesh = meshio.read(half_path)
    mirrored = mirror_half_topology_from_meshio(mesh)
    audit = audit_y_reflection(
        mirrored["points"], mirrored["tetrahedra"], mirrored["tetra_tags"], 1.0e-10
    )
    if not audit["passed"]:
        raise RuntimeError(f"symmetric full-domain mesh audit failed: {audit}")
    if mirrored["source_line_count"] != 1:
        raise RuntimeError("symmetric full-domain mesh must contain one source physical line")
    write_meshio_full_domain(output_path, mirrored)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                **audit,
                "source_line_count": mirrored["source_line_count"],
                "solver_domain": "full",
                "field_mirroring": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--half-mesh", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    build_symmetric_mesh(args.half_mesh, args.output_mesh, args.audit_json)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
