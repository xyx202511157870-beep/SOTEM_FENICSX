"""Pure topology utilities for auditable full-domain y-symmetric meshes."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_cells(values, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width})")
    return array


def mirror_half_topology(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    tetra_tags: np.ndarray,
    triangles: np.ndarray,
    triangle_tags: np.ndarray,
    lines: np.ndarray,
    line_tags: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Mirror a positive-y half mesh while reusing entities on y=0."""

    xyz = np.asarray(points, dtype=float)
    tetra = _as_cells(tetrahedra, 4, "tetrahedra")
    tri = _as_cells(triangles, 3, "triangles")
    line = _as_cells(lines, 2, "lines")
    tetra_tag_values = np.asarray(tetra_tags, dtype=np.int32).reshape(-1)
    tri_tag_values = np.asarray(triangle_tags, dtype=np.int32).reshape(-1)
    line_tag_values = np.asarray(line_tags, dtype=np.int32).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("points must be finite with shape (n, 3)")
    if np.any(xyz[:, 1] < -float(tolerance)):
        raise ValueError("input must be a positive-y half mesh")
    if tetra_tag_values.size != tetra.shape[0]:
        raise ValueError("tetra_tags must match tetrahedra")
    if tri_tag_values.size != tri.shape[0] or line_tag_values.size != line.shape[0]:
        raise ValueError("facet and line tags must match their entities")

    mirrored_index = np.arange(xyz.shape[0], dtype=np.int64)
    mirrored_points = []
    for index, point in enumerate(xyz):
        if point[1] > tolerance:
            mirrored_index[index] = xyz.shape[0] + len(mirrored_points)
            reflected = point.copy()
            reflected[1] *= -1.0
            mirrored_points.append(reflected)
    full_points = np.vstack((xyz, np.asarray(mirrored_points, dtype=float))) if mirrored_points else xyz.copy()

    mirrored_tetra = mirrored_index[tetra].copy()
    mirrored_tetra[:, [1, 2]] = mirrored_tetra[:, [2, 1]]
    full_tetra = np.vstack((tetra, mirrored_tetra))
    full_tetra_tags = np.concatenate((tetra_tag_values, tetra_tag_values))

    triangle_blocks = []
    triangle_tag_blocks = []
    for entity, tag in zip(tri, tri_tag_values):
        if np.all(np.abs(xyz[entity, 1]) <= tolerance):
            continue
        triangle_blocks.extend((entity, mirrored_index[entity]))
        triangle_tag_blocks.extend((tag, tag))

    line_blocks = []
    line_tag_blocks = []
    for entity, tag in zip(line, line_tag_values):
        line_blocks.append(entity)
        line_tag_blocks.append(tag)
        if not np.all(np.abs(xyz[entity, 1]) <= tolerance):
            line_blocks.append(mirrored_index[entity])
            line_tag_blocks.append(tag)

    return {
        "points": full_points,
        "tetrahedra": full_tetra,
        "tetra_tags": full_tetra_tags,
        "triangles": np.asarray(triangle_blocks, dtype=np.int64).reshape(-1, 3),
        "triangle_tags": np.asarray(triangle_tag_blocks, dtype=np.int32),
        "lines": np.asarray(line_blocks, dtype=np.int64).reshape(-1, 2),
        "line_tags": np.asarray(line_tag_blocks, dtype=np.int32),
    }


def _quantized(point: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    scale = max(float(tolerance), np.finfo(float).eps)
    return tuple(np.rint(np.asarray(point, dtype=float) / scale).astype(np.int64))


def audit_y_reflection(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    tetra_tags: np.ndarray,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Fail-closed audit of node, cell, material-tag, and volume reflection."""

    xyz = np.asarray(points, dtype=float)
    tetra = _as_cells(tetrahedra, 4, "tetrahedra")
    tags = np.asarray(tetra_tags, dtype=np.int32).reshape(-1)
    if tags.size != tetra.shape[0]:
        raise ValueError("tetra_tags must match tetrahedra")
    node_map = {_quantized(point, tolerance): index for index, point in enumerate(xyz)}
    node_matches = 0
    maximum_distance = 0.0
    for point in xyz:
        reflected = point.copy()
        reflected[1] *= -1.0
        match = node_map.get(_quantized(reflected, tolerance))
        if match is not None:
            distance = float(np.linalg.norm(xyz[match] - reflected))
            maximum_distance = max(maximum_distance, distance)
            node_matches += int(distance <= tolerance)

    centroids = np.mean(xyz[tetra], axis=1)
    cell_geometry: dict[tuple[int, int, int], list[int]] = {}
    for index, centroid in enumerate(centroids):
        cell_geometry.setdefault(_quantized(centroid, tolerance), []).append(index)
    centroid_matches = 0
    tag_mismatch_count = 0
    for index, centroid in enumerate(centroids):
        reflected = centroid.copy()
        reflected[1] *= -1.0
        matches = cell_geometry.get(_quantized(reflected, tolerance), [])
        if matches:
            centroid_matches += 1
            if not any(tags[match] == tags[index] for match in matches):
                tag_mismatch_count += 1

    vertices = xyz[tetra]
    determinants = np.linalg.det(
        np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
    )
    volumes = np.abs(determinants) / 6.0
    positive_volume = float(np.sum(volumes[centroids[:, 1] > tolerance]))
    negative_volume = float(np.sum(volumes[centroids[:, 1] < -tolerance]))
    node_fraction = float(node_matches / xyz.shape[0]) if xyz.shape[0] else 0.0
    cell_fraction = float(centroid_matches / tetra.shape[0]) if tetra.shape[0] else 0.0
    volume_scale = max(positive_volume, negative_volume, np.finfo(float).eps)
    volume_residual = abs(positive_volume - negative_volume) / volume_scale
    passed = (
        node_fraction == 1.0
        and cell_fraction == 1.0
        and maximum_distance <= tolerance
        and tag_mismatch_count == 0
        and volume_residual <= 1.0e-12
    )
    return {
        "exact_node_pair_fraction": node_fraction,
        "exact_centroid_pair_fraction": cell_fraction,
        "maximum_reflection_distance": maximum_distance,
        "tag_mismatch_count": int(tag_mismatch_count),
        "original_volume": positive_volume,
        "mirrored_volume": negative_volume,
        "relative_volume_residual": float(volume_residual),
        "passed": bool(passed),
    }


__all__ = ["audit_y_reflection", "mirror_half_topology"]
