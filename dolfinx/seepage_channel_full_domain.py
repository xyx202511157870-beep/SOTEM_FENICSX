#!/usr/bin/env python3
"""Full-domain FEniCSx adapter for the seepage-channel benchmark.

The adapter intentionally contains no half-domain or receiver-mirroring path.
Pure geometry/material and receiver-set helpers are added here as their
test-driven implementation tasks are completed.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ReceiverEvaluationConfig:
    receiver: tuple[float, float, float]
    receiver_id: str
    provenance: str = "explicit_full_domain"


def receiver_configs(
    locations: Iterable[tuple[float, float, float]],
) -> tuple[ReceiverEvaluationConfig, ...]:
    configs = tuple(
        ReceiverEvaluationConfig(
            receiver=tuple(float(value) for value in location),  # type: ignore[arg-type]
            receiver_id=f"Rx{index + 1}",
        )
        for index, location in enumerate(locations)
    )
    if any(len(config.receiver) != 3 for config in configs):
        raise ValueError("each receiver location must contain x, y, z")
    if len({config.receiver for config in configs}) != len(configs):
        raise ValueError("receiver locations must be unique")
    return configs


def box_mask(
    centers: ArrayLike,
    volumes: ArrayLike,
    bounds: ArrayLike,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    cell_centers = np.asarray(centers, dtype=float)
    cell_volumes = np.asarray(volumes, dtype=float)
    box_bounds = np.asarray(bounds, dtype=float)
    if cell_centers.ndim != 2 or cell_centers.shape[1] != 3:
        raise ValueError("centers must have shape (n, 3)")
    if cell_volumes.shape != (cell_centers.shape[0],):
        raise ValueError("volumes must contain one value per center")
    if box_bounds.shape != (3, 2) or np.any(box_bounds[:, 1] <= box_bounds[:, 0]):
        raise ValueError("bounds must have shape (3, 2) with upper > lower")
    mask = np.logical_and.reduce(
        [
            (cell_centers[:, axis] >= box_bounds[axis, 0])
            & (cell_centers[:, axis] <= box_bounds[axis, 1])
            for axis in range(3)
        ]
    )
    return mask, {
        "local_cell_count": int(np.count_nonzero(mask)),
        "local_discrete_volume_m3": float(np.sum(cell_volumes[mask])),
    }


def evaluate_receiver_set(
    electric_field: Any,
    dbdt: Any,
    mesh: Any,
    config: Any,
    *,
    evaluator: Any,
) -> list[dict[str, Any]]:
    raw_locations = getattr(config, "receiver_locations", ()) or (config.receiver,)
    records: list[dict[str, Any]] = []
    for receiver in receiver_configs(raw_locations):
        local_config = copy.copy(config)
        local_config.receiver = receiver.receiver
        local_config.receiver_locations = ()
        record = dict(evaluator(electric_field, dbdt, mesh, local_config))
        record.update(
            {
                "receiver_id": receiver.receiver_id,
                "receiver_x_m": receiver.receiver[0],
                "receiver_y_m": receiver.receiver[1],
                "receiver_z_m": receiver.receiver[2],
                "provenance": receiver.provenance,
            }
        )
        records.append(record)
    return records


def records_to_array(
    records: Iterable[dict[str, Any]],
    components: Iterable[str],
) -> NDArray[np.float64]:
    component_names = tuple(components)
    return np.asarray(
        [
            [float(record[component]) for component in component_names]
            for record in records
        ],
        dtype=float,
    )


def write_predictions_5rx(
    path: str | Path,
    *,
    times: ArrayLike,
    receiver_locations: ArrayLike,
    data: ArrayLike,
    components: Iterable[str],
) -> Path:
    output_path = Path(path)
    time_values = np.asarray(times, dtype=float).reshape(-1)
    locations = np.asarray(receiver_locations, dtype=float)
    component_names = tuple(str(component) for component in components)
    values = np.asarray(data, dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (n_rx, 3)")
    expected_shape = (locations.shape[0], time_values.size, len(component_names))
    if values.shape != expected_shape:
        raise ValueError(f"data must have shape {expected_shape}; got {values.shape}")
    required = ("Ex", "dBzdt", "Hz")
    if set(component_names) != set(required):
        raise ValueError("components must contain exactly Ex, dBzdt, and Hz")
    if not np.all(np.isfinite(values)):
        raise ValueError("all receiver predictions must be finite")
    indices = {component: component_names.index(component) for component in required}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "receiver_id",
                "receiver_x_m",
                "receiver_y_m",
                "receiver_z_m",
                "time_obs",
                "Ex",
                "dBzdt",
                "Hz",
                "provenance",
            ]
        )
        for receiver_index, location in enumerate(locations):
            for time_index, time_obs in enumerate(time_values):
                writer.writerow(
                    [
                        f"Rx{receiver_index + 1}",
                        *location.tolist(),
                        float(time_obs),
                        float(values[receiver_index, time_index, indices["Ex"]]),
                        float(values[receiver_index, time_index, indices["dBzdt"]]),
                        float(values[receiver_index, time_index, indices["Hz"]]),
                        "explicit_full_domain",
                    ]
                )
    return output_path


def write_receiver_set_npz(
    path: str | Path,
    *,
    times: ArrayLike,
    receiver_locations: ArrayLike,
    components: ArrayLike,
    data: ArrayLike,
    receiver_provenance: ArrayLike,
    material_audit: dict[str, Any],
) -> Path:
    output = Path(path)
    time_values = np.asarray(times, dtype=float).reshape(-1)
    locations = np.asarray(receiver_locations, dtype=float)
    component_names = np.asarray(components).astype(str).reshape(-1)
    values = np.asarray(data, dtype=float)
    provenance = np.asarray(receiver_provenance).astype(str).reshape(-1)
    expected_shape = (locations.shape[0], time_values.size, component_names.size)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (n_rx, 3)")
    if values.shape != expected_shape:
        raise ValueError(f"data must have shape {expected_shape}; got {values.shape}")
    if provenance.shape != (locations.shape[0],):
        raise ValueError("receiver_provenance must contain one value per receiver")
    if not np.all(provenance == "explicit_full_domain"):
        raise ValueError("all receiver provenance must be explicit_full_domain")
    if not np.all(np.isfinite(values)):
        raise ValueError("receiver-set data must be finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=time_values,
        receiver_locations=locations,
        components=component_names,
        data=values,
        receiver_provenance=provenance,
        material_audit_json=np.asarray(json.dumps(material_audit, sort_keys=True)),
    )
    return output


__all__ = [
    "ReceiverEvaluationConfig",
    "box_mask",
    "evaluate_receiver_set",
    "receiver_configs",
    "records_to_array",
    "write_predictions_5rx",
    "write_receiver_set_npz",
]
