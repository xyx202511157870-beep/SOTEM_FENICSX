"""Postprocess HDF5 receiver data with source-history corrections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .config import build_simulation
from .hj import HJMagneticSimulation, hj_dc_initial_electric_field
from .io import save_result_hdf5
from .simulation import ReceiverDataResult, TDEMIPSimulation


@dataclass(frozen=True)
class SourceHistoryPostprocessResult:
    """Receiver-only data after applying a source-history magnetic correction."""

    times: np.ndarray
    data: np.ndarray
    correction_delta: np.ndarray
    magnetic_receiver_indices: tuple[int, ...]


def load_config_from_result(path: str | Path) -> dict[str, Any]:
    """Read the YAML configuration stored in a result HDF5 file."""

    with h5py.File(path, "r") as h5:
        if "config_yaml" not in h5.attrs:
            raise ValueError("input result does not contain a config_yaml attribute")
        config = yaml.safe_load(h5.attrs["config_yaml"])
    if not isinstance(config, dict):
        raise ValueError("stored config_yaml must decode to a mapping")
    return config


def postprocess_source_history_receiver_data(
    input_path: str | Path,
    config: dict[str, Any],
) -> SourceHistoryPostprocessResult:
    """Apply configured source-history receiver correction to HDF5 receiver data."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r") as h5:
        if "times" not in h5 or "data" not in h5:
            raise ValueError("input result must contain 'times' and 'data' datasets")
        times = np.asarray(h5["times"][:], dtype=float)
        base_data = np.asarray(h5["data"][:], dtype=float)

    simulation = build_simulation(config)
    if getattr(simulation, "magnetic_recovery_source_history", None) is None:
        raise ValueError("config must define magnetic_recovery_source_history")
    if times.shape != simulation.times.shape or not np.allclose(
        times,
        simulation.times,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError("input time grid does not match the supplied configuration")
    if base_data.shape != (times.size, len(simulation.receivers)):
        raise ValueError("input data shape does not match the supplied receiver configuration")

    magnetic_indices = tuple(
        index
        for index, receiver in enumerate(simulation.receivers)
        if receiver.uses_magnetic_field_vector
    )
    delta = np.zeros_like(base_data)
    if magnetic_indices:
        locations = np.asarray(
            [simulation.receivers[index].location for index in magnetic_indices],
            dtype=float,
        )
        initial_memories = _initial_memories_for_source_history(simulation)
        for time_index, time in enumerate(times):
            field = simulation._source_history_magnetic_field(  # noqa: SLF001
                locations,
                float(time),
                initial_memories,
            )
            for local_index, receiver_index in enumerate(magnetic_indices):
                receiver = simulation.receivers[receiver_index]
                delta[time_index, receiver_index] = receiver.sample_magnetic_field_vector(
                    field[local_index],
                    simulation.mu,
                )

    return SourceHistoryPostprocessResult(
        times=times,
        data=base_data + delta,
        correction_delta=delta,
        magnetic_receiver_indices=magnetic_indices,
    )


def save_postprocessed_result_hdf5(
    output_path: str | Path,
    result: SourceHistoryPostprocessResult,
    config: dict[str, Any],
    *,
    input_path: str | Path,
) -> None:
    """Save a postprocessed receiver-only result with provenance metadata."""

    receiver_result = ReceiverDataResult(
        times=result.times,
        data=result.data,
        memories=[],
    )
    save_result_hdf5(output_path, receiver_result, config)
    with h5py.File(output_path, "a") as h5:
        h5.create_dataset("source_history_correction_delta", data=result.correction_delta)
        h5.attrs["source_history_postprocessed"] = True
        h5.attrs["source_history_postprocess_input"] = str(Path(input_path))
        h5.attrs["source_history_postprocess_magnetic_receiver_indices"] = ",".join(
            str(index) for index in result.magnetic_receiver_indices
        )


def _initial_memories_for_source_history(
    simulation: TDEMIPSimulation | HJMagneticSimulation,
) -> list[np.ndarray]:
    if isinstance(simulation, HJMagneticSimulation):
        if simulation.initial_e is not None:
            initial_e = np.asarray(simulation.initial_e, dtype=float)
        else:
            initial_e = hj_dc_initial_electric_field(
                simulation.mesh,
                simulation.initial_ip_model,
                simulation.sources,
            )
        return simulation._initial_memories(initial_e)  # noqa: SLF001

    initial_e = simulation.initial_electric_field()
    return simulation.ip_model.initial_memory(simulation.mesh.n_edges, initial_e)
