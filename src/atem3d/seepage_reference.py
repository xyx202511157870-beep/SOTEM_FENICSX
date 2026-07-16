"""Background-reference adapters used by seepage-channel benchmarks.

This module deliberately contains no crossline mirroring.  SimPEG results are
normalized from their explicit five-receiver HDF5 payload, and empymod is used
only for the layered background reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from .seepage_channel_model import MODEL, SeepageChannelBenchmark


COMPONENTS = ("Ex", "dBzdt", "Hz")
MU0 = float(mu_0)


def _resample_log_time(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    source_time_array = np.asarray(source_times, dtype=float).reshape(-1)
    target_time_array = np.asarray(target_times, dtype=float).reshape(-1)
    value_array = np.asarray(source_values, dtype=float)
    if value_array.shape[0] != source_time_array.size:
        raise ValueError("source_values first axis must match source_times")
    if np.any(source_time_array <= 0.0) or np.any(target_time_array <= 0.0):
        raise ValueError("log-time interpolation requires positive times")
    flattened = value_array.reshape(source_time_array.size, -1)
    output = np.empty((target_time_array.size, flattened.shape[1]), dtype=float)
    log_source = np.log10(source_time_array)
    log_target = np.log10(target_time_array)
    for column in range(flattened.shape[1]):
        output[:, column] = np.interp(log_target, log_source, flattened[:, column])
    return output.reshape((target_time_array.size, *value_array.shape[1:]))


def load_simpeg_values(
    path: str | Path,
    *,
    target_times: np.ndarray | None = None,
    model: SeepageChannelBenchmark = MODEL,
) -> np.ndarray:
    """Load and contract-check an explicit five-receiver SimPEG result."""

    import h5py  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    with h5py.File(Path(path), "r") as h5:
        source_times = np.asarray(h5["times"][:], dtype=float)
        data = np.asarray(h5["data"][:], dtype=float)
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))
    if config.get("coordinate_system") != "z_up":
        raise ValueError("SimPEG result must use the internal z_up coordinate system")
    expected_receivers = [
        {
            "location": [float(x), float(y), float(-z)],
            "component": component,
        }
        for x, y, z in model.receiver_locations
        for component in COMPONENTS
    ]
    if config.get("receivers") != expected_receivers:
        raise ValueError("SimPEG receiver coordinates or component ordering do not match")
    if data.shape != (source_times.size, len(expected_receivers)):
        raise ValueError(
            f"SimPEG data must have shape ({source_times.size}, {len(expected_receivers)}); "
            f"got {data.shape}"
        )
    values = data.reshape(source_times.size, len(model.receiver_locations), len(COMPONENTS))
    positive = source_times > 0.0
    output_times = model.times if target_times is None else np.asarray(target_times, dtype=float)
    return _resample_log_time(source_times[positive], values[positive], output_times).transpose(1, 0, 2)


def run_empymod_reference(
    times: np.ndarray | None = None,
    *,
    backend=None,
    srcpts: int = 129,
    model: SeepageChannelBenchmark = MODEL,
) -> np.ndarray:
    """Run the finite-wire empymod background in physical z-down coordinates."""

    if backend is None:
        import empymod as backend  # noqa: PLC0415
    time_array = model.times if times is None else np.asarray(times, dtype=float).reshape(-1)
    start, end = model.source_endpoints
    source = [start[0], end[0], start[1], end[1], start[2], end[2]]
    mappings = {
        "Ex": (False, -1, 0.0, 0.0, 1.0),
        "dBzdt": (True, 0, 0.0, 90.0, -MU0),
        "Hz": (True, -1, 0.0, 90.0, 1.0),
    }
    values = np.empty((len(model.receiver_locations), time_array.size, len(COMPONENTS)), dtype=float)
    for receiver_index, (receiver_x, receiver_y, receiver_z) in enumerate(model.receiver_locations):
        for component_index, component in enumerate(COMPONENTS):
            mrec, signal, azimuth, dip, factor = mappings[component]
            response = backend.bipole(
                src=source,
                rec=[float(receiver_x), float(receiver_y), float(receiver_z), azimuth, dip],
                depth=[0.0],
                res=[1.0 / model.air_conductivity, 1.0 / model.background_conductivity],
                freqtime=time_array,
                signal=signal,
                strength=model.source_current_a,
                mrec=mrec,
                srcpts=int(srcpts),
                verb=0,
            )
            values[receiver_index, :, component_index] = np.asarray(response, dtype=float).reshape(-1) * factor
    return values


__all__ = ["load_simpeg_values", "run_empymod_reference"]
