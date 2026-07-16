"""Shared contract for the 100 m wire, five-receiver four-way benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.constants import mu_0


COMPONENTS = ("Ex", "dBzdt", "Hz")
MU0 = float(mu_0)
_COMSOL_TIME_RE = re.compile(r"@\s*t=([0-9.+\-Ee]+)")


@dataclass(frozen=True)
class FourWayModel:
    source_start: tuple[float, float, float] = (-50.0, 0.0, 0.1)
    source_end: tuple[float, float, float] = (50.0, 0.0, 0.1)
    receiver_y: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    receiver_z: float = -0.1
    earth_resistivity: float = 100.0
    air_resistivity: float = 1.0e8
    current: float = 1.0
    memory_limit_gb: float = 30.0
    time_min: float = 1.0e-5
    time_max: float = 1.0e-2
    time_count: int = 31

    @property
    def times(self) -> np.ndarray:
        return np.logspace(np.log10(self.time_min), np.log10(self.time_max), self.time_count)

    @property
    def receiver_locations(self) -> np.ndarray:
        return np.asarray([(0.0, y, self.receiver_z) for y in self.receiver_y], dtype=float)


MODEL = FourWayModel()


def build_long_rows(method: str, times: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    """Return canonical signed component rows for one solver."""

    time_array = np.asarray(times, dtype=float).reshape(-1)
    value_array = np.asarray(values, dtype=float)
    expected_shape = (len(MODEL.receiver_y), time_array.size, len(COMPONENTS))
    if value_array.shape != expected_shape:
        raise ValueError(f"expected values shape {expected_shape}, got {value_array.shape}")

    rows: list[dict[str, Any]] = []
    for receiver_index, receiver_y in enumerate(MODEL.receiver_y):
        for time_index, time_s in enumerate(time_array):
            component_values = value_array[receiver_index, time_index]
            rows.append(
                {
                    "method": str(method),
                    "receiver": f"Rx{receiver_index + 1}",
                    "receiver_y_m": float(receiver_y),
                    "receiver_z_m": float(MODEL.receiver_z),
                    "time_s": float(time_s),
                    **{
                        component: float(component_values[component_index])
                        for component_index, component in enumerate(COMPONENTS)
                    },
                }
            )
    return rows


def ordinary_relative_error(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return pointwise ordinary relative error, with zero references undefined."""

    predicted_array = np.asarray(predicted, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if predicted_array.shape != reference_array.shape:
        raise ValueError("predicted and reference arrays must have the same shape")
    output = np.full(reference_array.shape, np.nan, dtype=float)
    nonzero = np.isfinite(reference_array) & (reference_array != 0.0)
    output[nonzero] = (
        np.abs(predicted_array[nonzero] - reference_array[nonzero])
        / np.abs(reference_array[nonzero])
    )
    return output


def resample_log_time(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate signed values along log10 time."""

    source_time_array = np.asarray(source_times, dtype=float).reshape(-1)
    target_time_array = np.asarray(target_times, dtype=float).reshape(-1)
    value_array = np.asarray(source_values, dtype=float)
    if value_array.ndim == 0 or value_array.shape[0] != source_time_array.size:
        raise ValueError("source_values first axis must match source_times")
    if np.any(source_time_array <= 0.0) or np.any(target_time_array <= 0.0):
        raise ValueError("source and target times must be positive")
    if np.any(np.diff(source_time_array) <= 0.0):
        raise ValueError("source_times must be strictly increasing")
    if target_time_array.size:
        epsilon = np.finfo(float).eps
        lower_tolerance = 64.0 * epsilon * max(
            abs(source_time_array[0]), abs(target_time_array[0])
        )
        upper_tolerance = 64.0 * epsilon * max(
            abs(source_time_array[-1]), abs(target_time_array[-1])
        )
        if (
            target_time_array[0] < source_time_array[0] - lower_tolerance
            or target_time_array[-1] > source_time_array[-1] + upper_tolerance
        ):
            raise ValueError("target_times must lie within the source time range")
        target_time_array = np.clip(
            target_time_array,
            source_time_array[0],
            source_time_array[-1],
        )

    flat_values = value_array.reshape(source_time_array.size, -1)
    output = np.empty((target_time_array.size, flat_values.shape[1]), dtype=float)
    source_log_time = np.log10(source_time_array)
    target_log_time = np.log10(target_time_array)
    for column in range(flat_values.shape[1]):
        output[:, column] = np.interp(
            target_log_time,
            source_log_time,
            flat_values[:, column],
        )
    return output.reshape((target_time_array.size, *value_array.shape[1:]))


def mirror_crossline_values(positive_values: np.ndarray) -> np.ndarray:
    """Expand y=0, +10, +20 m responses to the five-point symmetric line."""

    values = np.asarray(positive_values, dtype=float)
    expected_shape = (3, values.shape[1] if values.ndim >= 2 else 0, len(COMPONENTS))
    if values.ndim != 3 or values.shape != expected_shape:
        raise ValueError("positive_values must have shape (3, n_times, 3)")

    output = np.empty((5, values.shape[1], len(COMPONENTS)), dtype=float)
    output[2:] = values
    output[1] = values[1]
    output[0] = values[2]
    output[:2, :, 1:] *= -1.0
    output[2, :, 1:] = 0.0
    return output


def load_simpeg_values(
    path: str | Path,
    *,
    target_times: np.ndarray | None = None,
) -> np.ndarray:
    """Load and contract-check the five-receiver SimPEG HDF5 result."""

    import h5py  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    with h5py.File(Path(path), "r") as h5:
        source_times = np.asarray(h5["times"][:], dtype=float)
        data = np.asarray(h5["data"][:], dtype=float)
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))

    if config.get("coordinate_system") != "z_up":
        raise ValueError("SimPEG result must use the internal z_up coordinate system")
    expected_receivers = [
        {"location": [0.0, receiver_y, 0.1], "component": component}
        for receiver_y in MODEL.receiver_y
        for component in COMPONENTS
    ]
    if config.get("receivers") != expected_receivers:
        raise ValueError("SimPEG receiver coordinates or component ordering do not match the contract")
    if data.shape != (source_times.size, len(expected_receivers)):
        raise ValueError(
            "SimPEG data must have shape "
            f"({source_times.size}, {len(expected_receivers)}); got {data.shape}"
        )

    values = data.reshape(source_times.size, len(MODEL.receiver_y), len(COMPONENTS))
    positive = source_times > 0.0
    source_times = source_times[positive]
    values = values[positive]
    output_times = MODEL.times if target_times is None else np.asarray(target_times, dtype=float)
    resampled = resample_log_time(source_times, values, output_times)
    return resampled.transpose(1, 0, 2)


def _load_fenicsx_prediction_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"FEniCSx prediction file is empty: {path}")
    required = {"time_obs", *COMPONENTS}
    if not required.issubset(rows[0]):
        raise ValueError(f"FEniCSx prediction file lacks required columns: {path}")
    times = np.asarray([float(row["time_obs"]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[component]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    return times, values


def load_fenicsx_crossline_values(
    positive_prediction_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    target_times: np.ndarray | None = None,
) -> np.ndarray:
    """Load y=0,+10,+20 m FEniCSx predictions and mirror the crossline."""

    if len(positive_prediction_paths) != 3:
        raise ValueError("expected FEniCSx prediction paths for y=0, +10, and +20 m")
    output_times = MODEL.times if target_times is None else np.asarray(target_times, dtype=float)
    positive_values = []
    for path in positive_prediction_paths:
        source_times, source_values = _load_fenicsx_prediction_csv(path)
        positive_values.append(resample_log_time(source_times, source_values, output_times))
    return mirror_crossline_values(np.asarray(positive_values, dtype=float))


def _comsol_component(header: str) -> str | None:
    normalized = header.strip()
    if normalized.startswith("mef.Ex"):
        return "Ex"
    if normalized.startswith("d(mef.Bz;") or normalized.startswith("d(mef.Bz,"):
        return "dBzdt"
    if normalized.startswith("mef.Bz/mu0_const"):
        return "Hz"
    return None


def load_comsol_wide_values(
    path: str | Path,
    *,
    target_times: np.ndarray | None = None,
) -> np.ndarray:
    """Load the five-point COMSOL wide CSV export in canonical component order."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("% x,y,z,")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"COMSOL CSV lacks an x,y,z result header: {path}")
    header_line = lines[header_index].lstrip()[1:].strip()
    headers = next(csv.reader([header_line]))
    column_map: list[tuple[int, float, str]] = []
    for column_index, header in enumerate(headers):
        component = _comsol_component(header)
        match = _COMSOL_TIME_RE.search(header)
        if component is not None and match is not None:
            time_s = float(match.group(1))
            if time_s > 0.0:
                column_map.append((column_index, time_s, component))
    source_times = np.asarray(sorted({entry[1] for entry in column_map}), dtype=float)
    if source_times.size == 0:
        raise ValueError(f"COMSOL CSV lacks the required result expressions: {path}")
    time_indices = {float(time_s): index for index, time_s in enumerate(source_times)}
    component_indices = {component: index for index, component in enumerate(COMPONENTS)}
    values = np.full(
        (len(MODEL.receiver_y), source_times.size, len(COMPONENTS)),
        np.nan,
        dtype=float,
    )

    for line in lines[header_index + 1 :]:
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        fields = next(csv.reader([line]))
        if len(fields) < 3:
            continue
        x, receiver_y, receiver_z = (float(fields[index]) for index in range(3))
        if not np.isclose(x, 0.0, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"COMSOL receiver x coordinate does not match the contract: {x}")
        if not np.isclose(receiver_z, MODEL.receiver_z, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"COMSOL receiver z coordinate does not match the contract: {receiver_z}"
            )
        receiver_matches = np.flatnonzero(
            np.isclose(np.asarray(MODEL.receiver_y), receiver_y, rtol=0.0, atol=1.0e-6)
        )
        if receiver_matches.size != 1:
            raise ValueError(f"unexpected COMSOL receiver y coordinate: {receiver_y}")
        receiver_index = int(receiver_matches[0])
        for column_index, time_s, component in column_map:
            values[
                receiver_index,
                time_indices[time_s],
                component_indices[component],
            ] = float(fields[column_index])

    if not np.all(np.isfinite(values)):
        raise ValueError("COMSOL CSV does not contain every receiver, time, and component")
    output_times = MODEL.times if target_times is None else np.asarray(target_times, dtype=float)
    resampled = resample_log_time(source_times, values.transpose(1, 0, 2), output_times)
    return resampled.transpose(1, 0, 2)


def run_empymod_reference(
    times: np.ndarray | None = None,
    *,
    backend=None,
    srcpts: int = 129,
) -> np.ndarray:
    """Run the finite-wire empymod reference in physical z-positive-down coordinates."""

    if backend is None:
        import empymod as backend  # noqa: PLC0415

    time_array = MODEL.times if times is None else np.asarray(times, dtype=float).reshape(-1)
    source = [
        MODEL.source_start[0],
        MODEL.source_end[0],
        MODEL.source_start[1],
        MODEL.source_end[1],
        MODEL.source_start[2],
        MODEL.source_end[2],
    ]
    mappings = {
        "Ex": (False, -1, 0.0, 0.0, 1.0),
        "dBzdt": (True, 0, 0.0, 90.0, -MU0),
        "Hz": (True, -1, 0.0, 90.0, 1.0),
    }
    values = np.empty((len(MODEL.receiver_y), time_array.size, len(COMPONENTS)), dtype=float)
    for receiver_index, receiver_y in enumerate(MODEL.receiver_y):
        for component_index, component in enumerate(COMPONENTS):
            mrec, signal, azimuth, dip, factor = mappings[component]
            response = backend.bipole(
                src=source,
                rec=[0.0, float(receiver_y), float(MODEL.receiver_z), azimuth, dip],
                depth=[0.0],
                res=[MODEL.air_resistivity, MODEL.earth_resistivity],
                freqtime=time_array,
                signal=signal,
                strength=MODEL.current,
                mrec=mrec,
                srcpts=int(srcpts),
                verb=0,
            )
            values[receiver_index, :, component_index] = (
                np.asarray(response, dtype=float).reshape(-1) * factor
            )
    return values
