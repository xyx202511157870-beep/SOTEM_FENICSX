from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import mu_0


_REQUIRED_COMPONENTS = ("Ex", "Ey", "Hz", "dBzdt")
_CANONICAL_COLUMNS = (
    "Ex_V_per_m",
    "Ey_V_per_m",
    "Hz_A_per_m",
    "Bz_T",
    "dBzdt_T_per_s",
)
_CANONICAL_INDEX = {name: index for index, name in enumerate(_CANONICAL_COLUMNS)}


def _immutable_array(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=float)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _real_array(values, *, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric {ndim}-D array") from exc
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}-D array")
    return np.array(array, dtype=float, copy=True)


def _validated_times(times) -> np.ndarray:
    normalized = _real_array(times, name="times", ndim=1)
    if normalized.size == 0:
        raise ValueError("times must be nonempty")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("times must contain only finite values")
    if np.any(normalized <= 0.0):
        raise ValueError("times must contain only positive values")
    if np.any(np.diff(normalized) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return normalized


def _validated_values(values, *, row_count: int, column_count: int) -> np.ndarray:
    normalized = _real_array(values, name="values", ndim=2)
    if normalized.shape[0] != row_count:
        raise ValueError("values row count must equal times length")
    if normalized.shape[1] != column_count:
        raise ValueError(f"values must contain exactly {column_count} columns")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("values must contain only finite values")
    return normalized


def _component_names(components) -> tuple[str, ...]:
    message = "components must be a one-dimensional iterable of strings"
    if isinstance(components, (str, bytes)):
        raise ValueError(message)
    try:
        names = tuple(components)
    except TypeError as exc:
        raise ValueError(message) from exc
    if any(not isinstance(name, str) for name in names):
        raise ValueError(message)

    for required_name in _REQUIRED_COMPONENTS:
        if names.count(required_name) > 1:
            raise ValueError(f"duplicate canonical component: {required_name}")
    missing = [name for name in _REQUIRED_COMPONENTS if name not in names]
    if missing:
        raise ValueError(f"missing canonical components: {', '.join(missing)}")
    return names


@dataclass(frozen=True, eq=False)
class CanonicalResponse:
    times: np.ndarray
    values: np.ndarray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.columns) is not tuple or self.columns != _CANONICAL_COLUMNS:
            raise ValueError("columns must equal the canonical response columns")
        times = _validated_times(self.times)
        values = _validated_values(
            self.values,
            row_count=times.size,
            column_count=len(_CANONICAL_COLUMNS),
        )
        object.__setattr__(self, "times", _immutable_array(times))
        object.__setattr__(self, "values", _immutable_array(values))

    @property
    def ey_to_ex_peak_ratio(self) -> float:
        """Return the absolute peak ratio, defining zero/zero as no violation."""
        ex_peak = float(
            np.max(np.abs(self.values[:, _CANONICAL_INDEX["Ex_V_per_m"]]))
        )
        ey_peak = float(
            np.max(np.abs(self.values[:, _CANONICAL_INDEX["Ey_V_per_m"]]))
        )
        if ex_peak == 0.0:
            return 0.0 if ey_peak == 0.0 else float("inf")
        return ey_peak / ex_peak

    def write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("time_obs_s", *self.columns))
            rows = np.column_stack((self.times, self.values))
            writer.writerows(
                tuple(format(float(value), ".17g") for value in row) for row in rows
            )


def canonical_response(times, values, components) -> CanonicalResponse:
    names = _component_names(components)
    normalized_times = _validated_times(times)
    source = _validated_values(
        values,
        row_count=normalized_times.size,
        column_count=len(names),
    )

    component_indices = {name: index for index, name in enumerate(names)}
    hz = source[:, component_indices["Hz"]]
    canonical_values = np.column_stack(
        (
            source[:, component_indices["Ex"]],
            source[:, component_indices["Ey"]],
            hz,
            mu_0 * hz,
            source[:, component_indices["dBzdt"]],
        )
    )
    return CanonicalResponse(
        times=normalized_times,
        values=canonical_values,
        columns=_CANONICAL_COLUMNS,
    )
