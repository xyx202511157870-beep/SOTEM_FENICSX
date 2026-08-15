"""Waveform-aware empymod references for the magnetic six-channel contract.

The empymod magnetic reference routines naturally provide ideal switch-off
responses.  A real transmitter commonly ramps from its on-current to zero over
several microseconds.  For a linear time-invariant system, the post-ramp
response is a weighted superposition of delayed ideal switch-off responses:

    y(t) = integral[-d i(tau)/d tau * y_stepoff(t - tau) d tau],

where ``tau <= 0`` is measured relative to the end of the ramp and ``i`` is the
source-current scale.  This module evaluates that integral with Gauss-Legendre
quadrature for Hx/Hy/Hz and dBxdt/dBydt/dBzdt simultaneously.

中文说明：观测时间统一以关断结束时刻为零点。对于 5 us 线性关断，波形
节点为 ``[-5e-6, 0]``，电流比例为 ``[1, 0]``。卷积不会改变源几何、
坐标系或单位，只把理想阶跃参考转换为实际关断波形参考。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .empymod_compare import EmpymodSurvey
from .empymod_magnetic6 import (
    MAGNETIC6_COMPONENTS,
    MAGNETIC6_UNITS,
    MagneticSixReferenceResult,
    run_empymod_magnetic6_reference,
)


@dataclass(frozen=True)
class PiecewiseLinearTurnOff:
    """Piecewise-linear current scale relative to the end of turn-off.

    ``times`` must be strictly increasing, non-positive, and end at zero.
    ``values`` are dimensionless source-current scales and must end at zero.
    The current is assumed constant at ``values[0]`` before ``times[0]`` and
    zero after the final node.
    """

    times: np.ndarray
    values: np.ndarray
    name: str = "piecewise_linear_turnoff"

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
            raise ValueError("turn-off times and values must be matching 1D arrays")
        if times.size < 2:
            raise ValueError("turn-off waveform requires at least two nodes")
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(values)):
            raise ValueError("turn-off waveform nodes must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("turn-off times must be strictly increasing")
        scale = max(1.0, float(np.max(np.abs(times))))
        tolerance = 128.0 * np.finfo(float).eps * scale
        if np.any(times > tolerance):
            raise ValueError("turn-off times must be relative to ramp end and <= 0")
        if abs(float(times[-1])) > tolerance:
            raise ValueError("the final turn-off time must be zero")
        value_scale = max(1.0, float(np.max(np.abs(values))))
        value_tolerance = 128.0 * np.finfo(float).eps * value_scale
        if abs(float(values[-1])) > value_tolerance:
            raise ValueError("the final turn-off current scale must be zero")
        if abs(float(values[0])) <= value_tolerance:
            raise ValueError("the pre-ramp current scale must be non-zero")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "name", str(self.name))

    @classmethod
    def linear(
        cls,
        duration: float,
        *,
        initial_scale: float = 1.0,
        final_scale: float = 0.0,
        name: str = "linear_ramp_off",
    ) -> "PiecewiseLinearTurnOff":
        """Return a two-node linear ramp ending at time zero."""

        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("linear turn-off duration must be finite and positive")
        return cls(
            times=np.asarray([-duration, 0.0], dtype=float),
            values=np.asarray([initial_scale, final_scale], dtype=float),
            name=name,
        )

    @property
    def duration(self) -> float:
        return float(-self.times[0])

    @property
    def total_drop(self) -> float:
        return float(self.values[0] - self.values[-1])

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe waveform description."""

        return {
            "name": self.name,
            "time_origin": "ramp_end",
            "times_relative_to_ramp_end_s": self.times.tolist(),
            "current_scales": self.values.tolist(),
            "duration_s": self.duration,
            "total_current_scale_drop": self.total_drop,
            "segment_count": int(self.times.size - 1),
        }


@dataclass(frozen=True)
class WaveformMagneticSixReferenceResult:
    """Magnetic-six reference after convolution with a finite turn-off."""

    times: np.ndarray
    receiver_locations: tuple[tuple[float, float, float], ...]
    data: np.ndarray
    dbdt_native: np.ndarray | None
    dbdt_impulse: np.ndarray | None
    primary_dbdt_reference: str
    empymod_version: str | None
    audit: dict[str, Any]
    waveform: dict[str, Any]
    convolution: dict[str, Any]

    @property
    def components(self) -> tuple[str, ...]:
        return MAGNETIC6_COMPONENTS

    @property
    def units(self) -> tuple[str, ...]:
        return MAGNETIC6_UNITS

    def flat_data(self) -> np.ndarray:
        return np.asarray(self.data, dtype=float).reshape(self.times.size, -1)


def turnoff_waveform_from_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> PiecewiseLinearTurnOff | None:
    """Build the finite turn-off declared in a YAML-style source config.

    ``step_off`` returns ``None`` because no waveform convolution is required.
    ``linear_ramp_off`` and inline/file-backed ``tabulated`` waveforms are
    converted to times relative to the final waveform node.
    """

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("config must contain a source mapping")
    waveform = source.get("waveform", {"type": "step_off"})
    if not isinstance(waveform, Mapping):
        raise ValueError("source.waveform must be a mapping")
    waveform_type = str(waveform.get("type", "step_off")).strip().lower()
    source_current = float(source.get("current", 1.0))
    if source_current == 0.0:
        raise ValueError("source.current must be non-zero")

    if waveform_type == "step_off":
        return None
    if waveform_type == "linear_ramp_off":
        duration = float(waveform.get("t_off", waveform.get("off_time", 0.0)))
        initial_scale = _waveform_scale(
            waveform,
            source_current=source_current,
            current_key="current_initial",
            scale_keys=("initial_value_scale", "initial_value"),
            default=1.0,
        )
        final_scale = _waveform_scale(
            waveform,
            source_current=source_current,
            current_key="current_final",
            scale_keys=("final_value_scale", "final_value"),
            default=0.0,
        )
        return PiecewiseLinearTurnOff.linear(
            duration,
            initial_scale=initial_scale,
            final_scale=final_scale,
        )
    if waveform_type != "tabulated":
        raise ValueError(f"unsupported source waveform type: {waveform_type}")

    times, values = _tabulated_waveform_arrays(
        waveform,
        source_current=source_current,
        base_dir=base_dir,
    )
    times = times - float(times[-1])
    return PiecewiseLinearTurnOff(times=times, values=values, name="tabulated_turnoff")


def load_turnoff_csv(path: str | Path) -> PiecewiseLinearTurnOff:
    """Load ``time_s,current_scale`` (or ``time,current``) waveform CSV.

    Input times may start at zero; they are shifted so the final sample is the
    ramp-end time zero.  A ``current`` column is treated as already normalized
    unless the caller constructs the object through ``turnoff_waveform_from_config``.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("waveform CSV contains no rows")
    headers = set(rows[0])
    time_key = "time_s" if "time_s" in headers else "time"
    value_key = "current_scale" if "current_scale" in headers else "current"
    if time_key not in headers or value_key not in headers:
        raise ValueError(
            "waveform CSV must contain time_s,current_scale or time,current columns"
        )
    times = np.asarray([float(row[time_key]) for row in rows], dtype=float)
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    times -= float(times[-1])
    return PiecewiseLinearTurnOff(times=times, values=values, name=path.stem)


def run_empymod_magnetic6_waveform_reference(
    survey: EmpymodSurvey,
    waveform: PiecewiseLinearTurnOff,
    *,
    quadrature_order: int = 8,
    reference_runner: Callable[..., MagneticSixReferenceResult] = (
        run_empymod_magnetic6_reference
    ),
    **reference_kwargs,
) -> WaveformMagneticSixReferenceResult:
    """Convolve ideal switch-off magnetic-six responses with a finite ramp.

    The survey times must be positive times after the end of the turn-off.
    One empymod call is made on the sorted union of all delayed evaluation
    times, after which the quadrature is assembled for every receiver and
    component.
    """

    times = np.asarray(survey.times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("survey.times must be a non-empty 1D array")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("waveform-convolved survey times must be positive after ramp end")
    quadrature_order = int(quadrature_order)
    if quadrature_order < 2:
        raise ValueError("waveform quadrature_order must be at least 2")

    delays, weights, segment_indices = _turnoff_quadrature(waveform, quadrature_order)
    shifted = times[:, None] - delays[None, :]
    flat = shifted.reshape(-1)
    unique_times, inverse = np.unique(flat, return_inverse=True)
    delayed_survey = replace(survey, times=unique_times, signal=-1)
    delayed_reference = reference_runner(delayed_survey, **reference_kwargs)

    data = _weighted_delayed_values(
        delayed_reference.data,
        inverse,
        n_observation_times=times.size,
        n_nodes=delays.size,
        weights=weights,
    )
    native = _weighted_optional_dbdt(
        delayed_reference.dbdt_native,
        inverse,
        times.size,
        delays.size,
        weights,
    )
    impulse = _weighted_optional_dbdt(
        delayed_reference.dbdt_impulse,
        inverse,
        times.size,
        delays.size,
        weights,
    )
    audit = _convolved_dbdt_audit(
        native,
        impulse,
        delayed_reference.audit,
    )
    convolution = {
        "method": "gauss_legendre_superposition_of_delayed_step_off_responses",
        "quadrature_order_per_segment": quadrature_order,
        "quadrature_node_count": int(delays.size),
        "unique_empymod_time_count": int(unique_times.size),
        "weight_sum": float(np.sum(weights)),
        "expected_weight_sum": waveform.total_drop,
        "weight_sum_error": float(np.sum(weights) - waveform.total_drop),
        "delay_min_s": float(np.min(delays)),
        "delay_max_s": float(np.max(delays)),
        "segment_indices": segment_indices.tolist(),
    }
    return WaveformMagneticSixReferenceResult(
        times=times.copy(),
        receiver_locations=tuple(delayed_reference.receiver_locations),
        data=data,
        dbdt_native=native,
        dbdt_impulse=impulse,
        primary_dbdt_reference=delayed_reference.primary_dbdt_reference,
        empymod_version=delayed_reference.empymod_version,
        audit=audit,
        waveform=waveform.metadata(),
        convolution=convolution,
    )


def _turnoff_quadrature(
    waveform: PiecewiseLinearTurnOff,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, base_weights = np.polynomial.legendre.leggauss(quadrature_order)
    delays: list[float] = []
    weights: list[float] = []
    segment_indices: list[int] = []
    for index, (left, right, value_left, value_right) in enumerate(
        zip(
            waveform.times[:-1],
            waveform.times[1:],
            waveform.values[:-1],
            waveform.values[1:],
        )
    ):
        width = float(right - left)
        slope = float((value_right - value_left) / width)
        if slope == 0.0:
            continue
        half = 0.5 * width
        midpoint = 0.5 * float(left + right)
        segment_nodes = midpoint + half * nodes
        segment_weights = (-slope) * half * base_weights
        delays.extend(float(item) for item in segment_nodes)
        weights.extend(float(item) for item in segment_weights)
        segment_indices.extend([index] * quadrature_order)
    if not delays:
        raise ValueError("turn-off waveform has no current change")
    delay_array = np.asarray(delays, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    expected = waveform.total_drop
    error = abs(float(np.sum(weight_array)) - expected)
    tolerance = 1.0e-12 * max(1.0, abs(expected))
    if error > tolerance:
        raise RuntimeError(
            "turn-off quadrature does not conserve the total current change: "
            f"error={error:.6e}, tolerance={tolerance:.6e}"
        )
    return delay_array, weight_array, np.asarray(segment_indices, dtype=int)


def _weighted_delayed_values(
    values: np.ndarray,
    inverse: np.ndarray,
    *,
    n_observation_times: int,
    n_nodes: int,
    weights: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sampled = array[np.asarray(inverse, dtype=int)].reshape(
        n_observation_times,
        n_nodes,
        *array.shape[1:],
    )
    return np.einsum("k,tk...->t...", weights, sampled)


def _weighted_optional_dbdt(
    values: np.ndarray | None,
    inverse: np.ndarray,
    n_observation_times: int,
    n_nodes: int,
    weights: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    return _weighted_delayed_values(
        values,
        inverse,
        n_observation_times=n_observation_times,
        n_nodes=n_nodes,
        weights=weights,
    )


def _convolved_dbdt_audit(
    native: np.ndarray | None,
    impulse: np.ndarray | None,
    base_audit: Mapping[str, Any],
) -> dict[str, Any]:
    tolerance = float(base_audit.get("tolerance", 0.01))
    floor_fraction = float(base_audit.get("floor_fraction", 0.01))
    output = {
        "empymod_version": base_audit.get("empymod_version"),
        "primary_dbdt_reference": base_audit.get("primary_dbdt_reference"),
        "native_b_available": bool(base_audit.get("native_b_available", native is not None)),
        "requested": bool(base_audit.get("requested", True)),
        "tolerance": tolerance,
        "floor_fraction": floor_fraction,
        "waveform_convolved": True,
    }
    if native is None or impulse is None:
        return {
            **output,
            "performed": False,
            "passed": True,
            "reason": "both native-b and impulse-H routes are required for the convolved audit",
            "global_max_floor_relative_error": 0.0,
            "component_metrics": {},
        }

    metrics: dict[str, Any] = {}
    global_max = 0.0
    for index, component in enumerate(MAGNETIC6_COMPONENTS[3:]):
        primary = np.asarray(native[..., index], dtype=float)
        secondary = np.asarray(impulse[..., index], dtype=float)
        peak = float(np.max(np.abs(primary)))
        floor = max(floor_fraction * peak, np.finfo(float).tiny)
        scaled = np.abs(primary - secondary) / np.maximum(np.abs(primary), floor)
        max_error = float(np.max(scaled))
        active = np.abs(primary) >= floor
        sign_agreement = (
            float(np.mean(np.sign(primary[active]) == np.sign(secondary[active])))
            if np.any(active)
            else 1.0
        )
        global_max = max(global_max, max_error)
        metrics[component] = {
            "unit": "T/s",
            "native_peak_abs": peak,
            "floor": floor,
            "max_floor_relative_error": max_error,
            "sign_agreement_fraction": sign_agreement,
            "passed": bool(max_error <= tolerance),
        }
    return {
        **output,
        "performed": True,
        "identity": (
            "waveform-convolved native mrec='b' dB/dt versus "
            "waveform-convolved -mu0*H impulse"
        ),
        "global_max_floor_relative_error": global_max,
        "component_metrics": metrics,
        "passed": bool(all(item["passed"] for item in metrics.values())),
    }


def _waveform_scale(
    waveform: Mapping[str, Any],
    *,
    source_current: float,
    current_key: str,
    scale_keys: Sequence[str],
    default: float,
) -> float:
    if current_key in waveform:
        return float(waveform[current_key]) / source_current
    for key in scale_keys:
        if key in waveform:
            return float(waveform[key])
    return float(default)


def _tabulated_waveform_arrays(
    waveform: Mapping[str, Any],
    *,
    source_current: float,
    base_dir: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    path_value = waveform.get("path", waveform.get("csv_path"))
    if path_value is not None:
        path = Path(path_value)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        table = np.genfromtxt(path, delimiter=",", names=True)
        if table.dtype.names is None or "time" not in table.dtype.names:
            raise ValueError("tabulated waveform CSV must contain a time column")
        names = set(table.dtype.names)
        times = np.atleast_1d(np.asarray(table["time"], dtype=float))
        if "current" in names:
            values = np.atleast_1d(np.asarray(table["current"], dtype=float)) / source_current
        elif "current_scale" in names:
            values = np.atleast_1d(np.asarray(table["current_scale"], dtype=float))
        else:
            raise ValueError("tabulated waveform CSV requires current or current_scale")
        return times, values

    times = np.asarray(waveform.get("times"), dtype=float)
    if "currents" in waveform:
        values = np.asarray(waveform["currents"], dtype=float) / source_current
    else:
        values = np.asarray(waveform.get("values"), dtype=float)
    return times, values
