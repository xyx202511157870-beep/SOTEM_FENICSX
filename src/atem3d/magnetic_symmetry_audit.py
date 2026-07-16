"""Symmetry and zero-point audits for the five-receiver magnetic benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MAGNETIC_COMPONENTS = ("dBzdt", "Hz")


def _normalised_residual(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[float | None, float]:
    absolute_numerator = float(np.max(np.abs(numerator)))
    scale = float(np.max(np.abs(denominator)))
    ratio = None if scale == 0.0 else absolute_numerator / scale
    return ratio, absolute_numerator


def audit_receiver_symmetry(
    values: np.ndarray,
    components: Sequence[str],
    times: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Return even/odd residuals for a -20,-10,0,10,20 m receiver line."""

    data = np.asarray(values, dtype=float)
    names = tuple(str(name) for name in components)
    if (
        data.ndim != 3
        or data.shape[0] != 5
        or data.shape[1] == 0
        or data.shape[2] != len(names)
    ):
        raise ValueError("values must have shape (5, n_times, n_components)")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    if times is not None:
        time_values = np.asarray(times, dtype=float)
        if time_values.shape != (data.shape[1],):
            raise ValueError("times must contain one value per time sample")
        if not np.all(np.isfinite(time_values)):
            raise ValueError("times must be finite")

    result: dict[str, dict[str, Any]] = {}
    for component_index, name in enumerate(names):
        field = data[:, :, component_index]
        odd = name in MAGNETIC_COMPONENTS
        pair_24 = field[1] + field[3] if odd else field[1] - field[3]
        pair_15 = field[0] + field[4] if odd else field[0] - field[4]
        flank_24 = np.maximum(np.abs(field[1]), np.abs(field[3]))
        flank_15 = np.maximum(np.abs(field[0]), np.abs(field[4]))
        zero_ratio, zero_absolute = _normalised_residual(field[2], flank_24)
        residual_24, absolute_24 = _normalised_residual(pair_24, flank_24)
        residual_15, absolute_15 = _normalised_residual(pair_15, flank_15)
        result[name] = {
            "parity": "odd" if odd else "even",
            "rx3_zero_ratio": zero_ratio,
            "rx3_abs_max": zero_absolute,
            "pair_24_residual": residual_24,
            "pair_24_abs_max": absolute_24,
            "pair_15_residual": residual_15,
            "pair_15_abs_max": absolute_15,
        }
    return result


def audit_model_triplet(
    background: np.ndarray,
    channel: np.ndarray,
    components: Sequence[str],
    times: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Audit background, channel, and the signed channel-minus-background field."""

    background_values = np.asarray(background, dtype=float)
    channel_values = np.asarray(channel, dtype=float)
    if background_values.shape != channel_values.shape:
        raise ValueError("background and channel values must have the same shape")
    return {
        "background": audit_receiver_symmetry(
            background_values,
            components,
            times,
        ),
        "channel": audit_receiver_symmetry(
            channel_values,
            components,
            times,
        ),
        "delta": audit_receiver_symmetry(
            channel_values - background_values,
            components,
            times,
        ),
    }


def evaluate_magnetic_gates(
    metrics: Mapping[str, Mapping[str, float | None]],
    *,
    threshold: float = 0.01,
) -> dict[str, object]:
    """Apply the symmetry gate only to dBzdt and Hz metrics."""

    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    failures: list[str] = []
    for component in MAGNETIC_COMPONENTS:
        component_metrics = metrics.get(component)
        if component_metrics is None:
            failures.append(f"{component}.missing")
            continue
        for key in ("rx3_zero_ratio", "pair_24_residual", "pair_15_residual"):
            value = component_metrics.get(key)
            if value is None or not np.isfinite(float(value)) or float(value) > threshold_value:
                failures.append(f"{component}.{key}")
    return {
        "passed": not failures,
        "threshold": threshold_value,
        "failures": failures,
    }


__all__ = [
    "MAGNETIC_COMPONENTS",
    "audit_model_triplet",
    "audit_receiver_symmetry",
    "evaluate_magnetic_gates",
]
