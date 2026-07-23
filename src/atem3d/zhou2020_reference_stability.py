"""Pure evidence metrics for Zhou 2020 reference-transform stability."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np


SCHEMA = "atem3d.zhou2020.reference-stability/v1"


def sign_change_count(values) -> int:
    """Count sign changes after ignoring exact zero-valued samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("values must be a finite one-dimensional array")
    signs = np.sign(array)
    nonzero_signs = signs[signs != 0.0]
    return int(np.count_nonzero(np.diff(nonzero_signs)))


def _relative_l2(values, reference) -> float:
    """Return the L2 residual relative to a non-zero reference norm."""

    value_array = np.asarray(values, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if value_array.shape != reference_array.shape:
        raise ValueError("values and reference must have the same shape")
    if not np.isfinite(value_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("values and reference must be finite")
    if reference_array.size == 0 or not np.any(reference_array):
        raise ValueError("reference norm must be non-zero")
    scale = max(
        float(np.max(np.abs(value_array))),
        float(np.max(np.abs(reference_array))),
    )
    scaled_values = value_array / scale
    scaled_reference = reference_array / scale
    numerator = float(np.linalg.norm(scaled_values - scaled_reference))
    denominator = float(np.linalg.norm(scaled_reference))
    result = numerator / denominator
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0.0
        or not np.isfinite(result)
    ):
        raise ValueError("relative L2 calculation must be finite")
    return result


def _validated_inputs(times, *arrays) -> tuple[np.ndarray, ...]:
    """Validate a time vector and finite, matching one-dimensional signals."""

    time_values = np.asarray(times, dtype=float)
    if (
        time_values.ndim != 1
        or time_values.size < 1
        or not np.isfinite(time_values).all()
        or np.any(time_values <= 0.0)
        or np.any(np.diff(time_values) <= 0.0)
    ):
        raise ValueError("times must be finite, positive, and strictly increasing")

    result = [time_values]
    for values in arrays:
        array = np.asarray(values, dtype=float)
        if array.shape != time_values.shape or not np.isfinite(array).all():
            raise ValueError("arrays must be finite and have the same shape as times")
        result.append(array)
    return tuple(result)


def first_stable_sample(
    times,
    candidates,
    signal_to_spread: float = 3.0,
    consecutive: int = 5,
) -> int:
    """Return the first sample in a consecutive signal-dominated window."""

    time_values = _validated_inputs(times)[0]
    candidate_values = np.asarray(candidates, dtype=float)
    if (
        candidate_values.ndim != 2
        or candidate_values.shape[0] != time_values.size
        or candidate_values.shape[1] < 2
        or not np.isfinite(candidate_values).all()
    ):
        raise ValueError("candidates must have shape (n_times, n_methods>=2)")
    if isinstance(signal_to_spread, (bool, np.bool_)) or not isinstance(
        signal_to_spread, Real
    ):
        raise ValueError("signal_to_spread must be a finite real scalar")
    ratio = float(signal_to_spread)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("signal_to_spread must be finite and positive")
    if isinstance(consecutive, bool) or not isinstance(consecutive, (int, np.integer)):
        raise ValueError("consecutive must be a positive integer")
    if consecutive <= 0 or consecutive > time_values.size:
        raise ValueError("consecutive must be between 1 and the number of samples")

    centre = np.median(candidate_values, axis=1)
    spread = np.ptp(candidate_values, axis=1)
    stable = np.abs(centre) >= ratio * spread
    for start_index in range(time_values.size - consecutive + 1):
        if np.all(stable[start_index : start_index + consecutive]):
            return start_index
    raise ValueError("no consecutive stable window was found")


def build_reference_stability_audit(
    *,
    times,
    default_dlf,
    separate_total_qwe,
    direct_frequency_qwe,
    direct_qwe_converged,
    fenicsx_increment,
    signal_to_spread: float = 3.0,
    consecutive: int = 5,
) -> dict[str, Any]:
    """Build retained, non-promoting reference-stability audit evidence."""

    (
        time_values,
        default_values,
        separate_values,
        direct_values,
        fenicsx_values,
    ) = _validated_inputs(
        times,
        default_dlf,
        separate_total_qwe,
        direct_frequency_qwe,
        fenicsx_increment,
    )
    if not isinstance(direct_qwe_converged, (bool, np.bool_)):
        raise ValueError("direct_qwe_converged must be a boolean")
    start_index = first_stable_sample(
        time_values,
        np.column_stack((default_values, separate_values, direct_values)),
        signal_to_spread=signal_to_spread,
        consecutive=consecutive,
    )
    first_twenty = slice(0, min(20, time_values.size))
    converged = bool(direct_qwe_converged)

    return {
        "schema": SCHEMA,
        "status": "audited" if converged else "inconclusive",
        "formal_gate_decision": None,
        "all_samples_retained": True,
        "sample_count": int(time_values.size),
        "default_dlf": {
            "sign_changes_all": sign_change_count(default_values),
            "sign_changes_first20": sign_change_count(default_values[first_twenty]),
        },
        "qwe": {"converged": converged},
        "stable_window": {
            "start_index": start_index,
            "start_s": float(time_values[start_index]),
            "signal_to_spread": float(signal_to_spread),
            "consecutive": int(consecutive),
        },
        "transform_difference": {
            "default_dlf_vs_direct_qwe_relative_l2_full": _relative_l2(
                default_values, direct_values
            ),
            "default_dlf_vs_direct_qwe_relative_l2_first20": _relative_l2(
                default_values[first_twenty], direct_values[first_twenty]
            ),
        },
        "fenicsx_vs_direct_qwe": {
            "relative_l2_full": _relative_l2(fenicsx_values, direct_values),
            "relative_l2_stable_window": _relative_l2(
                fenicsx_values[start_index:], direct_values[start_index:]
            ),
        },
    }
