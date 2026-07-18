"""Signed-response acceptance metrics for SOTEM benchmarks."""

from __future__ import annotations

import numpy as np

from atem3d.metrics import robust_component_errors


_ZERO_CROSSING_TIME_TOLERANCE = 0.05


def linear_zero_crossings(times, values) -> np.ndarray:
    """Return times where a signed curve crosses zero.

    Strict sign changes are linearly interpolated.  An isolated exact zero
    bracketed by opposite signs is reported at its sample time.  A consecutive
    zero plateau bracketed by opposite signs is reported once at the midpoint
    of its first and last sample times.  Same-sign touches and leading or
    trailing zero runs are not crossings.
    """

    time_values = _validated_times(times)
    response_values = _real_array("values", values)
    if response_values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if response_values.shape != time_values.shape:
        raise ValueError("values must have the same length as times")
    if not np.all(np.isfinite(response_values)):
        raise ValueError("values must be finite")

    crossings: list[float] = []
    index = 0
    while index < response_values.size:
        if response_values[index] == 0.0:
            zero_start = index
            while index < response_values.size and response_values[index] == 0.0:
                index += 1
            zero_end = index
            if (
                zero_start > 0
                and zero_end < response_values.size
                and _opposite_nonzero_signs(
                    response_values[zero_start - 1], response_values[zero_end]
                )
            ):
                crossings.append(
                    time_values[zero_start]
                    + 0.5 * (time_values[zero_end - 1] - time_values[zero_start])
                )
            continue

        if index + 1 < response_values.size and response_values[index + 1] != 0.0:
            left = response_values[index]
            right = response_values[index + 1]
            if _opposite_nonzero_signs(left, right):
                left_magnitude = abs(float(left))
                right_magnitude = abs(float(right))
                scale = max(left_magnitude, right_magnitude)
                left_scaled = left_magnitude / scale
                right_scaled = right_magnitude / scale
                fraction = left_scaled / (left_scaled + right_scaled)
                crossings.append(
                    time_values[index]
                    + fraction * (time_values[index + 1] - time_values[index])
                )
        index += 1
    return np.asarray(crossings, dtype=float)


def compare_signed_response(
    times,
    prediction,
    reference,
    components,
    *,
    threshold: float = 0.10,
) -> dict:
    """Compare signed response curves using a one-percent reference-peak floor."""

    time_values = _validated_times(times)
    prediction_values = _real_array("prediction", prediction)
    reference_values = _real_array("reference", reference)
    if prediction_values.ndim != 2 or reference_values.ndim != 2:
        raise ValueError("prediction and reference must be two-dimensional")
    if prediction_values.shape != reference_values.shape:
        raise ValueError("prediction and reference must have the same shape")
    if prediction_values.shape[0] != time_values.size:
        raise ValueError("response rows must match the number of times")
    if not np.all(np.isfinite(prediction_values)):
        raise ValueError("prediction must be finite")
    if not np.all(np.isfinite(reference_values)):
        raise ValueError("reference must be finite")

    if isinstance(components, (str, bytes)):
        raise ValueError("components must be a sequence of strings, not str or bytes")
    try:
        component_names = list(components)
    except TypeError as exc:
        raise ValueError("components must be a sequence of strings") from exc
    if not component_names or any(
        not isinstance(component, str) or not component.strip()
        for component in component_names
    ):
        raise ValueError("components must be nonempty strings")
    if len(set(component_names)) != len(component_names):
        raise ValueError("components must be unique")
    if len(component_names) != prediction_values.shape[1]:
        raise ValueError("components must match response columns")

    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be finite and positive, not boolean")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite and positive") from exc
    if not np.isfinite(threshold_value) or threshold_value <= 0.0:
        raise ValueError("threshold must be finite and positive")

    floor_by_component: dict[str, float] = {}
    for index, component in enumerate(component_names):
        reference_peak = float(np.max(np.abs(reference_values[:, index])))
        if not np.isfinite(reference_peak) or reference_peak <= 0.0:
            raise ValueError(
                f"reference peak for component {component!r} must be finite and positive"
            )
        floor_by_component[component] = 0.01 * reference_peak
    diagnostic_rows, summary = robust_component_errors(
        time_values,
        prediction_values,
        reference_values,
        component_names,
        threshold=threshold_value,
        floor_overrides=floor_by_component,
    )
    rows = _acceptance_rows(
        diagnostic_rows,
        floor_by_component=floor_by_component,
        threshold=threshold_value,
    )
    max_robust_error_by_component = {
        component: float(summary[f"max_error_{component}"])
        for component in component_names
    }
    max_acceptance_error_by_component: dict[str, float] = {}
    amplitude_failed_components: list[str] = []
    amplitude_failed_times: set[float] = set()
    for component in component_names:
        component_rows = rows[rows["component"] == component]
        max_acceptance_error_by_component[component] = float(
            np.max(component_rows["acceptance_error"])
        )
        failed_rows = component_rows[~component_rows["pass_threshold"]]
        if failed_rows.size:
            amplitude_failed_components.append(component)
            amplitude_failed_times.update(float(time) for time in failed_rows["time_obs"])

    zero_crossings = {}
    zero_crossing_failed_components: list[str] = []
    for index, component in enumerate(component_names):
        prediction_crossings = linear_zero_crossings(
            time_values, prediction_values[:, index]
        )
        reference_crossings = linear_zero_crossings(
            time_values, reference_values[:, index]
        )
        count_match = prediction_crossings.size == reference_crossings.size
        if not count_match:
            max_error = float("inf")
        elif prediction_crossings.size == 0:
            max_error = 0.0
        else:
            max_error = float(
                np.max(
                    np.abs(prediction_crossings - reference_crossings)
                    / np.abs(reference_crossings)
                )
            )
        crossing_passed = bool(
            count_match and max_error <= _ZERO_CROSSING_TIME_TOLERANCE
        )
        if not crossing_passed:
            zero_crossing_failed_components.append(component)
        zero_crossings[component] = {
            "prediction": prediction_crossings.tolist(),
            "reference": reference_crossings.tolist(),
            "count_match": bool(count_match),
            "max_relative_time_error": max_error,
            "passed": crossing_passed,
        }

    robust_failed_components = list(summary["failed_components"])
    robust_failed_times = list(summary["failed_times"])
    amplitude_passed = not amplitude_failed_components
    zero_crossings_passed = not zero_crossing_failed_components
    failed_components = sorted(
        set(amplitude_failed_components) | set(zero_crossing_failed_components)
    )
    summary.update(
        {
            "robust_diagnostic_pass_all_components": bool(summary["pass_all_components"]),
            "robust_diagnostic_failed_components": robust_failed_components,
            "robust_diagnostic_failed_times": robust_failed_times,
            "legacy_pass_5pct_is_threshold_alias": True,
            "amplitude_threshold": threshold_value,
            "amplitude_pass_all_components": amplitude_passed,
            "amplitude_failed_components": amplitude_failed_components,
            "amplitude_failed_times": sorted(amplitude_failed_times),
            "max_acceptance_error_by_component": max_acceptance_error_by_component,
            "zero_crossing_time_tolerance": _ZERO_CROSSING_TIME_TOLERANCE,
            "zero_crossings_pass_all_components": zero_crossings_passed,
            "zero_crossing_failed_components": zero_crossing_failed_components,
            "pass_all_components": bool(amplitude_passed and zero_crossings_passed),
            "failed_components": failed_components,
            "failed_times": sorted(amplitude_failed_times),
        }
    )
    return {
        "rows": rows,
        "summary": summary,
        "floor_by_component": floor_by_component,
        "max_robust_error_by_component": max_robust_error_by_component,
        "max_acceptance_error_by_component": max_acceptance_error_by_component,
        "zero_crossings": zero_crossings,
    }


def _acceptance_rows(
    diagnostic_rows: np.ndarray,
    *,
    floor_by_component: dict[str, float],
    threshold: float,
) -> np.ndarray:
    dtype = np.dtype(
        [
            *diagnostic_rows.dtype.descr,
            ("response_strength", "U6"),
            ("acceptance_error", "f8"),
            ("pass_threshold", "?"),
        ]
    )
    rows = np.empty(diagnostic_rows.shape, dtype=dtype)
    for field in diagnostic_rows.dtype.names:
        rows[field] = diagnostic_rows[field]

    for index, row in enumerate(diagnostic_rows):
        component = str(row["component"])
        strong = abs(float(row["ref"])) >= floor_by_component[component]
        acceptance_error = float(
            row["relative_error_with_floor"]
            if strong
            else row["peak_normalized_error"]
        )
        passed = bool(acceptance_error <= threshold)
        rows[index]["response_strength"] = "strong" if strong else "weak"
        rows[index]["acceptance_error"] = acceptance_error
        rows[index]["pass_threshold"] = passed
        rows[index]["pass_5pct"] = passed
    return rows


def _validated_times(times) -> np.ndarray:
    time_values = _real_array("times", times)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if time_values.size == 0:
        raise ValueError("times must not be empty")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must be finite")
    if np.any(time_values <= 0.0):
        raise ValueError("times must be positive")
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return time_values


def _opposite_nonzero_signs(left: float, right: float) -> bool:
    return bool(
        left != 0.0
        and right != 0.0
        and np.signbit(left) != np.signbit(right)
    )


def _real_array(name: str, values) -> np.ndarray:
    raw_values = np.asarray(values)
    if np.iscomplexobj(raw_values):
        raise ValueError(f"{name} must be real")
    if not np.issubdtype(raw_values.dtype, np.number):
        raise ValueError(f"{name} must contain real numbers")
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
