"""Strict signed-response metrics for the Zhou 2020 TEM-IP benchmark."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np


SCHEMA = "atem3d.zhou2020.strict-comparison/v1"
COMPONENTS = ("Ex", "Hz", "dBzdt")
TOTAL_FIELD_L2_GATE = 0.05
IP_INCREMENT_L2_GATE = 0.10
ZERO_CROSSING_TIME_GATE = 0.10
EXPECTED_TIME_BOUNDS = (1.0e-4, 3.0)
EXPECTED_SAMPLE_COUNT = 101


def log_time_zero_crossings(times, values) -> np.ndarray:
    """Return signed zero crossings interpolated on log10(time)."""

    time_values = _validated_times(times)
    response = np.asarray(values, dtype=float)
    if response.ndim != 1 or response.shape != time_values.shape:
        raise ValueError("values must be one-dimensional and match times")
    if not np.isfinite(response).all():
        raise ValueError("values must be finite")

    crossings: list[float] = []
    index = 0
    log_times = np.log10(time_values)
    while index < response.size:
        if response[index] == 0.0:
            start = index
            while index < response.size and response[index] == 0.0:
                index += 1
            end = index
            if (
                start > 0
                and end < response.size
                and _opposite(response[start - 1], response[end])
            ):
                crossings.append(
                    10.0 ** (0.5 * (log_times[start] + log_times[end - 1]))
                )
            continue

        if index + 1 < response.size and response[index + 1] != 0.0:
            left = float(response[index])
            right = float(response[index + 1])
            if _opposite(left, right):
                fraction = abs(left) / (abs(left) + abs(right))
                log_crossing = log_times[index] + fraction * (
                    log_times[index + 1] - log_times[index]
                )
                crossings.append(10.0**log_crossing)
        index += 1
    return np.asarray(crossings, dtype=float)


def compare_zhou_responses(
    *,
    times,
    prediction_noip,
    reference_noip,
    prediction_ip,
    reference_ip,
    components: Sequence[str] = COMPONENTS,
) -> dict[str, Any]:
    """Compare full-window no-IP, IP, and pure-IP signed responses."""

    time_values = _validated_times(times)
    names = _validated_components(components)
    arrays = {
        "prediction_noip": _validated_response(
            prediction_noip, time_values.size, len(names)
        ),
        "reference_noip": _validated_response(
            reference_noip, time_values.size, len(names)
        ),
        "prediction_ip": _validated_response(
            prediction_ip, time_values.size, len(names)
        ),
        "reference_ip": _validated_response(
            reference_ip, time_values.size, len(names)
        ),
    }
    full_window = _full_window_record(time_values)
    point_errors: list[dict[str, Any]] = []
    failed_components: set[str] = set()
    failed_times: set[float] = set()

    total_field: dict[str, dict[str, Any]] = {}
    for variant in ("noip", "ip"):
        prediction = arrays[f"prediction_{variant}"]
        reference = arrays[f"reference_{variant}"]
        total_field[variant] = {}
        for index, component in enumerate(names):
            record, rows = _component_metrics(
                time_values,
                prediction[:, index],
                reference[:, index],
                gate=TOTAL_FIELD_L2_GATE,
                variant=variant,
                component=component,
            )
            total_field[variant][component] = record
            point_errors.extend(rows)
            if not record["passed"]:
                failed_components.add(component)
            for row in rows:
                if row["robust_relative_error"] > TOTAL_FIELD_L2_GATE:
                    failed_times.add(float(row["time_s"]))

    prediction_delta = arrays["prediction_ip"] - arrays["prediction_noip"]
    reference_delta = arrays["reference_ip"] - arrays["reference_noip"]
    ip_increment: dict[str, Any] = {}
    for index, component in enumerate(names):
        record, rows = _component_metrics(
            time_values,
            prediction_delta[:, index],
            reference_delta[:, index],
            gate=IP_INCREMENT_L2_GATE,
            variant="ip_increment",
            component=component,
        )
        ip_increment[component] = record
        point_errors.extend(rows)
        if not record["passed"]:
            failed_components.add(component)
        for row in rows:
            if row["robust_relative_error"] > IP_INCREMENT_L2_GATE:
                failed_times.add(float(row["time_s"]))

    zero_crossings: dict[str, dict[str, Any]] = {}
    zero_crossings_passed = True
    for variant in ("noip", "ip", "ip_increment"):
        if variant == "ip_increment":
            prediction = prediction_delta
            reference = reference_delta
        else:
            prediction = arrays[f"prediction_{variant}"]
            reference = arrays[f"reference_{variant}"]
        zero_crossings[variant] = {}
        for index, component in enumerate(names):
            record = _zero_crossing_metrics(
                time_values,
                prediction[:, index],
                reference[:, index],
            )
            zero_crossings[variant][component] = record
            if not record["passed"]:
                zero_crossings_passed = False
                failed_components.add(component)

    numerical_gates_passed = bool(
        all(
            record["passed"]
            for variant in total_field.values()
            for record in variant.values()
        )
        and all(record["passed"] for record in ip_increment.values())
        and zero_crossings_passed
    )
    if not full_window["passed"]:
        status = "incomplete_time_window"
    elif numerical_gates_passed:
        status = "ip_internally_validated"
    else:
        status = "failed_with_reproducible_evidence"

    return {
        "schema": SCHEMA,
        "status": status,
        "sample_count": int(time_values.size),
        "components": list(names),
        "gates": {
            "total_field_relative_l2": TOTAL_FIELD_L2_GATE,
            "ip_increment_relative_l2": IP_INCREMENT_L2_GATE,
            "zero_crossing_relative_time": ZERO_CROSSING_TIME_GATE,
        },
        "full_time_window": full_window,
        "total_field": total_field,
        "ip_increment": ip_increment,
        "zero_crossings": zero_crossings,
        "point_errors": point_errors,
        "failed_components": sorted(failed_components),
        "failed_times_s": sorted(failed_times),
        "numerical_gates_passed": numerical_gates_passed,
    }


def _component_metrics(
    times: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    gate: float,
    variant: str,
    component: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    peak = float(np.max(np.abs(reference)))
    if peak <= 0.0:
        raise ValueError(f"reference peak for {variant}/{component} must be positive")
    floor = 0.01 * peak
    residual = prediction - reference
    reference_norm = float(np.linalg.norm(reference))
    relative_l2 = float(np.linalg.norm(residual) / reference_norm)
    robust_errors = np.abs(residual) / np.maximum(np.abs(reference), floor)
    maximum_index = int(np.argmax(robust_errors))
    rows = [
        {
            "variant": variant,
            "component": component,
            "time_s": float(time),
            "prediction": float(predicted),
            "reference": float(expected),
            "response_strength": "strong"
            if abs(float(expected)) >= floor
            else "weak",
            "robust_relative_error": float(error),
        }
        for time, predicted, expected, error in zip(
            times,
            prediction,
            reference,
            robust_errors,
        )
    ]
    return (
        {
            "relative_l2": relative_l2,
            "max_robust_relative_error": float(robust_errors[maximum_index]),
            "time_of_max_robust_error_s": float(times[maximum_index]),
            "max_absolute_error": float(np.max(np.abs(residual))),
            "reference_peak": peak,
            "denominator_floor": floor,
            "gate": float(gate),
            "passed": bool(relative_l2 <= gate),
        },
        rows,
    )


def _zero_crossing_metrics(
    times: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    predicted = log_time_zero_crossings(times, prediction)
    expected = log_time_zero_crossings(times, reference)
    count_match = predicted.size == expected.size
    if count_match and expected.size:
        relative_errors = np.abs(predicted - expected) / expected
        max_error: float | None = float(np.max(relative_errors))
    elif count_match:
        max_error = 0.0
    else:
        max_error = None
    passed = bool(
        count_match
        and max_error is not None
        and max_error <= ZERO_CROSSING_TIME_GATE
    )
    return {
        "prediction": predicted.tolist(),
        "reference": expected.tolist(),
        "count_match": bool(count_match),
        "max_relative_time_error": max_error,
        "gate": ZERO_CROSSING_TIME_GATE,
        "passed": passed,
    }


def _full_window_record(times: np.ndarray) -> dict[str, Any]:
    start_passed = bool(
        np.isclose(times[0], EXPECTED_TIME_BOUNDS[0], rtol=0.0, atol=1.0e-15)
    )
    stop_passed = bool(
        np.isclose(times[-1], EXPECTED_TIME_BOUNDS[1], rtol=0.0, atol=1.0e-12)
    )
    count_passed = times.size == EXPECTED_SAMPLE_COUNT
    return {
        "expected_bounds_s": list(EXPECTED_TIME_BOUNDS),
        "actual_bounds_s": [float(times[0]), float(times[-1])],
        "expected_count": EXPECTED_SAMPLE_COUNT,
        "actual_count": int(times.size),
        "passed": bool(start_passed and stop_passed and count_passed),
    }


def _validated_times(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two values")
    if (
        not np.isfinite(result).all()
        or np.any(result <= 0.0)
        or np.any(np.diff(result) <= 0.0)
    ):
        raise ValueError("times must be finite, positive, and strictly increasing")
    return result


def _validated_components(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("components must be a sequence")
    result = tuple(values)
    if result != COMPONENTS:
        raise ValueError(f"components must equal {COMPONENTS}")
    return result


def _validated_response(values, rows: int, columns: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (rows, columns):
        raise ValueError(f"response must have shape {(rows, columns)}")
    if not np.isfinite(result).all():
        raise ValueError("response must contain only finite values")
    return result


def _opposite(left: float, right: float) -> bool:
    return bool(left != 0.0 and right != 0.0 and np.signbit(left) != np.signbit(right))
