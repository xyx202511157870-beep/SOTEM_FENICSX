import numpy as np
import pytest

from atem3d.zhou2020_metrics import (
    compare_zhou_responses,
    log_time_zero_crossings,
)


def test_log_time_zero_crossing_interpolates_on_logarithmic_time():
    crossings = log_time_zero_crossings(
        np.array([1.0, 100.0]),
        np.array([1.0, -1.0]),
    )

    np.testing.assert_allclose(crossings, [10.0])


def test_log_time_zero_crossing_counts_a_zero_plateau_once():
    crossings = log_time_zero_crossings(
        np.array([1.0, 10.0, 100.0, 1000.0]),
        np.array([1.0, 0.0, 0.0, -1.0]),
    )

    np.testing.assert_allclose(crossings, [np.sqrt(10.0 * 100.0)])


def _responses(times):
    log_time = np.log10(times)
    noip = np.column_stack(
        [
            2.0 / (1.0 + times),
            3.0 / (1.0 + times),
            -4.0 / (1.0 + times),
        ]
    )
    ip = noip.copy()
    ip[:, 0] = -log_time - 1.0
    ip[:, 1] *= 1.002
    ip[:, 2] *= 0.99
    return noip, ip


def test_compare_zhou_responses_passes_exact_full_window_and_reports_ip_increment():
    times = np.geomspace(1.0e-4, 3.0, 101)
    noip, ip = _responses(times)

    result = compare_zhou_responses(
        times=times,
        prediction_noip=noip,
        reference_noip=noip,
        prediction_ip=ip,
        reference_ip=ip,
    )

    assert result["schema"] == "atem3d.zhou2020.strict-comparison/v1"
    assert result["status"] == "ip_internally_validated"
    assert result["full_time_window"]["passed"] is True
    assert result["gates"] == {
        "total_field_relative_l2": 0.05,
        "ip_increment_relative_l2": 0.10,
        "zero_crossing_relative_time": 0.10,
    }
    assert result["total_field"]["noip"]["Ex"]["relative_l2"] == pytest.approx(0.0)
    assert result["ip_increment"]["Ex"]["relative_l2"] == pytest.approx(0.0)
    assert result["zero_crossings"]["ip"]["Ex"]["count_match"] is True
    assert result["zero_crossings"]["ip"]["Ex"]["reference"]


def test_compare_zhou_responses_marks_partial_window_incomplete():
    times = np.geomspace(1.0e-4, 1.0e-1, 21)
    noip, ip = _responses(times)

    result = compare_zhou_responses(
        times=times,
        prediction_noip=noip,
        reference_noip=noip,
        prediction_ip=ip,
        reference_ip=ip,
    )

    assert result["status"] == "incomplete_time_window"
    assert result["full_time_window"]["passed"] is False
    assert result["full_time_window"]["actual_count"] == 21


def test_compare_zhou_responses_preserves_failed_component_and_time():
    times = np.geomspace(1.0e-4, 3.0, 101)
    noip, ip = _responses(times)
    bad_ip = ip.copy()
    bad_ip[30, 0] += 100.0

    result = compare_zhou_responses(
        times=times,
        prediction_noip=noip,
        reference_noip=noip,
        prediction_ip=bad_ip,
        reference_ip=ip,
    )

    assert result["status"] == "failed_with_reproducible_evidence"
    assert "Ex" in result["failed_components"]
    assert times[30] in result["failed_times_s"]
    assert result["total_field"]["ip"]["Ex"]["passed"] is False


def test_compare_zhou_responses_does_not_drop_near_zero_samples():
    times = np.geomspace(1.0e-4, 3.0, 101)
    noip, ip = _responses(times)
    reference_ip = ip.copy()
    reference_ip[50, 0] = 0.0
    prediction_ip = reference_ip.copy()
    prediction_ip[50, 0] = 1.0e-3

    result = compare_zhou_responses(
        times=times,
        prediction_noip=noip,
        reference_noip=noip,
        prediction_ip=prediction_ip,
        reference_ip=reference_ip,
    )

    row = next(
        item
        for item in result["point_errors"]
        if item["variant"] == "ip"
        and item["component"] == "Ex"
        and item["time_s"] == times[50]
    )
    assert row["response_strength"] == "weak"
    assert row["robust_relative_error"] == pytest.approx(
        1.0e-3 / (0.01 * np.max(np.abs(reference_ip[:, 0])))
    )
    assert result["sample_count"] == 101
