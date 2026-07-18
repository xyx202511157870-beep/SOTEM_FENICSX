import numpy as np
import pytest

from atem3d.sotem_metrics import compare_signed_response, linear_zero_crossings


def test_linear_zero_crossings_preserve_topology_and_interpolate_time():
    times = np.array([1.0, 2.0, 4.0])
    values = np.array([2.0, -2.0, -1.0])

    assert linear_zero_crossings(times, values).tolist() == pytest.approx([1.5])


@pytest.mark.parametrize("amplitude", [1.0e-200, 1.0e-300, 1.0e308])
def test_linear_zero_crossings_are_stable_for_finite_extremes(amplitude):
    crossings = linear_zero_crossings(
        np.array([1.0, 2.0]),
        np.array([amplitude, -amplitude]),
    )

    assert crossings.tolist() == pytest.approx([1.5])


def test_linear_zero_crossings_detect_tiny_zero_plateau_without_underflow():
    crossings = linear_zero_crossings(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0e-200, 0.0, -1.0e-200]),
    )

    assert crossings.tolist() == pytest.approx([2.0])


def test_linear_zero_crossings_ignore_same_sign_tiny_touch():
    crossings = linear_zero_crossings(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0e-300, 0.0, 1.0e-300]),
    )

    assert crossings.tolist() == []


def test_signed_response_uses_one_percent_peak_floor():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [0.0]]),
        np.array([[1.0], [0.005]]),
        ["Ex"],
        threshold=0.10,
    )

    assert result["floor_by_component"]["Ex"] == pytest.approx(0.01)
    assert result["max_robust_error_by_component"]["Ex"] == pytest.approx(0.5)


def test_signed_response_weak_sample_uses_only_peak_normalized_acceptance_error():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [0.05]]),
        np.array([[1.0], [0.0]]),
        ["Ex"],
        threshold=0.10,
    )

    weak_row = result["rows"][1]
    assert weak_row["response_strength"] == "weak"
    assert weak_row["relative_error_with_floor"] == pytest.approx(5.0)
    assert weak_row["peak_normalized_error"] == pytest.approx(0.05)
    assert weak_row["acceptance_error"] == pytest.approx(0.05)
    assert weak_row["pass_threshold"]
    assert weak_row["pass_5pct"]
    assert result["max_acceptance_error_by_component"]["Ex"] == pytest.approx(0.05)
    assert result["summary"]["amplitude_pass_all_components"] is True
    assert result["summary"]["pass_all_components"] is True
    assert result["summary"]["legacy_pass_5pct_is_threshold_alias"] is True


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 0.0, -1.0], [2.0]),
        ([1.0, 0.0, 1.0], []),
        ([1.0, 0.0, 0.0, -1.0], [2.5]),
        ([1.0, 0.0, 0.0, 1.0], []),
        ([0.0, 0.0, -1.0, -2.0], []),
        ([1.0, 2.0, 0.0, 0.0], []),
    ],
)
def test_linear_zero_crossings_handle_exact_zero_runs_once(values, expected):
    times = np.arange(1.0, len(values) + 1.0)

    crossings = linear_zero_crossings(times, np.asarray(values))

    assert crossings.tolist() == pytest.approx(expected)
    assert np.all(np.isfinite(crossings))
    assert np.all(np.diff(crossings) > 0.0)


def test_linear_zero_crossings_returns_independent_array():
    times = np.array([1.0, 2.0])
    values = np.array([1.0, -1.0])

    crossings = linear_zero_crossings(times, values)
    crossings[0] = 99.0

    assert times.tolist() == [1.0, 2.0]
    assert values.tolist() == [1.0, -1.0]


@pytest.mark.parametrize(
    ("times", "values"),
    [
        ([], []),
        ([[1.0]], [1.0]),
        ([1.0], [[1.0]]),
        ([1.0, 2.0], [1.0]),
        ([0.0, 1.0], [1.0, -1.0]),
        ([1.0, 1.0], [1.0, -1.0]),
        ([2.0, 1.0], [1.0, -1.0]),
        ([1.0, np.inf], [1.0, -1.0]),
        ([1.0, 2.0], [1.0, np.nan]),
        ([1.0, 2.0], [1.0 + 1.0j, -1.0]),
        ([1.0, 2.0], ["1.0", "-1.0"]),
    ],
)
def test_linear_zero_crossings_reject_invalid_arrays(times, values):
    with pytest.raises((TypeError, ValueError)):
        linear_zero_crossings(times, values)


def test_signed_response_reports_matched_crossing_time_error():
    result = compare_signed_response(
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0], [-1.0], [-1.0]]),
        np.array([[1.0], [-3.0], [-3.0]]),
        ["Ex"],
        threshold=0.10,
    )

    crossing = result["zero_crossings"]["Ex"]
    assert crossing["prediction"] == pytest.approx([1.5])
    assert crossing["reference"] == pytest.approx([1.25])
    assert crossing["count_match"] is True
    assert crossing["max_relative_time_error"] == pytest.approx(0.2)
    assert crossing["passed"] is False


def test_signed_response_reports_crossing_count_mismatch_as_infinity():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [-1.0]]),
        np.array([[1.0], [1.0]]),
        ["Ex"],
        threshold=0.10,
    )

    crossing = result["zero_crossings"]["Ex"]
    assert crossing["count_match"] is False
    assert crossing["max_relative_time_error"] == float("inf")
    assert crossing["passed"] is False
    assert result["summary"]["zero_crossings_pass_all_components"] is False


def test_signed_response_reports_zero_error_when_neither_curve_crosses_zero():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [2.0]]),
        np.array([[1.0], [2.0]]),
        ["Ex"],
        threshold=0.10,
    )

    crossing = result["zero_crossings"]["Ex"]
    assert crossing == {
        "prediction": [],
        "reference": [],
        "count_match": True,
        "max_relative_time_error": 0.0,
        "passed": True,
    }


def test_signed_response_zero_crossing_count_is_hard_gate_after_amplitude_passes():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[1.0], [-0.0004]]),
        np.array([[1.0], [0.0004]]),
        ["Ex"],
        threshold=0.10,
    )

    assert result["summary"]["amplitude_pass_all_components"] is True
    assert result["summary"]["zero_crossings_pass_all_components"] is False
    assert result["summary"]["zero_crossing_failed_components"] == ["Ex"]
    assert result["summary"]["pass_all_components"] is False
    assert result["summary"]["failed_components"] == ["Ex"]


def test_signed_response_crossing_time_over_five_percent_is_hard_gate():
    result = compare_signed_response(
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0], [-0.009], [-0.01]]),
        np.array([[1.0], [0.009], [-0.01]]),
        ["Ex"],
        threshold=0.10,
    )

    crossing = result["zero_crossings"]["Ex"]
    assert result["summary"]["amplitude_pass_all_components"] is True
    assert crossing["count_match"] is True
    assert crossing["max_relative_time_error"] > 0.05
    assert crossing["passed"] is False
    assert result["summary"]["zero_crossing_time_tolerance"] == 0.05
    assert result["summary"]["pass_all_components"] is False


def test_signed_response_preserves_negative_and_near_zero_signed_samples():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[-2.0], [0.009]]),
        np.array([[-2.0], [-0.001]]),
        ["Ex"],
        threshold=0.10,
    )

    assert result["floor_by_component"]["Ex"] == pytest.approx(0.02)
    assert result["max_robust_error_by_component"]["Ex"] == pytest.approx(0.5)
    assert result["rows"][1]["pred"] == pytest.approx(0.009)
    assert result["rows"][1]["ref"] == pytest.approx(-0.001)


def test_signed_response_rejects_zero_reference_peak():
    with pytest.raises(ValueError, match="reference peak.*positive"):
        compare_signed_response(
            np.array([1.0]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            ["Ex"],
            threshold=0.10,
        )


@pytest.mark.parametrize(
    ("times", "prediction", "reference", "components", "threshold"),
    [
        ([], np.empty((0, 1)), np.empty((0, 1)), ["Ex"], 0.10),
        ([[1.0]], [[1.0]], [[1.0]], ["Ex"], 0.10),
        ([0.0], [[1.0]], [[1.0]], ["Ex"], 0.10),
        ([1.0, 1.0], [[1.0], [1.0]], [[1.0], [1.0]], ["Ex"], 0.10),
        ([2.0, 1.0], [[1.0], [1.0]], [[1.0], [1.0]], ["Ex"], 0.10),
        ([1.0, np.inf], [[1.0], [1.0]], [[1.0], [1.0]], ["Ex"], 0.10),
        ([1.0], [1.0], [[1.0]], ["Ex"], 0.10),
        ([1.0], [[1.0]], [1.0], ["Ex"], 0.10),
        ([1.0], [[1.0, 2.0]], [[1.0]], ["Ex"], 0.10),
        ([1.0], [[np.nan]], [[1.0]], ["Ex"], 0.10),
        ([1.0], [[1.0]], [[np.inf]], ["Ex"], 0.10),
        ([1.0], [[1.0 + 1.0j]], [[1.0]], ["Ex"], 0.10),
        ([1.0], [["1.0"]], [[1.0]], ["Ex"], 0.10),
        ([1.0], [[1.0]], [[1.0]], [], 0.10),
        ([1.0], [[1.0]], [[1.0]], [""], 0.10),
        ([1.0], [[1.0, 1.0]], [[1.0, 1.0]], ["Ex", "Ex"], 0.10),
        ([1.0], [[1.0]], [[1.0]], [1], 0.10),
        ([1.0], [[1.0]], [[1.0]], ["Ex"], 0.0),
        ([1.0], [[1.0]], [[1.0]], ["Ex"], np.inf),
        ([1.0], [[1.0]], [[1.0]], ["Ex"], True),
        ([1.0], [[1.0, 1.0]], [[1.0, 1.0]], "Ex", 0.10),
        ([1.0], [[1.0, 1.0]], [[1.0, 1.0]], b"Ex", 0.10),
    ],
)
def test_signed_response_rejects_invalid_inputs(
    times, prediction, reference, components, threshold
):
    with pytest.raises((TypeError, ValueError)):
        compare_signed_response(
            times,
            prediction,
            reference,
            components,
            threshold=threshold,
        )


def test_signed_response_catches_sign_inversion_without_absolute_value_masking():
    result = compare_signed_response(
        np.array([1.0, 2.0]),
        np.array([[-1.0], [1.0]]),
        np.array([[1.0], [-1.0]]),
        ["Ex"],
        threshold=0.10,
    )

    assert result["summary"]["pass_all_components"] is False
    assert result["max_robust_error_by_component"]["Ex"] == pytest.approx(2.0)
    assert result["rows"][0]["pred"] == -1.0
    assert result["rows"][0]["ref"] == 1.0
