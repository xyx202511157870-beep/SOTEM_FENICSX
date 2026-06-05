import numpy as np
import pytest

from atem3d.metrics import robust_component_errors, robust_relative_error


def test_robust_relative_error_uses_floor_for_near_zero_reference():
    assert robust_relative_error(1.0e-15, 0.0, 1.0e-14) == pytest.approx(0.1)


def test_robust_component_errors_report_all_required_columns():
    times = np.array([1.0e-5, 1.0e-4])
    pred = np.array([[1.1, 1.0e-15], [2.2, 2.0e-15]])
    ref = np.array([[1.0, 0.0], [2.0, 0.0]])

    rows, summary = robust_component_errors(
        times,
        pred,
        ref,
        ["Ex", "Ey"],
        threshold=0.05,
    )

    assert set(rows.dtype.names) == {
        "time_obs",
        "component",
        "pred",
        "ref",
        "abs_error",
        "ordinary_relative_error",
        "relative_error_with_floor",
        "peak_normalized_error",
        "pass_5pct",
    }
    assert summary["max_error_Ex"] == pytest.approx(0.1)
    assert summary["max_error_Ey"] == pytest.approx(0.2)
    assert summary["pass_all_components"] is False


def test_robust_component_errors_reports_physical_pass_for_weak_horizontal_component():
    times = np.array([1.0e-5, 1.0e-4])
    ref = np.array(
        [
            [10.0, 1.0e-12, 2.0e-9],
            [5.0, 2.0e-12, 1.0e-9],
        ]
    )
    pred = ref.copy()
    pred[:, 1] += np.array([0.1, -0.2])

    rows, summary = robust_component_errors(
        times,
        pred,
        ref,
        ["Ex", "Ey", "dBzdt"],
        threshold=0.05,
    )

    ey_rows = rows[rows["component"] == "Ey"]
    assert bool(np.all(ey_rows["pass_5pct"])) is False
    assert summary["pass_all_components"] is False
    assert summary["physical_pass_all_components"] is True
    assert summary["weak_component_passed"] is True
    assert summary["weak_components"] == ["Ey"]
    assert summary["physical_failed_components"] == []


def test_robust_component_errors_keeps_hz_and_dbdt_errors_separate():
    times = np.asarray([1.0e-5])
    ref = np.asarray([[1.0, 0.0, 2.0, 4.0]])
    pred = np.asarray([[1.0, 0.0, 3.0, 4.4]])

    _rows, summary = robust_component_errors(
        times,
        pred,
        ref,
        ["Ex", "Ey", "Hz", "dBzdt"],
        threshold=0.05,
    )

    assert summary["magnetic_quantity"] == "dBzdt"
    assert summary["magnetic_components"] == ["Hz", "dBzdt"]
    assert summary["max_error_Hz"] == pytest.approx(0.5)
    assert summary["max_error_dBzdt"] == pytest.approx(0.1)
    assert summary["max_error_Hz_or_dBzdt"] == pytest.approx(0.1)
