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

