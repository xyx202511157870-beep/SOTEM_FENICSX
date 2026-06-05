import numpy as np

import atem3d
from atem3d.metrics import (
    component_diagnostics,
    fit_linear_response_components,
    relative_l2,
    relative_linf,
    summarize_errors,
)


def test_relative_error_metrics_use_reference_norms():
    numerical = np.array([1.0, 2.0, 3.0])
    reference = np.array([1.0, 1.0, 1.0])

    assert relative_l2(numerical, reference) == np.sqrt(5.0) / np.sqrt(3.0)
    assert relative_linf(numerical, reference) == 2.0


def test_absolute_linf_is_available_from_package_namespace():
    assert atem3d.absolute_linf([1.0, 3.0], [1.5, 1.0]) == 2.0


def test_summarize_errors_reports_component_columns():
    numerical = np.array([[1.0, 2.0], [2.0, 4.0]])
    reference = np.array([[1.0, 1.0], [1.0, 2.0]])

    summary = summarize_errors(numerical, reference, ["Ex", "Hz"])

    assert set(summary) == {"Ex", "Hz"}
    assert summary["Ex"]["relative_linf"] == 1.0
    assert summary["Hz"]["relative_linf"] == 1.0
    assert summary["Ex"]["absolute_linf"] == 1.0
    assert summary["Hz"]["absolute_linf"] == 2.0


def test_component_diagnostics_report_scaling_and_endpoint_values():
    numerical = np.array([[2.0, -1.0], [4.0, -3.0], [6.0, -5.0]])
    reference = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])

    diagnostics = component_diagnostics(numerical, reference, ["scaled", "offset"])

    assert diagnostics["scaled"]["least_squares_scale_numerical_over_reference"] == 2.0
    assert diagnostics["scaled"]["relative_l2_after_optimal_scale"] == 0.0
    assert diagnostics["scaled"]["first_numerical"] == 2.0
    assert diagnostics["scaled"]["last_reference"] == 3.0
    assert diagnostics["offset"]["least_squares_scale_numerical_over_reference"] == -3.0
    assert diagnostics["offset"]["first_ratio_numerical_over_reference"] == -1.0


def test_fit_linear_response_components_recovers_known_weights():
    base = np.array(
        [
            [10.0, 5.0],
            [8.0, 4.0],
            [6.0, 3.0],
        ]
    )
    first = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    second = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    reference = base - 1.25 * first + 0.5 * second

    fit = fit_linear_response_components(
        base,
        [first, second],
        reference,
        signs=[-1.0, 1.0],
    )

    np.testing.assert_allclose(fit.weights, [1.25, 0.5])
    np.testing.assert_allclose(fit.fitted, reference)
    np.testing.assert_allclose(fit.residual, np.zeros_like(reference), atol=1.0e-14)


def test_fit_linear_response_components_can_use_time_mask():
    base = np.array([[100.0], [10.0], [10.0]])
    component = np.array([[50.0], [2.0], [4.0]])
    reference = np.array([[0.0], [9.0], [8.0]])

    fit = fit_linear_response_components(
        base,
        [component],
        reference,
        signs=[-1.0],
        mask=np.array([False, True, True]),
    )

    np.testing.assert_allclose(fit.weights, [0.5])
    np.testing.assert_allclose(fit.fitted, [[75.0], [9.0], [8.0]])
