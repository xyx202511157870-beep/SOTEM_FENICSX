from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load_module():
    path = Path("dolfinx/magnetic_receiver_operators.py")
    spec = importlib.util.spec_from_file_location(
        "magnetic_receiver_operators_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_horizontal_loop_points_and_tangents_are_pairwise_symmetric() -> None:
    module = load_module()

    rule = module.horizontal_loop_rule(
        (1.0, 2.0, 3.0),
        radius=2.0,
        point_count=32,
    )

    np.testing.assert_allclose(
        rule.points[:16, :2] + rule.points[16:, :2],
        np.repeat([[2.0, 4.0]], 16, axis=0),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        rule.tangents[:16] + rule.tangents[16:],
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(np.sum(rule.line_weights), 4.0 * np.pi)
    np.testing.assert_allclose(rule.points[:, 2], 3.0)


def test_constant_electric_field_has_zero_closed_loop_integral() -> None:
    module = load_module()
    rule = module.horizontal_loop_rule(
        (0.0, 0.0, 0.1),
        radius=2.0,
        point_count=32,
    )
    electric = np.repeat([[3.0, -4.0, 0.0]], 32, axis=0)

    value = module.faraday_loop_dbdt(electric, rule)

    assert abs(value) < 1.0e-14


def test_linear_field_matches_known_vertical_curl() -> None:
    module = load_module()
    curl_z = 7.5
    rule = module.horizontal_loop_rule(
        (0.0, 0.0, 0.1),
        radius=2.0,
        point_count=64,
    )
    x, y = rule.points[:, 0], rule.points[:, 1]
    electric = np.column_stack(
        (-0.5 * curl_z * y, 0.5 * curl_z * x, np.zeros_like(x))
    )

    np.testing.assert_allclose(
        module.faraday_loop_dbdt(electric, rule),
        -curl_z,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "radius,point_count,message",
    [
        (0.0, 32, "radius must be positive"),
        (-1.0, 32, "radius must be positive"),
        (2.0, 4, "multiple of 4 and at least 8"),
        (2.0, 10, "multiple of 4 and at least 8"),
    ],
)
def test_horizontal_loop_rule_rejects_invalid_geometry(
    radius: float,
    point_count: int,
    message: str,
) -> None:
    module = load_module()

    with pytest.raises(ValueError, match=message):
        module.horizontal_loop_rule(
            (0.0, 0.0, 0.1),
            radius=radius,
            point_count=point_count,
        )


def test_faraday_loop_rejects_nonfinite_electric_values() -> None:
    module = load_module()
    rule = module.horizontal_loop_rule(
        (0.0, 0.0, 0.1),
        radius=2.0,
        point_count=32,
    )
    electric = np.zeros((32, 3), dtype=float)
    electric[4, 1] = np.nan

    with pytest.raises(ValueError, match="finite and match"):
        module.faraday_loop_dbdt(electric, rule)


def test_tetra4_rule_integrates_constant_and_linear_coordinates() -> None:
    module = load_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    points, weights = module.tetra4_rule(vertices)

    np.testing.assert_allclose(weights.sum(), 1.0 / 6.0)
    np.testing.assert_allclose(
        np.sum(weights[:, None] * points, axis=0),
        [1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0],
    )
    assert np.all(points > 0.0)
    assert np.all(np.sum(points, axis=1) < 1.0)


def test_neumaier_vector_sum_retains_small_residual_after_cancellation() -> None:
    module = load_module()
    values = np.array(
        [
            [1.0e16, -1.0e16, 0.0],
            [1.0, -2.0, 3.0],
            [-1.0e16, 1.0e16, 0.0],
        ]
    )

    np.testing.assert_allclose(
        module.neumaier_vector_sum(values),
        [1.0, -2.0, 3.0],
    )


def test_biot_volume_integral_has_expected_sign_for_x_current_above_receiver() -> None:
    module = load_module()
    points = np.array([[0.0, 1.0, 0.0]])
    currents = np.array([[1.0, 0.0, 0.0]])

    result, audit = module.biot_savart_volume_h(
        receiver=(0.0, 0.0, 0.0),
        points=points,
        current_density=currents,
        weights=np.array([2.0]),
    )

    assert result[2] < 0.0
    assert audit["sample_count"] == 1
    assert audit["cancellation_ratio"][2] == 1.0


def test_biot_volume_integral_rejects_receiver_on_integration_point() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="coincides"):
        module.biot_savart_volume_h(
            receiver=(0.0, 0.0, 0.0),
            points=np.array([[0.0, 0.0, 0.0]]),
            current_density=np.array([[1.0, 0.0, 0.0]]),
            weights=np.array([1.0]),
        )
