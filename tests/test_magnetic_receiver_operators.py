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
