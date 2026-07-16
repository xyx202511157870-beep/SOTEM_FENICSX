from __future__ import annotations

import numpy as np
import pytest

from atem3d.magnetic_symmetry_audit import (
    audit_model_triplet,
    audit_receiver_symmetry,
    evaluate_magnetic_gates,
)


COMPONENTS = ("Ex", "dBzdt", "Hz")


def _exact_parity_values(time_count: int = 2) -> np.ndarray:
    values = np.zeros((5, time_count, 3), dtype=float)
    values[:, :, 0] = np.array([[1], [2], [3], [2], [1]])
    values[:, :, 1] = np.array([[-4], [-2], [0], [2], [4]])
    values[:, :, 2] = np.array([[-8], [-3], [0], [3], [8]])
    return values


def test_exact_odd_magnetic_and_even_electric_fields_have_zero_residual() -> None:
    times = np.array([1.0e-5, 2.0e-5])

    audit = audit_receiver_symmetry(
        _exact_parity_values(),
        components=COMPONENTS,
        times=times,
    )

    assert audit["Ex"]["parity"] == "even"
    assert audit["Ex"]["pair_24_residual"] == 0.0
    assert audit["dBzdt"]["parity"] == "odd"
    assert audit["dBzdt"]["rx3_zero_ratio"] == 0.0
    assert audit["dBzdt"]["pair_24_residual"] == 0.0
    assert audit["Hz"]["pair_15_residual"] == 0.0


def test_zero_denominator_returns_absolute_residual_without_infinite_ratio() -> None:
    values = np.zeros((5, 3, 3), dtype=float)
    values[2, :, 1] = 2.5e-12

    audit = audit_receiver_symmetry(values, COMPONENTS)

    assert audit["dBzdt"]["rx3_abs_max"] == 2.5e-12
    assert audit["dBzdt"]["rx3_zero_ratio"] is None


def test_triplet_audits_signed_channel_minus_background() -> None:
    background = np.zeros((5, 2, 3), dtype=float)
    channel = background.copy()
    channel[:, :, 1] = np.array([[-2], [-1], [0], [1], [2]])

    audit = audit_model_triplet(background, channel, COMPONENTS)

    assert audit["delta"]["dBzdt"]["rx3_zero_ratio"] == 0.0
    assert audit["delta"]["dBzdt"]["pair_24_residual"] == 0.0
    assert audit["delta"]["dBzdt"]["pair_15_residual"] == 0.0


def test_missing_scale_fails_magnetic_gate() -> None:
    metrics = {
        "Ex": {
            "rx3_zero_ratio": 99.0,
            "pair_24_residual": 0.0,
            "pair_15_residual": 0.0,
        },
        "dBzdt": {
            "rx3_zero_ratio": None,
            "pair_24_residual": 0.0,
            "pair_15_residual": 0.0,
        },
        "Hz": {
            "rx3_zero_ratio": 0.0,
            "pair_24_residual": 0.0,
            "pair_15_residual": 0.0,
        },
    }

    result = evaluate_magnetic_gates(metrics, threshold=0.01)

    assert result["passed"] is False
    assert "dBzdt.rx3_zero_ratio" in result["failures"]
    assert not any(item.startswith("Ex.") for item in result["failures"])


def test_magnetic_gate_accepts_values_at_threshold() -> None:
    metrics = {
        component: {
            "rx3_zero_ratio": 0.01,
            "pair_24_residual": 0.01,
            "pair_15_residual": 0.01,
        }
        for component in ("dBzdt", "Hz")
    }

    assert evaluate_magnetic_gates(metrics, threshold=0.01) == {
        "passed": True,
        "threshold": 0.01,
        "failures": [],
    }


@pytest.mark.parametrize(
    "values,components,times,message",
    [
        (np.zeros((4, 2, 3)), COMPONENTS, None, "shape"),
        (np.zeros((5, 2, 2)), COMPONENTS, None, "shape"),
        (np.full((5, 2, 3), np.nan), COMPONENTS, None, "finite"),
        (np.zeros((5, 2, 3)), COMPONENTS, np.array([1.0]), "times"),
    ],
)
def test_receiver_symmetry_rejects_invalid_payloads(
    values: np.ndarray,
    components: tuple[str, ...],
    times: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_receiver_symmetry(values, components, times)
