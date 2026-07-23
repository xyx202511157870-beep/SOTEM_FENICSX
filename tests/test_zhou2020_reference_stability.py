import numpy as np
import pytest

from atem3d.zhou2020_reference_stability import (
    _relative_l2,
    build_reference_stability_audit,
    first_stable_sample,
    sign_change_count,
)


def _reference_candidates():
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.array([0.1, 0.1, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    default = direct + np.array([0.2, -0.2, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02])
    separate = direct + np.array([-0.2, 0.2, -0.2, -0.02, -0.02, -0.02, -0.02, -0.02])
    return times, default, separate, direct


def _long_audit_inputs():
    times = np.geomspace(1.0e-4, 1.0e-1, 25)
    direct = np.arange(1.0, 26.0)
    default = direct * np.r_[np.full(20, 1.1), np.full(5, 1.2)]
    default[:3] = [3.0, -3.0, 3.0]
    separate = direct * np.r_[np.full(20, -1.0), np.full(5, 0.99)]
    fenicsx = direct * np.r_[np.full(20, 1.3), np.full(5, 1.4)]
    return times, default, separate, direct, fenicsx


def test_sign_change_count_ignores_exact_zeros():
    assert sign_change_count([1.0, 0.0, -2.0, -3.0, 0.0, 4.0]) == 2


def test_first_stable_sample_requires_consecutive_signal_dominance():
    times, default, separate, direct = _reference_candidates()

    assert first_stable_sample(
        times,
        np.column_stack([default, separate, direct]),
        signal_to_spread=3.0,
        consecutive=3,
    ) == 3


def test_first_stable_sample_rejects_boolean_signal_to_spread():
    times, default, separate, direct = _reference_candidates()

    with pytest.raises(ValueError, match="signal_to_spread"):
        first_stable_sample(
            times,
            np.column_stack([default, separate, direct]),
            signal_to_spread=True,
        )


def test_build_reference_stability_audit_retains_nonconverged_evidence():
    times, default, separate, direct, fenicsx = _long_audit_inputs()
    result = build_reference_stability_audit(
        times=times,
        default_dlf=default,
        separate_total_qwe=separate,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=fenicsx,
        signal_to_spread=3.0,
        consecutive=3,
    )

    assert result["schema"] == "atem3d.zhou2020.reference-stability/v1"
    assert result["status"] == "inconclusive"
    assert result["qwe"]["converged"] is False
    assert result["all_samples_retained"] is True
    assert result["sample_count"] == times.size
    assert result["default_dlf"]["sign_changes_first20"] == 2
    assert result["stable_window"]["start_s"] == times[20]
    assert result["transform_difference"][
        "default_dlf_vs_direct_qwe_relative_l2_full"
    ] == pytest.approx(np.linalg.norm(default - direct) / np.linalg.norm(direct))
    assert result["transform_difference"][
        "default_dlf_vs_direct_qwe_relative_l2_first20"
    ] == pytest.approx(
        np.linalg.norm(default[:20] - direct[:20]) / np.linalg.norm(direct[:20])
    )
    assert result["fenicsx_vs_direct_qwe"]["relative_l2_full"] == pytest.approx(
        np.linalg.norm(fenicsx - direct) / np.linalg.norm(direct)
    )
    assert result["fenicsx_vs_direct_qwe"]["relative_l2_stable_window"] == pytest.approx(
        np.linalg.norm(fenicsx[20:] - direct[20:]) / np.linalg.norm(direct[20:])
    )


def test_nonconverged_qwe_cannot_be_promoted_to_passed():
    times, default, separate, direct = _reference_candidates()
    result = build_reference_stability_audit(
        times=times,
        default_dlf=default,
        separate_total_qwe=separate,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=direct,
    )

    assert result["status"] == "inconclusive"
    assert result["qwe"]["converged"] is False
    assert result["formal_gate_decision"] is None


@pytest.mark.parametrize("value", ["False", 1, np.nan, [False]])
def test_build_reference_stability_audit_rejects_nonboolean_qwe_convergence(value):
    times, default, separate, direct = _reference_candidates()

    with pytest.raises((TypeError, ValueError), match="direct_qwe_converged"):
        build_reference_stability_audit(
            times=times,
            default_dlf=default,
            separate_total_qwe=separate,
            direct_frequency_qwe=direct,
            direct_qwe_converged=value,
            fenicsx_increment=direct,
        )


def test_relative_l2_remains_finite_for_large_finite_values():
    result = _relative_l2(
        np.array([1.0e308, -1.0e308]),
        np.array([1.0e308, 1.0e308]),
    )

    assert np.isfinite(result)
    assert result == pytest.approx(np.sqrt(2.0))
