import numpy as np
import pytest

from atem3d.zhou2020_reference_stability import (
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


def test_build_reference_stability_audit_retains_nonconverged_evidence():
    times, default, separate, direct = _reference_candidates()
    result = build_reference_stability_audit(
        times=times,
        default_dlf=default,
        separate_total_qwe=separate,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=1.05 * direct,
        signal_to_spread=3.0,
        consecutive=3,
    )

    assert result["schema"] == "atem3d.zhou2020.reference-stability/v1"
    assert result["status"] == "inconclusive"
    assert result["qwe"]["converged"] is False
    assert result["all_samples_retained"] is True
    assert result["sample_count"] == times.size
    assert result["default_dlf"]["sign_changes_first20"] == 2
    assert result["stable_window"]["start_s"] == times[3]
    assert result["fenicsx_vs_direct_qwe"]["relative_l2_full"] == pytest.approx(0.05)


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

    assert result["formal_gate_decision"] is None
