import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.bootstrap import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP,
    bootstrap_candidate_comparison,
    bootstrap_case_statistic,
    case_resample_indices,
    paired_case_bootstrap,
)
from atem3d.adaptive_debye_mvp.receiver_metrics import CaseMetrics, evaluate_candidate


def _case(case_id, total_p95, *, n_valid=10, passed=True):
    return CaseMetrics(
        case_id=case_id,
        groups=(),
        output_dt=1.0e-4,
        channels={},
        case_total_median=total_p95,
        case_total_p95=total_p95,
        case_total_nrmse_by_channel={},
        case_ip_increment_nrmse=0.01,
        peak_time_error_steps_max=0.0,
        zero_crossing_time_error_steps_max=0.0,
        unexplained_sign_flips=0,
        n_valid_samples=n_valid,
        passed=passed,
        failure_reasons=(),
    )


def test_resample_indices_are_case_level():
    indices = case_resample_indices(5, 10, 123)
    assert indices.shape == (10, 5)
    assert np.all((indices >= 0) & (indices < 5))
    np.testing.assert_array_equal(indices, case_resample_indices(5, 10, 123))
    assert not np.array_equal(indices, case_resample_indices(5, 10, 124))


def test_default_seed_and_size():
    result = bootstrap_case_statistic(np.array([0.1, 0.2, 0.3, 0.4]))
    assert result.n_bootstrap == DEFAULT_N_BOOTSTRAP == 2000
    assert result.seed == DEFAULT_BOOTSTRAP_SEED == 202609116
    assert result.replicates.shape == (2000,)


def test_replicates_only_take_case_subset_values():
    result = bootstrap_case_statistic(np.array([0.1, 0.2, 0.3]), statistic="max", n_bootstrap=200, seed=7)
    assert set(np.unique(result.replicates)).issubset({0.1, 0.2, 0.3})


def test_time_sample_permutation_does_not_change_bootstrap():
    left_cases = [_case("a", 0.11, n_valid=10), _case("b", 0.22, n_valid=10), _case("c", 0.33, n_valid=10)]
    right_cases = [_case("a", 0.11, n_valid=1000), _case("b", 0.22, n_valid=1000), _case("c", 0.33, n_valid=1000)]
    left = evaluate_candidate("L", 8, left_cases, spectral_error=0.01, condition_number=3.0)
    right = evaluate_candidate("R", 8, right_cases, spectral_error=0.01, condition_number=3.0)
    other = evaluate_candidate(
        "O",
        8,
        [_case("a", 0.12, n_valid=10), _case("b", 0.23, n_valid=10), _case("c", 0.34, n_valid=10)],
        spectral_error=0.01,
        condition_number=3.0,
    )
    first = bootstrap_candidate_comparison(left, other, n_bootstrap=64, seed=11)
    second = bootstrap_candidate_comparison(right, other, n_bootstrap=64, seed=11)
    np.testing.assert_allclose(first.replicates, second.replicates)
    assert first.n_cases == second.n_cases == 3


def test_two_d_values_rejected():
    with pytest.raises(ValueError, match="one statistic per case"):
        bootstrap_case_statistic(np.ones((3, 100)))


def test_paired_bootstrap_uses_same_indices():
    values = np.array([0.10, 0.20, 0.30, 0.40])
    result = paired_case_bootstrap(values, values + 0.01, statistic="mean", n_bootstrap=50, seed=3)
    np.testing.assert_allclose(result.replicates, -0.01, atol=1.0e-15)
    assert result.difference == pytest.approx(-0.01)
    assert result.probability_a_better == 1.0
    assert result.significant


def test_candidate_comparison_requires_matching_cases():
    left = evaluate_candidate("L", 8, [_case("a", 0.1), _case("b", 0.2)], spectral_error=0.01, condition_number=2.0)
    right = evaluate_candidate("R", 8, [_case("a", 0.1), _case("c", 0.2)], spectral_error=0.01, condition_number=2.0)
    with pytest.raises(ValueError):
        bootstrap_candidate_comparison(left, right, n_bootstrap=8, seed=1)


def test_ci_contains_point_estimate_for_median():
    rng = np.random.default_rng(0)
    values = rng.normal(size=20)
    wide = bootstrap_case_statistic(values, statistic="median", n_bootstrap=400, seed=9, confidence_level=0.99)
    narrow = bootstrap_case_statistic(values, statistic="median", n_bootstrap=400, seed=9, confidence_level=0.9)
    assert wide.ci_low <= wide.point_estimate <= wide.ci_high
    assert (narrow.ci_high - narrow.ci_low) < (wide.ci_high - wide.ci_low)
