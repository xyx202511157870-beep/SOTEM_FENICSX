import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.receiver_metrics import (
    CHANNEL_UNITS,
    CHANNELS,
    CaseMetrics,
    MetricThresholds,
    ReceiverCase,
    effective_mask,
    evaluate_candidate,
    evaluate_case,
    ip_increment_nrmse,
    is_qualifying,
    peak_errors,
    qualifying_failures,
    rank_candidates,
    ranking_key,
    robust_total_field_error,
    unexplained_sign_flips,
    zero_crossing_absolute_time_error,
    zero_crossing_times,
)
from atem3d.metrics import robust_relative_error


REF10 = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.005, 0.0005])
TIMES10 = np.linspace(1.0e-4, 1.0e-3, REF10.size)


def _single_channel_case(candidate, *, baseline=None, channel="Hz", groups=(), output_dt=None, reference=None):
    reference = REF10 if reference is None else np.asarray(reference, dtype=float)
    payload = {channel: np.asarray(candidate, dtype=float)}
    return ReceiverCase(
        case_id="case0",
        times=TIMES10 if reference.size == TIMES10.size else np.linspace(0.0, 1.0, reference.size),
        reference={channel: reference},
        candidate=payload,
        reference_no_ip=None if baseline is None else {channel: np.asarray(baseline, dtype=float)},
        groups=groups,
        output_dt=output_dt,
    )


def _case_metrics(
    case_id,
    *,
    passed,
    total_p95,
    ip_nrmse=0.0,
    median=0.0,
    groups=(),
    peak_steps=0.0,
    zc_steps=0.0,
    sign_flips=0,
):
    return CaseMetrics(
        case_id=case_id,
        groups=groups,
        output_dt=1.0e-4,
        channels={},
        case_total_median=median,
        case_total_p95=total_p95,
        case_total_nrmse_by_channel={},
        case_ip_increment_nrmse=ip_nrmse,
        peak_time_error_steps_max=peak_steps,
        zero_crossing_time_error_steps_max=zc_steps,
        unexplained_sign_flips=sign_flips,
        n_valid_samples=10,
        passed=passed,
        failure_reasons=() if passed else ("total_p95",),
    )


def test_effective_mask_and_scale_hand_computed():
    mask, scale = effective_mask(REF10)
    assert scale == pytest.approx(9.1, rel=1.0e-12)
    assert np.array_equal(mask, np.array([True] * 8 + [False, False]))


def test_ordinary_point_relative_error():
    error = robust_total_field_error([2.1], [2.0], amplitude_scale=10.0, alpha=1.0e-2, d_floor=1.0e-16)
    assert error[0] == pytest.approx(0.05)


def test_near_zero_point_uses_alpha_floor():
    error = robust_total_field_error([6.0e-3], [5.0e-3], amplitude_scale=1.0, alpha=1.0e-2, d_floor=1.0e-16)
    assert error[0] == pytest.approx(0.1)
    assert error[0] == pytest.approx(robust_relative_error(6.0e-3, 5.0e-3, 1.0e-2))


def test_total_field_case_metrics_hand_computed():
    metrics = evaluate_case(_single_channel_case(1.01 * REF10))
    assert metrics.case_total_median == pytest.approx(0.01)
    assert metrics.case_total_p95 == pytest.approx(0.01)
    assert metrics.n_valid_samples == 8
    assert metrics.channels["Hz"].total_nrmse == pytest.approx(0.01)


def test_ip_increment_nrmse_hand_computed():
    metrics = evaluate_case(_single_channel_case(1.01 * REF10, baseline=0.9 * REF10))
    assert metrics.case_ip_increment_nrmse == pytest.approx(0.1, rel=1.0e-12)
    assert metrics.passed is False
    assert "ip_increment_nrmse" in metrics.failure_reasons
    missing = evaluate_case(_single_channel_case(1.01 * REF10))
    assert np.isnan(missing.case_ip_increment_nrmse)
    assert "ip_increment_nrmse" not in missing.failure_reasons


def test_ip_increment_floor_bounds_error():
    mask, scale = effective_mask(REF10)
    increment = ip_increment_nrmse(
        1.01 * REF10,
        REF10,
        REF10,
        mask=mask,
        amplitude_scale=scale,
        ip_floor_fraction=1.0e-2,
    )
    residual = 0.01 * REF10[mask]
    expected = float(np.sqrt(np.mean(residual**2)) / (1.0e-2 * scale))
    assert increment == pytest.approx(expected)
    assert np.isfinite(increment)


def test_peak_time_error_one_step():
    times = np.arange(7) * 1.0e-4
    reference = np.array([1.0, 1.0, 1.0, 1.0001, 1.0, 1.0, 1.0])
    one_step = np.array([1.0, 1.0, 1.0, 1.0, 1.0001, 1.0, 1.0])
    two_step = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0001, 1.0])
    amplitude, time_error = peak_errors(times, one_step, reference, mask=np.ones(7, dtype=bool))
    assert time_error == pytest.approx(1.0e-4)
    assert amplitude == pytest.approx(0.0)
    one = evaluate_case(
        ReceiverCase(
            case_id="peak1",
            times=times,
            reference={"Hz": reference},
            candidate={"Hz": one_step},
            output_dt=1.0e-4,
        )
    )
    assert one.peak_time_error_steps_max == pytest.approx(1.0)
    assert "peak_time" not in one.failure_reasons
    two = evaluate_case(
        ReceiverCase(
            case_id="peak2",
            times=times,
            reference={"Hz": reference},
            candidate={"Hz": two_step},
            output_dt=1.0e-4,
        )
    )
    assert "peak_time" in two.failure_reasons


def test_zero_crossing_time_error_linear_interpolation():
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    reference = times - 0.5
    candidate = times - 0.55
    error, n_stable = zero_crossing_absolute_time_error(
        times,
        candidate,
        reference,
        amplitude_scale=float(np.quantile(np.abs(reference), 0.95)),
        alpha=1.0e-2,
    )
    assert error == pytest.approx(0.05, rel=1.0e-12)
    assert n_stable == 1
    none, n_none = zero_crossing_absolute_time_error(
        times,
        candidate,
        np.ones_like(times),
        amplitude_scale=1.0,
        alpha=1.0e-2,
    )
    assert np.isnan(none)
    assert n_none == 0
    lost, n_lost = zero_crossing_absolute_time_error(
        times,
        np.ones_like(times),
        reference,
        amplitude_scale=float(np.quantile(np.abs(reference), 0.95)),
        alpha=1.0e-2,
    )
    assert np.isinf(lost)
    assert n_lost == 1
    np.testing.assert_allclose(zero_crossing_times(times, reference), [0.5])


def test_unexplained_sign_flips():
    mask, scale = effective_mask(REF10)
    assert unexplained_sign_flips(-REF10, REF10, mask=mask, amplitude_scale=scale, alpha=1.0e-2) == 8
    flipped = REF10.copy()
    flipped[-2:] *= -1.0
    assert unexplained_sign_flips(flipped, REF10, mask=mask, amplitude_scale=scale, alpha=1.0e-2) == 0


def test_six_channel_unit_separation():
    base = REF10
    reference = {name: (1.0e-3 if name.startswith("H") else 1.0e-9) * base for name in CHANNELS}
    candidate = {name: 1.01 * values for name, values in reference.items()}
    metrics = evaluate_case(
        ReceiverCase(
            case_id="six",
            times=TIMES10,
            reference=reference,
            candidate=candidate,
        )
    )
    scales = [metrics.channels[name].amplitude_scale for name in CHANNELS]
    for name in CHANNELS:
        assert metrics.channels[name].total_p95 == pytest.approx(0.01)
        assert metrics.channels[name].unit == CHANNEL_UNITS[name]
    assert scales[0] / scales[3] == pytest.approx(1.0e6)
    with pytest.raises(ValueError):
        evaluate_case(
            ReceiverCase(
                case_id="bad",
                times=TIMES10,
                reference={"Ex": base},
                candidate={"Ex": 1.01 * base},
            )
        )


def test_all_zero_reference_channel_is_skipped():
    reference = {"Hz": REF10, "Hy": np.zeros_like(REF10)}
    candidate = {"Hz": REF10.copy(), "Hy": np.zeros_like(REF10)}
    metrics = evaluate_case(ReceiverCase(case_id="hy0", times=TIMES10, reference=reference, candidate=candidate))
    assert metrics.channels["Hy"].n_masked == 0
    assert np.isnan(metrics.channels["Hy"].total_p95)
    assert metrics.passed
    empty = evaluate_case(
        ReceiverCase(
            case_id="empty",
            times=TIMES10,
            reference={"Hz": np.zeros_like(REF10)},
            candidate={"Hz": np.zeros_like(REF10)},
        )
    )
    assert empty.passed is False
    assert empty.failure_reasons == ("empty_mask",)


def test_ranking_key_lexicographic():
    better_fail = evaluate_candidate(
        "b",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.02), _case_metrics("c1", passed=True, total_p95=0.02)],
        spectral_error=0.05,
        condition_number=10.0,
    )
    worse_fail = evaluate_candidate(
        "a",
        8,
        [_case_metrics("c0", passed=False, total_p95=0.001), _case_metrics("c1", passed=True, total_p95=0.001)],
        spectral_error=0.01,
        condition_number=2.0,
    )
    ranked = rank_candidates([worse_fail, better_fail])
    assert ranked[0].candidate_id == "b"
    assert ranking_key(better_fail)[0] < ranking_key(worse_fail)[0]
    tied_z = evaluate_candidate(
        "z",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.02), _case_metrics("c1", passed=True, total_p95=0.02)],
        spectral_error=0.05,
        condition_number=10.0,
    )
    tied_m = evaluate_candidate(
        "m",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.02), _case_metrics("c1", passed=True, total_p95=0.02)],
        spectral_error=0.05,
        condition_number=10.0,
    )
    assert [item.candidate_id for item in rank_candidates([tied_z, tied_m])] == ["m", "z"]
    nan_spectral = evaluate_candidate(
        "nan",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.02), _case_metrics("c1", passed=True, total_p95=0.02)],
        spectral_error=float("nan"),
        condition_number=10.0,
    )
    assert rank_candidates([nan_spectral, better_fail])[0].candidate_id == "b"


def test_qualifying_predicate():
    passing_cases = [
        _case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
        _case_metrics("c1", passed=True, total_p95=0.005, ip_nrmse=0.012, groups=("g",)),
        _case_metrics("c2", passed=True, total_p95=0.006, ip_nrmse=0.011, groups=("g",)),
        _case_metrics("c3", passed=True, total_p95=0.005, ip_nrmse=0.01, groups=("g",)),
        _case_metrics("c4", passed=True, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
    ]
    passing = evaluate_candidate("ok", 8, passing_cases, spectral_error=0.01, condition_number=5.0)
    assert is_qualifying(passing)
    assert qualifying_failures(passing) == ()

    high_total = evaluate_candidate(
        "tot",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.02, ip_nrmse=0.01)],
        spectral_error=0.01,
        condition_number=5.0,
    )
    assert "total_p95_q95" in qualifying_failures(high_total)

    missing_ip = evaluate_candidate(
        "noip",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=float("nan"))],
        spectral_error=0.01,
        condition_number=5.0,
    )
    assert "ip_increment_unavailable" in qualifying_failures(missing_ip)

    high_ip = evaluate_candidate(
        "ip",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=0.04)],
        spectral_error=0.01,
        condition_number=5.0,
    )
    assert "ip_increment_nrmse_q95" in qualifying_failures(high_ip)

    flips = evaluate_candidate(
        "flip",
        8,
        [_case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=0.01, sign_flips=1)],
        spectral_error=0.01,
        condition_number=5.0,
    )
    assert "sign_flips" in qualifying_failures(flips)

    small_group = evaluate_candidate(
        "tiny",
        8,
        [
            _case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
            _case_metrics("c1", passed=False, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
        ],
        spectral_error=0.01,
        condition_number=5.0,
    )
    assert "group_fail_rate" not in qualifying_failures(small_group)
    grouped = evaluate_candidate(
        "grp",
        8,
        [
            _case_metrics("c0", passed=True, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
            _case_metrics("c1", passed=False, total_p95=0.004, ip_nrmse=0.01, groups=("g",)),
        ],
        spectral_error=0.01,
        condition_number=5.0,
        thresholds=MetricThresholds(min_group_size=1),
    )
    assert "group_fail_rate" in qualifying_failures(grouped, MetricThresholds(min_group_size=1))
