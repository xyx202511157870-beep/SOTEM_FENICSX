"""Receiver-oriented total-field and IP-increment metrics for Debye MVP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atem3d.metrics import robust_relative_error


CHANNELS = ("Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt")
PROJECTED_CHANNELS = ("Hn", "dBndt")
CHANNEL_UNITS = {
    "Hx": "A/m",
    "Hy": "A/m",
    "Hz": "A/m",
    "dBxdt": "T/s",
    "dBydt": "T/s",
    "dBzdt": "T/s",
    "Hn": "A/m",
    "dBndt": "T/s",
}
DEFAULT_D_FLOOR = {
    "Hx": 1.0e-16,
    "Hy": 1.0e-16,
    "Hz": 1.0e-16,
    "dBxdt": 1.0e-18,
    "dBydt": 1.0e-18,
    "dBzdt": 1.0e-18,
    "Hn": 1.0e-16,
    "dBndt": 1.0e-18,
}


def _as_1d(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validated_times(times) -> np.ndarray:
    values = _as_1d("times", times)
    if values.size < 2:
        raise ValueError("times must contain at least two samples")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("times must be strictly increasing")
    return values


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _nan_to_inf(value: float) -> float:
    number = float(value)
    return number if np.isfinite(number) else float("inf")


@dataclass(frozen=True)
class MetricThresholds:
    """Acceptance thresholds and robust-error floors."""

    quantile: float = 0.95
    mask_fraction: float = 1.0e-3
    alpha: float = 1.0e-2
    d_floor: tuple[tuple[str, float], ...] = tuple(DEFAULT_D_FLOOR.items())
    ip_floor_fraction: float = 1.0e-2
    total_p95_max: float = 0.01
    ip_increment_nrmse_max: float = 0.03
    peak_time_error_steps_max: float = 1.0
    zero_crossing_time_error_steps_max: float = 1.0
    group_fail_rate_max: float = 0.10
    min_group_size: int = 5

    def d_floor_for(self, channel: str) -> float:
        mapping = dict(self.d_floor)
        if channel not in mapping:
            raise ValueError(f"unknown channel: {channel}")
        return float(mapping[channel])


@dataclass(frozen=True)
class ReceiverCase:
    """One survey case of reference and candidate six-channel waveforms."""

    case_id: str
    times: np.ndarray
    reference: dict[str, np.ndarray]
    candidate: dict[str, np.ndarray]
    reference_no_ip: dict[str, np.ndarray] | None = None
    groups: tuple[str, ...] = ()
    output_dt: float | None = None


@dataclass(frozen=True)
class ChannelMetrics:
    """Per-channel robust errors for one case."""

    channel: str
    unit: str
    amplitude_scale: float
    n_masked: int
    total_median: float
    total_p95: float
    total_nrmse: float
    ip_increment_nrmse: float
    peak_amplitude_error: float
    peak_time_error: float
    peak_time_error_steps: float
    zero_crossing_absolute_time_error: float
    zero_crossing_time_error_steps: float
    n_stable_zero_crossings: int
    unexplained_sign_flips: int


@dataclass(frozen=True)
class CaseMetrics:
    """Case-level aggregation of channel metrics."""

    case_id: str
    groups: tuple[str, ...]
    output_dt: float
    channels: dict[str, ChannelMetrics]
    case_total_median: float
    case_total_p95: float
    case_total_nrmse_by_channel: dict[str, float]
    case_ip_increment_nrmse: float
    peak_time_error_steps_max: float
    zero_crossing_time_error_steps_max: float
    unexplained_sign_flips: int
    n_valid_samples: int
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    """Cross-case ranking and qualifying summary for one Debye candidate."""

    candidate_id: str
    K: int
    case_metrics: tuple[CaseMetrics, ...]
    n_cases: int
    fail_rate: float
    total_p95_q95: float
    ip_increment_nrmse_q95: float
    worst_group: str
    worst_group_fail_rate: float
    total_median_median: float
    spectral_error: float
    condition_number: float
    group_fail_rates: dict[str, float]
    group_sizes: dict[str, int]
    qualifies: bool
    disqualifiers: tuple[str, ...]


def effective_mask(reference, *, quantile: float = 0.95, mask_fraction: float = 1.0e-3) -> tuple[np.ndarray, float]:
    """Return ``(|d_ref| >= mask_fraction * A_qc, A_qc)`` with ``A_qc = Q_q(|d_ref|)``."""

    values = _as_1d("reference", reference)
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    if float(mask_fraction) <= 0.0:
        raise ValueError("mask_fraction must be positive")
    amplitude_scale = float(np.quantile(np.abs(values), float(quantile))) if values.size else 0.0
    if amplitude_scale <= 0.0:
        return np.zeros(values.shape, dtype=bool), 0.0
    mask = np.abs(values) >= float(mask_fraction) * amplitude_scale
    return mask, amplitude_scale


def robust_total_field_error(
    candidate,
    reference,
    *,
    amplitude_scale: float,
    alpha: float,
    d_floor: float,
) -> np.ndarray:
    """Return pointwise ``|dK-dref| / max(|dref|, alpha A, d_floor)``."""

    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    if pred.shape != ref.shape:
        raise ValueError("candidate and reference must have the same shape")
    floor = max(float(alpha) * float(amplitude_scale), float(d_floor))
    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError("robust error floor must be finite and positive")
    return np.array(
        [robust_relative_error(float(left), float(right), floor) for left, right in zip(pred, ref)],
        dtype=float,
    )


def nrmse_with_floor(candidate, reference, *, floor: float) -> float:
    """Return RMS residual over ``max(RMS(reference), floor)``."""

    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    if pred.shape != ref.shape:
        raise ValueError("candidate and reference must have the same shape")
    denom = max(_rms(ref), float(floor))
    if denom <= 0.0:
        return float(_rms(pred - ref))
    return float(_rms(pred - ref) / denom)


def ip_increment_nrmse(
    candidate,
    reference,
    baseline,
    *,
    mask,
    amplitude_scale: float,
    ip_floor_fraction: float,
) -> float:
    """Return floored NRMSE of the IP increment ``d - d0``."""

    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    base = _as_1d("baseline", baseline)
    keep = np.asarray(mask, dtype=bool)
    if pred.shape != ref.shape or base.shape != ref.shape or keep.shape != ref.shape:
        raise ValueError("candidate, reference, baseline, and mask must share one shape")
    if not np.any(keep):
        return float("nan")
    delta_pred = pred[keep] - base[keep]
    delta_ref = ref[keep] - base[keep]
    return nrmse_with_floor(
        delta_pred,
        delta_ref,
        floor=float(ip_floor_fraction) * float(amplitude_scale),
    )


def peak_errors(times, candidate, reference, *, mask) -> tuple[float, float]:
    """Return ``(peak_amplitude_error, peak_time_error)`` on the effective mask."""

    time_values = _validated_times(times)
    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    keep = np.asarray(mask, dtype=bool)
    if pred.shape != time_values.shape or ref.shape != time_values.shape or keep.shape != time_values.shape:
        raise ValueError("times, candidate, reference, and mask must share one shape")
    if not np.any(keep):
        return float("nan"), float("nan")
    masked_index = np.flatnonzero(keep)
    ref_peak = masked_index[int(np.argmax(np.abs(ref[keep])))]
    pred_peak = masked_index[int(np.argmax(np.abs(pred[keep])))]
    ref_amp = abs(float(ref[ref_peak]))
    if ref_amp <= 0.0:
        return float("nan"), float("nan")
    amplitude_error = abs(abs(float(pred[pred_peak])) - ref_amp) / ref_amp
    time_error = abs(float(time_values[pred_peak]) - float(time_values[ref_peak]))
    return float(amplitude_error), float(time_error)


def zero_crossing_times(times, values) -> np.ndarray:
    """Return linearly interpolated sign-change times."""

    time_values = _validated_times(times)
    response = _as_1d("values", values)
    if response.shape != time_values.shape:
        raise ValueError("values must have the same length as times")
    signs = np.sign(response)
    nonzero = np.flatnonzero(signs != 0.0)
    if nonzero.size == 0:
        return np.asarray([], dtype=float)
    filled = signs.copy()
    filled[: nonzero[0]] = signs[nonzero[0]]
    last = signs[nonzero[0]]
    for index in range(nonzero[0] + 1, filled.size):
        if filled[index] == 0.0:
            filled[index] = last
        else:
            last = filled[index]
    crossings: list[float] = []
    for index in range(filled.size - 1):
        if filled[index] * filled[index + 1] < 0.0:
            left = float(response[index])
            right = float(response[index + 1])
            denom = right - left
            if denom == 0.0:
                fraction = 0.5
            else:
                fraction = -left / denom
            fraction = min(max(fraction, 0.0), 1.0)
            crossings.append(float(time_values[index] + fraction * (time_values[index + 1] - time_values[index])))
    return np.asarray(crossings, dtype=float)


def _segment_peak(values: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        return 0.0
    return float(np.max(np.abs(values[start:stop])))


def stable_zero_crossing_times(times, values, *, amplitude_scale: float, alpha: float) -> np.ndarray:
    """Return crossings whose adjacent lobes reach ``alpha * A_qc``."""

    time_values = _validated_times(times)
    response = _as_1d("values", values)
    crossings = zero_crossing_times(time_values, response)
    if crossings.size == 0:
        return crossings
    threshold = float(alpha) * float(amplitude_scale)
    signs = np.sign(response)
    nonzero = np.flatnonzero(signs != 0.0)
    filled = np.zeros(response.shape, dtype=float)
    if nonzero.size:
        filled[: nonzero[0]] = signs[nonzero[0]]
        last = signs[nonzero[0]]
        for index in range(nonzero[0], filled.size):
            if signs[index] == 0.0:
                filled[index] = last
            else:
                filled[index] = signs[index]
                last = signs[index]
    stable: list[float] = []
    change_indices = [index for index in range(filled.size - 1) if filled[index] * filled[index + 1] < 0.0]
    for crossing, index in zip(crossings, change_indices):
        left_start = index
        while left_start > 0 and filled[left_start - 1] == filled[index]:
            left_start -= 1
        right_stop = index + 1
        while right_stop < filled.size and filled[right_stop] == filled[index + 1]:
            right_stop += 1
        if _segment_peak(response, left_start, index + 1) >= threshold and _segment_peak(response, index + 1, right_stop) >= threshold:
            stable.append(float(crossing))
    return np.asarray(stable, dtype=float)


def zero_crossing_absolute_time_error(
    times,
    candidate,
    reference,
    *,
    amplitude_scale: float,
    alpha: float,
) -> tuple[float, int]:
    """Return ``(max |dt|, n_stable_ref_crossings)`` for stable reference crossings."""

    ref_crossings = stable_zero_crossing_times(times, reference, amplitude_scale=amplitude_scale, alpha=alpha)
    if ref_crossings.size == 0:
        return float("nan"), 0
    cand_crossings = zero_crossing_times(times, candidate)
    if cand_crossings.size == 0:
        return float("inf"), int(ref_crossings.size)
    errors = [float(np.min(np.abs(cand_crossings - crossing))) for crossing in ref_crossings]
    return float(np.max(errors)), int(ref_crossings.size)


def unexplained_sign_flips(
    candidate,
    reference,
    *,
    mask,
    amplitude_scale: float,
    alpha: float,
) -> int:
    """Count masked samples with opposite sign outside the explained band."""

    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    keep = np.asarray(mask, dtype=bool)
    if pred.shape != ref.shape or keep.shape != ref.shape:
        raise ValueError("candidate, reference, and mask must share one shape")
    threshold = float(alpha) * float(amplitude_scale)
    pred_sign = np.sign(pred)
    ref_sign = np.sign(ref)
    opposite = (pred_sign != ref_sign) & (pred_sign != 0.0) & (ref_sign != 0.0)
    explained_band = np.abs(ref) < threshold
    return int(np.count_nonzero(keep & opposite & ~explained_band))


def _empty_channel_metrics(channel: str, amplitude_scale: float) -> ChannelMetrics:
    return ChannelMetrics(
        channel=channel,
        unit=CHANNEL_UNITS[channel],
        amplitude_scale=float(amplitude_scale),
        n_masked=0,
        total_median=float("nan"),
        total_p95=float("nan"),
        total_nrmse=float("nan"),
        ip_increment_nrmse=float("nan"),
        peak_amplitude_error=float("nan"),
        peak_time_error=float("nan"),
        peak_time_error_steps=float("nan"),
        zero_crossing_absolute_time_error=float("nan"),
        zero_crossing_time_error_steps=float("nan"),
        n_stable_zero_crossings=0,
        unexplained_sign_flips=0,
    )


def evaluate_channel(
    channel: str,
    times,
    candidate,
    reference,
    baseline,
    *,
    output_dt: float,
    thresholds: MetricThresholds,
) -> tuple[ChannelMetrics, np.ndarray]:
    """Evaluate one channel and return ``(metrics, masked e_total)``."""

    if channel not in CHANNEL_UNITS:
        raise ValueError(f"unknown channel: {channel}")
    time_values = _validated_times(times)
    pred = _as_1d("candidate", candidate)
    ref = _as_1d("reference", reference)
    if pred.shape != time_values.shape or ref.shape != time_values.shape:
        raise ValueError("times, candidate, and reference must share one shape")
    mask, amplitude_scale = effective_mask(
        ref,
        quantile=thresholds.quantile,
        mask_fraction=thresholds.mask_fraction,
    )
    if not np.any(mask):
        return _empty_channel_metrics(channel, amplitude_scale), np.asarray([], dtype=float)

    errors = robust_total_field_error(
        pred,
        ref,
        amplitude_scale=amplitude_scale,
        alpha=thresholds.alpha,
        d_floor=thresholds.d_floor_for(channel),
    )
    masked_errors = errors[mask]
    peak_amp, peak_time = peak_errors(time_values, pred, ref, mask=mask)
    zc_error, n_stable = zero_crossing_absolute_time_error(
        time_values,
        pred,
        ref,
        amplitude_scale=amplitude_scale,
        alpha=thresholds.alpha,
    )
    if baseline is None:
        increment = float("nan")
    else:
        increment = ip_increment_nrmse(
            pred,
            ref,
            baseline,
            mask=mask,
            amplitude_scale=amplitude_scale,
            ip_floor_fraction=thresholds.ip_floor_fraction,
        )
    dt = float(output_dt)
    metrics = ChannelMetrics(
        channel=channel,
        unit=CHANNEL_UNITS[channel],
        amplitude_scale=amplitude_scale,
        n_masked=int(np.count_nonzero(mask)),
        total_median=float(np.median(masked_errors)),
        total_p95=float(np.quantile(masked_errors, thresholds.quantile)),
        total_nrmse=nrmse_with_floor(
            pred[mask],
            ref[mask],
            floor=thresholds.d_floor_for(channel),
        ),
        ip_increment_nrmse=float(increment),
        peak_amplitude_error=float(peak_amp),
        peak_time_error=float(peak_time),
        peak_time_error_steps=float(peak_time / dt) if np.isfinite(peak_time) else float("nan"),
        zero_crossing_absolute_time_error=float(zc_error),
        zero_crossing_time_error_steps=(
            float(zc_error / dt) if np.isfinite(zc_error) or np.isinf(zc_error) else float("nan")
        ),
        n_stable_zero_crossings=int(n_stable),
        unexplained_sign_flips=unexplained_sign_flips(
            pred,
            ref,
            mask=mask,
            amplitude_scale=amplitude_scale,
            alpha=thresholds.alpha,
        ),
    )
    return metrics, np.asarray(masked_errors, dtype=float)


def _case_output_dt(case: ReceiverCase, times: np.ndarray) -> float:
    if case.output_dt is None:
        return float(np.median(np.diff(times)))
    dt = float(case.output_dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("output_dt must be positive")
    return dt


def evaluate_case(case: ReceiverCase, thresholds: MetricThresholds = MetricThresholds()) -> CaseMetrics:
    """Evaluate all supplied channels of one receiver case."""

    times = _validated_times(case.times)
    if set(case.candidate) != set(case.reference):
        raise ValueError("candidate and reference must provide the same channels")
    if not case.reference:
        raise ValueError("at least one channel is required")
    unknown = [name for name in case.reference if name not in CHANNEL_UNITS]
    if unknown:
        raise ValueError(f"unknown channel: {unknown[0]}")
    if case.reference_no_ip is not None and set(case.reference_no_ip) != set(case.reference):
        raise ValueError("reference_no_ip must provide the same channels as reference")

    dt = _case_output_dt(case, times)
    channels: dict[str, ChannelMetrics] = {}
    pooled: list[np.ndarray] = []
    ordered_names = list(CHANNELS) + [name for name in PROJECTED_CHANNELS if name not in CHANNELS]
    for name in ordered_names:
        if name not in case.reference:
            continue
        baseline = None if case.reference_no_ip is None else case.reference_no_ip[name]
        metrics, masked_errors = evaluate_channel(
            name,
            times,
            case.candidate[name],
            case.reference[name],
            baseline,
            output_dt=dt,
            thresholds=thresholds,
        )
        channels[name] = metrics
        if masked_errors.size:
            pooled.append(masked_errors)

    if pooled:
        all_errors = np.concatenate(pooled)
        case_total_median = float(np.median(all_errors))
        case_total_p95 = float(np.quantile(all_errors, thresholds.quantile))
        n_valid = int(all_errors.size)
    else:
        case_total_median = float("nan")
        case_total_p95 = float("nan")
        n_valid = 0

    increments = [metrics.ip_increment_nrmse for metrics in channels.values()]
    finite_increments = [value for value in increments if np.isfinite(value)]
    case_ip = float(np.max(finite_increments)) if finite_increments else float("nan")

    peak_steps = [metrics.peak_time_error_steps for metrics in channels.values() if np.isfinite(metrics.peak_time_error_steps)]
    peak_time_error_steps_max = float(np.max(peak_steps)) if peak_steps else float("nan")

    zc_steps = [
        metrics.zero_crossing_time_error_steps
        for metrics in channels.values()
        if np.isfinite(metrics.zero_crossing_time_error_steps) or np.isinf(metrics.zero_crossing_time_error_steps)
    ]
    zero_crossing_time_error_steps_max = float(np.max(zc_steps)) if zc_steps else float("nan")
    sign_flips = int(sum(metrics.unexplained_sign_flips for metrics in channels.values()))

    reasons: list[str] = []
    if n_valid == 0:
        reasons.append("empty_mask")
    if np.isfinite(case_total_p95) and case_total_p95 > thresholds.total_p95_max:
        reasons.append("total_p95")
    if np.isfinite(case_ip) and case_ip > thresholds.ip_increment_nrmse_max:
        reasons.append("ip_increment_nrmse")
    if sign_flips > 0:
        reasons.append("sign_flips")
    if np.isfinite(peak_time_error_steps_max) and peak_time_error_steps_max > thresholds.peak_time_error_steps_max + 1.0e-9:
        reasons.append("peak_time")
    if (np.isinf(zero_crossing_time_error_steps_max) or (
        np.isfinite(zero_crossing_time_error_steps_max)
        and zero_crossing_time_error_steps_max > thresholds.zero_crossing_time_error_steps_max + 1.0e-9
    )):
        reasons.append("zero_crossing_time")

    return CaseMetrics(
        case_id=str(case.case_id),
        groups=tuple(str(group) for group in case.groups),
        output_dt=dt,
        channels=channels,
        case_total_median=case_total_median,
        case_total_p95=case_total_p95,
        case_total_nrmse_by_channel={name: metrics.total_nrmse for name, metrics in channels.items()},
        case_ip_increment_nrmse=case_ip,
        peak_time_error_steps_max=peak_time_error_steps_max,
        zero_crossing_time_error_steps_max=zero_crossing_time_error_steps_max,
        unexplained_sign_flips=sign_flips,
        n_valid_samples=n_valid,
        passed=len(reasons) == 0,
        failure_reasons=tuple(reasons),
    )


def _quantile_with_inf(values: Sequence[float], quantile: float) -> float:
    prepared = np.array([_nan_to_inf(value) for value in values], dtype=float)
    if prepared.size == 0:
        return float("nan")
    return float(np.quantile(prepared, quantile))


def _group_statistics(
    case_metrics: Sequence[CaseMetrics],
    *,
    min_group_size: int,
) -> tuple[dict[str, float], dict[str, int], str, float]:
    sizes: dict[str, int] = {}
    fails: dict[str, int] = {}
    for metrics in case_metrics:
        for group in metrics.groups:
            sizes[group] = sizes.get(group, 0) + 1
            fails[group] = fails.get(group, 0) + int(not metrics.passed)
    rates = {group: fails[group] / sizes[group] for group in sizes}
    eligible = {group: rates[group] for group, size in sizes.items() if size >= int(min_group_size)}
    if not eligible:
        return rates, sizes, "", 0.0
    worst_rate = max(eligible.values())
    worst_groups = sorted(group for group, rate in eligible.items() if rate == worst_rate)
    return rates, sizes, worst_groups[0], float(worst_rate)


def qualifying_failures(
    evaluation: CandidateEvaluation,
    thresholds: MetricThresholds = MetricThresholds(),
) -> tuple[str, ...]:
    """Return lexicographic qualifying-gate failures."""

    reasons: list[str] = []
    if evaluation.n_cases == 0:
        reasons.append("no_cases")
        return tuple(reasons)
    if not np.isfinite(evaluation.total_p95_q95) or evaluation.total_p95_q95 > thresholds.total_p95_max:
        reasons.append("total_p95_q95")
    if all(not np.isfinite(metrics.case_ip_increment_nrmse) for metrics in evaluation.case_metrics):
        reasons.append("ip_increment_unavailable")
    elif evaluation.ip_increment_nrmse_q95 > thresholds.ip_increment_nrmse_max:
        reasons.append("ip_increment_nrmse_q95")
    if any(metrics.unexplained_sign_flips > 0 for metrics in evaluation.case_metrics):
        reasons.append("sign_flips")
    peak_steps = [metrics.peak_time_error_steps_max for metrics in evaluation.case_metrics]
    if any(np.isfinite(value) and value > thresholds.peak_time_error_steps_max + 1.0e-9 for value in peak_steps):
        reasons.append("peak_time")
    zc_steps = [metrics.zero_crossing_time_error_steps_max for metrics in evaluation.case_metrics]
    if any(
        np.isinf(value) or (np.isfinite(value) and value > thresholds.zero_crossing_time_error_steps_max + 1.0e-9)
        for value in zc_steps
    ):
        reasons.append("zero_crossing_time")
    if evaluation.worst_group_fail_rate > thresholds.group_fail_rate_max:
        reasons.append("group_fail_rate")
    return tuple(reasons)


def is_qualifying(
    evaluation: CandidateEvaluation,
    thresholds: MetricThresholds = MetricThresholds(),
) -> bool:
    return len(qualifying_failures(evaluation, thresholds)) == 0


def evaluate_candidate(
    candidate_id: str,
    K: int,
    case_metrics: Sequence[CaseMetrics],
    *,
    spectral_error: float,
    condition_number: float,
    thresholds: MetricThresholds = MetricThresholds(),
) -> CandidateEvaluation:
    """Aggregate case metrics into ranking and qualifying summaries."""

    metrics = tuple(case_metrics)
    n_cases = len(metrics)
    fail_rate = float(np.mean([not item.passed for item in metrics])) if n_cases else float("nan")
    total_p95_q95 = _quantile_with_inf([item.case_total_p95 for item in metrics], thresholds.quantile) if n_cases else float("nan")
    increments = [item.case_ip_increment_nrmse for item in metrics]
    if n_cases == 0 or all(not np.isfinite(value) for value in increments):
        ip_q95 = float("nan")
    else:
        ip_q95 = _quantile_with_inf(increments, thresholds.quantile)
    medians = [item.case_total_median for item in metrics]
    total_median_median = float(np.median(medians)) if medians else float("nan")
    group_rates, group_sizes, worst_group, worst_rate = _group_statistics(
        metrics,
        min_group_size=thresholds.min_group_size,
    )
    evaluation = CandidateEvaluation(
        candidate_id=str(candidate_id),
        K=int(K),
        case_metrics=metrics,
        n_cases=n_cases,
        fail_rate=fail_rate,
        total_p95_q95=total_p95_q95,
        ip_increment_nrmse_q95=ip_q95,
        worst_group=worst_group,
        worst_group_fail_rate=float(worst_rate),
        total_median_median=total_median_median,
        spectral_error=float(spectral_error),
        condition_number=float(condition_number),
        group_fail_rates=group_rates,
        group_sizes=group_sizes,
        qualifies=False,
        disqualifiers=(),
    )
    disqualifiers = qualifying_failures(evaluation, thresholds)
    return CandidateEvaluation(
        candidate_id=evaluation.candidate_id,
        K=evaluation.K,
        case_metrics=evaluation.case_metrics,
        n_cases=evaluation.n_cases,
        fail_rate=evaluation.fail_rate,
        total_p95_q95=evaluation.total_p95_q95,
        ip_increment_nrmse_q95=evaluation.ip_increment_nrmse_q95,
        worst_group=evaluation.worst_group,
        worst_group_fail_rate=evaluation.worst_group_fail_rate,
        total_median_median=evaluation.total_median_median,
        spectral_error=evaluation.spectral_error,
        condition_number=evaluation.condition_number,
        group_fail_rates=evaluation.group_fail_rates,
        group_sizes=evaluation.group_sizes,
        qualifies=len(disqualifiers) == 0,
        disqualifiers=disqualifiers,
    )


def ranking_key(evaluation: CandidateEvaluation) -> tuple[float, float, float, float, float, float, float]:
    """Lexicographic lower-is-better ranking tuple."""

    return (
        _nan_to_inf(evaluation.fail_rate),
        _nan_to_inf(evaluation.total_p95_q95),
        _nan_to_inf(evaluation.ip_increment_nrmse_q95),
        _nan_to_inf(evaluation.worst_group_fail_rate),
        _nan_to_inf(evaluation.total_median_median),
        _nan_to_inf(evaluation.spectral_error),
        _nan_to_inf(evaluation.condition_number),
    )


def rank_candidates(evaluations: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """Sort candidates by ranking key, then by ``candidate_id``."""

    return sorted(evaluations, key=lambda item: (ranking_key(item), item.candidate_id))
