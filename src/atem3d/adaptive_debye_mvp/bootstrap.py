"""Case-level paired bootstrap for Debye-MVP receiver metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .receiver_metrics import CandidateEvaluation, CaseMetrics


DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 202609116

STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "median": lambda values: float(np.median(values)),
    "mean": lambda values: float(np.mean(values)),
    "p95": lambda values: float(np.quantile(values, 0.95)),
    "max": lambda values: float(np.max(values)),
    "fail_rate": lambda values: float(np.mean(values)),
}


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile CI of a case-level statistic."""

    statistic: str
    n_cases: int
    n_bootstrap: int
    seed: int
    confidence_level: float
    point_estimate: float
    ci_low: float
    ci_high: float
    replicates: np.ndarray
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired case-level bootstrap comparison of two candidates."""

    statistic: str
    n_cases: int
    n_bootstrap: int
    seed: int
    confidence_level: float
    estimate_a: float
    estimate_b: float
    difference: float
    ci_low: float
    ci_high: float
    replicates: np.ndarray
    probability_a_better: float
    significant: bool
    case_ids: tuple[str, ...]


def case_resample_indices(n_cases: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Return ``(n_bootstrap, n_cases)`` case-level resample indices."""

    n_cases = int(n_cases)
    n_bootstrap = int(n_bootstrap)
    if n_cases < 2:
        raise ValueError("n_cases must be at least 2")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    return np.random.default_rng(int(seed)).integers(0, n_cases, size=(n_bootstrap, n_cases))


def _resolve_statistic(statistic) -> tuple[str, Callable[[np.ndarray], float]]:
    if callable(statistic):
        return statistic.__name__, statistic
    name = str(statistic)
    if name not in STATISTICS:
        raise ValueError(f"unknown statistic: {name}")
    return name, STATISTICS[name]


def _as_case_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must contain exactly one statistic per case")
    if array.size < 2:
        raise ValueError("n_cases must be at least 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def bootstrap_case_statistic(
    values,
    statistic="median",
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
    case_ids: Sequence[str] = (),
) -> BootstrapResult:
    """Bootstrap one case-level statistic."""

    samples = _as_case_values(values)
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    name, func = _resolve_statistic(statistic)
    indices = case_resample_indices(samples.size, n_bootstrap, seed)
    replicates = np.array([func(samples[row]) for row in indices], dtype=float)
    point = float(func(samples))
    alpha = 0.5 * (1.0 - float(confidence_level))
    ci_low, ci_high = np.quantile(replicates, [alpha, 1.0 - alpha])
    return BootstrapResult(
        statistic=name,
        n_cases=int(samples.size),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        confidence_level=float(confidence_level),
        point_estimate=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        replicates=replicates,
        case_ids=tuple(str(item) for item in case_ids),
    )


def paired_case_bootstrap(
    values_a,
    values_b,
    statistic="p95",
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
    lower_is_better: bool = True,
    case_ids: Sequence[str] = (),
) -> PairedBootstrapResult:
    """Bootstrap the paired difference of a case-level statistic."""

    left = _as_case_values(values_a)
    right = _as_case_values(values_b)
    if left.shape != right.shape:
        raise ValueError("values_a and values_b must have the same length")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    name, func = _resolve_statistic(statistic)
    indices = case_resample_indices(left.size, n_bootstrap, seed)
    replicates = np.array([func(left[row]) - func(right[row]) for row in indices], dtype=float)
    estimate_a = float(func(left))
    estimate_b = float(func(right))
    difference = estimate_a - estimate_b
    alpha = 0.5 * (1.0 - float(confidence_level))
    ci_low, ci_high = np.quantile(replicates, [alpha, 1.0 - alpha])
    if lower_is_better:
        probability = float(np.mean(replicates < 0.0))
    else:
        probability = float(np.mean(replicates > 0.0))
    return PairedBootstrapResult(
        statistic=name,
        n_cases=int(left.size),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        confidence_level=float(confidence_level),
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        difference=float(difference),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        replicates=replicates,
        probability_a_better=probability,
        significant=not (float(ci_low) <= 0.0 <= float(ci_high)),
        case_ids=tuple(str(item) for item in case_ids),
    )


def case_metric_values(
    case_metrics: Sequence[CaseMetrics],
    metric: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Extract one case-level metric, aligned by ``case_id``."""

    if metric == "case_total_p95":
        values = [float(item.case_total_p95) for item in case_metrics]
    elif metric == "case_total_median":
        values = [float(item.case_total_median) for item in case_metrics]
    elif metric == "case_ip_increment_nrmse":
        values = [float(item.case_ip_increment_nrmse) for item in case_metrics]
    elif metric == "passed":
        values = [1.0 if not item.passed else 0.0 for item in case_metrics]
    else:
        raise ValueError(f"unknown case metric: {metric}")
    case_ids = tuple(str(item.case_id) for item in case_metrics)
    return case_ids, np.asarray(values, dtype=float)


def bootstrap_candidate_comparison(
    a: CandidateEvaluation,
    b: CandidateEvaluation,
    *,
    metric: str = "case_total_p95",
    statistic="p95",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Compare two candidate evaluations on a shared case set."""

    ids_a, values_a = case_metric_values(a.case_metrics, metric)
    ids_b, values_b = case_metric_values(b.case_metrics, metric)
    if set(ids_a) != set(ids_b):
        raise ValueError("candidate evaluations must share the same case_id set")
    order = tuple(sorted(ids_a))
    index_a = {case_id: index for index, case_id in enumerate(ids_a)}
    index_b = {case_id: index for index, case_id in enumerate(ids_b)}
    aligned_a = np.asarray([values_a[index_a[case_id]] for case_id in order], dtype=float)
    aligned_b = np.asarray([values_b[index_b[case_id]] for case_id in order], dtype=float)
    resolved_statistic = "mean" if metric == "passed" and statistic == "p95" else statistic
    return paired_case_bootstrap(
        aligned_a,
        aligned_b,
        resolved_statistic,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence_level=confidence_level,
        case_ids=order,
    )
