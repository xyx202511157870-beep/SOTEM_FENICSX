"""Pilot oracle-gap evaluation and L0 gate (no selector, no 3-D)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bootstrap import paired_case_bootstrap
from .candidates import CandidateSpec
from .case_bridge import (
    assert_shared_survey_hash,
    case_geometry,
    case_time_grid,
    case_waveform,
    channel_column,
    debye_material,
    disk_receivers,
    exact_pelton_material,
    forward_response,
    nonpolarizable_material,
    point_receivers,
    polarizable_material,
    evaluation_transform,
)
from .io import to_record, write_json, write_records_csv
from .layered_forward import BlockedBySoftwareOrResourcesError
from .passive_fit import fit_pelton_passive_hard_dc
from .protocol_constants import (
    BOOTSTRAP_SEED,
    K_PILOT,
    K_PRACTICAL,
    PILOT_RECEIVERS,
    PILOT_WAVEFORMS,
    SPECTRAL_FREQUENCIES,
    spectral_weights,
)
from .receiver_metrics import (
    MetricThresholds,
    ReceiverCase,
    evaluate_case,
)
from .registry import LayeredCase, instantiate_candidates


@dataclass(frozen=True)
class FitRecord:
    candidate_id: str
    K: int
    valid: bool
    spectral_error_s0: float
    spectral_error_s1: float
    condition_number: float
    relative_dc_error: float
    optimizer_success: bool
    fit: Any


@dataclass(frozen=True)
class TaskMetrics:
    case_id: str
    candidate_id: str
    K: int
    waveform_id: str
    receiver_id: str
    receiver_index: int
    total_p95: float
    ip_increment_nrmse: float
    passed: bool
    unexplained_sign_flips: int
    peak_time_error_steps_max: float
    h_p95: float
    dbdt_p95: float


@dataclass(frozen=True)
class CaseKChoice:
    case_id: str
    K: int
    spectral_id: str
    oracle_id: str
    ids_differ: bool
    e_b2: float
    e_or: float
    ratio: float
    gap: float
    ip_b2: float
    ip_or: float
    log_hausdorff: float
    qualifies_b2: bool
    qualifies_or: bool


def log_hausdorff_pole_distance(tau_a, tau_b) -> float:
    """Return the log10 Hausdorff distance between two pole grids."""

    left = np.log10(np.asarray(tau_a, dtype=float))
    right = np.log10(np.asarray(tau_b, dtype=float))
    if left.size == 0 or right.size == 0:
        return float("inf")
    d_ab = np.max([float(np.min(np.abs(right - value))) for value in left])
    d_ba = np.max([float(np.min(np.abs(left - value))) for value in right])
    return float(max(d_ab, d_ba))


def fit_case_candidates(
    case: LayeredCase,
    candidates: tuple[CandidateSpec, ...],
) -> list[FitRecord]:
    """Fit every frozen template on the polarizable Cole-Cole spectrum."""

    material = polarizable_material(case)
    frequencies = SPECTRAL_FREQUENCIES
    target = material.complex_conductivity(frequencies)
    records: list[FitRecord] = []
    for spec in candidates:
        fit_s0 = fit_pelton_passive_hard_dc(
            material.rho0,
            material.chargeability,
            material.tau,
            material.c,
            frequencies,
            spec.tau_grid,
            weights=spectral_weights("S0", frequencies),
        )
        fit_s1 = fit_pelton_passive_hard_dc(
            material.rho0,
            material.chargeability,
            material.tau,
            material.c,
            frequencies,
            spec.tau_grid,
            weights=spectral_weights("S1", frequencies),
        )
        valid = fit_s0.passes_hard_gates() and fit_s1.passes_hard_gates()
        records.append(
            FitRecord(
                candidate_id=spec.candidate_id,
                K=spec.K,
                valid=valid,
                spectral_error_s0=float(fit_s0.spectral_error),
                spectral_error_s1=float(fit_s1.spectral_error),
                condition_number=float(fit_s0.condition_number),
                relative_dc_error=float(fit_s0.relative_dc_error),
                optimizer_success=bool(fit_s0.optimizer_status.success and fit_s1.optimizer_status.success),
                fit=fit_s0,
            )
        )
    return records


def choose_official_spectral_variant(records: list[FitRecord]) -> str:
    """Pick S0 vs S1 from material-spectrum metrics only."""

    valid = [item for item in records if item.valid]
    if not valid:
        return "S0"
    median_s0 = float(np.median([item.spectral_error_s0 for item in valid]))
    median_s1 = float(np.median([item.spectral_error_s1 for item in valid]))
    return "S0" if median_s0 <= median_s1 else "S1"


def spectral_error(record: FitRecord, variant: str) -> float:
    return record.spectral_error_s0 if variant == "S0" else record.spectral_error_s1


def _channel_p95(metrics, names: tuple[str, ...]) -> float:
    values = [metrics.channels[name].total_p95 for name in names if name in metrics.channels]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else float("nan")


def evaluate_task(
    *,
    case: LayeredCase,
    candidate_id: str,
    K: int,
    waveform_id: str,
    receiver_id: str,
    receiver_index: int,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    times: np.ndarray,
) -> TaskMetrics:
    groups = (case.case_id, waveform_id, receiver_id)
    metrics = evaluate_case(
        ReceiverCase(
            case_id=f"{case.case_id}:{waveform_id}:{receiver_id}:{receiver_index}",
            times=times,
            reference=reference,
            candidate=candidate,
            reference_no_ip=baseline,
            groups=groups,
            output_dt=float(np.median(np.diff(times))),
        )
    )
    return TaskMetrics(
        case_id=case.case_id,
        candidate_id=candidate_id,
        K=K,
        waveform_id=waveform_id,
        receiver_id=receiver_id,
        receiver_index=receiver_index,
        total_p95=metrics.case_total_p95,
        ip_increment_nrmse=metrics.case_ip_increment_nrmse,
        passed=metrics.passed,
        unexplained_sign_flips=metrics.unexplained_sign_flips,
        peak_time_error_steps_max=metrics.peak_time_error_steps_max,
        h_p95=_channel_p95(metrics, ("Hx", "Hy", "Hz")),
        dbdt_p95=_channel_p95(metrics, ("dBxdt", "dBydt", "dBzdt")),
    )


def reduce_case_error(tasks: list[TaskMetrics]) -> float:
    values = [item.total_p95 for item in tasks if np.isfinite(item.total_p95)]
    return float(np.quantile(values, 0.95)) if values else float("nan")


def reduce_ip_error(tasks: list[TaskMetrics]) -> float:
    values = [item.ip_increment_nrmse for item in tasks if np.isfinite(item.ip_increment_nrmse)]
    return float(np.quantile(values, 0.95)) if values else float("nan")


def tasks_qualify(tasks: list[TaskMetrics], thresholds: MetricThresholds = MetricThresholds()) -> bool:
    if not tasks:
        return False
    if reduce_case_error(tasks) > thresholds.total_p95_max:
        return False
    if reduce_ip_error(tasks) > thresholds.ip_increment_nrmse_max:
        return False
    if any(item.unexplained_sign_flips > 0 for item in tasks):
        return False
    if any(
        np.isfinite(item.peak_time_error_steps_max)
        and item.peak_time_error_steps_max > thresholds.peak_time_error_steps_max + 1.0e-9
        for item in tasks
    ):
        return False
    return True


def _label_matches(label: str, receiver_id: str) -> bool:
    kind, _sep, index_text = str(label).partition(":")
    if not index_text:
        return False
    return kind == receiver_id


def collect_tasks_from_responses(
    case: LayeredCase,
    *,
    candidate_id: str,
    K: int,
    reference_models: dict[str, Any],
    candidate_models: dict[str, Any],
    baseline_models: dict[str, Any],
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
    receiver_ids: tuple[str, ...] = PILOT_RECEIVERS,
) -> list[TaskMetrics]:
    """Score one constitutive model from official wrapper responses."""

    tasks: list[TaskMetrics] = []
    for waveform_id in waveform_ids:
        reference = reference_models[waveform_id]
        candidate = candidate_models[waveform_id]
        baseline = baseline_models[waveform_id]
        times = np.asarray(reference["times"], dtype=float)
        labels = [str(label) for label in reference["receiver_labels"]]
        if labels != [str(label) for label in candidate["receiver_labels"]]:
            raise RuntimeError("candidate receiver labels do not match the reference survey")
        if labels != [str(label) for label in baseline["receiver_labels"]]:
            raise RuntimeError("baseline receiver labels do not match the reference survey")
        assert_shared_survey_hash([reference, candidate, baseline])
        for column, label in enumerate(labels):
            for receiver_id in receiver_ids:
                if not _label_matches(label, receiver_id):
                    continue
                receiver_index = int(label.split(":", 1)[1])
                tasks.append(
                    evaluate_task(
                        case=case,
                        candidate_id=candidate_id,
                        K=K,
                        waveform_id=waveform_id,
                        receiver_id=receiver_id,
                        receiver_index=receiver_index,
                        reference=channel_column(reference, column),
                        candidate=channel_column(candidate, column),
                        baseline=channel_column(baseline, column),
                        times=times,
                    )
                )
    return tasks


def pick_spectral_best(records: list[FitRecord], variant: str) -> FitRecord:
    valid = [item for item in records if item.valid]
    if not valid:
        raise RuntimeError("no valid spectral candidates")
    return min(valid, key=lambda item: (spectral_error(item, variant), item.condition_number, item.candidate_id))


def pick_oracle_best(task_map: dict[str, list[TaskMetrics]], records: list[FitRecord]) -> FitRecord:
    valid = [item for item in records if item.valid and item.candidate_id in task_map]
    if not valid:
        raise RuntimeError("no valid oracle candidates")

    def key(record: FitRecord) -> tuple:
        tasks = task_map[record.candidate_id]
        fail_rate = float(np.mean([not item.passed for item in tasks]))
        return (
            fail_rate,
            reduce_case_error(tasks),
            reduce_ip_error(tasks),
            record.spectral_error_s0,
            record.condition_number,
            record.candidate_id,
        )

    return min(valid, key=key)


def _run_model_waveforms(
    case: LayeredCase,
    material,
    *,
    receivers,
    waveform_ids: tuple[str, ...],
    cache_dir,
    backend,
    model_id: str,
    times,
    geometry,
    transform,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for waveform_id in waveform_ids:
        models[waveform_id] = forward_response(
            material,
            geometry,
            case_waveform(waveform_id),
            receivers,
            times,
            transform,
            cache_dir=cache_dir,
            cache_key=f"{case.case_id}:{model_id}:{waveform_id}",
            backend=backend,
        )
    return models


def _forward_and_tasks(
    case: LayeredCase,
    record: FitRecord,
    *,
    reference_models,
    baseline_models,
    waveform_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...],
    cache_dir,
    backend,
    include_disks: bool,
    times,
    geometry,
    transform,
) -> list[TaskMetrics]:
    receivers = disk_receivers(case) if include_disks else point_receivers(case)
    models = _run_model_waveforms(
        case,
        debye_material(case, record.fit, candidate_id=record.candidate_id),
        receivers=receivers,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        backend=backend,
        model_id=f"{record.candidate_id}:{'disk' if include_disks else 'point'}",
        times=times,
        geometry=geometry,
        transform=transform,
    )
    return collect_tasks_from_responses(
        case,
        candidate_id=record.candidate_id,
        K=record.K,
        reference_models=reference_models,
        candidate_models=models,
        baseline_models=baseline_models,
        waveform_ids=waveform_ids,
        receiver_ids=receiver_ids,
    )


def evaluate_pilot_case(
    case: LayeredCase,
    *,
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
    cache_dir: str | Path | None = None,
    backend=None,
    k_values: tuple[int, ...] = K_PILOT,
    official_variant: str | None = None,
    disk_shortlist: int = 2,
    include_disks: bool = True,
) -> dict[str, Any]:
    """Evaluate spectral-best vs receiver-oracle on one pilot case.

    All valid templates are compared on point receivers via
    ``compute_layered_response``. Official PR 9 ``disk_average`` receivers are
    then computed for the spectral-best, the point-oracle, and the top
    ``disk_shortlist`` point-ranked templates. The shortlist is 2 because
    AverageReceiver 36-point x 3-axis disks cost ~140 s each on lagged DLF.
    """

    candidates = tuple(spec for spec in instantiate_candidates(case) if spec.K in k_values)
    fits = fit_case_candidates(case, candidates)
    variant = official_variant or choose_official_spectral_variant(fits)
    geometry = case_geometry(case)
    times = case_time_grid()
    transform = evaluation_transform()
    points = point_receivers(case)
    disks = disk_receivers(case)
    exact = exact_pelton_material(case)
    noip = nonpolarizable_material(case)
    try:
        reference_point = _run_model_waveforms(
            case,
            exact,
            receivers=points,
            waveform_ids=waveform_ids,
            cache_dir=cache_dir,
            backend=backend,
            model_id="exact:point",
            times=times,
            geometry=geometry,
            transform=transform,
        )
        baseline_point = _run_model_waveforms(
            case,
            noip,
            receivers=points,
            waveform_ids=waveform_ids,
            cache_dir=cache_dir,
            backend=backend,
            model_id="noip:point",
            times=times,
            geometry=geometry,
            transform=transform,
        )
        for waveform_id in waveform_ids:
            assert_shared_survey_hash([reference_point[waveform_id], baseline_point[waveform_id]])
        print(f"[{case.case_id}] point exact/noip finished", flush=True)
    except BlockedBySoftwareOrResourcesError:
        raise
    choices: list[CaseKChoice] = []
    task_rows: list[TaskMetrics] = []
    by_k: dict[int, list[FitRecord]] = {}
    for record in fits:
        by_k.setdefault(record.K, []).append(record)

    point_maps: dict[int, dict[str, list[TaskMetrics]]] = {}
    shortlists: dict[int, set[str]] = {}
    spectrals: dict[int, FitRecord] = {}
    for poles, records in by_k.items():
        point_map: dict[str, list[TaskMetrics]] = {}
        print(
            f"[{case.case_id}] K={poles} point forwards "
            f"{sum(1 for item in records if item.valid)}/{len(records)}",
            flush=True,
        )
        for record in records:
            if not record.valid:
                continue
            print(f"[{case.case_id}] point {record.candidate_id}", flush=True)
            tasks = _forward_and_tasks(
                case,
                record,
                reference_models=reference_point,
                baseline_models=baseline_point,
                waveform_ids=waveform_ids,
                receiver_ids=("point",),
                cache_dir=cache_dir,
                backend=backend,
                include_disks=False,
                times=times,
                geometry=geometry,
                transform=transform,
            )
            point_map[record.candidate_id] = tasks
            task_rows.extend(tasks)
        if not point_map:
            continue
        point_maps[poles] = point_map
        spectrals[poles] = pick_spectral_best(records, variant)
        point_oracle = pick_oracle_best(point_map, records)
        ranked = sorted(point_map, key=lambda cid: (reduce_case_error(point_map[cid]), cid))
        shortlists[poles] = {
            spectrals[poles].candidate_id,
            point_oracle.candidate_id,
            *ranked[: int(disk_shortlist)],
        }

    if not include_disks:
        for poles, records in by_k.items():
            if poles not in point_maps:
                continue
            point_map = point_maps[poles]
            spectral = spectrals[poles]
            oracle = pick_oracle_best(point_map, records)
            e_b2 = reduce_case_error(point_map[spectral.candidate_id])
            e_or = reduce_case_error(point_map[oracle.candidate_id])
            ratio = (
                float(e_or / e_b2)
                if e_b2 > 0.0 and np.isfinite(e_b2) and np.isfinite(e_or)
                else float("nan")
            )
            choices.append(
                CaseKChoice(
                    case_id=case.case_id,
                    K=poles,
                    spectral_id=spectral.candidate_id,
                    oracle_id=oracle.candidate_id,
                    ids_differ=spectral.candidate_id != oracle.candidate_id,
                    e_b2=e_b2,
                    e_or=e_or,
                    ratio=ratio,
                    gap=1.0 - ratio if np.isfinite(ratio) else float("nan"),
                    ip_b2=reduce_ip_error(point_map[spectral.candidate_id]),
                    ip_or=reduce_ip_error(point_map[oracle.candidate_id]),
                    log_hausdorff=log_hausdorff_pole_distance(
                        next(spec.tau_grid for spec in candidates if spec.candidate_id == spectral.candidate_id),
                        next(spec.tau_grid for spec in candidates if spec.candidate_id == oracle.candidate_id),
                    ),
                    qualifies_b2=tasks_qualify(point_map[spectral.candidate_id]),
                    qualifies_or=tasks_qualify(point_map[oracle.candidate_id]),
                )
            )
        print(f"[{case.case_id}] point-only stage finished", flush=True)
        return {
            "case_id": case.case_id,
            "official_variant": variant,
            "fits": fits,
            "choices": choices,
            "tasks": task_rows,
            "point_only": True,
            "disk_shortlist": {str(k): sorted(ids) for k, ids in shortlists.items()},
        }

    print(f"[{case.case_id}] computing exact/noip disks", flush=True)
    reference_disk = _run_model_waveforms(
        case,
        exact,
        receivers=disks,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        backend=backend,
        model_id="exact:disk",
        times=times,
        geometry=geometry,
        transform=transform,
    )
    baseline_disk = _run_model_waveforms(
        case,
        noip,
        receivers=disks,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        backend=backend,
        model_id="noip:disk",
        times=times,
        geometry=geometry,
        transform=transform,
    )
    for waveform_id in waveform_ids:
        assert_shared_survey_hash([reference_disk[waveform_id], baseline_disk[waveform_id]])

    for poles, records in by_k.items():
        if poles not in point_maps:
            continue
        point_map = point_maps[poles]
        shortlist_ids = shortlists[poles]
        spectral = spectrals[poles]
        disk_map: dict[str, list[TaskMetrics]] = {}
        print(
            f"[{case.case_id}] K={poles} disk shortlist {sorted(shortlist_ids)}",
            flush=True,
        )
        for record in records:
            if record.candidate_id not in shortlist_ids:
                continue
            disk_tasks = _forward_and_tasks(
                case,
                record,
                reference_models=reference_disk,
                baseline_models=baseline_disk,
                waveform_ids=waveform_ids,
                receiver_ids=("disk_1.0", "disk_4.0"),
                cache_dir=cache_dir,
                backend=backend,
                include_disks=True,
                times=times,
                geometry=geometry,
                transform=transform,
            )
            combined = point_map[record.candidate_id] + disk_tasks
            disk_map[record.candidate_id] = combined
            task_rows.extend(disk_tasks)
        oracle = pick_oracle_best(disk_map, records)
        task_map = disk_map
        e_b2 = reduce_case_error(task_map[spectral.candidate_id])
        e_or = reduce_case_error(task_map[oracle.candidate_id])
        ratio = float(e_or / e_b2) if e_b2 > 0.0 and np.isfinite(e_b2) and np.isfinite(e_or) else float("nan")
        choices.append(
            CaseKChoice(
                case_id=case.case_id,
                K=poles,
                spectral_id=spectral.candidate_id,
                oracle_id=oracle.candidate_id,
                ids_differ=spectral.candidate_id != oracle.candidate_id,
                e_b2=e_b2,
                e_or=e_or,
                ratio=ratio,
                gap=1.0 - ratio if np.isfinite(ratio) else float("nan"),
                ip_b2=reduce_ip_error(task_map[spectral.candidate_id]),
                ip_or=reduce_ip_error(task_map[oracle.candidate_id]),
                log_hausdorff=log_hausdorff_pole_distance(
                    next(spec.tau_grid for spec in candidates if spec.candidate_id == spectral.candidate_id),
                    next(spec.tau_grid for spec in candidates if spec.candidate_id == oracle.candidate_id),
                ),
                qualifies_b2=tasks_qualify(task_map[spectral.candidate_id]),
                qualifies_or=tasks_qualify(task_map[oracle.candidate_id]),
            )
        )
    return {
        "case_id": case.case_id,
        "official_variant": variant,
        "fits": fits,
        "choices": choices,
        "tasks": task_rows,
        "point_only": False,
        "disk_shortlist": {str(k): sorted(ids) for k, ids in shortlists.items()},
    }


def qualifying_k(choices: list[CaseKChoice], *, oracle: bool) -> float:
    qualified = [item.K for item in choices if (item.qualifies_or if oracle else item.qualifies_b2)]
    return float(min(qualified)) if qualified else float("inf")


def _finite_median(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def _group_ratio(tasks_or: list[TaskMetrics], tasks_b2: list[TaskMetrics], predicate) -> float:
    left = [item.total_p95 for item in tasks_or if predicate(item) and np.isfinite(item.total_p95)]
    right = [item.total_p95 for item in tasks_b2 if predicate(item) and np.isfinite(item.total_p95)]
    if not left or not right:
        return float("nan")
    return float(np.median(left) / np.median(right)) if np.median(right) > 0.0 else float("nan")


def evaluate_l0(pilot_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen L0 A/B predicates to pilot oracle-gap results."""

    pilot_results = [hydrate_result(result) for result in pilot_results]
    by_k: dict[int, list[CaseKChoice]] = {k: [] for k in K_PILOT}
    k_qual_diffs: list[float] = []
    for result in pilot_results:
        choices = result["choices"]
        for choice in choices:
            by_k[choice.K].append(choice)
        k_qual_diffs.append(qualifying_k(choices, oracle=False) - qualifying_k(choices, oracle=True))

    same_k: dict[str, Any] = {}
    passed_a = False
    best_k = None
    for poles in K_PRACTICAL:
        rows = by_k[poles]
        ratios = np.asarray([item.ratio for item in rows], dtype=float)
        finite = ratios[np.isfinite(ratios)]
        if finite.size == 0:
            continue
        bootstrap = paired_case_bootstrap(
            [item.e_or for item in rows],
            [item.e_b2 for item in rows],
            statistic="median",
            seed=BOOTSTRAP_SEED,
            case_ids=[item.case_id for item in rows],
        )
        ratio_ci = paired_case_bootstrap(
            [item.e_or / item.e_b2 if item.e_b2 > 0 else np.nan for item in rows],
            np.ones(len(rows)),
            statistic="median",
            seed=BOOTSTRAP_SEED,
            case_ids=[item.case_id for item in rows],
        )
        win_rate = float(np.mean([item.e_or < item.e_b2 for item in rows]))
        median_ratio = float(np.median(finite))
        record = {
            "K": poles,
            "median_ratio": median_ratio,
            "bootstrap_ci_low": float(ratio_ci.ci_low),
            "bootstrap_ci_high": float(ratio_ci.ci_high),
            "error_difference_ci_high": float(bootstrap.ci_high),
            "win_rate": win_rate,
            "n_cases": len(rows),
            "ids_differ_rate": float(np.mean([item.ids_differ for item in rows])),
        }
        same_k[str(poles)] = record
        if (
            median_ratio <= 0.80
            and float(ratio_ci.ci_high) < 1.00
            and win_rate >= 0.70
        ):
            passed_a = True
            best_k = poles

    # Group checks use the best practical K if A is otherwise close, else K=8.
    focus_k = best_k if best_k is not None else 8
    group_ok = {"waveforms": False, "receivers": False, "components": False, "ip": False}
    if focus_k in by_k and by_k[focus_k]:
        # Reconstruct group ratios from stored choice-level fields only when
        # full task tables are present.
        waveform_improvements = 0
        receiver_point = []
        receiver_disk = []
        h_ratios = []
        dbdt_ratios = []
        ip_ratios = []
        for result in pilot_results:
            choice = next(item for item in result["choices"] if item.K == focus_k)
            tasks = result["tasks"]
            or_tasks = [item for item in tasks if item.candidate_id == choice.oracle_id and item.K == focus_k]
            b2_tasks = [item for item in tasks if item.candidate_id == choice.spectral_id and item.K == focus_k]
            improved_waveforms = 0
            for waveform_id in PILOT_WAVEFORMS:
                ratio = _group_ratio(or_tasks, b2_tasks, lambda item, wid=waveform_id: item.waveform_id == wid)
                if np.isfinite(ratio) and ratio < 1.0:
                    improved_waveforms += 1
            waveform_improvements += int(improved_waveforms >= 2)
            receiver_point.append(_group_ratio(or_tasks, b2_tasks, lambda item: item.receiver_id == "point"))
            receiver_disk.append(
                _group_ratio(or_tasks, b2_tasks, lambda item: item.receiver_id in {"disk_1.0", "disk_4.0"})
            )
            h_ratios.append(_group_ratio(or_tasks, b2_tasks, lambda item: np.isfinite(item.h_p95)))
            dbdt_ratios.append(_group_ratio(or_tasks, b2_tasks, lambda item: np.isfinite(item.dbdt_p95)))
            if choice.ip_b2 > 0:
                ip_ratios.append(choice.ip_or / choice.ip_b2)
        group_ok["waveforms"] = waveform_improvements >= int(0.70 * len(pilot_results))
        point_med = _finite_median(receiver_point)
        disk_med = _finite_median(receiver_disk)
        point_ok = np.isfinite(point_med) and point_med < 1.0
        disk_ok = np.isfinite(disk_med) and disk_med < 1.0
        group_ok["receivers"] = bool(point_ok and disk_ok)
        h_med = _finite_median(h_ratios)
        db_med = _finite_median(dbdt_ratios)
        group_ok["components"] = (h_med <= 0.90 or db_med <= 0.90) and h_med <= 1.10 and db_med <= 1.10
        group_ok["ip"] = float(np.nanmedian(ip_ratios)) <= 1.05 if ip_ratios else False
        if best_k is not None:
            passed_a = passed_a and all(group_ok.values())

    diffs = np.asarray(k_qual_diffs, dtype=float)
    finite_diffs = diffs[np.isfinite(diffs)]
    passed_b = False
    if finite_diffs.size:
        passed_b = float(np.median(finite_diffs)) >= 2.0 and float(np.mean(finite_diffs >= 0.0)) >= 0.70

    passed = bool(passed_a or passed_b)
    best_ratio = min((row["median_ratio"] for row in same_k.values()), default=float("nan"))
    best_ci = next(
        (row for row in same_k.values() if row["median_ratio"] == best_ratio),
        {"bootstrap_ci_low": float("nan"), "bootstrap_ci_high": float("nan"), "win_rate": float("nan"), "K": None},
    )
    return {
        "passed": passed,
        "passed_A": bool(passed_a),
        "passed_B": bool(passed_b),
        "same_k": same_k,
        "group_ok": group_ok,
        "focus_k": focus_k,
        "best_same_k_median_ratio": best_ratio,
        "best_same_k": best_ci.get("K"),
        "bootstrap_ci_low": best_ci.get("bootstrap_ci_low"),
        "bootstrap_ci_high": best_ci.get("bootstrap_ci_high"),
        "win_rate": best_ci.get("win_rate"),
        "median_k_qual_diff": float(np.median(finite_diffs)) if finite_diffs.size else float("nan"),
        "nonnegative_k_qual_rate": float(np.mean(finite_diffs >= 0.0)) if finite_diffs.size else float("nan"),
        "status": "L0_PASS" if passed else "L0_FAIL",
    }


def hydrate_choice(item: Any) -> CaseKChoice:
    if isinstance(item, CaseKChoice):
        return item
    fields = {name: item[name] for name in CaseKChoice.__dataclass_fields__}
    return CaseKChoice(**fields)


def hydrate_task(item: Any) -> TaskMetrics:
    if isinstance(item, TaskMetrics):
        return item
    fields = {name: item[name] for name in TaskMetrics.__dataclass_fields__}
    return TaskMetrics(**fields)


def hydrate_result(payload: dict[str, Any]) -> dict[str, Any]:
    choices = [hydrate_choice(item) for item in payload.get("choices", [])]
    case_id = payload.get("case_id")
    if not case_id and choices:
        case_id = choices[0].case_id
    return {
        "case_id": case_id,
        "official_variant": payload.get("official_variant", "S0"),
        "choices": choices,
        "tasks": [hydrate_task(item) for item in payload.get("tasks", [])],
        "point_only": bool(payload.get("point_only", False)),
    }


def write_case_result(path: str | Path, result: dict[str, Any]) -> None:
    """Write one case so sibling VMs can assemble L0 without rerunning empymod."""

    write_json(
        path,
        {
            "case_id": result["case_id"],
            "official_variant": result.get("official_variant", "S0"),
            "point_only": bool(result.get("point_only", False)),
            "choices": [to_record(item) for item in result["choices"]],
            "tasks": [to_record(item) for item in result["tasks"]],
        },
    )


def load_case_results(paths) -> list[dict[str, Any]]:
    from .io import read_json

    return [hydrate_result(read_json(path)) for path in paths]


PILOT_CASE_RESULT_SCHEMA = "atem3d.adaptive_debye_mvp.pilot_case_result.v1"

_CASE_BY_ID: dict[str, LayeredCase] | None = None
_FIT_CACHE: dict[tuple[str, str], Any] = {}


def forward_unit_cache_key(case_id: str, model_id: str, waveform_id: str) -> str:
    return f"{case_id}:{model_id}:{waveform_id}"


def plan_point_forward_units(
    case: LayeredCase,
    *,
    k_values: tuple[int, ...] = K_PILOT,
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
) -> list[dict[str, Any]]:
    """List point-receiver cache units matching ``evaluate_pilot_case`` keys."""

    candidates = tuple(spec for spec in instantiate_candidates(case) if spec.K in k_values)
    fits = fit_case_candidates(case, candidates)
    units: list[dict[str, Any]] = []
    for model_id in ("exact:point", "noip:point"):
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case_id": case.case_id,
                    "model_id": model_id,
                    "waveform_id": waveform_id,
                    "receiver_kind": "point",
                    "cache_key": forward_unit_cache_key(case.case_id, model_id, waveform_id),
                }
            )
    for record in fits:
        if not record.valid:
            continue
        model_id = f"{record.candidate_id}:point"
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case_id": case.case_id,
                    "model_id": model_id,
                    "waveform_id": waveform_id,
                    "receiver_kind": "point",
                    "candidate_id": record.candidate_id,
                    "cache_key": forward_unit_cache_key(case.case_id, model_id, waveform_id),
                }
            )
    return units


def plan_disk_forward_units(
    case: LayeredCase,
    shortlist_ids: Iterable[str],
    *,
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
) -> list[dict[str, Any]]:
    """List official disk-average cache units for a candidate shortlist."""

    ids = sorted({str(item) for item in shortlist_ids})
    units: list[dict[str, Any]] = []
    for model_id in ("exact:disk", "noip:disk"):
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case_id": case.case_id,
                    "model_id": model_id,
                    "waveform_id": waveform_id,
                    "receiver_kind": "disk",
                    "cache_key": forward_unit_cache_key(case.case_id, model_id, waveform_id),
                }
            )
    for candidate_id in ids:
        model_id = f"{candidate_id}:disk"
        for waveform_id in waveform_ids:
            units.append(
                {
                    "case_id": case.case_id,
                    "model_id": model_id,
                    "waveform_id": waveform_id,
                    "receiver_kind": "disk",
                    "candidate_id": candidate_id,
                    "cache_key": forward_unit_cache_key(case.case_id, model_id, waveform_id),
                }
            )
    return units


def _case_by_id(case_id: str) -> LayeredCase:
    global _CASE_BY_ID
    if _CASE_BY_ID is None:
        from .registry import generate_all_cases

        _CASE_BY_ID = {item.case_id: item for item in generate_all_cases()}
    return _CASE_BY_ID[case_id]


def material_for_model_id(case: LayeredCase, model_id: str):
    """Rebuild the constitutive model encoded in a cache ``model_id``."""

    kind, _sep, _kind = model_id.rpartition(":")
    if kind == "exact":
        return exact_pelton_material(case)
    if kind == "noip":
        return nonpolarizable_material(case)
    cache_key = (case.case_id, kind)
    if cache_key not in _FIT_CACHE:
        spec = next(item for item in instantiate_candidates(case) if item.candidate_id == kind)
        records = fit_case_candidates(case, (spec,))
        if not records or not records[0].valid:
            raise RuntimeError(f"cannot build Debye material for {model_id}")
        _FIT_CACHE[cache_key] = records[0].fit
    return debye_material(case, _FIT_CACHE[cache_key], candidate_id=kind)


def execute_forward_unit(unit: dict[str, Any], cache_dir: str | Path) -> dict[str, Any]:
    """Compute or load one ``forward_response`` cache entry (worker entry)."""

    import os
    import time

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    from .case_bridge import load_cached_response

    started = time.perf_counter()
    cache_key = str(unit["cache_key"])
    cached = load_cached_response(cache_dir, cache_key)
    if cached is not None:
        return {
            "cache_key": cache_key,
            "from_cache": True,
            "seconds": time.perf_counter() - started,
            "shared_survey_hash": cached["hashes"]["shared_survey_hash"],
        }
    case = _case_by_id(str(unit["case_id"]))
    receivers = disk_receivers(case) if unit["receiver_kind"] == "disk" else point_receivers(case)
    result = forward_response(
        material_for_model_id(case, str(unit["model_id"])),
        case_geometry(case),
        case_waveform(str(unit["waveform_id"])),
        receivers,
        case_time_grid(),
        evaluation_transform(),
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    return {
        "cache_key": cache_key,
        "from_cache": bool(result.get("from_cache")),
        "seconds": time.perf_counter() - started,
        "shared_survey_hash": result["hashes"]["shared_survey_hash"],
    }


def _fit_summary(fits) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in fits:
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "K": item.K,
                "valid": item.valid,
                "spectral_error_s0": item.spectral_error_s0,
                "spectral_error_s1": item.spectral_error_s1,
                "condition_number": item.condition_number,
                "relative_dc_error": item.relative_dc_error,
                "optimizer_success": item.optimizer_success,
            }
        )
    return rows


def pilot_case_result_record(
    result: dict[str, Any],
    *,
    case: LayeredCase,
    provenance: dict[str, Any],
    k_values: tuple[int, ...] = K_PILOT,
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
) -> dict[str, Any]:
    """Compact per-case JSON with every field ``evaluate_l0`` reads."""

    shortlist = result.get("disk_shortlist") or {}
    shortlist = {str(key): list(value) for key, value in shortlist.items()}
    return {
        "schema": PILOT_CASE_RESULT_SCHEMA,
        "case_id": case.case_id,
        "case_hash": case.case_hash(),
        "split": case.split,
        "official_variant": result.get("official_variant", "S0"),
        "point_only": bool(result.get("point_only", False)),
        "k_values": [int(value) for value in k_values],
        "waveform_ids": list(waveform_ids),
        "receiver_ids": ["point", "disk_1.0", "disk_4.0"],
        "disk_shortlist": shortlist,
        "choices": [to_record(item) for item in result["choices"]],
        "tasks": [to_record(item) for item in result["tasks"]],
        "fit_summary": _fit_summary(result.get("fits", [])),
        "provenance": provenance,
    }


def write_pilot_case_result(
    path: str | Path,
    result: dict[str, Any],
    *,
    case: LayeredCase,
    provenance: dict[str, Any],
    k_values: tuple[int, ...] = K_PILOT,
    waveform_ids: tuple[str, ...] = PILOT_WAVEFORMS,
) -> Path:
    return write_json(
        path,
        pilot_case_result_record(
            result,
            case=case,
            provenance=provenance,
            k_values=k_values,
            waveform_ids=waveform_ids,
        ),
    )


def load_pilot_case_result(path) -> dict[str, Any]:
    """Load ``case_PG*.json`` into the dataclass form ``evaluate_l0`` expects."""

    from .io import read_json

    payload = read_json(path)

    def _finite_or_nan(value: Any) -> Any:
        return float("nan") if value is None else value

    def _hydrate_choice(item: Any) -> CaseKChoice:
        if isinstance(item, CaseKChoice):
            return item
        fields = {name: _finite_or_nan(item[name]) for name in CaseKChoice.__dataclass_fields__}
        return CaseKChoice(**fields)

    def _hydrate_task(item: Any) -> TaskMetrics:
        if isinstance(item, TaskMetrics):
            return item
        fields = {name: _finite_or_nan(item[name]) for name in TaskMetrics.__dataclass_fields__}
        return TaskMetrics(**fields)

    choices = [_hydrate_choice(item) for item in payload.get("choices", [])]
    case_id = payload.get("case_id")
    if not case_id and choices:
        case_id = choices[0].case_id
    return {
        "case_id": case_id,
        "official_variant": payload.get("official_variant", "S0"),
        "choices": choices,
        "tasks": [_hydrate_task(item) for item in payload.get("tasks", [])],
        "fits": [],
        "point_only": bool(payload.get("point_only", False)),
        "disk_shortlist": payload.get("disk_shortlist", {}),
        "case_hash": payload.get("case_hash"),
        "provenance": payload.get("provenance", {}),
    }


def write_oracle_gap_artifacts(output_dir: str | Path, pilot_results: list[dict[str, Any]], l0: dict[str, Any]) -> None:
    """Write Flow-2 machine-readable tables and the L0 decision."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pilot_results = [hydrate_result(result) for result in pilot_results]
    choices = [to_record(choice) for result in pilot_results for choice in result["choices"]]
    tasks = [to_record(task) for result in pilot_results for task in result["tasks"]]
    write_records_csv(root / "oracle_gap_choices.csv", choices)
    write_records_csv(root / "oracle_gap_tasks.csv", tasks)
    write_json(root / "L0_summary.json", l0)
    write_json(
        root / "official_spectral_variant.json",
        {"variant": pilot_results[0]["official_variant"] if pilot_results else "S0"},
    )
