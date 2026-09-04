"""Independent-test layered precompute (Flow 4 acceleration, no selector, no L2).

This module writes per-case JSON that PR 10 can consume after L1 freeze.
It does not train or select templates, does not evaluate the L2 gate, and
does not authorize 3-D.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .case_bridge import (
    case_geometry,
    case_time_grid,
    disk_receivers,
    evaluation_transform,
    exact_pelton_material,
    load_cached_response,
    nonpolarizable_material,
)
from .guards import ThreeDNotAuthorizedError, assert_split_readable
from .io import read_json, to_record, write_json
from .oracle_gap import (
    FitRecord,
    TaskMetrics,
    _forward_and_tasks,
    _run_model_waveforms,
    evaluate_pilot_case,
    fit_case_candidates,
    hydrate_choice,
    hydrate_task,
    load_pilot_case_result,
)
from .protocol_constants import (
    K_PILOT,
    TEST_RECEIVERS,
    TEST_WAVEFORMS,
    TILTED_RECEIVER_ID,
    normalize_tilted_normal,
)
from .receiver_metrics import ReceiverCase, evaluate_case
from .registry import LayeredCase, cases_for_split, generate_all_cases, instantiate_candidates


STAGE = "flow4_precompute"
SCHEMA = "atem3d.adaptive_debye_mvp.independent_test_case_result.v1"
DISK_COVERAGE = "l0_shortlist"
OFFICIAL_DIR_NAME = "flow4_independent_test"
PROJECTED_TASK_CHANNELS = ("Hn", "dBndt")


def independent_test_cases() -> tuple[LayeredCase, ...]:
    """Return the frozen TE01-TE10 cases. Readable only at the precompute stage."""

    cases = cases_for_split("independent_test", generate_all_cases())
    if len(cases) != 10:
        raise RuntimeError(f"expected 10 independent_test cases, got {len(cases)}")
    for case in cases:
        assert_split_readable(case.split, stage=STAGE)
        if case.split != "independent_test":
            raise ValueError(f"{case.case_id} is not independent_test")
        if case.case_id[:2] != "TE":
            raise ValueError(f"unexpected independent_test id {case.case_id}")
    return cases


def frozen_official_variant(flow2_dir: str | Path) -> str:
    """Return the frozen L0 official spectral variant (material-spectrum only)."""

    payload = read_json(Path(flow2_dir) / "official_spectral_variant.json")
    variant = str(payload.get("variant", "")).strip()
    if variant not in {"S0", "S1"}:
        raise ValueError("official_spectral_variant.json must contain S0 or S1")
    return variant


def registry_hashes(generated_root: str | Path) -> dict[str, str]:
    manifest = read_json(Path(generated_root) / "registry_manifest.json")
    return {
        "case_registry_sha256": str(manifest["case_registry_sha256"]),
        "candidate_registry_sha256": str(manifest["candidate_registry_sha256"]),
        "candidate_config_hash": str(manifest["candidate_config_hash"]),
    }


def project_onto_normal(response: dict[str, Any], normal) -> dict[str, Any]:
    """Project six-channel H and dB/dt onto a unit coil normal."""

    vector = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("tilted normal must be a finite non-zero vector")
    unit = vector / norm
    data = np.asarray(response["data"], dtype=float)
    if data.ndim != 3 or data.shape[-1] != 6:
        raise ValueError("response data must have shape (n_times, n_receivers, 6)")
    channels = [str(name) for name in response.get("channels", ("Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt"))]
    index = {name: i for i, name in enumerate(channels)}
    for name in ("Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt"):
        if name not in index:
            raise ValueError(f"response missing channel {name}")
    h = (
        data[:, :, index["Hx"]] * unit[0]
        + data[:, :, index["Hy"]] * unit[1]
        + data[:, :, index["Hz"]] * unit[2]
    )
    dbdt = (
        data[:, :, index["dBxdt"]] * unit[0]
        + data[:, :, index["dBydt"]] * unit[1]
        + data[:, :, index["dBzdt"]] * unit[2]
    )
    stacked = np.stack([h, dbdt], axis=-1)
    return {
        "data": stacked,
        "channels": list(PROJECTED_TASK_CHANNELS),
        "receiver_labels": [str(label) for label in response["receiver_labels"]],
        "times": np.asarray(response["times"], dtype=float),
        "hashes": dict(response.get("hashes", {})),
    }


def projected_channel_column(projected: dict[str, Any], receiver_index: int) -> dict[str, np.ndarray]:
    data = np.asarray(projected["data"], dtype=float)
    names = list(projected["channels"])
    return {name: data[:, int(receiver_index), index].copy() for index, name in enumerate(names)}


def _channel_p95(metrics, names: tuple[str, ...]) -> float:
    values = [metrics.channels[name].total_p95 for name in names if name in metrics.channels]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else float("nan")


def _evaluate_projected_task(
    *,
    case: LayeredCase,
    candidate_id: str,
    K: int,
    waveform_id: str,
    receiver_index: int,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    times: np.ndarray,
) -> TaskMetrics:
    metrics = evaluate_case(
        ReceiverCase(
            case_id=f"{case.case_id}:{waveform_id}:{TILTED_RECEIVER_ID}:{receiver_index}",
            times=times,
            reference=reference,
            candidate=candidate,
            reference_no_ip=baseline,
            groups=(case.case_id, waveform_id, TILTED_RECEIVER_ID),
            output_dt=float(np.median(np.diff(times))),
        )
    )
    return TaskMetrics(
        case_id=case.case_id,
        candidate_id=candidate_id,
        K=K,
        waveform_id=waveform_id,
        receiver_id=TILTED_RECEIVER_ID,
        receiver_index=int(receiver_index),
        total_p95=metrics.case_total_p95,
        ip_increment_nrmse=metrics.case_ip_increment_nrmse,
        passed=metrics.passed,
        unexplained_sign_flips=metrics.unexplained_sign_flips,
        peak_time_error_steps_max=metrics.peak_time_error_steps_max,
        h_p95=_channel_p95(metrics, ("Hn",)),
        dbdt_p95=_channel_p95(metrics, ("dBndt",)),
    )


def _cache_key(case_id: str, model_id: str, waveform_id: str) -> str:
    return f"{case_id}:{model_id}:{waveform_id}"


def _require_cached_point(cache_dir: str | Path, case_id: str, model_id: str, waveform_id: str) -> dict[str, Any]:
    cached = load_cached_response(cache_dir, _cache_key(case_id, model_id, waveform_id))
    if cached is None:
        raise FileNotFoundError(
            f"missing point cache {case_id}:{model_id}:{waveform_id}; "
            "tilted-coil projection refuses to skip"
        )
    return cached


def tilted_tasks_from_cache(
    case: LayeredCase,
    fits: list[FitRecord],
    *,
    waveform_ids: tuple[str, ...] = TEST_WAVEFORMS,
    cache_dir: str | Path,
) -> list[TaskMetrics]:
    """Score the holdout tilted coil from cached point six-channel responses."""

    if case.sensor_frame != "tilted":
        return []
    normal = case.sensor_normal
    tasks: list[TaskMetrics] = []
    for waveform_id in waveform_ids:
        reference = project_onto_normal(
            _require_cached_point(cache_dir, case.case_id, "exact:point", waveform_id),
            normal,
        )
        baseline = project_onto_normal(
            _require_cached_point(cache_dir, case.case_id, "noip:point", waveform_id),
            normal,
        )
        n_receivers = int(np.asarray(reference["data"]).shape[1])
        times = np.asarray(reference["times"], dtype=float)
        for record in fits:
            if not record.valid:
                continue
            candidate = project_onto_normal(
                _require_cached_point(
                    cache_dir,
                    case.case_id,
                    f"{record.candidate_id}:point",
                    waveform_id,
                ),
                normal,
            )
            for receiver_index in range(n_receivers):
                tasks.append(
                    _evaluate_projected_task(
                        case=case,
                        candidate_id=record.candidate_id,
                        K=record.K,
                        waveform_id=waveform_id,
                        receiver_index=receiver_index,
                        reference=projected_channel_column(reference, receiver_index),
                        candidate=projected_channel_column(candidate, receiver_index),
                        baseline=projected_channel_column(baseline, receiver_index),
                        times=times,
                    )
                )
    return tasks


def disk_shortlist_from_tasks(tasks: list[TaskMetrics]) -> dict[str, list[str]]:
    collected: dict[int, set[str]] = {}
    for item in tasks:
        if item.receiver_id in {"disk_1.0", "disk_4.0"}:
            collected.setdefault(int(item.K), set()).add(str(item.candidate_id))
    return {str(key): sorted(collected[key]) for key in sorted(collected)}


def fit_summary(fits: list[FitRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in fits:
        rows.append(
            {
                "K": int(record.K),
                "candidate_id": str(record.candidate_id),
                "condition_number": float(record.condition_number),
                "optimizer_success": bool(record.optimizer_success),
                "relative_dc_error": float(record.relative_dc_error),
                "spectral_error_s0": float(record.spectral_error_s0),
                "spectral_error_s1": float(record.spectral_error_s1),
                "valid": bool(record.valid),
            }
        )
    return rows


def _git_field(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_provenance(
    *,
    started_at: datetime,
    wall_seconds: float,
    roads_workers: int,
    point_hash: str,
    disk_hash: str,
    official_variant_source: str,
    empymod_version: str,
) -> dict[str, Any]:
    transform = evaluation_transform()
    return {
        "empymod_version": str(empymod_version),
        "ft_pts_per_dec": int(transform.ft_pts_per_dec),
        "git_branch": _git_field(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git_field(["git", "rev-parse", "HEAD"]),
        "l2_evaluated": False,
        "official_variant_source": str(official_variant_source),
        "recorded_at": started_at.astimezone(timezone.utc).isoformat(),
        "roads_workers": int(roads_workers),
        "selected_template_written": False,
        "selector_read": False,
        "shared_survey_hash": {"disk": str(disk_hash), "point": str(point_hash)},
        "three_d_run": False,
        "transform_label": str(transform.label),
        "wall_seconds": float(wall_seconds),
    }


def _survey_hash_from_cache(cache_dir: str | Path, case_id: str, model_id: str, waveform_id: str) -> str:
    cached = load_cached_response(cache_dir, _cache_key(case_id, model_id, waveform_id))
    if cached is None:
        raise FileNotFoundError(f"missing cache for survey hash {case_id}:{model_id}:{waveform_id}")
    return str(cached["hashes"]["shared_survey_hash"])


def tilted_projection_record(case: LayeredCase) -> dict[str, Any]:
    return {
        "channels": list(PROJECTED_TASK_CHANNELS),
        "derived_from": "point",
        "disk_projection": False,
        "normal": list(normalize_tilted_normal() if case.sensor_frame == "tilted" else case.sensor_normal),
    }


def write_independent_case_result(
    path: str | Path,
    result: dict[str, Any],
    *,
    case: LayeredCase,
    k_values: tuple[int, ...],
    waveform_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...] = TEST_RECEIVERS,
    disk_shortlist: dict[str, list[str]] | None = None,
    provenance: dict[str, Any],
    tilted_projection: dict[str, Any] | None = None,
    registry: dict[str, str] | None = None,
) -> None:
    """Write one independent_test case JSON. Never writes selected_template files."""

    destination = Path(path)
    if destination.name.startswith("selected_template"):
        raise ValueError("independent_test precompute must not write selected_template files")
    tasks = result["tasks"]
    payload = {
        "schema": SCHEMA,
        "case_id": case.case_id,
        "split": case.split,
        "case_hash": case.case_hash(),
        "official_variant": result.get("official_variant", "S0"),
        "point_only": False,
        "k_values": [int(value) for value in k_values],
        "waveform_ids": list(waveform_ids),
        "receiver_ids": list(receiver_ids),
        "disk_coverage": DISK_COVERAGE,
        "disk_shortlist": disk_shortlist if disk_shortlist is not None else disk_shortlist_from_tasks(tasks),
        "fit_summary": result.get("fit_summary") or fit_summary(result.get("fits", [])),
        "choices": [to_record(item) for item in result["choices"]],
        "tasks": [to_record(item) for item in tasks],
        "tilted_projection": tilted_projection if tilted_projection is not None else tilted_projection_record(case),
        "registry_hashes": registry or {},
        "provenance": provenance,
    }
    write_json(destination, payload)


def load_independent_case_result(path: str | Path) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict) or str(raw.get("schema")) != SCHEMA:
        raise ValueError(f"{path} is not {SCHEMA}")
    loaded = load_pilot_case_result(path)
    loaded["schema"] = SCHEMA
    loaded["split"] = raw.get("split", "independent_test")
    loaded["disk_coverage"] = raw.get("disk_coverage", DISK_COVERAGE)
    loaded["tilted_projection"] = raw.get("tilted_projection", {})
    loaded["registry_hashes"] = raw.get("registry_hashes", {})
    loaded["provenance"] = raw.get("provenance", {})
    loaded["waveform_ids"] = raw.get("waveform_ids", list(TEST_WAVEFORMS))
    loaded["receiver_ids"] = raw.get("receiver_ids", list(TEST_RECEIVERS))
    loaded["k_values"] = raw.get("k_values", list(K_PILOT))
    loaded["fit_summary"] = raw.get("fit_summary", [])
    return loaded


def case_is_complete(
    path: str | Path,
    *,
    waveform_ids: tuple[str, ...] = TEST_WAVEFORMS,
    receiver_ids: tuple[str, ...] = TEST_RECEIVERS,
) -> bool:
    destination = Path(path)
    if not destination.is_file():
        return False
    try:
        payload = read_json(destination)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict) or str(payload.get("schema")) != SCHEMA:
        return False
    if bool(payload.get("point_only", False)):
        return False
    have_wave = {str(item) for item in payload.get("waveform_ids", [])}
    have_rx = {str(item) for item in payload.get("receiver_ids", [])}
    if any(item not in have_wave for item in waveform_ids):
        return False
    if any(item not in have_rx for item in receiver_ids):
        return False
    tasks = payload.get("tasks", [])
    choices = payload.get("choices", [])
    return bool(tasks) and bool(choices) and bool(payload.get("case_hash"))


def extend_disk_tasks(
    case: LayeredCase,
    result: dict[str, Any],
    *,
    candidate_ids: tuple[str, ...],
    waveform_ids: tuple[str, ...] = TEST_WAVEFORMS,
    cache_dir: str | Path,
    backend=None,
) -> list[TaskMetrics]:
    """Compute missing official disk tasks for named templates (post-L1 top-up)."""

    wanted = {str(item) for item in candidate_ids}
    existing = {
        (item.candidate_id, item.waveform_id, item.receiver_id, item.receiver_index)
        for item in result.get("tasks", [])
        if isinstance(item, TaskMetrics) and item.receiver_id in {"disk_1.0", "disk_4.0"}
    }
    candidates = tuple(spec for spec in instantiate_candidates(case) if spec.candidate_id in wanted)
    fits = [record for record in fit_case_candidates(case, candidates) if record.valid]
    if not fits:
        return []
    geometry = case_geometry(case)
    times = case_time_grid()
    transform = evaluation_transform()
    disks = disk_receivers(case)
    reference_disk = _run_model_waveforms(
        case,
        exact_pelton_material(case),
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
        nonpolarizable_material(case),
        receivers=disks,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        backend=backend,
        model_id="noip:disk",
        times=times,
        geometry=geometry,
        transform=transform,
    )
    extra: list[TaskMetrics] = []
    for record in fits:
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
        for item in disk_tasks:
            key = (item.candidate_id, item.waveform_id, item.receiver_id, item.receiver_index)
            if key not in existing:
                extra.append(item)
    return extra


def refuse_3d_always() -> None:
    """Independent-test precompute never starts 3-D, even if a later gate passed."""

    raise ThreeDNotAuthorizedError(
        "independent_test precompute is not a 3-D authorization step; 3-D was not run"
    )


def evaluate_independent_case(
    case: LayeredCase,
    *,
    cache_dir: str | Path,
    k_values: tuple[int, ...] = K_PILOT,
    waveform_ids: tuple[str, ...] = TEST_WAVEFORMS,
    official_variant: str,
    disk_shortlist: int = 2,
    include_disks: bool = True,
    backend=None,
) -> dict[str, Any]:
    """Run the official lagged-DLF matrix for one independent_test case."""

    if case.split != "independent_test":
        raise ValueError(f"{case.case_id} is not an independent_test case")
    assert_split_readable(case.split, stage=STAGE)
    result = evaluate_pilot_case(
        case,
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
        backend=backend,
        k_values=k_values,
        official_variant=official_variant,
        disk_shortlist=disk_shortlist,
        include_disks=include_disks,
    )
    tilted = tilted_tasks_from_cache(
        case,
        result["fits"],
        waveform_ids=waveform_ids,
        cache_dir=cache_dir,
    )
    tasks = list(result["tasks"]) + list(tilted)
    return {
        "case_id": case.case_id,
        "official_variant": result.get("official_variant", official_variant),
        "fits": result.get("fits", []),
        "choices": result["choices"],
        "tasks": tasks,
        "point_only": False,
        "fit_summary": fit_summary(result.get("fits", [])),
        "disk_shortlist": disk_shortlist_from_tasks(tasks),
    }


def survey_hashes(case_id: str, cache_dir: str | Path, waveform_id: str = "W0") -> dict[str, str]:
    point = _survey_hash_from_cache(cache_dir, case_id, "exact:point", waveform_id)
    disk = _survey_hash_from_cache(cache_dir, case_id, "exact:disk", waveform_id)
    return {"point": point, "disk": disk}


def empymod_version_text() -> str:
    try:
        import empymod

        return str(empymod.__version__)
    except ImportError:
        return "missing"


def hydrate_independent_tasks(items: list[Any]) -> list[TaskMetrics]:
    return [hydrate_task(item) if not isinstance(item, TaskMetrics) else item for item in items]


def hydrate_independent_choices(items: list[Any]) -> list[Any]:
    return [hydrate_choice(item) if not hasattr(item, "spectral_id") else item for item in items]
