"""Cheap empymod reference audit for the layered Cole-Cole / Debye wrapper.

These checks are a smoke-geometry convergence audit, not an 8-case
oracle-gap sweep. ``reference_type`` is set to
``empymod_converged_cole_cole`` only when the cheap audit passes for an
exact Pelton/Cole-Cole material on the smoke geometry and the numbers
are tied to the approved production identity.

Disk-order variants reuse the same Gauss-Legendre-in-area × mid-azimuth
algorithm as ``receivers.AverageReceiver`` with configurable order. This
module does not modify ``receivers.py``.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Sequence

import numpy as np

from atem3d.adaptive_debye_mvp import layered_forward as lf
from atem3d.adaptive_debye_mvp.layered_forward import (
    CHANNELS,
    DebyeCandidateMaterial,
    ExactPeltonMaterial,
    LayeredGeometry,
    Material,
    NonPolarizableMaterial,
    ReceiverSpec,
    TimeGrid,
    TransformSettings,
    WaveformSpec,
    compute_layered_response,
    disk_average_channel_pair,
)
from atem3d.receivers import _component_axis, _disk_basis, _normalised_vector


DEFAULT_TOLERANCE = 1.0e-3
DEFAULT_FLOOR_FRACTION = 1.0e-2
NEAR_ZERO_FRACTION = 1.0e-3
GATING_CHECKS = (
    "frequency_range_expansion",
    "frequency_sampling_density",
    "fourier_method_pair",
    "source_quadrature_9_vs_17",
    "hankel_filter_pair",
    "disk_quadrature_order",
    "six_channel_signs_and_near_zero",
)
REQUIRED_PERFORMED_CHECKS = (
    "fourier_method_pair",
    "source_quadrature_9_vs_17",
    "frequency_range_expansion",
    "frequency_sampling_density",
    "six_channel_signs_and_near_zero",
)
H_CHANNELS = ("Hx", "Hy", "Hz")
DBDT_CHANNELS = ("dBxdt", "dBydt", "dBzdt")


def configurable_disk_quadrature(
    center,
    normal_axis: str,
    radius_m: float,
    *,
    radial_order: int,
    azimuth_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same disk rule as ``AverageReceiver``, with configurable orders."""

    radial_order = int(radial_order)
    azimuth_count = int(azimuth_count)
    if radial_order < 1 or azimuth_count < 1:
        raise ValueError("radial_order and azimuth_count must be positive")
    center = np.asarray(center, dtype=float)
    radius = float(radius_m)
    if radius <= 0.0:
        raise ValueError("radius_m must be positive")
    normal = _normalised_vector(_component_axis(f"H{normal_axis}"), "normal")
    basis_u, basis_v = _disk_basis(normal)
    nodes, radial_weights = np.polynomial.legendre.leggauss(radial_order)
    unit_area_nodes = 0.5 * (nodes + 1.0)
    unit_area_weights = 0.5 * radial_weights
    angles = 2.0 * np.pi * (
        np.arange(azimuth_count, dtype=float) + 0.5
    ) / azimuth_count
    points = []
    weights = []
    for unit_area, radial_weight in zip(unit_area_nodes, unit_area_weights):
        radial_distance = radius * np.sqrt(unit_area)
        for angle in angles:
            direction = np.cos(angle) * basis_u + np.sin(angle) * basis_v
            points.append(center + radial_distance * direction)
            weights.append(radial_weight / azimuth_count)
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def floor_relative_error(
    reference: np.ndarray,
    variant: np.ndarray,
    *,
    floor_fraction: float = DEFAULT_FLOOR_FRACTION,
    channel_names: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Component-wise floor-relative error matching ``compare_magnetic6``."""

    reference = np.asarray(reference, dtype=float)
    variant = np.asarray(variant, dtype=float)
    if reference.shape != variant.shape:
        raise ValueError("reference and variant shapes must match")
    if channel_names is None:
        if reference.shape[-1] != len(CHANNELS):
            raise ValueError("channel_names required when the last axis is not 6")
        channel_names = CHANNELS
    exclude_set = set(exclude)
    per_channel: dict[str, float] = {}
    global_max = 0.0
    for index, name in enumerate(channel_names):
        ref = reference[..., index]
        other = variant[..., index]
        peak = float(np.max(np.abs(ref)))
        floor = max(float(floor_fraction) * peak, np.finfo(float).tiny)
        scaled = np.abs(other - ref) / np.maximum(np.abs(ref), floor)
        max_error = float(np.max(scaled))
        per_channel[name] = max_error
        if name not in exclude_set:
            global_max = max(global_max, max_error)
    return {
        "per_channel_max_floor_relative_error": per_channel,
        "global_max": global_max,
    }


def run_reference_audit(
    material: Material,
    geometry: LayeredGeometry,
    waveform: WaveformSpec,
    receivers: Sequence[ReceiverSpec],
    times: TimeGrid,
    transform_settings: TransformSettings,
    *,
    baseline: dict[str, Any] | None = None,
    checks: Sequence[str] = GATING_CHECKS,
    tolerance: float = DEFAULT_TOLERANCE,
    floor_fraction: float = DEFAULT_FLOOR_FRACTION,
    point_receiver_index: int = 0,
    disk_receiver_index: int | None = None,
    disk_variant: tuple[int, int] = (4, 16),
    disk_axes: Sequence[str] = ("z",),
    include_informational_fftlog: bool = False,
    backend=None,
) -> dict[str, Any]:
    """Run the cheap gating checks on one point (and optional disk) receiver."""

    receivers = tuple(receivers)
    requested = tuple(checks)
    point = receivers[point_receiver_index]
    if point.kind != "point":
        raise ValueError("point_receiver_index must select a point receiver")
    point_receivers = (point,)
    if baseline is None:
        baseline = compute_layered_response(
            material,
            geometry,
            waveform,
            point_receivers,
            times,
            transform_settings,
            backend=backend,
        )
    baseline_data = _point_column(baseline, 0)
    near_zero = _near_zero_channel_names(baseline_data)

    report_checks: dict[str, Any] = {}
    if "frequency_range_expansion" in requested:
        report_checks["frequency_range_expansion"] = _check_frequency_range(
            material,
            geometry,
            waveform,
            point_receivers,
            times,
            transform_settings,
            baseline_data,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            exclude=near_zero,
        )
    if "frequency_sampling_density" in requested:
        report_checks["frequency_sampling_density"] = _check_frequency_sampling(
            material,
            geometry,
            waveform,
            point_receivers,
            times,
            transform_settings,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            exclude=near_zero,
        )
    if "fourier_method_pair" in requested:
        report_checks["fourier_method_pair"] = _check_fourier_pair(
            material,
            geometry,
            waveform,
            point_receivers,
            times,
            transform_settings,
            baseline_data,
            baseline_transform=transform_settings,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            exclude=near_zero,
        )
    if "source_quadrature_9_vs_17" in requested:
        report_checks["source_quadrature_9_vs_17"] = _compare_transform_variant(
            name="source_quadrature_9_vs_17",
            material=material,
            geometry=geometry,
            waveform=waveform,
            receivers=point_receivers,
            times=times,
            baseline_settings=transform_settings,
            variant_settings=replace(transform_settings, srcpts=17, label="srcpts_17"),
            baseline_data=baseline_data,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            note="finite-source quadrature 9 vs 17",
            exclude=near_zero,
        )
    if "hankel_filter_pair" in requested:
        report_checks["hankel_filter_pair"] = _compare_transform_variant(
            name="hankel_filter_pair",
            material=material,
            geometry=geometry,
            waveform=waveform,
            receivers=point_receivers,
            times=times,
            baseline_settings=transform_settings,
            variant_settings=replace(
                transform_settings,
                ht_filter="key_401_2009",
                label="hankel_key_401_2009",
            ),
            baseline_data=baseline_data,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            note="Hankel DLF key_201_2009 vs key_401_2009",
            exclude=near_zero,
        )
    if "disk_quadrature_order" in requested:
        report_checks["disk_quadrature_order"] = _check_disk_quadrature(
            material,
            geometry,
            waveform,
            receivers,
            times,
            transform_settings,
            disk_receiver_index=disk_receiver_index,
            disk_variant=disk_variant,
            disk_axes=disk_axes,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            exclude=near_zero,
        )
    if "six_channel_signs_and_near_zero" in requested:
        report_checks["six_channel_signs_and_near_zero"] = _check_signs_and_near_zero(
            baseline,
            material,
            floor_fraction=floor_fraction,
        )
    if include_informational_fftlog:
        report_checks["fftlog_informational"] = _compare_transform_variant(
            name="fftlog_informational",
            material=material,
            geometry=geometry,
            waveform=waveform,
            receivers=point_receivers,
            times=times,
            baseline_settings=transform_settings,
            variant_settings=replace(
                transform_settings,
                ft="fftlog",
                label="fftlog_informational",
            ),
            baseline_data=baseline_data,
            backend=backend,
            tolerance=tolerance,
            floor_fraction=floor_fraction,
            gating=False,
            note="fftlog is informational only; not a gating cross-check",
            exclude=near_zero,
        )

    if "six_channel_signs_and_near_zero" in report_checks:
        near_zero = list(
            report_checks["six_channel_signs_and_near_zero"].get(
                "near_zero_channels", near_zero
            )
        )
    required_ok = all(
        bool(report_checks.get(name, {}).get("performed"))
        and bool(report_checks.get(name, {}).get("passed"))
        for name in REQUIRED_PERFORMED_CHECKS
    )
    gating_failed = any(
        item.get("gating", True) and item.get("performed") and not item.get("passed")
        for item in report_checks.values()
    )
    all_gating_passed = required_ok and not gating_failed
    approved_identity_tied = _approved_identity_tied(
        transform_settings,
        report_checks.get("fourier_method_pair"),
    )
    reference_type = _classify_reference_type(
        material,
        report_checks,
        requested,
        all_gating_passed=all_gating_passed,
        approved_identity_tied=approved_identity_tied,
        required_ok=required_ok,
    )
    return {
        "schema": "atem3d.adaptive_debye_mvp.reference_audit.v1",
        "baseline_hashes": dict(baseline.get("hashes", {})),
        "tolerance": float(tolerance),
        "floor_fraction": float(floor_fraction),
        "checks": report_checks,
        "near_zero_channels": list(near_zero),
        "all_gating_passed": bool(all_gating_passed),
        "approved_identity_tied": bool(approved_identity_tied),
        "reference_type": reference_type,
    }


def annotate_reference_type(response: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``response`` stamped with the audit classification."""

    annotated = dict(response)
    annotated["reference_type"] = str(audit["reference_type"])
    annotated["audit_summary"] = {
        "all_gating_passed": bool(audit["all_gating_passed"]),
        "approved_identity_tied": bool(audit["approved_identity_tied"]),
        "near_zero_channels": list(audit.get("near_zero_channels", [])),
        "checks": {
            name: {
                "performed": bool(item.get("performed")),
                "passed": bool(item.get("passed")),
                "gating": bool(item.get("gating", True)),
                "global_max": item.get("global_max"),
            }
            for name, item in audit.get("checks", {}).items()
        },
    }
    return annotated


def _check_frequency_range(
    material,
    geometry,
    waveform,
    receivers,
    times,
    transform_settings,
    baseline_data,
    *,
    backend,
    tolerance,
    floor_fraction,
    exclude=(),
) -> dict[str, Any]:
    expanded = TimeGrid(
        times_s=tuple(
            float(value)
            for value in np.unique(
                np.concatenate(
                    [
                        times.as_array(),
                        np.logspace(-6, -1, 31),
                    ]
                )
            )
        )
    )
    started = time.perf_counter()
    variant = compute_layered_response(
        material,
        geometry,
        waveform,
        receivers,
        expanded,
        transform_settings,
        backend=backend,
    )
    cost = time.perf_counter() - started
    shared = _values_at_times(
        variant["times"],
        _point_column(variant, 0),
        times.as_array(),
    )
    metrics = floor_relative_error(
        baseline_data,
        shared,
        floor_fraction=floor_fraction,
        exclude=exclude,
    )
    note = "expanded time set logspace(-6,-1) union, compared on the 31 shared times"
    if transform_settings.ft_pts_per_dec == 0:
        note += "; standard DLF per-time sets are unchanged by padding"
    return _check_record(
        "frequency_range_expansion",
        performed=True,
        passed=metrics["global_max"] <= tolerance,
        baseline="shared 31 times",
        variant="padded times union logspace(-6,-1,31)",
        metrics=metrics,
        cost_s=cost,
        note=note,
    )


def _check_frequency_sampling(
    material,
    geometry,
    waveform,
    receivers,
    times,
    transform_settings,
    *,
    backend,
    tolerance,
    floor_fraction,
    exclude=(),
) -> dict[str, Any]:
    coarse = replace(
        transform_settings,
        ft_pts_per_dec=30,
        label="splined_dlf_30",
    )
    dense = replace(
        transform_settings,
        ft_pts_per_dec=90,
        label="splined_dlf_90",
    )
    started = time.perf_counter()
    coarse_response = compute_layered_response(
        material, geometry, waveform, receivers, times, coarse, backend=backend
    )
    dense_response = compute_layered_response(
        material, geometry, waveform, receivers, times, dense, backend=backend
    )
    cost = time.perf_counter() - started
    metrics = floor_relative_error(
        _point_column(coarse_response, 0),
        _point_column(dense_response, 0),
        floor_fraction=floor_fraction,
        exclude=exclude,
    )
    return _check_record(
        "frequency_sampling_density",
        performed=True,
        passed=metrics["global_max"] <= tolerance,
        baseline="ft pts_per_dec=30",
        variant="ft pts_per_dec=90",
        metrics=metrics,
        cost_s=cost,
        note="splined DLF density pair; not compared to the lagged/standard baseline",
    )


def _check_fourier_pair(
    material,
    geometry,
    waveform,
    receivers,
    times,
    transform_settings,
    baseline_data,
    *,
    baseline_transform,
    backend,
    tolerance,
    floor_fraction,
    exclude=(),
) -> dict[str, Any]:
    if baseline_transform.ft_pts_per_dec == 0:
        variant_settings = replace(
            baseline_transform,
            ft_pts_per_dec=-1,
            label="lagged_dlf",
        )
        variant_label = "ft pts_per_dec=-1"
    else:
        variant_settings = replace(
            baseline_transform,
            ft_pts_per_dec=0,
            label="approved_production_identity",
        )
        variant_label = "ft pts_per_dec=0 (approved identity)"
    return _compare_transform_variant(
        name="fourier_method_pair",
        material=material,
        geometry=geometry,
        waveform=waveform,
        receivers=receivers,
        times=times,
        baseline_settings=transform_settings,
        variant_settings=variant_settings,
        baseline_data=baseline_data,
        backend=backend,
        tolerance=tolerance,
        floor_fraction=floor_fraction,
        note=f"pairs the audited numbers with {variant_label}",
        exclude=exclude,
    )


def _check_disk_quadrature(
    material,
    geometry,
    waveform,
    receivers,
    times,
    transform_settings,
    *,
    disk_receiver_index,
    disk_variant,
    disk_axes,
    backend,
    tolerance,
    floor_fraction,
    exclude=(),
) -> dict[str, Any]:
    if disk_receiver_index is None:
        disk_receiver_index = next(
            (index for index, receiver in enumerate(receivers) if receiver.kind == "disk_average"),
            None,
        )
    if disk_receiver_index is None:
        return _check_record(
            "disk_quadrature_order",
            performed=False,
            passed=True,
            baseline=None,
            variant=None,
            metrics={"per_channel_max_floor_relative_error": {}, "global_max": 0.0},
            cost_s=0.0,
            note="no disk receiver available",
        )
    disk = receivers[disk_receiver_index]
    if disk.kind != "disk_average":
        raise ValueError("disk_receiver_index must select a disk_average receiver")
    resistivity = lf.build_empymod_resistivity(material)
    resolved = lf.resolve_waveform(waveform)
    call_kwargs = transform_settings.empymod_kwargs(geometry.n_layers)
    backend = lf._require_empymod(backend)
    axes = tuple(disk_axes)
    started = time.perf_counter()
    production_pairs = []
    variant_pairs = []
    channel_names: list[str] = []
    for axis in axes:
        production, _record = disk_average_channel_pair(
            axis,
            disk.location,
            float(disk.radius_m),
            geometry=geometry,
            times=times.as_array(),
            resistivity=resistivity,
            waveform=resolved,
            waveform_spec=waveform,
            transform_settings=transform_settings,
            call_kwargs=call_kwargs,
            backend=backend,
        )
        points, weights = configurable_disk_quadrature(
            disk.location,
            axis,
            float(disk.radius_m),
            radial_order=int(disk_variant[0]),
            azimuth_count=int(disk_variant[1]),
        )
        variant, _variant_record = disk_average_channel_pair(
            axis,
            disk.location,
            float(disk.radius_m),
            geometry=geometry,
            times=times.as_array(),
            resistivity=resistivity,
            waveform=resolved,
            waveform_spec=waveform,
            transform_settings=transform_settings,
            call_kwargs=call_kwargs,
            backend=backend,
            points=points,
            weights=weights,
        )
        production_pairs.append(production)
        variant_pairs.append(variant)
        channel_names.extend(lf._AXIS_TO_CHANNELS[axis])
    cost = time.perf_counter() - started
    production_data = np.concatenate(production_pairs, axis=1)
    variant_data = np.concatenate(variant_pairs, axis=1)
    metrics = floor_relative_error(
        production_data,
        variant_data,
        floor_fraction=floor_fraction,
        channel_names=channel_names,
        exclude=exclude,
    )
    return _check_record(
        "disk_quadrature_order",
        performed=True,
        passed=metrics["global_max"] <= tolerance,
        baseline="radial_order=3, azimuth_count=12",
        variant=f"radial_order={disk_variant[0]}, azimuth_count={disk_variant[1]}",
        metrics=metrics,
        cost_s=cost,
        note=f"component-normal disks on axes {list(axes)}",
    )


def _check_signs_and_near_zero(
    baseline: dict[str, Any],
    material: Material,
    *,
    floor_fraction: float,
) -> dict[str, Any]:
    data = np.asarray(baseline["data"], dtype=float)
    if data.ndim != 3:
        raise ValueError("baseline data must have shape (n_times, n_receivers, 6)")
    column = data[:, 0, :]
    peaks, near_zero = _channel_peaks_and_near_zero(column)
    route_ok = True
    sign_agreement = {}
    for label, audit in baseline.get("dbdt_route_audit", {}).items():
        if not audit:
            continue
        if audit.get("performed") and not audit.get("passed", True):
            route_ok = False
        metrics = audit.get("component_metrics") or {}
        for component, item in metrics.items():
            sign_agreement[f"{label}:{component}"] = item.get("sign_agreement_fraction")
            if (
                audit.get("performed")
                and item.get("sign_agreement_fraction") is not None
                and float(item["sign_agreement_fraction"]) < 1.0
            ):
                route_ok = False

    decay_ok = True
    decay_gating = isinstance(material, NonPolarizableMaterial) or (
        isinstance(material, ExactPeltonMaterial)
        and all(value == 0.0 for value in material.chargeability)
    )
    decay_details = {}
    last = column[-1]
    for magnetic, derivative in zip(H_CHANNELS, DBDT_CHANNELS):
        h_value = float(last[CHANNELS.index(magnetic)])
        dbdt_value = float(last[CHANNELS.index(derivative)])
        if magnetic in near_zero or derivative in near_zero:
            decay_details[magnetic] = "skipped_near_zero"
            continue
        agreed = np.sign(dbdt_value) == -np.sign(h_value) or h_value == 0.0
        decay_details[magnetic] = bool(agreed)
        if decay_gating and not agreed:
            decay_ok = False

    passed = route_ok and (decay_ok if decay_gating else True) and True
    return {
        "performed": True,
        "passed": bool(passed),
        "gating": True,
        "baseline": "point six-channel response",
        "variant": "native-b vs -mu0*H impulse plus late-time decay sign",
        "per_channel_max_floor_relative_error": {},
        "global_max": 0.0,
        "cost_s": 0.0,
        "note": "near-zero channels are excluded from relative gating",
        "near_zero_channels": near_zero,
        "sign_agreement_fraction": sign_agreement,
        "decay_sign_gating": decay_gating,
        "decay_sign": decay_details,
        "dbdt_route_ok": route_ok,
        "channel_peaks": peaks,
    }


def _compare_transform_variant(
    *,
    name: str,
    material,
    geometry,
    waveform,
    receivers,
    times,
    baseline_settings,
    variant_settings,
    baseline_data,
    backend,
    tolerance,
    floor_fraction,
    note: str,
    gating: bool = True,
    exclude=(),
) -> dict[str, Any]:
    started = time.perf_counter()
    variant = compute_layered_response(
        material,
        geometry,
        waveform,
        receivers,
        times,
        variant_settings,
        backend=backend,
    )
    cost = time.perf_counter() - started
    metrics = floor_relative_error(
        baseline_data,
        _point_column(variant, 0),
        floor_fraction=floor_fraction,
        exclude=exclude,
    )
    return _check_record(
        name,
        performed=True,
        passed=metrics["global_max"] <= tolerance,
        baseline=baseline_settings.label,
        variant=variant_settings.label,
        metrics=metrics,
        cost_s=cost,
        note=note,
        gating=gating,
    )


def _check_record(
    name: str,
    *,
    performed: bool,
    passed: bool,
    baseline,
    variant,
    metrics: dict[str, Any],
    cost_s: float,
    note: str,
    gating: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "performed": bool(performed),
        "passed": bool(passed),
        "gating": bool(gating),
        "baseline": baseline,
        "variant": variant,
        "per_channel_max_floor_relative_error": metrics[
            "per_channel_max_floor_relative_error"
        ],
        "global_max": metrics["global_max"],
        "cost_s": float(cost_s),
        "note": note,
    }


def _point_column(response: dict[str, Any], index: int) -> np.ndarray:
    data = np.asarray(response["data"], dtype=float)
    if data.ndim != 3:
        raise ValueError("response data must have shape (n_times, n_receivers, 6)")
    return data[:, index, :]


def _values_at_times(
    source_times: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    indices = []
    for time_value in query_times:
        matches = np.flatnonzero(
            np.isclose(source_times, float(time_value), rtol=0.0, atol=1.0e-15)
        )
        if matches.size == 0:
            raise ValueError(f"shared time {time_value:g} is missing from the variant")
        indices.append(int(matches[0]))
    return np.asarray(values, dtype=float)[np.asarray(indices, dtype=int)]


def _approved_identity_tied(
    transform_settings: TransformSettings,
    fourier_check: dict[str, Any] | None,
) -> bool:
    if transform_settings.is_approved_production_identity():
        return True
    if fourier_check is None:
        return False
    return bool(fourier_check.get("performed")) and bool(fourier_check.get("passed"))


def _classify_reference_type(
    material: Material,
    checks: dict[str, Any],
    requested: Sequence[str],
    *,
    all_gating_passed: bool,
    approved_identity_tied: bool,
    required_ok: bool | None = None,
) -> str:
    if required_ok is None:
        required_ok = all(
            bool(checks.get(name, {}).get("performed"))
            and bool(checks.get(name, {}).get("passed"))
            for name in REQUIRED_PERFORMED_CHECKS
        )
    gating_failed = any(
        item.get("gating", True) and item.get("performed") and not item.get("passed")
        for item in checks.values()
    )
    if gating_failed:
        return "empymod_audit_failed"
    if not required_ok or not all_gating_passed or not approved_identity_tied:
        return "empymod_unaudited"
    if isinstance(material, ExactPeltonMaterial):
        return "empymod_converged_cole_cole"
    if isinstance(material, DebyeCandidateMaterial):
        return "empymod_audited_debye_candidate"
    return "empymod_unaudited"


def _channel_peaks_and_near_zero(column: np.ndarray) -> tuple[dict[str, float], list[str]]:
    peaks = {
        name: float(np.max(np.abs(column[:, index])))
        for index, name in enumerate(CHANNELS)
    }
    h_group = max(peaks[name] for name in H_CHANNELS)
    dbdt_group = max(peaks[name] for name in DBDT_CHANNELS)
    near_zero = [
        name
        for name in H_CHANNELS
        if peaks[name] < NEAR_ZERO_FRACTION * max(h_group, np.finfo(float).tiny)
    ] + [
        name
        for name in DBDT_CHANNELS
        if peaks[name] < NEAR_ZERO_FRACTION * max(dbdt_group, np.finfo(float).tiny)
    ]
    return peaks, near_zero


def _near_zero_channel_names(column: np.ndarray) -> list[str]:
    _peaks, near_zero = _channel_peaks_and_near_zero(np.asarray(column, dtype=float))
    return near_zero
