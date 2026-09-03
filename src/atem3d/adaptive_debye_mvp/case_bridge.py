"""Map frozen LayeredCase objects onto the PR 9 empymod wrapper types.

Oracle-gap and Flow 0 call ``compute_layered_response`` only. This module
does not reimplement empymod internals, disk quadrature, or the smoke audit.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from atem3d.materials.cole_cole import PeltonColeColeResistivity

from .layered_forward import (
    APPROVED_PRODUCTION_TRANSFORM,
    CHANNELS,
    SMOKE_FAST_TRANSFORM,
    DebyeCandidateMaterial,
    DebyeTermSpec,
    ExactPeltonMaterial,
    LayeredGeometry,
    NonPolarizableMaterial,
    ReceiverSpec,
    TimeGrid,
    TransformSettings,
    W0_IDEAL_STEP_OFF,
    W1_LINEAR_RAMP_5US,
    W2_LINEAR_RAMP_20US,
    WaveformSpec,
    compute_layered_response,
)
from .passive_fit import PassiveDebyeFit
from .protocol_constants import (
    COORDINATE_SYSTEM,
    SOURCE_CURRENT,
    WAVEFORM_BY_ID,
    WAVEFORM_QUADRATURE_ORDER,
    observation_times,
)
from .registry import LayeredCase


PRODUCTION_TRANSFORM = replace(
    APPROVED_PRODUCTION_TRANSFORM,
    audit_impulse_on_points=False,
    dbdt_reference="impulse_h",
    label="approved_production_no_impulse_audit",
)

# Official PR 9 lagged-DLF identity. The approved production identity
# (ft_pts_per_dec=0) timed at ~84 s per 2-point W0 call versus ~5 s lagged;
# the frozen 8-case x 180-candidate L0 matrix is not completable with
# production DLF on 4 CPUs. Exact Cole-Cole and Debye still share
# hashes["shared_survey_hash"] because only the constitutive model changes.
EVALUATION_TRANSFORM = SMOKE_FAST_TRANSFORM


def polarizable_material(case: LayeredCase) -> PeltonColeColeResistivity:
    """Return the polarizable earth layer as a Pelton resistivity material."""

    rho0 = float(case.resistivities[case.polarizable_layer_index])
    return PeltonColeColeResistivity(
        rho0=rho0,
        chargeability=float(case.m),
        tau=float(case.tau),
        c=float(case.c),
    )


def case_geometry(case: LayeredCase) -> LayeredGeometry:
    return LayeredGeometry(
        source_start=case.source_start,
        source_end=case.source_end,
        source_current_a=SOURCE_CURRENT,
        depths=case.depths,
        coordinate_system=COORDINATE_SYSTEM,
        label=case.case_id,
    )


def case_time_grid(times: np.ndarray | None = None) -> TimeGrid:
    values = observation_times() if times is None else np.asarray(times, dtype=float)
    return TimeGrid(times_s=tuple(float(value) for value in values))


def case_waveform(waveform_id: str) -> WaveformSpec:
    """Translate a frozen protocol waveform id into the PR 9 WaveformSpec."""

    if waveform_id == "W0":
        return W0_IDEAL_STEP_OFF
    if waveform_id == "W1":
        return W1_LINEAR_RAMP_5US
    if waveform_id == "W2":
        return W2_LINEAR_RAMP_20US
    proto = WAVEFORM_BY_ID[waveform_id]
    if proto.kind == "linear_ramp":
        return WaveformSpec(
            kind="linear_ramp",
            ramp_duration_s=float(proto.duration_s),
            quadrature_order=WAVEFORM_QUADRATURE_ORDER,
            label=f"{waveform_id}_linear_ramp",
        )
    if proto.kind == "tabulated":
        return WaveformSpec(
            kind="tabulated",
            config={
                "source": {
                    "current": 1.0,
                    "waveform": {
                        "type": "tabulated",
                        "times": list(proto.times_s or ()),
                        "values": list(proto.current_scales or ()),
                    },
                }
            },
            quadrature_order=WAVEFORM_QUADRATURE_ORDER,
            label="W3_tabulated",
        )
    raise ValueError(f"unsupported waveform id: {waveform_id}")


def point_receivers(case: LayeredCase) -> tuple[ReceiverSpec, ...]:
    return tuple(
        ReceiverSpec(location=location, kind="point", label=f"point:{index}")
        for index, location in enumerate(case.receivers)
    )


def disk_receivers(case: LayeredCase, *, station: int = 0) -> tuple[ReceiverSpec, ...]:
    location = case.receivers[station]
    return tuple(
        ReceiverSpec(
            location=location,
            kind="disk_average",
            radius_m=float(radius),
            label=f"disk_{float(radius)}:{station}",
        )
        for radius in case.disk_radii
    )


def exact_pelton_material(case: LayeredCase) -> ExactPeltonMaterial:
    n_layers = len(case.resistivities)
    chargeability = [0.0] * n_layers
    chargeability[case.polarizable_layer_index] = float(case.m)
    return ExactPeltonMaterial(
        rho0_ohm_m=tuple(float(value) for value in case.resistivities),
        chargeability=tuple(chargeability),
        tau_s=tuple(float(case.tau) for _ in range(n_layers)),
        c=tuple(float(case.c) for _ in range(n_layers)),
    )


def nonpolarizable_material(case: LayeredCase) -> NonPolarizableMaterial:
    return NonPolarizableMaterial(
        resistivity_ohm_m=tuple(float(value) for value in case.resistivities)
    )


def debye_material(
    case: LayeredCase,
    fit: PassiveDebyeFit,
    *,
    candidate_id: str,
) -> DebyeCandidateMaterial:
    """Build an explicit Debye candidate from a hard-DC polarizable-layer fit."""

    n_layers = len(case.resistivities)
    sigma_infinity = []
    for index, rho in enumerate(case.resistivities):
        if index == case.polarizable_layer_index:
            sigma_infinity.append(float(fit.sigma_infinity))
        else:
            sigma_infinity.append(1.0 / float(rho))
    terms = []
    for delta, tau in zip(np.asarray(fit.delta_sigma, dtype=float), np.asarray(fit.tau_grid, dtype=float)):
        deltas = [0.0] * n_layers
        deltas[case.polarizable_layer_index] = max(float(delta), 0.0)
        terms.append(DebyeTermSpec(delta_sigma_s_per_m=tuple(deltas), tau_s=float(tau)))
    if not terms:
        raise ValueError(f"Debye fit for {candidate_id} has no poles")
    return DebyeCandidateMaterial(
        sigma_infinity_s_per_m=tuple(sigma_infinity),
        terms=tuple(terms),
        candidate_id=str(candidate_id),
    )


def production_transform() -> TransformSettings:
    return PRODUCTION_TRANSFORM


def evaluation_transform() -> TransformSettings:
    """Return the official PR 9 transform used for L0 evaluation."""

    return EVALUATION_TRANSFORM


def channel_column(result: dict[str, Any], receiver_index: int) -> dict[str, np.ndarray]:
    """Return one receiver as the metric library's channel-name mapping."""

    data = np.asarray(result["data"], dtype=float)
    names = list(result.get("channels", CHANNELS))
    return {name: data[:, receiver_index, index].copy() for index, name in enumerate(names)}


def labeled_columns(result: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {
        str(label): channel_column(result, index)
        for index, label in enumerate(result["receiver_labels"])
    }


def assert_shared_survey_hash(results: list[dict[str, Any]]) -> str:
    """Exact Cole-Cole and Debye must share the survey hash."""

    hashes = [str(item["hashes"]["shared_survey_hash"]) for item in results]
    unique = set(hashes)
    if len(unique) != 1:
        raise RuntimeError(f"shared_survey_hash mismatch across constitutive models: {unique}")
    return hashes[0]


def _cache_file(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.npz"


def load_cached_response(cache_dir: str | Path | None, cache_key: str | None) -> dict[str, Any] | None:
    if cache_dir is None or not cache_key:
        return None
    path = _cache_file(cache_dir, cache_key)
    if not path.is_file():
        return None
    loaded = np.load(path, allow_pickle=False)
    channels = [str(value) for value in loaded["channels"].tolist()]
    labels = [str(value) for value in loaded["receiver_labels"].tolist()]
    return {
        "data": np.asarray(loaded["data"], dtype=float),
        "channels": channels,
        "receiver_labels": labels,
        "times": np.asarray(loaded["times"], dtype=float),
        "hashes": {
            "shared_survey_hash": str(np.asarray(loaded["shared_survey_hash"]).reshape(-1)[0])
        },
        "from_cache": True,
    }


def store_cached_response(cache_dir: str | Path | None, cache_key: str | None, result: dict[str, Any]) -> None:
    if cache_dir is None or not cache_key:
        return
    import os

    path = _cache_file(cache_dir, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                data=np.asarray(result["data"], dtype="<f8"),
                channels=np.asarray(list(result["channels"]), dtype="U16"),
                receiver_labels=np.asarray(list(result["receiver_labels"]), dtype="U32"),
                times=np.asarray(result["times"], dtype=float),
                shared_survey_hash=np.asarray(result["hashes"]["shared_survey_hash"]),
            )
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def forward_response(
    material,
    geometry: LayeredGeometry,
    waveform: WaveformSpec,
    receivers: tuple[ReceiverSpec, ...],
    times: TimeGrid,
    transform_settings: TransformSettings,
    *,
    cache_dir: str | Path | None = None,
    cache_key: str | None = None,
    backend=None,
) -> dict[str, Any]:
    """Call ``compute_layered_response`` with an optional on-disk cache."""

    cached = load_cached_response(cache_dir, cache_key)
    if cached is not None:
        return cached
    result = compute_layered_response(
        material,
        geometry,
        waveform,
        receivers,
        times,
        transform_settings,
        backend=backend,
    )
    store_cached_response(cache_dir, cache_key, result)
    return result
