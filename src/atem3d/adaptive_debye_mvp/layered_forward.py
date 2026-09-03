"""Layered empymod six-channel forward for exact Cole-Cole and Debye candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import mu_0

from atem3d.empymod_compare import (
    EmpymodSurvey,
    _component_coordinate_factor,
    _receiver_mapping,
    _source_mapping,
    make_debye_resistivity_model,
    make_exact_pelton_resistivity_model,
)
from atem3d.empymod_magnetic6 import MAGNETIC6_COMPONENTS, _real_time_values
from atem3d.empymod_waveform import (
    PiecewiseLinearTurnOff,
    _turnoff_quadrature,
    _weighted_delayed_values,
)
from atem3d.materials.cole_cole import PeltonColeColeResistivity
from atem3d.receivers import AverageReceiver, _disk_basis

from .io import canonical_json, sha256_hex
from .passive_fit import PassiveDebyeFit
from .protocol_constants import (
    CHANNELS,
    COORDINATE_SYSTEM,
    EMPYMOD_RECPTS,
    EMPYMOD_SRCPTS,
    SOURCE_CURRENT,
    WAVEFORM_BY_ID,
    WAVEFORM_QUADRATURE_ORDER,
    observation_times,
)
from .registry import LayeredCase


_H_CHANNELS = ("Hx", "Hy", "Hz")
_DBDT_CHANNELS = ("dBxdt", "dBydt", "dBzdt")


@dataclass(frozen=True)
class LayeredResponse:
    """Six-channel magnetic data on the frozen time grid."""

    times: np.ndarray
    locations: tuple[tuple[float, float, float], ...]
    data: np.ndarray
    model_id: str
    waveform_id: str

    def at_location(self, index: int) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(self.data[:, index, channel_index], dtype=float)
            for channel_index, name in enumerate(CHANNELS)
        }


def waveform_from_id(waveform_id: str) -> PiecewiseLinearTurnOff | None:
    """Return the frozen waveform, or ``None`` for ideal step-off."""

    spec = WAVEFORM_BY_ID[str(waveform_id)]
    if spec.kind == "ideal_step_off":
        return None
    if spec.kind == "linear_ramp":
        return PiecewiseLinearTurnOff.linear(float(spec.duration_s), name=spec.waveform_id)
    if spec.kind == "tabulated":
        return PiecewiseLinearTurnOff(
            times=np.asarray(spec.times_s, dtype=float),
            values=np.asarray(spec.current_scales, dtype=float),
            name=spec.waveform_id,
        )
    raise ValueError(f"unsupported waveform: {waveform_id}")


def pelton_layers(case: LayeredCase, *, chargeability: float | None = None) -> dict[str, Any]:
    """Return an empymod exact-Pelton ``res`` model for one case."""

    m_values = np.zeros(len(case.resistivities), dtype=float)
    tau_values = np.ones(len(case.resistivities), dtype=float)
    c_values = np.ones(len(case.resistivities), dtype=float)
    charge = case.m if chargeability is None else float(chargeability)
    m_values[case.polarizable_layer_index] = charge
    tau_values[case.polarizable_layer_index] = case.tau
    c_values[case.polarizable_layer_index] = case.c
    return make_exact_pelton_resistivity_model(
        rho0=case.resistivities,
        chargeability=m_values,
        tau=tau_values,
        c=c_values,
    )


def noip_resistivities(case: LayeredCase) -> list[float]:
    """Return the DC layered resistivity used for the no-IP baseline."""

    return [float(value) for value in case.resistivities]


def debye_layers(case: LayeredCase, fit: PassiveDebyeFit) -> dict[str, Any]:
    """Return an empymod Debye ``res`` model with IP only in the polarizable layer."""

    n_layers = len(case.resistivities)
    sigma_inf = np.array([1.0 / float(rho) for rho in case.resistivities], dtype=float)
    sigma_inf[case.polarizable_layer_index] = float(fit.sigma_infinity)
    terms = []
    for delta, tau in zip(fit.delta_sigma, fit.tau_grid):
        deltas = np.zeros(n_layers, dtype=float)
        deltas[case.polarizable_layer_index] = max(float(delta), 0.0)
        terms.append({"delta_sigma": deltas, "tau": float(tau)})
    return make_debye_resistivity_model(sigma_inf, terms)


def polarizable_material(case: LayeredCase) -> PeltonColeColeResistivity:
    rho0 = float(case.resistivities[case.polarizable_layer_index])
    return PeltonColeColeResistivity(rho0=rho0, chargeability=case.m, tau=case.tau, c=case.c)


def _disk_component_name(component: str) -> str:
    """Map six-channel names onto AverageReceiver H/B axes."""

    if component in {"Hx", "Hy", "Hz", "Bx", "By", "Bz"}:
        return component
    mapping = {"dBxdt": "Hx", "dBydt": "Hy", "dBzdt": "Hz"}
    if component not in mapping:
        raise ValueError(f"unsupported disk component: {component}")
    return mapping[component]


def disk_quadrature(location: tuple[float, float, float], radius: float, component: str):
    """Return AverageReceiver disk points and weights for one component."""

    receiver = AverageReceiver(
        location=location,
        component=_disk_component_name(component),
        receiver_type="disk_average",
        radius=float(radius),
    )
    return receiver.sample_points, receiver.sample_weights


def square4_disk_quadrature(
    location: tuple[float, float, float],
    radius: float,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a frozen 4-point in-plane square disk (layered production quadrature)."""

    center = np.asarray(location, dtype=float)
    axis = {"Hx": 0, "Hy": 1, "Hz": 2, "dBxdt": 0, "dBydt": 1, "dBzdt": 2}[_disk_component_name(component)]
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    basis_u, basis_v = _disk_basis(normal)
    offset = float(radius) / np.sqrt(2.0)
    points = np.vstack(
        [
            center + offset * basis_u,
            center - offset * basis_u,
            center + offset * basis_v,
            center - offset * basis_v,
        ]
    )
    weights = np.full(4, 0.25, dtype=float)
    return points, weights


def unique_evaluation_locations(
    case: LayeredCase,
    *,
    include_disks: bool = True,
    disk_rule: str = "square4",
    disk_receiver_index: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """Return unique point and disk-quadrature locations in deterministic order."""

    seen: set[tuple[float, float, float]] = set()
    locations: list[tuple[float, float, float]] = []

    def _add(point) -> None:
        key = (round(float(point[0]), 9), round(float(point[1]), 9), round(float(point[2]), 9))
        if key not in seen:
            seen.add(key)
            locations.append((float(point[0]), float(point[1]), float(point[2])))

    for receiver in case.receivers:
        _add(receiver)
    if include_disks:
        receiver = case.receivers[int(disk_receiver_index)]
        for radius in case.disk_radii:
            for component in ("Hx", "Hy", "Hz"):
                if disk_rule == "average_receiver":
                    points, _weights = disk_quadrature(receiver, radius, component)
                elif disk_rule == "square4":
                    points, _weights = square4_disk_quadrature(receiver, radius, component)
                else:
                    raise ValueError(f"unknown disk_rule: {disk_rule}")
                for point in points:
                    _add(point)
    return tuple(locations)


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def _model_cache_key(
    case: LayeredCase,
    *,
    model_id: str,
    resistivities,
    times: np.ndarray,
    locations: tuple[tuple[float, float, float], ...],
) -> str:
    payload = {
        "case_hash": case.case_hash(),
        "model_id": model_id,
        "depths": list(case.depths),
        "resistivity_kind": "dict" if isinstance(resistivities, dict) else "list",
        "resistivity_res": None if isinstance(resistivities, dict) else list(resistivities),
        "times": [float(value) for value in times],
        "locations": [list(item) for item in locations],
        "srcpts": EMPYMOD_SRCPTS,
        "recpts": EMPYMOD_RECPTS,
        "coordinate_system": COORDINATE_SYSTEM,
    }
    if isinstance(resistivities, dict):
        payload["resistivity_res"] = [float(value) for value in resistivities.get("res", [])]
        payload["has_func_eta"] = "func_eta" in resistivities
    return sha256_hex(canonical_json(payload))


def _batched_component(
    backend,
    *,
    source,
    depths: list[float],
    resistivities,
    times: np.ndarray,
    locations: tuple[tuple[float, float, float], ...],
    component: str,
    signal: int,
    mrec,
) -> np.ndarray:
    rec, _mapped_mrec = _receiver_mapping(locations[0], component, COORDINATE_SYSTEM)
    xs = [float(location[0]) for location in locations]
    ys = [float(location[1]) for location in locations]
    zs = [float(_receiver_mapping(location, component, COORDINATE_SYSTEM)[0][2]) for location in locations]
    azimuth = rec[3]
    dip = rec[4]
    response = backend.bipole(
        src=source,
        rec=[xs, ys, zs, azimuth, dip],
        depth=depths,
        res=resistivities,
        freqtime=times,
        signal=signal,
        strength=SOURCE_CURRENT,
        mrec=mrec,
        srcpts=EMPYMOD_SRCPTS,
        recpts=EMPYMOD_RECPTS,
        verb=0,
    )
    values = np.asarray(response)
    if values.ndim == 1:
        values = values.reshape(times.size, 1)
    elif values.shape[0] != times.size and values.shape[-1] == times.size:
        values = np.moveaxis(values, -1, 0)
    values = np.asarray(values, dtype=float).reshape(times.size, len(locations))
    scale = _component_coordinate_factor(component, COORDINATE_SYSTEM)
    return scale * values


def compute_stepoff_magnetic6(
    case: LayeredCase,
    resistivities,
    times: np.ndarray,
    locations: tuple[tuple[float, float, float], ...],
    *,
    backend=None,
    cache_dir: str | Path | None = None,
    model_id: str = "model",
) -> np.ndarray:
    """Return ``(n_times, n_locations, 6)`` ideal step-off magnetic-six data."""

    times = np.asarray(times, dtype=float)
    cache_key = _model_cache_key(
        case,
        model_id=model_id,
        resistivities=resistivities,
        times=times,
        locations=locations,
    )
    if cache_dir is not None:
        path = _cache_path(Path(cache_dir), cache_key)
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                return np.asarray(archive["data"], dtype=float)

    if backend is None:
        import empymod as backend  # noqa: PLC0415

    source = _source_mapping(case.source_start, case.source_end, COORDINATE_SYSTEM)
    depths = list(case.depths)
    data = np.zeros((times.size, len(locations), 6), dtype=float)
    for index, component in enumerate(_H_CHANNELS):
        data[:, :, index] = _batched_component(
            backend,
            source=source,
            depths=depths,
            resistivities=resistivities,
            times=times,
            locations=locations,
            component=component,
            signal=-1,
            mrec=True,
        )
    for index, component in enumerate(_DBDT_CHANNELS):
        impulse = _batched_component(
            backend,
            source=source,
            depths=depths,
            resistivities=resistivities,
            times=times,
            locations=locations,
            component=component,
            signal=0,
            mrec=True,
        )
        data[:, :, 3 + index] = -mu_0 * impulse

    if cache_dir is not None:
        path = _cache_path(Path(cache_dir), cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, data=data, times=times)
        tmp.replace(path)
    return data


def assemble_waveform(
    delayed_times: np.ndarray,
    delayed_data: np.ndarray,
    observation_times: np.ndarray,
    waveform: PiecewiseLinearTurnOff | None,
) -> np.ndarray:
    """Assemble W0 or a finite-ramp response from delayed step-off samples."""

    if waveform is None:
        index = {float(time): i for i, time in enumerate(delayed_times)}
        rows = [index[float(time)] for time in observation_times]
        return delayed_data[np.asarray(rows, dtype=int)]

    delays, weights, _segments = _turnoff_quadrature(waveform, WAVEFORM_QUADRATURE_ORDER)
    shifted = np.asarray(observation_times, dtype=float)[:, None] - delays[None, :]
    unique, inverse = np.unique(np.round(delayed_times, decimals=12), return_inverse=True)
    lookup = {float(time): int(i) for i, time in enumerate(unique)}
    rounded = np.round(delayed_times, decimals=12)
    delayed_map = {float(time): i for i, time in enumerate(rounded)}
    inverse_rows = []
    for value in shifted.reshape(-1):
        key = float(np.round(value, decimals=12))
        if key not in delayed_map:
            nearest = int(np.argmin(np.abs(delayed_times - value)))
            inverse_rows.append(nearest)
        else:
            inverse_rows.append(delayed_map[key])
    return _weighted_delayed_values(
        delayed_data,
        np.asarray(inverse_rows, dtype=int),
        n_observation_times=int(np.asarray(observation_times).size),
        n_nodes=int(delays.size),
        weights=weights,
    )


def delayed_time_union(times: np.ndarray, waveform_ids: tuple[str, ...]) -> np.ndarray:
    """Return the unique delayed times needed by the requested waveforms."""

    times = np.asarray(times, dtype=float)
    values = set(float(item) for item in times)
    for waveform_id in waveform_ids:
        waveform = waveform_from_id(waveform_id)
        if waveform is None:
            continue
        delays, _weights, _segments = _turnoff_quadrature(waveform, WAVEFORM_QUADRATURE_ORDER)
        for time in times:
            for delay in delays:
                values.add(float(time - delay))
    return np.asarray(sorted(value for value in values if value > 0.0), dtype=float)


def run_case_models(
    case: LayeredCase,
    *,
    resistivities,
    model_id: str,
    waveform_ids: tuple[str, ...] | None = None,
    times=None,
    backend=None,
    cache_dir: str | Path | None = None,
    include_disks: bool = False,
    disk_rule: str = "square4",
) -> dict[str, LayeredResponse]:
    """Compute all requested waveforms for one resistivity model."""

    times = observation_times() if times is None else np.asarray(times, dtype=float)
    waveform_ids = tuple(case.waveform_ids if waveform_ids is None else waveform_ids)
    locations = unique_evaluation_locations(case, include_disks=include_disks, disk_rule=disk_rule)
    delayed_times = delayed_time_union(times, waveform_ids)
    delayed = compute_stepoff_magnetic6(
        case,
        resistivities,
        delayed_times,
        locations,
        backend=backend,
        cache_dir=cache_dir,
        model_id=f"{model_id}|delayed",
    )
    responses: dict[str, LayeredResponse] = {}
    for waveform_id in waveform_ids:
        assembled = assemble_waveform(delayed_times, delayed, times, waveform_from_id(waveform_id))
        responses[waveform_id] = LayeredResponse(
            times=times.copy(),
            locations=locations,
            data=np.asarray(assembled, dtype=float),
            model_id=model_id,
            waveform_id=waveform_id,
        )
    return responses


def average_receiver_channels(
    response: LayeredResponse,
    location: tuple[float, float, float],
    *,
    kind: str,
    radius: float | None = None,
    normal: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Reduce a batched response to one point, disk, or projected coil."""

    locations = np.asarray(response.locations, dtype=float)
    if kind == "point":
        index = _nearest_location_index(locations, location)
        return response.at_location(index)
    if kind.startswith("disk"):
        if radius is None:
            raise ValueError("disk averaging requires a radius")
        channels: dict[str, np.ndarray] = {}
        for component in CHANNELS:
            points, weights = square4_disk_quadrature(location, radius, component)
            columns = [_nearest_location_index(locations, point) for point in points]
            samples = response.data[:, columns, CHANNELS.index(component)]
            channels[component] = samples @ np.asarray(weights, dtype=float)
        return channels
    if kind == "tilted_coil":
        if normal is None:
            raise ValueError("tilted coil requires a normal")
        unit = np.asarray(normal, dtype=float)
        unit = unit / float(np.linalg.norm(unit))
        index = _nearest_location_index(locations, location)
        hx, hy, hz = (response.data[:, index, i] for i in range(3))
        dbx, dby, dbz = (response.data[:, index, i] for i in range(3, 6))
        projection = unit[0] * dbx + unit[1] * dby + unit[2] * dbz
        h_proj = unit[0] * hx + unit[1] * hy + unit[2] * hz
        return {
            "Hx": hx,
            "Hy": hy,
            "Hz": hz,
            "dBxdt": dbx,
            "dBydt": dby,
            "dBzdt": dbz,
            "H_normal": h_proj,
            "dBdt_normal": projection,
        }
    raise ValueError(f"unknown receiver kind: {kind}")


def _nearest_location_index(locations: np.ndarray, point) -> int:
    target = np.asarray(point, dtype=float)
    return int(np.argmin(np.sum((locations - target) ** 2, axis=1)))


def build_survey(case: LayeredCase, times=None) -> EmpymodSurvey:
    """Build the existing EmpymodSurvey object for one case."""

    times = observation_times() if times is None else np.asarray(times, dtype=float)
    return EmpymodSurvey(
        source_start=case.source_start,
        source_end=case.source_end,
        receiver_locations=case.receivers,
        components=MAGNETIC6_COMPONENTS,
        times=times,
        depths=list(case.depths),
        resistivities=noip_resistivities(case),
        strength=SOURCE_CURRENT,
        signal=-1,
        coordinate_system=COORDINATE_SYSTEM,
    )
