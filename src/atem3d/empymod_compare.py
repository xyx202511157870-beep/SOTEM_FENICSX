"""empymod reference-response helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.constants import mu_0

from .fit import (
    DebyeFitResult,
    fit_cole_cole_conductivity_debye,
    fit_pelton_resistivity_debye,
)


@dataclass(frozen=True)
class EmpymodSurvey:
    """Geometry and layered model for empymod finite-bipole references."""

    source_start: tuple[float, float, float]
    source_end: tuple[float, float, float]
    receiver_locations: Sequence[tuple[float, float, float]]
    components: Sequence[str]
    times: np.ndarray
    depths: Sequence[float]
    resistivities: Sequence[float] | dict[str, Any]
    strength: float = 1.0
    signal: int | None = -1
    receiver_components: Sequence[tuple[tuple[float, float, float], str]] | None = None
    coordinate_system: str = "depth_down"


def run_empymod_reference(survey: EmpymodSurvey, backend=None, **kwargs) -> np.ndarray:
    """Compute empymod reference data for Ex/Ey/Hz-style point receivers.

    ``backend`` is injectable so tests can verify the geometry mapping without
    importing or running empymod. When omitted, empymod is imported lazily.
    """

    if backend is None:
        import empymod as backend  # noqa: PLC0415

    times = np.asarray(survey.times, dtype=float)
    responses: list[np.ndarray] = []
    for location, component in _iter_receiver_components(survey):
        rec, mrec = _receiver_mapping(location, component, survey.coordinate_system)
        response = backend.bipole(
            src=_source_mapping(survey.source_start, survey.source_end, survey.coordinate_system),
            rec=rec,
            depth=list(survey.depths),
            res=_resistivity_model(survey.resistivities),
            freqtime=times,
            signal=survey.signal,
            strength=survey.strength,
            mrec=mrec,
            verb=0,
            **kwargs,
        )
        values = np.asarray(response, dtype=float).reshape(-1)
        if component in {"Bx", "By", "Bz"}:
            values = mu_0 * values
        values = _component_coordinate_factor(component, survey.coordinate_system) * values
        responses.append(values)
    return np.column_stack(responses)


def build_empymod_survey_from_result(
    result_path: str | Path,
    depths: Sequence[float],
    resistivities: Sequence[float],
    signal: int | None = -1,
) -> tuple[EmpymodSurvey, list[str]]:
    """Build an empymod survey and column names from an ATEM3D result file."""

    import h5py
    import yaml

    with h5py.File(result_path, "r") as h5:
        times = h5["times"][:]
        config = yaml.safe_load(h5.attrs["config_yaml"])
    return build_empymod_survey_from_config(
        config,
        times=times,
        depths=depths,
        resistivities=resistivities,
        signal=signal,
    )


def build_empymod_survey_from_config(
    config: dict[str, Any],
    *,
    times: Sequence[float],
    depths: Sequence[float],
    resistivities: Sequence[float],
    signal: int | None = -1,
) -> tuple[EmpymodSurvey, list[str]]:
    """Build an empymod survey and column names from a config and time array."""

    if len(resistivities) != len(depths) + 1:
        raise ValueError("len(resistivities) must equal len(depths) + 1")

    locations, components, names, flat = _receiver_columns(config)
    source_cfg = config["source"]
    survey = EmpymodSurvey(
        source_start=tuple(float(v) for v in source_cfg["start"]),
        source_end=tuple(float(v) for v in source_cfg["end"]),
        receiver_locations=locations,
        components=components,
        times=np.asarray(times, dtype=float),
        depths=list(depths),
        resistivities=list(resistivities),
        strength=float(source_cfg["current"]),
        signal=signal,
        receiver_components=flat,
        coordinate_system=str(config.get("coordinate_system", "depth_down")),
    )
    return survey, names


def make_debye_resistivity_model(
    sigma_infinity: Sequence[float],
    debye_terms: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an empymod ``res`` dictionary for Debye conductivity dispersion."""

    sigma = np.asarray(sigma_infinity, dtype=float)
    if sigma.ndim != 1:
        raise ValueError("sigma_infinity must be a 1D sequence")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_infinity must be positive")

    terms = []
    for term in debye_terms or []:
        delta = np.asarray(term["delta_sigma"], dtype=float)
        if delta.ndim == 0:
            delta = np.full_like(sigma, float(delta))
        if delta.shape != sigma.shape:
            raise ValueError("each delta_sigma must be scalar or match sigma_infinity")
        tau = float(term["tau"])
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        terms.append({"delta_sigma": delta, "tau": tau})

    def func_eta(_model: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return _debye_eta_from_terms(context["freq"], sigma, terms)

    return {
        "res": (1.0 / sigma).tolist(),
        "func_eta": func_eta,
    }


def make_debye_resistivity_model_from_config(
    config: dict[str, Any],
    depths: Sequence[float],
) -> dict[str, Any]:
    """Build an empymod Debye ``res`` model from YAML-style config."""

    model_cfg = config.get("model", {})
    coordinate_system = str(config.get("coordinate_system", model_cfg.get("coordinate_system", "depth_down")))
    layers = model_cfg.get("layers")
    if layers is None:
        intervals = _empymod_layer_intervals(depths)
        sigma_value, term_specs = _layer_debye_model_specs(model_cfg, model_cfg)
        sigma = [sigma_value] * len(intervals)
        terms = [
            {
                "delta_sigma": np.full(len(intervals), delta_sigma),
                "tau": tau,
            }
            for delta_sigma, tau in term_specs
        ]
        return make_debye_resistivity_model(sigma, terms)

    intervals = _empymod_layer_intervals(depths)
    sigma = []
    raw_terms: list[tuple[int, float, float]] = []
    for layer_index, (top, bottom) in enumerate(intervals):
        layer = _find_config_layer(layers, top, bottom, coordinate_system)
        layer_sigma, term_specs = _layer_debye_model_specs(layer, model_cfg)
        sigma.append(layer_sigma)
        for delta_sigma, tau in term_specs:
            raw_terms.append((layer_index, delta_sigma, tau))

    tau_values = sorted({tau for _, _, tau in raw_terms})
    terms = []
    for tau in tau_values:
        delta = np.zeros(len(intervals), dtype=float)
        for layer_index, value, term_tau in raw_terms:
            if term_tau == tau:
                delta[layer_index] = value
        terms.append({"delta_sigma": delta, "tau": tau})
    return make_debye_resistivity_model(sigma, terms)


def _layer_debye_model_specs(
    layer: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[float, list[tuple[float, float]]]:
    sigma_infinity = float(layer["sigma_infinity"])
    terms = [
        (float(term["delta_sigma"]), float(term["tau"]))
        for term in layer.get("debye_terms", [])
    ]
    if "ip_model" not in layer:
        return sigma_infinity, terms

    ip_cfg = layer["ip_model"]
    frequency_cfg = ip_cfg.get("fit_frequencies", model_cfg.get("fit_frequencies"))
    if frequency_cfg is None:
        raise ValueError("ip_model requires fit_frequencies in the layer or model")
    frequencies = np.asarray(frequency_cfg, dtype=float)
    if frequencies.size == 0:
        raise ValueError("ip_model requires fit_frequencies in the layer or model")

    tau_grid = ip_cfg.get("tau_grid")
    if tau_grid is not None:
        tau_grid = np.asarray(tau_grid, dtype=float)
    n_terms = int(ip_cfg.get("n_terms", 10))
    model_type = ip_cfg["type"]
    fit: DebyeFitResult
    if model_type == "cole_cole_conductivity":
        fit = fit_cole_cole_conductivity_debye(
            sigma_infinity=float(ip_cfg.get("sigma_infinity", layer["sigma_infinity"])),
            eta=float(ip_cfg["eta"]),
            tau=float(ip_cfg["tau"]),
            c=float(ip_cfg["c"]),
            frequencies=frequencies,
            tau_grid=tau_grid,
            n_terms=n_terms,
        )
    elif model_type == "pelton":
        fit = fit_pelton_resistivity_debye(
            rho0=float(ip_cfg["rho0"]),
            chargeability=float(ip_cfg["chargeability"]),
            tau=float(ip_cfg["tau"]),
            c=float(ip_cfg["c"]),
            frequencies=frequencies,
            tau_grid=tau_grid,
            n_terms=n_terms,
        )
    else:
        raise ValueError("ip_model.type must be 'cole_cole_conductivity' or 'pelton'")

    terms.extend((float(term.delta_sigma[0]), float(term.tau)) for term in fit.terms)
    return fit.sigma_infinity, terms


def _source_mapping(start, end, coordinate_system: str = "depth_down") -> list[float]:
    sx, sy, sz = (float(v) for v in start)
    ex, ey, ez = (float(v) for v in end)
    if coordinate_system == "z_up":
        sz = -sz
        ez = -ez
    elif coordinate_system != "depth_down":
        raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")
    return [sx, ex, sy, ey, sz, ez]


def _empymod_layer_intervals(depths: Sequence[float]) -> list[tuple[float, float]]:
    values = [float(depth) for depth in depths]
    if any(values[i + 1] <= values[i] for i in range(len(values) - 1)):
        raise ValueError("depths must be strictly increasing")
    bounds = [-np.inf, *values, np.inf]
    return list(zip(bounds[:-1], bounds[1:]))


def _find_config_layer(
    layers: Sequence[dict[str, Any]],
    top: float,
    bottom: float,
    coordinate_system: str = "depth_down",
) -> dict[str, Any]:
    for layer in layers:
        layer_top, layer_bottom = _layer_depth_bounds(layer, coordinate_system)
        top_ok = layer_top < bottom if np.isneginf(top) else layer_top <= top
        bottom_ok = layer_bottom > top if np.isposinf(bottom) else layer_bottom >= bottom
        if top_ok and bottom_ok:
            return layer
    raise ValueError(f"no config layer covers empymod interval ({top}, {bottom})")


def _layer_depth_bounds(
    layer: dict[str, Any],
    coordinate_system: str,
) -> tuple[float, float]:
    top = float(layer["top"])
    bottom = float(layer["bottom"])
    if coordinate_system == "z_up":
        return -top, -bottom
    if coordinate_system == "depth_down":
        return top, bottom
    raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")


def _resistivity_model(resistivities: Sequence[float] | dict[str, Any]) -> list[float] | dict[str, Any]:
    if isinstance(resistivities, dict):
        return resistivities
    return list(resistivities)


def _debye_eta_from_terms(
    freq: Sequence[float],
    sigma_infinity: np.ndarray,
    terms: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(freq, dtype=float)
    sigma = np.asarray(sigma_infinity, dtype=float)[None, :].astype(complex)
    omega = 2.0 * np.pi * freq[:, None]
    eta = np.repeat(sigma, freq.size, axis=0)
    for term in terms:
        delta = np.asarray(term["delta_sigma"], dtype=float)[None, :]
        tau = float(term["tau"])
        eta = eta - delta / (1.0 + 1j * omega * tau)
    return eta, eta.copy()


def _receiver_mapping(
    location,
    component: str,
    coordinate_system: str = "depth_down",
) -> tuple[list[float], bool]:
    x, y, z = (float(v) for v in location)
    z = _receiver_depth(z, coordinate_system)
    electric_vertical_dip = _vertical_dip(coordinate_system, axial=False)
    magnetic_vertical_dip = _vertical_dip(coordinate_system, axial=True)
    if component == "Ex":
        return [x, y, z, 0.0, 0.0], False
    if component == "Ey":
        return [x, y, z, 90.0, 0.0], False
    if component == "Ez":
        return [x, y, z, 0.0, electric_vertical_dip], False
    if component == "Hx" or component == "Bx":
        return [x, y, z, 0.0, 0.0], True
    if component == "Hy" or component == "By":
        return [x, y, z, 90.0, 0.0], True
    if component == "Hz" or component == "Bz":
        return [x, y, z, 0.0, magnetic_vertical_dip], True
    if component == "dBxdt":
        return [x, y, z, 0.0, 0.0], "b"
    if component == "dBydt":
        return [x, y, z, 90.0, 0.0], "b"
    if component == "dBzdt":
        return [x, y, z, 0.0, magnetic_vertical_dip], "b"
    raise ValueError("unsupported receiver component")


def _receiver_depth(z: float, coordinate_system: str) -> float:
    if coordinate_system == "z_up":
        return -z
    if coordinate_system == "depth_down":
        return z
    raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")


def _vertical_dip(coordinate_system: str, axial: bool) -> float:
    if coordinate_system == "z_up":
        return 90.0 if axial else -90.0
    if coordinate_system == "depth_down":
        return 90.0
    raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")


def _component_coordinate_factor(component: str, coordinate_system: str) -> float:
    if coordinate_system != "z_up":
        return 1.0
    if component in {"Hx", "Hy", "Bx", "By", "dBxdt", "dBydt"}:
        return -1.0
    return 1.0


def _iter_receiver_components(survey: EmpymodSurvey):
    if survey.receiver_components is not None:
        return list(survey.receiver_components)
    return [
        (location, component)
        for location in survey.receiver_locations
        for component in survey.components
    ]


def _receiver_columns(
    config: dict,
) -> tuple[
    list[tuple[float, float, float]],
    list[str],
    list[str],
    list[tuple[tuple[float, float, float], str]],
]:
    if "receiver_line" in config:
        line = config["receiver_line"]
        locations = [(float(x), float(line["y"]), float(line["z"])) for x in line["x"]]
        components = [str(component) for component in line["components"]]
        flat = [(location, component) for location in locations for component in components]
        names = [
            f"{component}@x={float(x):g}"
            for x in line["x"]
            for component in components
        ]
        return locations, components, names, flat

    receivers = config.get("receivers", [])
    locations = [tuple(float(v) for v in rx["location"]) for rx in receivers]
    components = [str(rx["component"]) for rx in receivers]
    names = [f"{component}@{i}" for i, component in enumerate(components)]
    flat = list(zip(locations, components))
    return locations, components, names, flat
