"""Shared empymod layered forward for exact Cole-Cole and Debye candidates.

Oracle-gap / selector agents must call::

    compute_layered_response(
        material, geometry, waveform, receivers, times, transform_settings
    ) -> dict

The returned dict always contains six-channel arrays
``Hx, Hy, Hz, dBxdt, dBydt, dBzdt`` (A/m and T/s) plus provenance and
hashes. Exact Cole-Cole and every Debye candidate MUST be generated with
identical source, layered structure, waveform, receivers, numerical
transform settings, and time points. Only the constitutive model may
differ; assert that with ``hashes["shared_survey_hash"]``.

Time origin is the instant of complete current shut-off (empymod-waveform
``ramp_end``) for W0 ideal step-off, W1 5 us linear ramp, W2 20 us linear
ramp, and W3 tabulated holdout. Observation times default to 31
log-spaced samples in ``[1e-5, 1e-2]`` s.

This module wraps ``empymod_magnetic6``, ``empymod_waveform``,
``empymod_compare`` resistivity builders, and ``receivers.AverageReceiver``
disk quadrature. It does **not** import ``source_history_operator`` (that
module is a FEM Debye history basis, not a 1-D empymod path) and it does
**not** import ``dolfinx``.

If empymod is missing or older than 2.5.4 the call raises
``BlockedBySoftwareOrResourcesError`` whose message starts with
``BLOCKED_BY_SOFTWARE_OR_RESOURCES``. Numbers are never fabricated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atem3d.empymod_compare import (
    EmpymodSurvey,
    make_debye_resistivity_model,
    make_exact_pelton_resistivity_model,
)
from atem3d.empymod_magnetic6 import (
    MAGNETIC6_COMPONENTS,
    MAGNETIC6_UNITS,
    run_empymod_magnetic6_reference,
)
from atem3d.empymod_waveform import (
    PiecewiseLinearTurnOff,
    load_turnoff_csv,
    run_empymod_magnetic6_waveform_reference,
    turnoff_waveform_from_config,
)
from atem3d.receivers import AverageReceiver


SCHEMA = "atem3d.adaptive_debye_mvp.layered_response.v1"
CHANNELS = MAGNETIC6_COMPONENTS
UNITS = MAGNETIC6_UNITS
TIME_ORIGIN = "complete_current_shutoff"
DISK_RADIAL_ORDER = 3
DISK_AZIMUTH_COUNT = 12
_EMPYMOD_MIN_VERSION = (2, 5, 4)
_AXIS_TO_CHANNELS = {
    "x": ("Hx", "dBxdt"),
    "y": ("Hy", "dBydt"),
    "z": ("Hz", "dBzdt"),
}


class BlockedBySoftwareOrResourcesError(RuntimeError):
    """Raised when empymod cannot be imported or is too old."""

    code = "BLOCKED_BY_SOFTWARE_OR_RESOURCES"


@dataclass(frozen=True)
class LayeredGeometry:
    """Finite grounded bipole plus empymod layer interfaces."""

    source_start: tuple[float, float, float]
    source_end: tuple[float, float, float]
    source_current_a: float
    depths: tuple[float, ...]
    coordinate_system: str = "z_up"
    label: str = "oblique_grounded_bipole"

    def __post_init__(self) -> None:
        start = _finite_xyz(self.source_start, "source_start")
        end = _finite_xyz(self.source_end, "source_end")
        depths = tuple(float(value) for value in self.depths)
        if not depths:
            raise ValueError("depths must contain at least the air/earth interface")
        if any(not np.isfinite(value) for value in depths):
            raise ValueError("depths must be finite")
        if any(depths[index + 1] <= depths[index] for index in range(len(depths) - 1)):
            raise ValueError("depths must be strictly increasing")
        if not np.isfinite(self.source_current_a) or self.source_current_a == 0.0:
            raise ValueError("source_current_a must be finite and non-zero")
        if self.coordinate_system not in {"z_up", "depth_down"}:
            raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")
        object.__setattr__(self, "source_start", start)
        object.__setattr__(self, "source_end", end)
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "source_current_a", float(self.source_current_a))
        object.__setattr__(self, "coordinate_system", str(self.coordinate_system))
        object.__setattr__(self, "label", str(self.label))

    @property
    def n_layers(self) -> int:
        return len(self.depths) + 1


@dataclass(frozen=True)
class ReceiverSpec:
    """Point or disk-average six-channel magnetic receiver."""

    location: tuple[float, float, float]
    kind: str = "point"
    radius_m: float | None = None
    label: str = "rx0"

    def __post_init__(self) -> None:
        location = _finite_xyz(self.location, "location")
        kind = str(self.kind).strip().lower()
        if kind not in {"point", "disk_average"}:
            raise ValueError("receiver kind must be 'point' or 'disk_average'")
        radius = None if self.radius_m is None else float(self.radius_m)
        if kind == "disk_average":
            if radius is None or not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("disk_average receivers require a positive radius_m")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class WaveformSpec:
    """W0/W1/W2/W3 waveform declaration relative to complete shut-off."""

    kind: str
    ramp_duration_s: float | None = None
    path: str | None = None
    config: Mapping[str, Any] | None = None
    quadrature_order: int = 8
    label: str = "W0"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"ideal_step_off", "linear_ramp", "tabulated"}:
            raise ValueError(
                "waveform kind must be 'ideal_step_off', 'linear_ramp', or 'tabulated'"
            )
        order = int(self.quadrature_order)
        if order < 2:
            raise ValueError("quadrature_order must be at least 2")
        duration = (
            None if self.ramp_duration_s is None else float(self.ramp_duration_s)
        )
        if kind == "linear_ramp":
            if duration is None or not np.isfinite(duration) or duration <= 0.0:
                raise ValueError("linear_ramp requires a positive ramp_duration_s")
        if kind == "tabulated" and self.path is None and self.config is None:
            raise ValueError("tabulated waveform requires path or config")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "ramp_duration_s", duration)
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(self, "quadrature_order", order)
        object.__setattr__(self, "label", str(self.label))


W0_IDEAL_STEP_OFF = WaveformSpec(kind="ideal_step_off", label="W0_ideal_step_off")
W1_LINEAR_RAMP_5US = WaveformSpec(
    kind="linear_ramp",
    ramp_duration_s=5.0e-6,
    label="W1_linear_ramp_5us",
)
W2_LINEAR_RAMP_20US = WaveformSpec(
    kind="linear_ramp",
    ramp_duration_s=20.0e-6,
    label="W2_linear_ramp_20us",
)


def w3_tabulated(path: str | Path) -> WaveformSpec:
    """Return the W3 tabulated-holdout stub (loader only; no shipped table)."""

    return WaveformSpec(kind="tabulated", path=str(path), label="W3_tabulated")


@dataclass(frozen=True)
class TimeGrid:
    """Positive observation times after complete current shut-off."""

    times_s: tuple[float, ...]
    origin: str = TIME_ORIGIN

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.times_s)
        if not times:
            raise ValueError("times_s must be non-empty")
        if any(not np.isfinite(value) or value <= 0.0 for value in times):
            raise ValueError("times must be finite and positive")
        if any(times[index + 1] <= times[index] for index in range(len(times) - 1)):
            raise ValueError("times must be strictly increasing")
        origin = str(self.origin)
        if origin != TIME_ORIGIN:
            raise ValueError(f"time origin must be {TIME_ORIGIN!r}")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "origin", origin)

    @classmethod
    def default(cls) -> "TimeGrid":
        return cls(times_s=tuple(float(value) for value in np.logspace(-5, -2, 31)))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.times_s, dtype=float)


@dataclass(frozen=True)
class TransformSettings:
    """Numerical transform / quadrature identity passed to empymod."""

    equation: str = "quasistatic"
    eperm_h: float = 0.0
    eperm_v: float = 0.0
    mperm_h: float = 1.0
    mperm_v: float = 1.0
    ht: str = "dlf"
    ht_filter: str = "key_201_2009"
    ht_pts_per_dec: int = 0
    ft: str = "dlf"
    ft_filter: str = "key_201_2012"
    ft_pts_per_dec: int = 0
    srcpts: int = 9
    recpts: int = 1
    dbdt_reference: str = "auto"
    audit_impulse_on_points: bool = True
    ft_fftlog_pts_per_dec: int = 10
    ft_fftlog_add_dec: tuple[float, float] = (-2.0, 1.0)
    ft_fftlog_q: float = 0.0
    label: str = "approved_production_identity"

    def __post_init__(self) -> None:
        equation = str(self.equation).strip().lower()
        ht = str(self.ht).strip().lower()
        ft = str(self.ft).strip().lower()
        if equation != "quasistatic":
            raise ValueError("equation must be 'quasistatic'")
        if float(self.eperm_h) != 0.0 or float(self.eperm_v) != 0.0:
            raise ValueError("quasistatic identity requires eperm_h = eperm_v = 0")
        if ht not in {"dlf", "qwe", "quad"}:
            raise ValueError("ht must be 'dlf', 'qwe', or 'quad'")
        if ft not in {"dlf", "qwe", "fftlog", "fft"}:
            raise ValueError("ft must be 'dlf', 'qwe', 'fftlog', or 'fft'")
        if self.dbdt_reference not in {"auto", "native_b", "impulse_h"}:
            raise ValueError("dbdt_reference must be 'auto', 'native_b', or 'impulse_h'")
        if int(self.srcpts) <= 0 or int(self.recpts) <= 0:
            raise ValueError("srcpts and recpts must be positive")
        add_dec = tuple(float(value) for value in self.ft_fftlog_add_dec)
        if len(add_dec) != 2:
            raise ValueError("ft_fftlog_add_dec must contain exactly two values")
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "eperm_h", float(self.eperm_h))
        object.__setattr__(self, "eperm_v", float(self.eperm_v))
        object.__setattr__(self, "mperm_h", float(self.mperm_h))
        object.__setattr__(self, "mperm_v", float(self.mperm_v))
        object.__setattr__(self, "ht", ht)
        object.__setattr__(self, "ht_filter", str(self.ht_filter))
        object.__setattr__(self, "ht_pts_per_dec", int(self.ht_pts_per_dec))
        object.__setattr__(self, "ft", ft)
        object.__setattr__(self, "ft_filter", str(self.ft_filter))
        object.__setattr__(self, "ft_pts_per_dec", int(self.ft_pts_per_dec))
        object.__setattr__(self, "srcpts", int(self.srcpts))
        object.__setattr__(self, "recpts", int(self.recpts))
        object.__setattr__(self, "dbdt_reference", str(self.dbdt_reference))
        object.__setattr__(self, "audit_impulse_on_points", bool(self.audit_impulse_on_points))
        object.__setattr__(self, "ft_fftlog_pts_per_dec", int(self.ft_fftlog_pts_per_dec))
        object.__setattr__(self, "ft_fftlog_add_dec", add_dec)
        object.__setattr__(self, "ft_fftlog_q", float(self.ft_fftlog_q))
        object.__setattr__(self, "label", str(self.label))

    def hankel_parameters(self) -> dict[str, Any]:
        if self.ht == "dlf":
            return {"filter": self.ht_filter, "pts_per_dec": self.ht_pts_per_dec}
        raise ValueError(f"unsupported ht method for identity recording: {self.ht}")

    def fourier_parameters(self) -> dict[str, Any]:
        if self.ft == "dlf":
            return {"filter": self.ft_filter, "pts_per_dec": self.ft_pts_per_dec}
        if self.ft == "fftlog":
            return {
                "pts_per_dec": self.ft_fftlog_pts_per_dec,
                "add_dec": [self.ft_fftlog_add_dec[0], self.ft_fftlog_add_dec[1]],
                "q": self.ft_fftlog_q,
            }
        raise ValueError(f"unsupported ft method for identity recording: {self.ft}")

    def identity_dict(self) -> dict[str, Any]:
        """Reproduce ``sotem_pipeline._empymod_reference_identity`` shape."""

        return {
            "equation": self.equation,
            "electric_permittivity": {
                "horizontal": self.eperm_h,
                "vertical": self.eperm_v,
            },
            "magnetic_permeability": {
                "horizontal": self.mperm_h,
                "vertical": self.mperm_v,
            },
            "hankel_transform": {
                "method": self.ht,
                "parameters": self.hankel_parameters(),
            },
            "fourier_transform": {
                "method": self.ft,
                "parameters": self.fourier_parameters(),
            },
        }

    def is_approved_production_identity(self) -> bool:
        return _canonical_json(self.identity_dict()) == _canonical_json(
            {
                "equation": "quasistatic",
                "electric_permittivity": {"horizontal": 0.0, "vertical": 0.0},
                "magnetic_permeability": {"horizontal": 1.0, "vertical": 1.0},
                "hankel_transform": {
                    "method": "dlf",
                    "parameters": {"filter": "key_201_2009", "pts_per_dec": 0},
                },
                "fourier_transform": {
                    "method": "dlf",
                    "parameters": {"filter": "key_201_2012", "pts_per_dec": 0},
                },
            }
        )

    def empymod_kwargs(self, n_layers: int) -> dict[str, Any]:
        if type(n_layers) is not int or n_layers <= 0:
            raise ValueError("n_layers must be an explicit positive integer")
        htarg = dict(self.hankel_parameters())
        ftarg = dict(self.fourier_parameters())
        if self.ht == "dlf":
            htarg["dlf"] = htarg.pop("filter")
        if self.ft == "dlf":
            ftarg["dlf"] = ftarg.pop("filter")
        return {
            "epermH": [self.eperm_h] * n_layers,
            "epermV": [self.eperm_v] * n_layers,
            "mpermH": [self.mperm_h] * n_layers,
            "mpermV": [self.mperm_v] * n_layers,
            "ht": self.ht,
            "htarg": htarg,
            "ft": self.ft,
            "ftarg": ftarg,
        }

    def hash_payload(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "eperm_h": self.eperm_h,
            "eperm_v": self.eperm_v,
            "mperm_h": self.mperm_h,
            "mperm_v": self.mperm_v,
            "ht": self.ht,
            "ht_filter": self.ht_filter,
            "ht_pts_per_dec": self.ht_pts_per_dec,
            "ft": self.ft,
            "ft_filter": self.ft_filter,
            "ft_pts_per_dec": self.ft_pts_per_dec,
            "srcpts": self.srcpts,
            "recpts": self.recpts,
            "dbdt_reference": self.dbdt_reference,
            "audit_impulse_on_points": self.audit_impulse_on_points,
            "ft_fftlog_pts_per_dec": self.ft_fftlog_pts_per_dec,
            "ft_fftlog_add_dec": list(self.ft_fftlog_add_dec),
            "ft_fftlog_q": self.ft_fftlog_q,
        }


APPROVED_PRODUCTION_TRANSFORM = TransformSettings()
SMOKE_FAST_TRANSFORM = replace(
    APPROVED_PRODUCTION_TRANSFORM,
    ft_pts_per_dec=-1,
    label="smoke_fast_lagged_dlf",
)


@dataclass(frozen=True)
class NonPolarizableMaterial:
    resistivity_ohm_m: tuple[float, ...]
    kind: str = "non_polarizable"

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.resistivity_ohm_m)
        if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("resistivity_ohm_m must be finite and positive")
        object.__setattr__(self, "resistivity_ohm_m", values)
        object.__setattr__(self, "kind", "non_polarizable")


@dataclass(frozen=True)
class ExactPeltonMaterial:
    """Exact Pelton/Cole-Cole resistivity-form material for empymod."""

    rho0_ohm_m: tuple[float, ...]
    chargeability: tuple[float, ...]
    tau_s: tuple[float, ...]
    c: tuple[float, ...]
    kind: str = "exact_pelton_cole_cole"

    def __post_init__(self) -> None:
        rho0 = tuple(float(value) for value in self.rho0_ohm_m)
        chargeability = tuple(float(value) for value in self.chargeability)
        tau = tuple(float(value) for value in self.tau_s)
        c = tuple(float(value) for value in self.c)
        if len({len(rho0), len(chargeability), len(tau), len(c)}) != 1:
            raise ValueError("Pelton layer parameters must have matching lengths")
        if not rho0:
            raise ValueError("Pelton model must contain at least one layer")
        object.__setattr__(self, "rho0_ohm_m", rho0)
        object.__setattr__(self, "chargeability", chargeability)
        object.__setattr__(self, "tau_s", tau)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "kind", "exact_pelton_cole_cole")


@dataclass(frozen=True)
class DebyeTermSpec:
    delta_sigma_s_per_m: tuple[float, ...]
    tau_s: float

    def __post_init__(self) -> None:
        delta = tuple(float(value) for value in self.delta_sigma_s_per_m)
        tau = float(self.tau_s)
        if not delta or any(not np.isfinite(value) or value < 0.0 for value in delta):
            raise ValueError("delta_sigma_s_per_m must be finite and non-negative")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau_s must be finite and positive")
        object.__setattr__(self, "delta_sigma_s_per_m", delta)
        object.__setattr__(self, "tau_s", tau)


@dataclass(frozen=True)
class DebyeCandidateMaterial:
    """Explicit Debye/Prony candidate; this wrapper does not fit poles."""

    sigma_infinity_s_per_m: tuple[float, ...]
    terms: tuple[DebyeTermSpec, ...]
    kind: str = "debye_candidate"
    candidate_id: str = "debye"

    def __post_init__(self) -> None:
        sigma = tuple(float(value) for value in self.sigma_infinity_s_per_m)
        if not sigma or any(not np.isfinite(value) or value <= 0.0 for value in sigma):
            raise ValueError("sigma_infinity_s_per_m must be finite and positive")
        terms = tuple(self.terms)
        for term in terms:
            if len(term.delta_sigma_s_per_m) != len(sigma):
                raise ValueError("each Debye term must have one delta_sigma per layer")
        object.__setattr__(self, "sigma_infinity_s_per_m", sigma)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "kind", "debye_candidate")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))


Material = NonPolarizableMaterial | ExactPeltonMaterial | DebyeCandidateMaterial


def default_smoke_geometry() -> LayeredGeometry:
    """Asymmetric oblique finite grounded source plus 2-layer earth."""

    return LayeredGeometry(
        source_start=(-60.0, -25.0, -0.3),
        source_end=(70.0, 35.0, -0.8),
        source_current_a=10.0,
        depths=(0.0, 30.0),
        coordinate_system="z_up",
        label="smoke_oblique_grounded_bipole",
    )


def default_smoke_receivers() -> tuple[ReceiverSpec, ReceiverSpec, ReceiverSpec]:
    """Off-axis point plus production disk radii 1 m and 4 m."""

    location = (120.0, 80.0, 1.0)
    return (
        ReceiverSpec(location=location, kind="point", label="rx_point"),
        ReceiverSpec(
            location=location,
            kind="disk_average",
            radius_m=1.0,
            label="rx_disk_1m",
        ),
        ReceiverSpec(
            location=location,
            kind="disk_average",
            radius_m=4.0,
            label="rx_disk_4m",
        ),
    )


def default_smoke_materials() -> dict[str, Material]:
    """Polarizable-cover materials for the cheap smoke geometry."""

    air = 2.0e14
    cover = 50.0
    basement = 200.0
    chargeability = 0.2
    tau = 1.0e-3
    c = 0.5
    cover_sigma_inf = 1.0 / (cover * (1.0 - chargeability))
    cover_delta = cover_sigma_inf - (1.0 / cover)
    return {
        "exact_m0p2": ExactPeltonMaterial(
            rho0_ohm_m=(air, cover, basement),
            chargeability=(0.0, chargeability, 0.0),
            tau_s=(tau, tau, tau),
            c=(c, c, c),
        ),
        "exact_m0": ExactPeltonMaterial(
            rho0_ohm_m=(air, cover, basement),
            chargeability=(0.0, 0.0, 0.0),
            tau_s=(tau, tau, tau),
            c=(c, c, c),
        ),
        "non_polarizable": NonPolarizableMaterial(
            resistivity_ohm_m=(air, cover, basement)
        ),
        "debye_dummy": DebyeCandidateMaterial(
            sigma_infinity_s_per_m=(1.0 / air, cover_sigma_inf, 1.0 / basement),
            terms=(
                DebyeTermSpec(
                    delta_sigma_s_per_m=(0.0, cover_delta, 0.0),
                    tau_s=tau,
                ),
            ),
            candidate_id="dummy_one_pole",
        ),
    }


def default_smoke_case() -> dict[str, Any]:
    """Bundle used by the cheap smoke test and optional CLI."""

    return {
        "geometry": default_smoke_geometry(),
        "receivers": default_smoke_receivers(),
        "times": TimeGrid.default(),
        "waveforms": {
            "W0": W0_IDEAL_STEP_OFF,
            "W1": W1_LINEAR_RAMP_5US,
            "W2": W2_LINEAR_RAMP_20US,
        },
        "materials": default_smoke_materials(),
        "transform": SMOKE_FAST_TRANSFORM,
    }


def resolve_waveform(spec: WaveformSpec) -> PiecewiseLinearTurnOff | None:
    """Resolve W0/W1/W2/W3 to a shut-off table, or ``None`` for ideal step-off."""

    if spec.kind == "ideal_step_off":
        return None
    if spec.kind == "linear_ramp":
        return PiecewiseLinearTurnOff.linear(
            float(spec.ramp_duration_s),
            name=spec.label,
        )
    if spec.path is not None:
        return load_turnoff_csv(spec.path)
    return turnoff_waveform_from_config(spec.config)


def build_empymod_resistivity(material: Material) -> list[float] | dict[str, Any]:
    """Build an empymod ``res`` model, flattening 2-D DLF frequency grids."""

    if isinstance(material, NonPolarizableMaterial):
        return list(material.resistivity_ohm_m)
    if isinstance(material, ExactPeltonMaterial):
        raw = make_exact_pelton_resistivity_model(
            rho0=material.rho0_ohm_m,
            chargeability=material.chargeability,
            tau=material.tau_s,
            c=material.c,
        )
        return _flatten_frequency_context(raw)
    if isinstance(material, DebyeCandidateMaterial):
        raw = make_debye_resistivity_model(
            material.sigma_infinity_s_per_m,
            [
                {
                    "delta_sigma": list(term.delta_sigma_s_per_m),
                    "tau": term.tau_s,
                }
                for term in material.terms
            ],
        )
        return _flatten_frequency_context(raw)
    raise TypeError(f"unsupported material type: {type(material)!r}")


def _flatten_frequency_context(res: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``func_eta`` so DLF ``context['freq']`` of shape ``(n_t, n_f)`` works.

    empymod 2.6 with the approved ``pts_per_dec=0`` identity passes a 2-D
    frequency array. Existing ``empymod_compare`` builders require 1-D
    frequencies. This adapter flattens without editing those builders.
    """

    original = res["func_eta"]

    def func_eta(model: dict[str, Any], context: dict[str, Any]):
        ctx = dict(context)
        ctx["freq"] = np.asarray(ctx["freq"], dtype=float).reshape(-1)
        return original(model, ctx)

    return {**res, "func_eta": func_eta}


def compute_layered_response(
    material: Material,
    geometry: LayeredGeometry,
    waveform: WaveformSpec,
    receivers: Sequence[ReceiverSpec],
    times: TimeGrid,
    transform_settings: TransformSettings,
    *,
    backend=None,
    disk_axes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the shared six-channel layered empymod contract.

    Parameters
    ----------
    material
        Constitutive model. Exact Pelton/Cole-Cole, an explicit Debye
        candidate, or a non-polarizable resistivity list.
    geometry
        Finite grounded source, layer interfaces, current, coordinates.
    waveform
        W0 / W1 / W2 / W3 specification. Time origin is complete shut-off.
    receivers
        Point and/or disk-average six-channel receivers.
    times
        Positive times after complete shut-off.
    transform_settings
        Hankel/Fourier identity, source quadrature, dB/dt route.

    Returns
    -------
    dict
        Channel arrays, hashes, provenance, and ``reference_type``
        (``empymod_unaudited`` until ``reference_audit.annotate_reference_type``).
    """

    receivers = tuple(receivers)
    if not receivers:
        raise ValueError("at least one receiver is required")
    _validate_material_layers(material, geometry)
    backend = _require_empymod(backend)
    version_text = _backend_version_text(backend)
    resistivity = build_empymod_resistivity(material)
    resolved_waveform = resolve_waveform(waveform)
    call_kwargs = transform_settings.empymod_kwargs(geometry.n_layers)
    time_array = times.as_array()

    columns: list[np.ndarray] = []
    dbdt_audits: dict[str, Any] = {}
    receiver_records: list[dict[str, Any]] = []
    primary_reference = None
    convolution = None
    last_audit: dict[str, Any] | None = None
    python_call = (
        "run_empymod_magnetic6_waveform_reference"
        if resolved_waveform is not None
        else "run_empymod_magnetic6_reference"
    )

    for receiver in receivers:
        if receiver.kind == "point":
            result = _run_point_or_waveform(
                geometry=geometry,
                locations=(receiver.location,),
                times=time_array,
                resistivity=resistivity,
                waveform=resolved_waveform,
                waveform_spec=waveform,
                transform_settings=transform_settings,
                call_kwargs=call_kwargs,
                backend=backend,
                audit_impulse=transform_settings.audit_impulse_on_points,
            )
            columns.append(np.asarray(result.data, dtype=float)[:, 0, :])
            last_audit = dict(result.audit)
            dbdt_audits[receiver.label] = last_audit
            primary_reference = result.primary_dbdt_reference
            convolution = getattr(result, "convolution", None)
            receiver_records.append(_point_receiver_record(receiver))
            continue

        axes = tuple(disk_axes) if disk_axes is not None else ("x", "y", "z")
        stacked, disk_record = disk_average_six_channel(
            geometry=geometry,
            receiver=receiver,
            times=time_array,
            resistivity=resistivity,
            waveform=resolved_waveform,
            waveform_spec=waveform,
            transform_settings=transform_settings,
            call_kwargs=call_kwargs,
            backend=backend,
            axes=axes,
        )
        columns.append(stacked)
        receiver_records.append(disk_record)
        if convolution is None:
            convolution = disk_record.get("convolution")
        if primary_reference is None:
            primary_reference = disk_record.get("primary_dbdt_reference")
        dbdt_audits[receiver.label] = {
            "performed": False,
            "passed": True,
            "reason": "impulse cross-check is skipped for disk-average receivers",
            "primary_dbdt_reference": disk_record.get("primary_dbdt_reference"),
        }

    data = np.stack(columns, axis=1)
    hashes = compute_hashes(
        material,
        geometry,
        waveform,
        receivers,
        times,
        transform_settings,
        data,
        empymod_version=version_text,
    )
    frequencies = _frequency_provenance(
        time_array,
        resolved_waveform,
        transform_settings,
        quadrature_order=waveform.quadrature_order,
    )
    provenance = {
        "empymod_version": version_text,
        "transform": {
            **transform_settings.identity_dict(),
            "srcpts": transform_settings.srcpts,
            "recpts": transform_settings.recpts,
            "ft_pts_per_dec": transform_settings.ft_pts_per_dec,
            "ht_pts_per_dec": transform_settings.ht_pts_per_dec,
            "is_approved_production_identity": (
                transform_settings.is_approved_production_identity()
            ),
        },
        "frequencies": frequencies,
        "primary_dbdt_reference": primary_reference
        if primary_reference is not None
        else (last_audit or {}).get("primary_dbdt_reference"),
        "source": {
            "start": list(geometry.source_start),
            "end": list(geometry.source_end),
            "current_a": geometry.source_current_a,
            "coordinate_system": geometry.coordinate_system,
            "finite_source_quadrature_points": transform_settings.srcpts,
            "orientation": "finite_grounded_bipole",
        },
        "layers": {
            "depths": list(geometry.depths),
            "n_layers": geometry.n_layers,
        },
        "waveform": _waveform_provenance(waveform, resolved_waveform),
        "convolution": convolution,
        "receivers": receiver_records,
        "material": _jsonable(asdict(material)),
        "time_origin": TIME_ORIGIN,
        "units": list(UNITS),
        "coordinates": {
            "system": geometry.coordinate_system,
            "time_origin": TIME_ORIGIN,
        },
        "python_call": python_call,
    }
    by_channel = {
        channel: data[:, :, index]
        for index, channel in enumerate(CHANNELS)
    }
    return {
        "schema": SCHEMA,
        "channels": list(CHANNELS),
        "units": list(UNITS),
        "time_origin": TIME_ORIGIN,
        "times": time_array.copy(),
        "receiver_labels": [receiver.label for receiver in receivers],
        "data": data,
        "by_channel": by_channel,
        "hashes": hashes,
        "provenance": provenance,
        "dbdt_route_audit": dbdt_audits,
        "reference_type": "empymod_unaudited",
        "blocked_reason": None,
    }


def disk_average_channel_pair(
    axis: str,
    center: tuple[float, float, float],
    radius_m: float,
    *,
    geometry: LayeredGeometry,
    times: np.ndarray,
    resistivity,
    waveform: PiecewiseLinearTurnOff | None,
    waveform_spec: WaveformSpec,
    transform_settings: TransformSettings,
    call_kwargs: Mapping[str, Any],
    backend,
    points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Average Hx/dBxdt, Hy/dBydt, or Hz/dBzdt over a component-normal disk."""

    axis = str(axis).strip().lower()
    if axis not in _AXIS_TO_CHANNELS:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if points is None or weights is None:
        receiver = AverageReceiver(
            location=center,
            component=f"H{axis}",
            receiver_type="disk_average",
            radius=float(radius_m),
        )
        points = receiver.sample_points
        weights = receiver.sample_weights
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("disk points must have shape (n_points, 3)")
    if weights.shape != (points.shape[0],):
        raise ValueError("disk weights must have shape (n_points,)")
    locations = tuple(_finite_xyz(row, "disk_point") for row in points)
    result = _run_point_or_waveform(
        geometry=geometry,
        locations=locations,
        times=np.asarray(times, dtype=float),
        resistivity=resistivity,
        waveform=waveform,
        waveform_spec=waveform_spec,
        transform_settings=transform_settings,
        call_kwargs=call_kwargs,
        backend=backend,
        audit_impulse=False,
    )
    averaged = np.einsum("k,tkc->tc", weights, np.asarray(result.data, dtype=float))
    keep = _AXIS_TO_CHANNELS[axis]
    indices = [CHANNELS.index(name) for name in keep]
    record = {
        "axis": axis,
        "channels": list(keep),
        "n_points": int(points.shape[0]),
        "weight_sum": float(np.sum(weights)),
        "radial_order": DISK_RADIAL_ORDER,
        "azimuth_count": DISK_AZIMUTH_COUNT,
        "normal_per_channel": "component_axis",
        "convolution": getattr(result, "convolution", None),
        "primary_dbdt_reference": result.primary_dbdt_reference,
    }
    return averaged[:, indices], record


def disk_average_six_channel(
    *,
    geometry: LayeredGeometry,
    receiver: ReceiverSpec,
    times: np.ndarray,
    resistivity,
    waveform: PiecewiseLinearTurnOff | None,
    waveform_spec: WaveformSpec,
    transform_settings: TransformSettings,
    call_kwargs: Mapping[str, Any],
    backend,
    axes: Sequence[str] = ("x", "y", "z"),
    points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble a six-channel disk average from component-normal disks."""

    stacked = np.zeros((np.asarray(times).size, 6), dtype=float)
    used_axes: list[str] = []
    last_record: dict[str, Any] | None = None
    for axis in axes:
        pair, record = disk_average_channel_pair(
            axis,
            receiver.location,
            float(receiver.radius_m),
            geometry=geometry,
            times=times,
            resistivity=resistivity,
            waveform=waveform,
            waveform_spec=waveform_spec,
            transform_settings=transform_settings,
            call_kwargs=call_kwargs,
            backend=backend,
            points=points,
            weights=weights,
        )
        names = _AXIS_TO_CHANNELS[axis]
        stacked[:, CHANNELS.index(names[0])] = pair[:, 0]
        stacked[:, CHANNELS.index(names[1])] = pair[:, 1]
        used_axes.append(axis)
        last_record = record
    if last_record is None:
        raise ValueError("disk_axes must contain at least one axis")
    return stacked, {
        "label": receiver.label,
        "kind": "disk_average",
        "location": list(receiver.location),
        "radius_m": float(receiver.radius_m),
        "quadrature": {
            "radial_order": DISK_RADIAL_ORDER,
            "azimuth_count": DISK_AZIMUTH_COUNT,
            "n_points": last_record["n_points"],
            "normal_per_channel": "component_axis",
            "axes": used_axes,
        },
        "convolution": last_record.get("convolution"),
        "primary_dbdt_reference": last_record.get("primary_dbdt_reference"),
    }


def compute_hashes(
    material: Material,
    geometry: LayeredGeometry,
    waveform: WaveformSpec,
    receivers: Sequence[ReceiverSpec],
    times: TimeGrid,
    transform_settings: TransformSettings,
    data: np.ndarray,
    *,
    empymod_version: str | None = None,
) -> dict[str, str]:
    """Canonical JSON + SHA-256 hashes for the shared survey contract."""

    resolved = resolve_waveform(waveform)
    geometry_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.geometry.v1",
                "source_start": list(geometry.source_start),
                "source_end": list(geometry.source_end),
                "source_current_a": geometry.source_current_a,
                "depths": list(geometry.depths),
                "coordinate_system": geometry.coordinate_system,
            }
        )
    )
    if resolved is None:
        waveform_nodes = None
        waveform_values = None
    else:
        waveform_nodes = [float(value) for value in resolved.times]
        waveform_values = [float(value) for value in resolved.values]
    waveform_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.waveform.v1",
                "kind": waveform.kind,
                "times_relative_to_ramp_end_s": waveform_nodes,
                "current_scales": waveform_values,
                "quadrature_order": waveform.quadrature_order,
                "time_origin": "ramp_end",
            }
        )
    )
    times_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.times.v1",
                "times_s": list(times.times_s),
                "origin": times.origin,
            }
        )
    )
    transform_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.transform.v1",
                **transform_settings.hash_payload(),
                "empymod_version": empymod_version,
            }
        )
    )
    receiver_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.receivers.v1",
                "receivers": [
                    {
                        "location": list(receiver.location),
                        "kind": receiver.kind,
                        "radius_m": receiver.radius_m,
                        "disk_quadrature": (
                            None
                            if receiver.kind != "disk_average"
                            else {
                                "radial_order": DISK_RADIAL_ORDER,
                                "azimuth_count": DISK_AZIMUTH_COUNT,
                            }
                        ),
                    }
                    for receiver in receivers
                ],
            }
        )
    )
    material_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.material.v1",
                **_jsonable(asdict(material)),
            }
        )
    )
    shared_survey_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.shared_survey.v1",
                "geometry_hash": geometry_hash,
                "waveform_hash": waveform_hash,
                "times_hash": times_hash,
                "transform_hash": transform_hash,
                "receiver_hash": receiver_hash,
            }
        )
    )
    config_hash = sha256_hex(
        _canonical_json(
            {
                "schema": "atem3d.adaptive_debye_mvp.config.v1",
                "shared_survey_hash": shared_survey_hash,
                "material_hash": material_hash,
            }
        )
    )
    array = np.ascontiguousarray(np.asarray(data, dtype="<f8"))
    output_hash = sha256_hex(
        json.dumps(
            {"shape": list(array.shape), "dtype": "<f8"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + array.tobytes()
    )
    return {
        "geometry_hash": geometry_hash,
        "waveform_hash": waveform_hash,
        "times_hash": times_hash,
        "transform_hash": transform_hash,
        "receiver_hash": receiver_hash,
        "material_hash": material_hash,
        "shared_survey_hash": shared_survey_hash,
        "config_hash": config_hash,
        "output_hash": output_hash,
    }


def canonical_json(obj: Any) -> str:
    return _canonical_json(obj)


def sha256_hex(text_or_bytes: str | bytes) -> str:
    payload = (
        text_or_bytes.encode("utf-8")
        if isinstance(text_or_bytes, str)
        else text_or_bytes
    )
    return hashlib.sha256(payload).hexdigest()


def response_to_json_dict(response: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a response dict into JSON-safe Python objects."""

    return _jsonable(dict(response))


def _run_point_or_waveform(
    *,
    geometry: LayeredGeometry,
    locations: Sequence[tuple[float, float, float]],
    times: np.ndarray,
    resistivity,
    waveform: PiecewiseLinearTurnOff | None,
    waveform_spec: WaveformSpec,
    transform_settings: TransformSettings,
    call_kwargs: Mapping[str, Any],
    backend,
    audit_impulse: bool,
):
    survey = EmpymodSurvey(
        source_start=geometry.source_start,
        source_end=geometry.source_end,
        receiver_locations=tuple(locations),
        components=CHANNELS,
        times=np.asarray(times, dtype=float),
        depths=geometry.depths,
        resistivities=resistivity,
        strength=geometry.source_current_a,
        signal=-1,
        coordinate_system=geometry.coordinate_system,
    )
    reference_kwargs = {
        **dict(call_kwargs),
        "backend": backend,
        "srcpts": transform_settings.srcpts,
        "recpts": transform_settings.recpts,
        "dbdt_reference": transform_settings.dbdt_reference,
        "audit_impulse": bool(audit_impulse),
    }
    if waveform is None:
        return run_empymod_magnetic6_reference(survey, **reference_kwargs)
    return run_empymod_magnetic6_waveform_reference(
        survey,
        waveform,
        quadrature_order=waveform_spec.quadrature_order,
        **reference_kwargs,
    )


def _require_empymod(backend=None):
    if backend is not None:
        _assert_supported_version(backend)
        return backend
    try:
        backend = _import_empymod()
    except ImportError as exc:
        raise BlockedBySoftwareOrResourcesError(
            "BLOCKED_BY_SOFTWARE_OR_RESOURCES: empymod>=2.5.4 is required "
            "for the layered Cole-Cole / Debye forward wrapper"
        ) from exc
    _assert_supported_version(backend)
    return backend


def _import_empymod():
    import empymod  # noqa: PLC0415

    return empymod


def _assert_supported_version(backend) -> None:
    version = _backend_version_tuple(backend)
    if version is None:
        return
    if version < _EMPYMOD_MIN_VERSION:
        found = _backend_version_text(backend) or "unknown"
        raise BlockedBySoftwareOrResourcesError(
            "BLOCKED_BY_SOFTWARE_OR_RESOURCES: empymod>=2.5.4 is required "
            f"for the layered Cole-Cole / Debye forward wrapper; found {found}"
        )


def _backend_version_text(backend) -> str | None:
    version = getattr(backend, "__version__", None)
    return None if version is None else str(version)


def _backend_version_tuple(backend) -> tuple[int, int, int] | None:
    text = _backend_version_text(backend)
    if text is None:
        return None
    parts: list[int] = []
    for token in text.replace("+", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
        if len(parts) == 3:
            break
    if len(parts) < 2:
        return None
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _validate_material_layers(material: Material, geometry: LayeredGeometry) -> None:
    if isinstance(material, NonPolarizableMaterial):
        count = len(material.resistivity_ohm_m)
    elif isinstance(material, ExactPeltonMaterial):
        count = len(material.rho0_ohm_m)
    else:
        count = len(material.sigma_infinity_s_per_m)
    if count != geometry.n_layers:
        raise ValueError(
            "material layer count must equal geometry.n_layers "
            f"(got {count} vs {geometry.n_layers})"
        )


def _point_receiver_record(receiver: ReceiverSpec) -> dict[str, Any]:
    return {
        "label": receiver.label,
        "kind": "point",
        "location": list(receiver.location),
        "radius_m": None,
        "quadrature": {"n_points": 1},
    }


def _waveform_provenance(
    spec: WaveformSpec,
    resolved: PiecewiseLinearTurnOff | None,
) -> dict[str, Any]:
    if resolved is None:
        return {
            "name": spec.label,
            "kind": spec.kind,
            "time_origin": "ramp_end",
            "duration_s": 0.0,
            "quadrature_order": spec.quadrature_order,
        }
    metadata = resolved.metadata()
    metadata["kind"] = spec.kind
    metadata["label"] = spec.label
    metadata["quadrature_order"] = spec.quadrature_order
    return metadata


def _frequency_provenance(
    times: np.ndarray,
    waveform: PiecewiseLinearTurnOff | None,
    transform_settings: TransformSettings,
    *,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    eval_times = np.asarray(times, dtype=float)
    if waveform is not None:
        from atem3d.empymod_waveform import _turnoff_quadrature  # noqa: PLC0415

        delays, _weights, _segments = _turnoff_quadrature(waveform, int(quadrature_order))
        eval_times = np.unique((eval_times[:, None] - delays[None, :]).reshape(-1))
    kwargs = transform_settings.empymod_kwargs(2)
    frequencies = _empymod_frequency_grid(
        eval_times,
        kwargs["ft"],
        kwargs["ftarg"],
    )
    pts = int(transform_settings.ft_pts_per_dec)
    if transform_settings.ft == "fftlog":
        sampling = "fftlog"
    elif pts == 0:
        sampling = "per_time_standard_dlf"
    elif pts < 0:
        sampling = "lagged_convolution_dlf"
    else:
        sampling = "splined_dlf"
    return {
        "n_unique": int(frequencies.size),
        "min_hz": float(np.min(frequencies)) if frequencies.size else None,
        "max_hz": float(np.max(frequencies)) if frequencies.size else None,
        "shape": list(frequencies.shape),
        "sampling": sampling,
        "hankel_method": transform_settings.ht,
        "hankel_filter": transform_settings.ht_filter,
        "fourier_method": transform_settings.ft,
        "fourier_filter": transform_settings.ft_filter,
        "frequency_range_hz": [
            None if frequencies.size == 0 else float(np.min(frequencies)),
            None if frequencies.size == 0 else float(np.max(frequencies)),
        ],
        "frequency_sampling": sampling,
    }


def _empymod_frequency_grid(times: np.ndarray, ft: str, ftarg: Mapping[str, Any]) -> np.ndarray:
    try:
        import empymod  # noqa: PLC0415
    except ImportError:
        return np.asarray([], dtype=float)
    check_time = getattr(getattr(empymod, "utils", None), "check_time", None)
    if check_time is None:
        return np.asarray([], dtype=float)
    try:
        result = check_time(
            np.asarray(times, dtype=float),
            -1,
            ft,
            dict(ftarg),
            0,
            new=True,
        )
    except TypeError:
        try:
            result = check_time(np.asarray(times, dtype=float), -1, ft, dict(ftarg), 0)
        except Exception:
            return np.asarray([], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    frequencies = _extract_frequencies(result)
    return np.asarray(frequencies, dtype=float).reshape(-1)


def _extract_frequencies(result: Any) -> np.ndarray:
    if hasattr(result, "freq"):
        return np.asarray(result.freq, dtype=float)
    if isinstance(result, tuple) and len(result) >= 2:
        # empymod.utils.check_time returns (time, freq, ft, ftarg[, signal]).
        array = np.asarray(result[1])
        if array.size and np.issubdtype(array.dtype, np.number):
            return np.asarray(array, dtype=float)
    return np.asarray([], dtype=float)


def _finite_xyz(value, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in array)


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        _jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating):
            return float(value)
        if np.issubdtype(value.dtype, np.integer):
            return int(value)
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value)
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
