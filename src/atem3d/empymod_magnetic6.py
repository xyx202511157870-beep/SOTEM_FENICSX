"""Formal empymod validation helpers for H and dB/dt three-component data.

This module defines one canonical six-channel magnetic contract:

    Hx, Hy, Hz              [A/m]
    dBxdt, dBydt, dBzdt     [T/s]

For an ideal switch-off source, dB/dt can be constructed from the magnetic
H impulse response through

    dB/dt = -mu0 * H_impulse.

This route is supported by empymod 2.5.4 and later. empymod 2.6 and later
also expose a native magnetic-flux receiver mode (``mrec='b'``) that returns
dB/dt directly. When available, the two routes can be cross-checked to catch
receiver orientation, coordinate-system, sign, unit, and transform-setting
errors before a numerical solver is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from scipy.constants import mu_0

from .empymod_compare import (
    EmpymodSurvey,
    _component_coordinate_factor,
    _receiver_mapping,
    _resistivity_model,
    _source_mapping,
)


MAGNETIC6_COMPONENTS = (
    "Hx",
    "Hy",
    "Hz",
    "dBxdt",
    "dBydt",
    "dBzdt",
)
MAGNETIC6_UNITS = (
    "A/m",
    "A/m",
    "A/m",
    "T/s",
    "T/s",
    "T/s",
)
_H_COMPONENTS = MAGNETIC6_COMPONENTS[:3]
_DBDT_COMPONENTS = MAGNETIC6_COMPONENTS[3:]


@dataclass(frozen=True)
class MagneticSixReferenceResult:
    """empymod six-component reference and optional dB/dt cross-check."""

    times: np.ndarray
    receiver_locations: tuple[tuple[float, float, float], ...]
    data: np.ndarray
    dbdt_native: np.ndarray | None
    dbdt_impulse: np.ndarray | None
    primary_dbdt_reference: str
    empymod_version: str | None
    audit: dict[str, Any]

    @property
    def components(self) -> tuple[str, ...]:
        return MAGNETIC6_COMPONENTS

    @property
    def units(self) -> tuple[str, ...]:
        return MAGNETIC6_UNITS

    def flat_data(self) -> np.ndarray:
        """Return ``(n_times, n_locations*6)`` in location-major order."""

        return np.asarray(self.data, dtype=float).reshape(self.times.size, -1)


@dataclass(frozen=True)
class MagneticSixNumericalData:
    """Numerical six-component data loaded from CSV or NPZ."""

    times: np.ndarray
    data: np.ndarray
    receiver_locations: tuple[tuple[float, float, float], ...] | None = None


def build_magnetic6_survey_from_config(
    config: dict[str, Any],
    *,
    times: Sequence[float],
    depths: Sequence[float],
    resistivities: Sequence[float] | dict[str, Any],
    signal: int = -1,
) -> EmpymodSurvey:
    """Build a canonical six-component empymod survey from a YAML config."""

    if signal != -1:
        raise ValueError(
            "formal magnetic-six validation requires ideal switch-off signal=-1"
        )
    source_cfg = config.get("source")
    if not isinstance(source_cfg, dict):
        raise ValueError("config must contain a source mapping")
    locations = receiver_locations_from_config(config)
    flat = [
        (location, component)
        for location in locations
        for component in MAGNETIC6_COMPONENTS
    ]
    return EmpymodSurvey(
        source_start=tuple(float(value) for value in source_cfg["start"]),
        source_end=tuple(float(value) for value in source_cfg["end"]),
        receiver_locations=locations,
        components=MAGNETIC6_COMPONENTS,
        times=np.asarray(times, dtype=float),
        depths=list(depths),
        resistivities=resistivities,
        strength=float(source_cfg.get("current", 1.0)),
        signal=-1,
        receiver_components=flat,
        coordinate_system=str(config.get("coordinate_system", "depth_down")),
    )


def receiver_locations_from_config(
    config: dict[str, Any],
) -> tuple[tuple[float, float, float], ...]:
    """Return unique receiver locations in deterministic input order."""

    if "receiver_line" in config:
        line = config["receiver_line"]
        if not isinstance(line, dict):
            raise ValueError("receiver_line must be a mapping")
        x_values = _as_coordinate_sequence(line.get("x"), "receiver_line.x")
        y_values = _broadcast_coordinate(
            line.get("y"), len(x_values), "receiver_line.y"
        )
        z_values = _broadcast_coordinate(
            line.get("z"), len(x_values), "receiver_line.z"
        )
        return tuple(
            (float(x), float(y), float(z))
            for x, y, z in zip(x_values, y_values, z_values)
        )

    if "receivers" in config:
        raw = config["receivers"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("receivers must be a non-empty list")
        locations: list[tuple[float, float, float]] = []
        seen: set[tuple[float, float, float]] = set()
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each receiver must be a mapping")
            location_raw = item.get("location")
            if location_raw is None:
                location_raw = (item.get("x"), item.get("y"), item.get("z"))
            location = _validated_location(location_raw)
            if location not in seen:
                locations.append(location)
                seen.add(location)
        return tuple(locations)

    if "receiver" in config:
        receiver = config["receiver"]
        if isinstance(receiver, dict):
            raw = receiver.get("location")
            if raw is None:
                raw = (
                    receiver.get("x"),
                    receiver.get("y"),
                    receiver.get("z"),
                )
        else:
            raw = receiver
        return (_validated_location(raw),)

    raise ValueError("config must contain receiver_line, receivers, or receiver")


def run_empymod_magnetic6_reference(
    survey: EmpymodSurvey,
    *,
    backend=None,
    srcpts: int = 9,
    recpts: int = 1,
    dbdt_reference: str = "auto",
    audit_impulse: bool = True,
    audit_tolerance: float = 0.01,
    audit_floor_fraction: float = 0.01,
    require_audit_pass: bool = False,
    **kwargs,
) -> MagneticSixReferenceResult:
    """Compute Hx/Hy/Hz and dBxdt/dBydt/dBzdt with empymod.

    ``auto`` uses native ``mrec='b'`` on empymod >= 2.6 and otherwise falls
    back to ``-mu0*H_impulse``. ``native_b`` requires empymod >= 2.6, while
    ``impulse_h`` works with empymod 2.5.4 and later.
    """

    if survey.signal != -1:
        raise ValueError("magnetic-six validation requires survey.signal == -1")
    times = np.asarray(survey.times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("survey.times must be a non-empty 1D array")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("survey.times must be finite and positive")
    if dbdt_reference not in {"auto", "native_b", "impulse_h"}:
        raise ValueError(
            "dbdt_reference must be 'auto', 'native_b', or 'impulse_h'"
        )
    if srcpts <= 0 or recpts <= 0:
        raise ValueError("srcpts and recpts must be positive")
    if audit_tolerance <= 0.0 or audit_floor_fraction <= 0.0:
        raise ValueError("audit tolerances must be positive")

    if backend is None:
        import empymod as backend  # noqa: PLC0415

    version_text = _backend_version_text(backend)
    native_available = _native_b_available(backend)
    primary_reference = _resolve_dbdt_reference(
        requested=dbdt_reference,
        native_available=native_available,
        version_text=version_text,
    )

    locations = tuple(
        _validated_location(value) for value in survey.receiver_locations
    )
    if not locations:
        raise ValueError("at least one receiver location is required")

    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("srcpts", int(srcpts))
    call_kwargs.setdefault("recpts", int(recpts))
    call_kwargs.setdefault("verb", 0)
    source = _source_mapping(
        survey.source_start,
        survey.source_end,
        survey.coordinate_system,
    )
    resistivity_model = _resistivity_model(survey.resistivities)

    h_data = np.zeros((times.size, len(locations), 3), dtype=float)
    need_native = primary_reference == "native_b" or (
        bool(audit_impulse) and native_available
    )
    need_impulse = primary_reference == "impulse_h" or bool(audit_impulse)
    native_dbdt = np.zeros_like(h_data) if need_native else None
    impulse_dbdt = np.zeros_like(h_data) if need_impulse else None

    for location_index, location in enumerate(locations):
        for component_index, component in enumerate(_H_COMPONENTS):
            rec, _mrec = _receiver_mapping(
                location,
                component,
                survey.coordinate_system,
            )
            response = backend.bipole(
                src=source,
                rec=rec,
                depth=list(survey.depths),
                res=resistivity_model,
                freqtime=times,
                signal=-1,
                strength=survey.strength,
                mrec=True,
                **call_kwargs,
            )
            scale = _component_coordinate_factor(
                component,
                survey.coordinate_system,
            )
            h_data[:, location_index, component_index] = (
                scale * _real_time_values(response, component)
            )

        for component_index, component in enumerate(_DBDT_COMPONENTS):
            rec, _mrec = _receiver_mapping(
                location,
                component,
                survey.coordinate_system,
            )
            coordinate_factor = _component_coordinate_factor(
                component,
                survey.coordinate_system,
            )
            if native_dbdt is not None:
                native = backend.bipole(
                    src=source,
                    rec=rec,
                    depth=list(survey.depths),
                    res=resistivity_model,
                    freqtime=times,
                    signal=-1,
                    strength=survey.strength,
                    mrec="b",
                    **call_kwargs,
                )
                native_dbdt[:, location_index, component_index] = (
                    coordinate_factor * _real_time_values(native, component)
                )

            if impulse_dbdt is not None:
                impulse = backend.bipole(
                    src=source,
                    rec=rec,
                    depth=list(survey.depths),
                    res=resistivity_model,
                    freqtime=times,
                    signal=0,
                    strength=survey.strength,
                    mrec=True,
                    **call_kwargs,
                )
                impulse_dbdt[:, location_index, component_index] = (
                    -mu_0
                    * coordinate_factor
                    * _real_time_values(impulse, component)
                )

    audit = _dbdt_audit(
        native_dbdt,
        impulse_dbdt,
        tolerance=float(audit_tolerance),
        floor_fraction=float(audit_floor_fraction),
        empymod_version=version_text,
        primary_reference=primary_reference,
        native_available=native_available,
        requested=bool(audit_impulse),
    )
    if require_audit_pass:
        if not bool(audit["performed"]):
            raise RuntimeError(
                "empymod dB/dt cross-check was required but is unavailable: "
                + str(audit.get("reason", "unknown reason"))
            )
        if not bool(audit["passed"]):
            raise RuntimeError(
                "empymod native dB/dt and -mu0*H-impulse routes failed the "
                f"cross-check: max_error="
                f"{audit['global_max_floor_relative_error']:.6g}, "
                f"tolerance={audit_tolerance:.6g}"
            )

    if primary_reference == "native_b":
        if native_dbdt is None:  # pragma: no cover
            raise RuntimeError("native dB/dt reference was not computed")
        selected_dbdt = native_dbdt
    else:
        if impulse_dbdt is None:  # pragma: no cover
            raise RuntimeError("impulse-H dB/dt reference was not computed")
        selected_dbdt = impulse_dbdt

    data = np.concatenate([h_data, selected_dbdt], axis=2)
    return MagneticSixReferenceResult(
        times=times.copy(),
        receiver_locations=locations,
        data=data,
        dbdt_native=native_dbdt,
        dbdt_impulse=impulse_dbdt,
        primary_dbdt_reference=primary_reference,
        empymod_version=version_text,
        audit=audit,
    )


def load_magnetic6_numerical(path: str | Path) -> MagneticSixNumericalData:
    """Load canonical magnetic-six data from NPZ or a single-location CSV."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "times" in archive:
                times = np.asarray(archive["times"], dtype=float)
            elif "time_obs" in archive:
                times = np.asarray(archive["time_obs"], dtype=float)
            else:
                raise ValueError("NPZ must contain 'times' or 'time_obs'")
            if "data" not in archive:
                raise ValueError("NPZ must contain 'data'")
            data = np.asarray(archive["data"], dtype=float)
            raw_components = (
                archive["components"] if "components" in archive else None
            )
            components = _decoded_components(raw_components)
            locations = _npz_locations(archive)
        data = _canonicalize_data(data, components)
        return MagneticSixNumericalData(
            times=times,
            data=data,
            receiver_locations=locations,
        )

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("CSV contains no data rows")
        time_key = "time_obs" if "time_obs" in rows[0] else "time_s"
        if time_key not in rows[0]:
            raise ValueError("CSV must contain time_obs or time_s")
        missing = [
            component
            for component in MAGNETIC6_COMPONENTS
            if component not in rows[0]
        ]
        if missing:
            raise ValueError(
                "CSV is missing magnetic-six columns: " + ", ".join(missing)
            )
        times = np.asarray([float(row[time_key]) for row in rows], dtype=float)
        data = np.asarray(
            [
                [float(row[component]) for component in MAGNETIC6_COMPONENTS]
                for row in rows
            ],
            dtype=float,
        )[:, None, :]
        return MagneticSixNumericalData(times=times, data=data)

    raise ValueError("numerical data must be .npz or .csv")


def compare_magnetic6(
    numerical: MagneticSixNumericalData,
    reference: MagneticSixReferenceResult,
    *,
    tolerance: float = 0.05,
    floor_fraction: float = 0.01,
) -> dict[str, Any]:
    """Compare numerical and empymod data with component-specific floors."""

    if tolerance <= 0.0 or floor_fraction <= 0.0:
        raise ValueError("comparison tolerances must be positive")
    if numerical.times.shape != reference.times.shape or not np.allclose(
        numerical.times,
        reference.times,
        rtol=0.0,
        atol=max(1.0e-15, 1.0e-12 * float(np.max(reference.times))),
    ):
        raise ValueError("numerical and reference time axes differ")
    numerical_data = np.asarray(numerical.data, dtype=float)
    reference_data = np.asarray(reference.data, dtype=float)
    if numerical_data.shape != reference_data.shape:
        raise ValueError(
            "numerical data shape must equal reference shape; "
            f"got {numerical_data.shape} and {reference_data.shape}"
        )

    component_metrics: dict[str, Any] = {}
    global_max = 0.0
    for index, component in enumerate(MAGNETIC6_COMPONENTS):
        ref = reference_data[..., index]
        pred = numerical_data[..., index]
        peak = float(np.max(np.abs(ref)))
        floor = max(float(floor_fraction) * peak, np.finfo(float).tiny)
        error = np.abs(pred - ref)
        scaled = error / np.maximum(np.abs(ref), floor)
        peak_normalized = error / max(peak, np.finfo(float).tiny)
        max_scaled = float(np.max(scaled))
        global_max = max(global_max, max_scaled)
        component_metrics[component] = {
            "unit": MAGNETIC6_UNITS[index],
            "reference_peak_abs": peak,
            "floor": floor,
            "max_abs_error": float(np.max(error)),
            "max_floor_relative_error": max_scaled,
            "max_peak_normalized_error": float(np.max(peak_normalized)),
            "rms_error": float(np.sqrt(np.mean(error**2))),
            "passed": bool(max_scaled <= tolerance),
        }

    return {
        "artifact_schema": "atem3d.empymod_magnetic6_validation.v2",
        "components": list(MAGNETIC6_COMPONENTS),
        "units": list(MAGNETIC6_UNITS),
        "n_times": int(reference.times.size),
        "n_locations": int(reference.data.shape[1]),
        "empymod_version": reference.empymod_version,
        "primary_dbdt_reference": reference.primary_dbdt_reference,
        "tolerance": float(tolerance),
        "floor_fraction": float(floor_fraction),
        "global_max_floor_relative_error": global_max,
        "component_metrics": component_metrics,
        "passed": bool(
            all(item["passed"] for item in component_metrics.values())
        ),
        "dbdt_reference_audit": reference.audit,
    }


def write_magnetic6_artifacts(
    output_dir: str | Path,
    *,
    numerical: MagneticSixNumericalData,
    reference: MagneticSixReferenceResult,
    comparison: dict[str, Any],
) -> None:
    """Write CSV, JSON, and diagnostic plots for six-component validation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "magnetic6_comparison.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "time_obs",
            "location_index",
            "x",
            "y",
            "z",
            "component",
            "unit",
            "numerical",
            "empymod",
            "abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for time_index, time in enumerate(reference.times):
            for location_index, location in enumerate(
                reference.receiver_locations
            ):
                for component_index, component in enumerate(
                    MAGNETIC6_COMPONENTS
                ):
                    pred = float(
                        numerical.data[time_index, location_index, component_index]
                    )
                    ref = float(
                        reference.data[time_index, location_index, component_index]
                    )
                    writer.writerow(
                        {
                            "time_obs": f"{float(time):.16g}",
                            "location_index": location_index,
                            "x": f"{location[0]:.16g}",
                            "y": f"{location[1]:.16g}",
                            "z": f"{location[2]:.16g}",
                            "component": component,
                            "unit": MAGNETIC6_UNITS[component_index],
                            "numerical": f"{pred:.16e}",
                            "empymod": f"{ref:.16e}",
                            "abs_error": f"{abs(pred-ref):.16e}",
                        }
                    )

    (output_dir / "magnetic6_error_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "empymod_dbdt_crosscheck.json").write_text(
        json.dumps(reference.audit, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _write_plots(output_dir, numerical, reference)


def _dbdt_audit(
    native: np.ndarray | None,
    impulse: np.ndarray | None,
    *,
    tolerance: float,
    floor_fraction: float,
    empymod_version: str | None,
    primary_reference: str,
    native_available: bool,
    requested: bool,
) -> dict[str, Any]:
    base = {
        "empymod_version": empymod_version,
        "primary_dbdt_reference": primary_reference,
        "native_b_available": bool(native_available),
        "requested": bool(requested),
        "tolerance": tolerance,
        "floor_fraction": floor_fraction,
    }
    if not requested:
        return {
            **base,
            "performed": False,
            "passed": True,
            "reason": "cross-check disabled by caller",
            "global_max_floor_relative_error": 0.0,
            "component_metrics": {},
        }
    if native is None:
        return {
            **base,
            "performed": False,
            "passed": True,
            "reason": (
                "native mrec='b' dB/dt is unavailable; empymod >= 2.6 is required"
            ),
            "global_max_floor_relative_error": 0.0,
            "component_metrics": {},
        }
    if impulse is None:
        return {
            **base,
            "performed": False,
            "passed": True,
            "reason": "impulse-H reference was not computed",
            "global_max_floor_relative_error": 0.0,
            "component_metrics": {},
        }

    metrics: dict[str, Any] = {}
    global_max = 0.0
    for index, component in enumerate(_DBDT_COMPONENTS):
        primary = native[..., index]
        secondary = impulse[..., index]
        peak = float(np.max(np.abs(primary)))
        floor = max(floor_fraction * peak, np.finfo(float).tiny)
        scaled = np.abs(primary - secondary) / np.maximum(np.abs(primary), floor)
        max_error = float(np.max(scaled))
        global_max = max(global_max, max_error)
        active = np.abs(primary) >= floor
        sign_agreement = (
            float(np.mean(np.sign(primary[active]) == np.sign(secondary[active])))
            if np.any(active)
            else 1.0
        )
        metrics[component] = {
            "unit": "T/s",
            "native_peak_abs": peak,
            "floor": floor,
            "max_floor_relative_error": max_error,
            "sign_agreement_fraction": sign_agreement,
            "passed": bool(max_error <= tolerance),
        }
    return {
        **base,
        "performed": True,
        "identity": "native mrec='b' switch-off dB/dt versus -mu0*H impulse",
        "global_max_floor_relative_error": global_max,
        "component_metrics": metrics,
        "passed": bool(all(item["passed"] for item in metrics.values())),
    }


def _write_plots(
    output_dir: Path,
    numerical: MagneticSixNumericalData,
    reference: MagneticSixReferenceResult,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (("H3", (0, 1, 2)), ("dBdt3", (3, 4, 5)))
    for group_name, component_indices in groups:
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
        first_component = component_indices[0]
        for axis, component_index in zip(axes, component_indices):
            component = MAGNETIC6_COMPONENTS[component_index]
            for location_index in range(reference.data.shape[1]):
                axis.loglog(
                    reference.times,
                    np.abs(reference.data[:, location_index, component_index]),
                    "-",
                    label=(
                        f"empymod loc{location_index}"
                        if component_index == first_component
                        else None
                    ),
                )
                axis.loglog(
                    numerical.times,
                    np.abs(numerical.data[:, location_index, component_index]),
                    "o",
                    markersize=3,
                    label=(
                        f"numerical loc{location_index}"
                        if component_index == first_component
                        else None
                    ),
                )
            axis.set_title(
                f"{component} [{MAGNETIC6_UNITS[component_index]}]"
            )
            axis.set_xlabel("time [s]")
            axis.grid(True, which="both", alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels)
        fig.savefig(
            output_dir / f"magnetic6_{group_name}_comparison.png", dpi=180
        )
        plt.close(fig)


def _canonicalize_data(
    data: np.ndarray,
    components: tuple[str, ...] | None,
) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(
            "numerical data must have shape (time, 6) or (time, location, 6)"
        )
    if components is None:
        if array.shape[-1] != 6:
            raise ValueError(
                "NPZ without components must have six channels in canonical order"
            )
        return array
    if len(components) != array.shape[-1]:
        raise ValueError("components length does not match data channels")
    missing = [
        component
        for component in MAGNETIC6_COMPONENTS
        if component not in components
    ]
    if missing:
        raise ValueError(
            "NPZ is missing magnetic-six components: " + ", ".join(missing)
        )
    indices = [components.index(component) for component in MAGNETIC6_COMPONENTS]
    return array[..., indices]


def _decoded_components(raw) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1)
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _npz_locations(
    archive,
) -> tuple[tuple[float, float, float], ...] | None:
    if "receiver_locations" not in archive:
        return None
    locations = np.asarray(archive["receiver_locations"], dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (n_locations, 3)")
    return tuple(tuple(float(value) for value in row) for row in locations)


def _real_time_values(response, component: str) -> np.ndarray:
    values = np.asarray(response)
    values = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(values):
        imaginary = float(np.max(np.abs(values.imag)))
        real_scale = max(float(np.max(np.abs(values.real))), np.finfo(float).tiny)
        if imaginary > 1.0e-10 * real_scale:
            raise ValueError(
                f"empymod time response for {component} has non-negligible imaginary part"
            )
        values = values.real
    return np.asarray(values, dtype=float).reshape(-1)


def _backend_version_text(backend) -> str | None:
    version = getattr(backend, "__version__", None)
    return None if version is None else str(version)


def _backend_version_tuple(backend) -> tuple[int, int] | None:
    version = _backend_version_text(backend)
    if version is None:
        return None
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())


def _native_b_available(backend) -> bool:
    version = _backend_version_tuple(backend)
    if version is None:
        return True
    return version >= (2, 6)


def _resolve_dbdt_reference(
    *,
    requested: str,
    native_available: bool,
    version_text: str | None,
) -> str:
    if requested == "auto":
        return "native_b" if native_available else "impulse_h"
    if requested == "native_b" and not native_available:
        found = version_text or "unknown"
        raise RuntimeError(
            "native mrec='b' dB/dt validation requires "
            f"empymod >= 2.6; found {found}. "
            "Use dbdt_reference='impulse_h' or 'auto'."
        )
    return requested


def _as_coordinate_sequence(value, name: str) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if not values or any(not np.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite coordinates")
    return values


def _broadcast_coordinate(value, count: int, name: str) -> list[float]:
    values = _as_coordinate_sequence(value, name)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must be scalar or have length {count}")
    return values


def _validated_location(value) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError("receiver location must be a finite 3-vector")
    return tuple(float(item) for item in array)
