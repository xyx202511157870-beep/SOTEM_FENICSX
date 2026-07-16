"""Fail-closed scientific verification helpers for seepage-channel runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .seepage_channel_model import SeepageChannelBenchmark


class ModelContractMismatch(ValueError):
    """Raised when artifacts do not describe one identical physical model."""


class VerificationGateError(RuntimeError):
    """Raised when a formal artifact is requested before verification passes."""


FORMAL_RECEIVER_INDICES = (0, 1, 3, 4)
COMPONENTS = ("Ex", "dBzdt", "Hz")


def _vectors(values: Any) -> list[list[float]]:
    return [[float(component) for component in vector] for vector in values]


def canonical_model_contract(model: SeepageChannelBenchmark) -> dict[str, Any]:
    """Return the complete JSON-safe physical contract for a benchmark model."""

    return {
        "coordinate_convention": str(model.coordinate_convention),
        "source": {
            "endpoints_m": _vectors(model.source_endpoints),
            "current_a": float(model.source_current_a),
            "waveform": str(model.waveform),
        },
        "receivers_m": _vectors(model.receiver_locations),
        "components": [str(component) for component in model.components],
        "times_s": [float(value) for value in model.times],
        "air_conductivity_s_per_m": float(model.air_conductivity),
        "background_conductivity_s_per_m": float(model.background_conductivity),
        "channel": {
            "center_m": [float(value) for value in model.channel.center],
            "size_m": [float(value) for value in model.channel.size],
            "bounds_m": _vectors(model.channel.bounds),
            "conductivity_s_per_m": float(model.channel.conductivity),
        },
    }


def canonical_model_json(model: SeepageChannelBenchmark) -> str:
    """Serialize a model contract deterministically for hashing and manifests."""

    return json.dumps(
        canonical_model_contract(model),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def model_fingerprint(model: SeepageChannelBenchmark) -> str:
    """Return the SHA-256 fingerprint of the complete physical contract."""

    return hashlib.sha256(canonical_model_json(model).encode("utf-8")).hexdigest()


def require_consistent_fingerprints(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the common fingerprint or fail on missing/mixed artifact metadata."""

    fingerprints: dict[str, str] = {}
    for name, artifact in artifacts.items():
        raw = artifact.get("model_fingerprint")
        if raw is None or not str(raw).strip():
            raise ModelContractMismatch(f"missing model fingerprint for {name}")
        fingerprints[str(name)] = str(raw).strip().lower()
    unique = set(fingerprints.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={value}" for name, value in fingerprints.items())
        raise ModelContractMismatch(f"mixed model fingerprints: {details}")
    if not unique:
        raise ModelContractMismatch("missing model fingerprint artifacts")
    return next(iter(unique))


def _require_close(name: str, actual: Any, expected: Any, *, atol: float = 1e-12) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=atol,
    ):
        raise ModelContractMismatch(f"FEniCSx {name} does not match model contract")


def verify_fenicsx_run_contract(
    run_dir: str | Path,
    *,
    model: SeepageChannelBenchmark,
    case: str,
    expected_local_mesh_size: float | None = None,
) -> dict[str, Any]:
    """Verify FEniCSx resolved inputs before signing them with a model hash."""

    root = Path(run_dir)
    with (root / "run_config_resolved.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (root / "fenicsx_run_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    expected_source = [
        [float(x), float(y), -float(z)] for x, y, z in model.source_endpoints
    ]
    _require_close(
        "source endpoints",
        [config["source_start"], config["source_end"]],
        expected_source,
    )
    _require_close("source current", config["source_current"], model.source_current_a)
    _require_close("air conductivity", 1.0 / float(config["rho_air"]), model.air_conductivity)
    _require_close(
        "background conductivity",
        1.0 / float(config["rho_earth"]),
        model.background_conductivity,
    )
    _require_close("first output time", config["t_min"], model.times[0])
    _require_close("last output time", config["t_max"], model.times[-1])
    _require_close(
        "output time growth",
        config["time_growth"],
        float(model.times[1] / model.times[0]),
        atol=1e-14,
    )

    if int(summary.get("receiver_count", -1)) != len(model.receiver_locations):
        raise ModelContractMismatch("FEniCSx receiver count does not match model contract")
    if summary.get("receiver_provenance") != ["explicit_full_domain"] * len(
        model.receiver_locations
    ):
        raise ModelContractMismatch("FEniCSx receivers are not explicit full-domain values")

    material = summary.get("material_audit", {})
    if not material.get("enabled"):
        raise ModelContractMismatch("FEniCSx conductivity-box material audit is missing")
    _require_close(
        "conductivity-box bounds",
        material.get("bounds"),
        model.channel.to_z_up_bounds(),
    )
    expected_sigma = (
        model.background_conductivity if str(case) == "background" else model.channel.conductivity
    )
    _require_close(
        "conductivity-box conductivity",
        material.get("sigma_s_per_m"),
        expected_sigma,
    )
    _require_close(
        "conductivity-box theoretical volume",
        material.get("theoretical_volume_m3"),
        model.channel.volume_m3,
    )
    if expected_local_mesh_size is not None:
        _require_close(
            "conductivity-box mesh size",
            material.get("mesh_size_m"),
            expected_local_mesh_size,
        )

    return {
        "method": "FEniCSx",
        "case": str(case),
        "model_fingerprint": model_fingerprint(model),
        "magnetic_receiver_mode": str(config.get("magnetic_receiver_mode", "")),
        "magnetic_dbdt_mode": str(config.get("magnetic_dbdt_mode", "")),
        "material_audit": material,
    }


def verify_simpeg_config_contract(
    config_path: str | Path,
    *,
    model: SeepageChannelBenchmark,
    case: str,
    expected_local_mesh_size: float | None = None,
) -> dict[str, Any]:
    """Verify a SimPEG YAML input before signing it with a model hash."""

    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("coordinate_system") != "z_up":
        raise ModelContractMismatch("SimPEG coordinate system must be z_up")

    expected_source = [
        [float(x), float(y), -float(z)] for x, y, z in model.source_endpoints
    ]
    source = config["source"]
    _require_close(
        "source endpoints", [source["start"], source["end"]], expected_source
    )
    _require_close("source current", source["current"], model.source_current_a)
    if source.get("waveform", {}).get("type") != "step_off":
        raise ModelContractMismatch("SimPEG waveform does not match ideal step-off")

    layers = config["model"]["layers"]
    if len(layers) != 2:
        raise ModelContractMismatch("SimPEG model must contain air and earth layers")
    _require_close(
        "air conductivity", layers[0]["sigma_infinity"], model.air_conductivity
    )
    _require_close(
        "background conductivity",
        layers[1]["sigma_infinity"],
        model.background_conductivity,
    )

    expected_receivers = [
        ([float(x), float(y), -float(z)], str(component))
        for x, y, z in model.receiver_locations
        for component in model.components
    ]
    actual_receivers = [
        ([float(value) for value in receiver["location"]], str(receiver["component"]))
        for receiver in config["receivers"]
    ]
    if actual_receivers != expected_receivers:
        raise ModelContractMismatch("SimPEG receivers do not match model contract")

    boxes = config["model"].get("conductivity_boxes", [])
    if str(case) == "channel":
        if len(boxes) != 1:
            raise ModelContractMismatch("SimPEG channel config requires one conductivity box")
        box = boxes[0]
        _require_close(
            "conductivity-box bounds", box["bounds"], model.channel.to_z_up_bounds()
        )
        _require_close(
            "conductivity-box conductivity",
            box["sigma_infinity"],
            model.channel.conductivity,
        )
    elif str(case) == "background":
        if boxes:
            raise ModelContractMismatch("SimPEG background config must not add contrast")
    else:
        raise ValueError(f"unknown SimPEG seepage case: {case}")

    duration = sum(
        float(step) * int(count) for step, count, *_rest in config["time_steps"]
    )
    _require_close("simulation end time", duration, model.times[-1], atol=1e-14)
    for axis in ("hy", "hz"):
        minimum = min(float(segment[0]) for segment in config["mesh"][axis])
        if expected_local_mesh_size is not None and not np.isclose(
            minimum, expected_local_mesh_size
        ):
            raise ModelContractMismatch(
                f"SimPEG {axis} local mesh size does not match controlled case"
            )
        if expected_local_mesh_size is None and minimum > 0.25:
            raise ModelContractMismatch(f"SimPEG {axis} does not resolve the 1 m channel")

    return {
        "method": "SimPEG",
        "case": str(case),
        "model_fingerprint": model_fingerprint(model),
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _formal_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[0] != 5 or array.shape[2] != 3:
        raise ValueError("solver values must have shape (5, n_times, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("solver values must all be finite")
    return array[list(FORMAL_RECEIVER_INDICES)]


def zero_contrast_metrics(
    delta: Any,
    background: Any,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Measure zero-contrast residuals on the four formal receivers."""

    delta_values = _formal_values(delta)
    background_values = _formal_values(background)
    ratios = []
    for component in range(3):
        numerator = np.linalg.norm(delta_values[:, :, component])
        denominator = max(
            np.linalg.norm(background_values[:, :, component]), np.finfo(float).tiny
        )
        ratios.append(float(numerator / denominator))
    return {
        "available": True,
        "pass": bool(all(value <= threshold for value in ratios)),
        "threshold": float(threshold),
        "normalized_l2": ratios,
        "formal_receiver_indices": list(FORMAL_RECEIVER_INDICES),
    }


def anomaly_energy_trend(
    control_values: list[float] | tuple[float, ...],
    deltas: list[Any] | tuple[Any, ...],
    background: Any,
    *,
    relative_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check that normalized anomaly energy is non-decreasing in a sweep."""

    if len(control_values) != len(deltas) or len(deltas) < 2:
        raise ValueError("a trend requires matching control values and at least two deltas")
    background_values = _formal_values(background)
    scale = max(np.linalg.norm(background_values), np.finfo(float).tiny)
    energies = [float(np.linalg.norm(_formal_values(delta)) / scale) for delta in deltas]
    monotone = all(
        later + relative_tolerance * max(1.0, abs(earlier)) >= earlier
        for earlier, later in zip(energies[:-1], energies[1:])
    )
    return {
        "available": True,
        "pass": bool(monotone),
        "control_values": [float(value) for value in control_values],
        "energies": energies,
        "relative_tolerance": float(relative_tolerance),
    }


def _relative_error_samples(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    samples: list[np.ndarray] = []
    for receiver in range(first.shape[0]):
        for component in range(first.shape[2]):
            a = first[receiver, :, component]
            b = second[receiver, :, component]
            scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            if scale == 0.0:
                continue
            mask = (np.abs(a) >= 0.05 * scale) | (np.abs(b) >= 0.05 * scale)
            floor = max(scale * 1e-12, np.finfo(float).tiny)
            denominator = np.maximum(np.maximum(np.abs(a[mask]), np.abs(b[mask])), floor)
            samples.append(np.abs(a[mask] - b[mask]) / denominator)
    return np.concatenate(samples) if samples else np.empty(0, dtype=float)


def _error_stats(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    errors = _relative_error_samples(first, second)
    if errors.size == 0:
        return {"count": 0, "median": 0.0, "p95": 0.0}
    return {
        "count": int(errors.size),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95.0)),
    }


def three_level_convergence(
    coarse: Any,
    medium: Any,
    fine: Any,
    *,
    median_threshold: float,
    p95_threshold: float,
) -> dict[str, Any]:
    """Evaluate decreasing coarse/medium/fine changes on anomaly arrays."""

    coarse_values = _formal_values(coarse)
    medium_values = _formal_values(medium)
    fine_values = _formal_values(fine)
    medium_coarse = _error_stats(medium_values, coarse_values)
    fine_medium = _error_stats(fine_values, medium_values)
    passed = (
        fine_medium["median"] < medium_coarse["median"]
        and fine_medium["p95"] < medium_coarse["p95"]
        and fine_medium["median"] <= median_threshold
        and fine_medium["p95"] <= p95_threshold
    )
    return {
        "available": True,
        "pass": bool(passed),
        "medium_coarse": medium_coarse,
        "fine_medium": fine_medium,
        "median_threshold": float(median_threshold),
        "p95_threshold": float(p95_threshold),
    }


def parity_metrics(
    values: Any,
    *,
    pair_threshold: float,
    center_threshold: float,
) -> dict[str, Any]:
    """Evaluate Ex even parity and magnetic odd parity on the five-point array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[0] != 5 or array.shape[2] != 3:
        raise ValueError("solver values must have shape (5, n_times, 3)")
    component_metrics: dict[str, Any] = {}
    passed = True
    for component_index, component in enumerate(COMPONENTS):
        parity_sign = 1.0 if component == "Ex" else -1.0
        peak = max(float(np.max(np.abs(array[:, :, component_index]))), np.finfo(float).tiny)
        pair_15 = float(
            np.max(
                np.abs(
                    array[0, :, component_index]
                    - parity_sign * array[4, :, component_index]
                )
            )
            / peak
        )
        pair_24 = float(
            np.max(
                np.abs(
                    array[1, :, component_index]
                    - parity_sign * array[3, :, component_index]
                )
            )
            / peak
        )
        center_ratio = (
            0.0
            if component == "Ex"
            else float(np.max(np.abs(array[2, :, component_index])) / peak)
        )
        item_pass = max(pair_15, pair_24) <= pair_threshold and (
            component == "Ex" or center_ratio <= center_threshold
        )
        passed &= item_pass
        component_metrics[component] = {
            "parity": "even" if component == "Ex" else "odd",
            "pair_15_residual": pair_15,
            "pair_24_residual": pair_24,
            "center_ratio": center_ratio,
            "pass": bool(item_pass),
        }
    return {
        "available": True,
        "pass": bool(passed),
        "pair_threshold": float(pair_threshold),
        "center_threshold": float(center_threshold),
        "components": component_metrics,
    }


def cross_solver_agreement(
    first: Any,
    second: Any,
    *,
    median_threshold: float,
    p95_threshold: float,
) -> dict[str, Any]:
    """Compare two anomaly arrays on formal receivers and strong signals."""

    first_values = _formal_values(first)
    second_values = _formal_values(second)
    components: dict[str, Any] = {}
    passed = True
    for index, component in enumerate(COMPONENTS):
        stats = _error_stats(
            first_values[:, :, index : index + 1],
            second_values[:, :, index : index + 1],
        )
        stats["pass"] = bool(
            stats["median"] <= median_threshold and stats["p95"] <= p95_threshold
        )
        passed &= stats["pass"]
        components[component] = stats
    return {
        "available": True,
        "pass": bool(passed),
        "median_threshold": float(median_threshold),
        "p95_threshold": float(p95_threshold),
        "components": components,
        "formal_receiver_indices": list(FORMAL_RECEIVER_INDICES),
    }


def build_verification_summary(
    *,
    model_fingerprint_value: str,
    required_gates: tuple[str, ...] | list[str],
    gates: Mapping[str, Mapping[str, Any]],
    require_pass: bool = False,
) -> dict[str, Any]:
    """Build a deterministic fail-closed verification summary."""

    normalized: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for name in required_gates:
        item = dict(gates.get(name, {"available": False, "pass": False}))
        item.setdefault("available", False)
        item.setdefault("pass", False)
        normalized[str(name)] = item
        if not item["available"] or not item["pass"]:
            failed.append(str(name))
    summary = {
        "model_fingerprint": str(model_fingerprint_value),
        "required_gates": [str(name) for name in required_gates],
        "gates": normalized,
        "failed_gates": failed,
        "pass": not failed,
    }
    if require_pass and failed:
        raise VerificationGateError(
            "verification failed for mandatory gates: " + ", ".join(failed)
        )
    return summary


__all__ = [
    "ModelContractMismatch",
    "VerificationGateError",
    "FORMAL_RECEIVER_INDICES",
    "anomaly_energy_trend",
    "build_verification_summary",
    "canonical_model_contract",
    "canonical_model_json",
    "model_fingerprint",
    "parity_metrics",
    "require_consistent_fingerprints",
    "three_level_convergence",
    "cross_solver_agreement",
    "verify_fenicsx_run_contract",
    "verify_simpeg_config_contract",
    "zero_contrast_metrics",
]
