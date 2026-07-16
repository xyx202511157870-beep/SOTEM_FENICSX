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
        if minimum > 0.25:
            raise ModelContractMismatch(f"SimPEG {axis} does not resolve the 1 m channel")

    return {
        "method": "SimPEG",
        "case": str(case),
        "model_fingerprint": model_fingerprint(model),
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "ModelContractMismatch",
    "canonical_model_contract",
    "canonical_model_json",
    "model_fingerprint",
    "require_consistent_fingerprints",
    "verify_fenicsx_run_contract",
    "verify_simpeg_config_contract",
]
