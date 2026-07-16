#!/usr/bin/env python3
"""Audit five-receiver magnetic methods and select a formal output fail-closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.magnetic_symmetry_audit import audit_model_triplet


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def build_method_summary(
    methods: Mapping[str, Mapping[str, np.ndarray]],
    *,
    components: tuple[str, ...],
    times: np.ndarray,
    signal_floor: float,
) -> dict[str, dict[str, Any]]:
    """Aggregate background/channel/delta symmetry and cross-method errors."""

    if not methods:
        raise ValueError("method set must not be empty")
    time_values = _finite_array(times, "times").reshape(-1)
    floor = float(signal_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("signal_floor must be finite and nonnegative")
    result: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, models in methods.items():
        if set(models) != {"background", "channel"}:
            raise ValueError(f"method {name!r} requires background and channel")
        background = _finite_array(models["background"], f"{name}.background")
        channel = _finite_array(models["channel"], f"{name}.channel")
        metrics = audit_model_triplet(background, channel, components, time_values)
        magnetic_metrics = [
            metrics[model][component]
            for model in ("background", "channel", "delta")
            for component in ("dBzdt", "Hz")
        ]

        def worst(key: str) -> float | None:
            values = [item[key] for item in magnetic_metrics]
            if any(value is None for value in values):
                return None
            return float(max(float(value) for value in values))

        flank_scale = max(
            float(np.max(np.abs(background[[0, 1, 3, 4], :, 1:]))),
            float(np.max(np.abs(channel[[0, 1, 3, 4], :, 1:]))),
        )
        rx3_abs = max(
            float(np.max(np.abs(background[2, :, 1:]))),
            float(np.max(np.abs(channel[2, :, 1:]))),
        )
        result[str(name)] = {
            "models": metrics,
            "rx3_zero_ratio": worst("rx3_zero_ratio"),
            "pair_24_residual": worst("pair_24_residual"),
            "pair_15_residual": worst("pair_15_residual"),
            "rx3_abs_max": rx3_abs,
            "rx3_relative_error": None if flank_scale <= floor else rx3_abs / flank_scale,
            "strong_signal_median_error": 0.0,
            "strong_signal_error_increase_percentage_points": 0.0,
            "ex_median_change": 0.0,
        }
        arrays[str(name)] = {"background": background, "channel": channel}

    baseline_name = "curl" if "curl" in arrays else sorted(arrays)[0]
    baseline = arrays[baseline_name]
    for name, model_arrays in arrays.items():
        magnetic_errors = []
        ex_changes = []
        for model in ("background", "channel"):
            reference = baseline[model]
            candidate = model_arrays[model]
            magnetic_mask = np.abs(reference[:, :, 1:]) > floor
            if np.any(magnetic_mask):
                magnetic_errors.extend(
                    (
                        np.abs(candidate[:, :, 1:] - reference[:, :, 1:])[magnetic_mask]
                        / np.abs(reference[:, :, 1:])[magnetic_mask]
                    ).tolist()
                )
            ex_mask = np.abs(reference[:, :, 0]) > floor
            if np.any(ex_mask):
                ex_changes.extend(
                    (
                        np.abs(candidate[:, :, 0] - reference[:, :, 0])[ex_mask]
                        / np.abs(reference[:, :, 0])[ex_mask]
                    ).tolist()
                )
        median_error = float(np.median(magnetic_errors)) if magnetic_errors else 0.0
        result[name]["strong_signal_median_error"] = median_error
        result[name]["strong_signal_error_increase_percentage_points"] = 100.0 * median_error
        result[name]["ex_median_change"] = float(np.median(ex_changes)) if ex_changes else 0.0
    result[baseline_name]["strong_signal_error_increase_percentage_points"] = 0.0
    return result


def magnetic_method_failures(metrics: Mapping[str, Any]) -> list[str]:
    limits = {
        "rx3_zero_ratio": 0.01,
        "pair_24_residual": 0.01,
        "pair_15_residual": 0.01,
        "strong_signal_error_increase_percentage_points": 2.0,
        "ex_median_change": 0.005,
    }
    failures = []
    for key, limit in limits.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) > limit:
            failures.append(key)
    return failures


def select_formal_method(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    accepted = []
    rejected = {}
    for name, metrics in summary.items():
        failures = magnetic_method_failures(metrics)
        if failures:
            rejected[name] = failures
        else:
            accepted.append(name)
    if not accepted:
        return {"passed": False, "selected": None, "rejected": rejected}
    ranking = sorted(
        accepted,
        key=lambda name: (
            summary[name]["strong_signal_median_error"],
            summary[name]["rx3_zero_ratio"],
            name,
        ),
    )
    return {"passed": True, "selected": ranking[0], "rejected": rejected}


def load_method_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: np.asarray(stored[name]) for name in stored.files}
    validate_method_payload(payload)
    return payload


def validate_method_payload(payload: Mapping[str, np.ndarray]) -> None:
    required = {"times", "receiver_locations", "components", "receiver_provenance"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"method payload missing {sorted(missing)}")
    times = _finite_array(payload["times"], "times").reshape(-1)
    receivers = _finite_array(payload["receiver_locations"], "receiver_locations")
    components = np.asarray(payload["components"]).astype(str).reshape(-1)
    provenance = np.asarray(payload["receiver_provenance"]).astype(str).reshape(-1)
    if receivers.shape != (5, 3) or times.size == 0:
        raise ValueError("payload requires five receivers and nonempty times")
    if tuple(components) != ("Ex", "dBzdt", "Hz"):
        raise ValueError("components must be Ex,dBzdt,Hz")
    if provenance.shape != (5,) or not np.all(provenance == "explicit_full_domain"):
        raise ValueError("receiver provenance must be explicit_full_domain")
    method_keys = [key for key in payload if key.startswith("background__")]
    if not method_keys:
        raise ValueError("method payload contains no methods")
    for key in method_keys:
        method = key.split("__", 1)[1]
        channel_key = f"channel__{method}"
        if channel_key not in payload:
            raise ValueError(f"missing {channel_key}")
        expected = (5, times.size, 3)
        if np.asarray(payload[key]).shape != expected or np.asarray(payload[channel_key]).shape != expected:
            raise ValueError(f"method arrays must have shape {expected}")
        _finite_array(payload[key], key)
        _finite_array(payload[channel_key], channel_key)


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def run_audit(payload_path: Path, output_dir: Path, *, signal_floor: float = 1.0e-20) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    try:
        payload = load_method_payload(payload_path)
        components = tuple(np.asarray(payload["components"]).astype(str).tolist())
        methods = {
            key.split("__", 1)[1]: {
                "background": payload[key],
                "channel": payload[f"channel__{key.split('__', 1)[1]}"],
            }
            for key in payload
            if key.startswith("background__")
        }
        summary = build_method_summary(
            methods,
            components=components,
            times=payload["times"],
            signal_floor=signal_floor,
        )
        selection = select_formal_method(summary)
        metrics_path = output_dir / "magnetic_symmetry_metrics.json"
        convergence_path = output_dir / "magnetic_convergence_summary.json"
        write_json(metrics_path, summary)
        write_json(convergence_path, selection)
        table_path = output_dir / "magnetic_receiver_methods.csv"
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["method", "rx3_zero_ratio", "pair_24_residual", "pair_15_residual", "strong_signal_median_error", "passed"],
            )
            writer.writeheader()
            for name, metrics in sorted(summary.items()):
                writer.writerow({
                    "method": name,
                    "rx3_zero_ratio": metrics["rx3_zero_ratio"],
                    "pair_24_residual": metrics["pair_24_residual"],
                    "pair_15_residual": metrics["pair_15_residual"],
                    "strong_signal_median_error": metrics["strong_signal_median_error"],
                    "passed": not magnetic_method_failures(metrics),
                })
        artifacts = {path.name: _sha256(path) for path in (payload_path, metrics_path, convergence_path, table_path)}
        manifest = {**selection, "artifacts": artifacts}
        write_json(manifest_path, manifest)
        return manifest
    except Exception:
        write_json(manifest_path, {"passed": False, "selected": None, "rejected": {"input": ["invalid"]}})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-floor", type=float, default=1.0e-20)
    args = parser.parse_args()
    result = run_audit(args.payload, args.output_dir, signal_floor=args.signal_floor)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
