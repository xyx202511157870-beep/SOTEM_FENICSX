"""Validation and result I/O for the seepage-channel three-solver benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .four_way_validation import load_simpeg_values, run_empymod_reference
from .seepage_channel_model import MODEL


COMPONENTS = ("Ex", "dBzdt", "Hz")
EXPECTED_SHAPE = (5, 31, 3)


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("background_only_1d must be a scalar flag")
    return bool(array.reshape(()).item())


def channel_delta(channel: Any, background: Any) -> np.ndarray:
    channel_values = np.asarray(channel, dtype=float)
    background_values = np.asarray(background, dtype=float)
    if channel_values.shape != background_values.shape:
        raise ValueError("channel and background arrays must have the same shape")
    return channel_values - background_values


def ordinary_relative_error(predicted: Any, reference: Any) -> np.ndarray:
    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if predicted_values.shape != reference_values.shape:
        raise ValueError("predicted and reference arrays must have the same shape")
    errors = np.full(reference_values.shape, np.nan, dtype=float)
    defined = np.isfinite(reference_values) & (reference_values != 0.0)
    errors[defined] = (
        np.abs(predicted_values[defined] - reference_values[defined])
        / np.abs(reference_values[defined])
    )
    return errors


def strong_signal_mask(values: Any, peak_fraction: float = 0.05) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    fraction = float(peak_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("peak_fraction must lie in (0, 1]")
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=bool)
    peak = float(np.max(np.abs(array[finite])))
    if peak == 0.0:
        return np.zeros(array.shape, dtype=bool)
    return finite & (np.abs(array) >= fraction * peak)


def summarize_convergence(
    coarse: Any,
    refined: Any,
    *,
    strong_mask: Any,
) -> dict[str, Any]:
    coarse_values = np.asarray(coarse, dtype=float)
    refined_values = np.asarray(refined, dtype=float)
    mask = np.asarray(strong_mask, dtype=bool)
    if coarse_values.shape != refined_values.shape or mask.shape != coarse_values.shape:
        raise ValueError("coarse, refined, and strong_mask must have matching shapes")
    relative_change = ordinary_relative_error(coarse_values, refined_values)
    selected = mask & np.isfinite(relative_change)
    selected_values = relative_change[selected]
    return {
        "raw_count": int(relative_change.size),
        "strong_count": int(np.count_nonzero(mask)),
        "defined_strong_count": int(selected_values.size),
        "median_relative_change": (
            float(np.median(selected_values)) if selected_values.size else None
        ),
        "max_relative_change": (
            float(np.max(selected_values)) if selected_values.size else None
        ),
        "raw_relative_change": [
            None if not np.isfinite(value) else float(value)
            for value in relative_change.reshape(-1)
        ],
        "strong_mask": mask.reshape(-1).tolist(),
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _component_masks(reference: np.ndarray) -> dict[str, np.ndarray]:
    return {
        component: strong_signal_mask(reference[:, :, index], 0.05)
        for index, component in enumerate(COMPONENTS)
    }


def _error_summary(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    targets: Mapping[str, float],
) -> dict[str, Any]:
    errors = ordinary_relative_error(predicted, reference)
    masks = _component_masks(reference)
    summary: dict[str, Any] = {}
    for component_index, component in enumerate(COMPONENTS):
        mask = masks[component] & np.isfinite(errors[:, :, component_index])
        selected = errors[:, :, component_index][mask]
        median = float(np.median(selected)) if selected.size else None
        target = float(targets[component])
        summary[component] = {
            "strong_count": int(selected.size),
            "median_relative_error": median,
            "max_relative_error": float(np.max(selected)) if selected.size else None,
            "target": target,
            "pass": bool(median is not None and median <= target),
        }
    summary["pass_all_components"] = all(
        bool(summary[component]["pass"]) for component in COMPONENTS
    )
    return summary


def aggregate_payloads(
    output_root: str | Path,
    *,
    empymod_background: Mapping[str, Any],
    simpeg_background: Mapping[str, Any],
    simpeg_channel: Mapping[str, Any],
    fenicsx_background: Mapping[str, Any],
    fenicsx_channel: Mapping[str, Any],
    convergence_cases: Mapping[str, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        ("empymod", "background"): empymod_background,
        ("SimPEG", "background"): simpeg_background,
        ("SimPEG", "channel"): simpeg_channel,
        ("FEniCSx", "background"): fenicsx_background,
        ("FEniCSx", "channel"): fenicsx_channel,
    }
    for (method, _case), payload in payloads.items():
        validate_result_payload(method, payload)

    values = {
        key: np.asarray(payload["values"], dtype=float)
        for key, payload in payloads.items()
    }
    simpeg_delta = channel_delta(
        values[("SimPEG", "channel")],
        values[("SimPEG", "background")],
    )
    fenicsx_delta = channel_delta(
        values[("FEniCSx", "channel")],
        values[("FEniCSx", "background")],
    )
    times = np.asarray(MODEL.times, dtype=float)
    locations = np.asarray(MODEL.receiver_locations, dtype=float)

    benchmark_rows: list[dict[str, Any]] = []
    for (method, case), method_values in values.items():
        for receiver_index, location in enumerate(locations):
            for time_index, time_s in enumerate(times):
                for component_index, component in enumerate(COMPONENTS):
                    benchmark_rows.append(
                        {
                            "method": method,
                            "case": case,
                            "receiver_id": f"Rx{receiver_index + 1}",
                            "receiver_x_m": location[0],
                            "receiver_y_m": location[1],
                            "receiver_z_m": location[2],
                            "time_s": time_s,
                            "component": component,
                            "value": method_values[receiver_index, time_index, component_index],
                        }
                    )
    _write_csv(
        output / "benchmark_values.csv",
        (
            "method",
            "case",
            "receiver_id",
            "receiver_x_m",
            "receiver_y_m",
            "receiver_z_m",
            "time_s",
            "component",
            "value",
        ),
        benchmark_rows,
    )

    empymod_values = values[("empymod", "background")]
    background_masks = _component_masks(empymod_values)
    background_rows: list[dict[str, Any]] = []
    for method in ("SimPEG", "FEniCSx"):
        predicted = values[(method, "background")]
        errors = ordinary_relative_error(predicted, empymod_values)
        for receiver_index in range(5):
            for time_index, time_s in enumerate(times):
                for component_index, component in enumerate(COMPONENTS):
                    background_rows.append(
                        {
                            "method": method,
                            "reference": "empymod_background_1d",
                            "receiver_id": f"Rx{receiver_index + 1}",
                            "time_s": time_s,
                            "component": component,
                            "predicted": predicted[receiver_index, time_index, component_index],
                            "reference_value": empymod_values[receiver_index, time_index, component_index],
                            "relative_error": errors[receiver_index, time_index, component_index],
                            "strong_signal": bool(background_masks[component][receiver_index, time_index]),
                        }
                    )
    _write_csv(
        output / "background_errors.csv",
        (
            "method",
            "reference",
            "receiver_id",
            "time_s",
            "component",
            "predicted",
            "reference_value",
            "relative_error",
            "strong_signal",
        ),
        background_rows,
    )

    delta_rows: list[dict[str, Any]] = []
    for method, delta_values in (("SimPEG", simpeg_delta), ("FEniCSx", fenicsx_delta)):
        for receiver_index in range(5):
            for time_index, time_s in enumerate(times):
                for component_index, component in enumerate(COMPONENTS):
                    delta_rows.append(
                        {
                            "method": method,
                            "case": "channel_minus_background",
                            "receiver_id": f"Rx{receiver_index + 1}",
                            "time_s": time_s,
                            "component": component,
                            "value": delta_values[receiver_index, time_index, component_index],
                        }
                    )
    _write_csv(
        output / "channel_delta_values.csv",
        ("method", "case", "receiver_id", "time_s", "component", "value"),
        delta_rows,
    )

    delta_errors = ordinary_relative_error(simpeg_delta, fenicsx_delta)
    delta_masks = _component_masks(fenicsx_delta)
    delta_error_rows: list[dict[str, Any]] = []
    for receiver_index in range(5):
        for time_index, time_s in enumerate(times):
            for component_index, component in enumerate(COMPONENTS):
                delta_error_rows.append(
                    {
                        "method": "SimPEG_delta",
                        "reference": "FEniCSx_delta",
                        "receiver_id": f"Rx{receiver_index + 1}",
                        "time_s": time_s,
                        "component": component,
                        "predicted": simpeg_delta[receiver_index, time_index, component_index],
                        "reference_value": fenicsx_delta[receiver_index, time_index, component_index],
                        "relative_error": delta_errors[receiver_index, time_index, component_index],
                        "strong_signal": bool(delta_masks[component][receiver_index, time_index]),
                    }
                )
    _write_csv(
        output / "channel_delta_errors.csv",
        (
            "method",
            "reference",
            "receiver_id",
            "time_s",
            "component",
            "predicted",
            "reference_value",
            "relative_error",
            "strong_signal",
        ),
        delta_error_rows,
    )

    convergence_summary: dict[str, Any] = {"available": bool(convergence_cases)}
    for name, (coarse, refined) in (convergence_cases or {}).items():
        coarse_values = np.asarray(coarse, dtype=float)
        refined_values = np.asarray(refined, dtype=float)
        convergence_summary[name] = summarize_convergence(
            coarse_values,
            refined_values,
            strong_mask=strong_signal_mask(refined_values, 0.05),
        )

    model_audit = {
        "coordinate_convention": MODEL.coordinate_convention,
        "source_endpoints_m": MODEL.source_endpoints,
        "receiver_locations_m": MODEL.receiver_locations,
        "receiver_provenance": ["explicit_full_domain"] * 5,
        "channel": {
            "center_m": MODEL.channel.center,
            "size_m": MODEL.channel.size,
            "bounds_m": MODEL.channel.bounds,
            "conductivity_s_per_m": MODEL.channel.conductivity,
            "theoretical_volume_m3": MODEL.channel.volume_m3,
        },
        "empymod": {
            "background_only_1d": True,
            "reference_role": "layered_background_only",
        },
    }
    background_targets = {"Ex": 0.10, "dBzdt": 0.20, "Hz": 0.10}
    delta_targets = {component: 0.20 for component in COMPONENTS}
    benchmark_summary = {
        "background": {
            method: _error_summary(
                values[(method, "background")],
                empymod_values,
                targets=background_targets,
            )
            for method in ("SimPEG", "FEniCSx")
        },
        "channel_delta": _error_summary(
            simpeg_delta,
            fenicsx_delta,
            targets=delta_targets,
        ),
        "all_raw_values_retained": True,
        "channel_to_empymod_error_prohibited": True,
    }
    (output / "model_audit.json").write_text(
        json.dumps(model_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "convergence_summary.json").write_text(
        json.dumps(convergence_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        output / "benchmark_results.npz",
        times=times,
        receiver_locations=locations,
        components=np.asarray(COMPONENTS),
        empymod_background=empymod_values,
        simpeg_background=values[("SimPEG", "background")],
        simpeg_channel=values[("SimPEG", "channel")],
        fenicsx_background=values[("FEniCSx", "background")],
        fenicsx_channel=values[("FEniCSx", "channel")],
        simpeg_delta=simpeg_delta,
        fenicsx_delta=fenicsx_delta,
    )
    return benchmark_summary


def validate_result_payload(method: str, payload: Mapping[str, Any]) -> None:
    method_name = str(method).strip()
    times = np.asarray(payload["times"], dtype=float).reshape(-1)
    locations = np.asarray(payload["receiver_locations"], dtype=float)
    components = tuple(str(item) for item in np.asarray(payload["components"]).tolist())
    values = np.asarray(payload["values"], dtype=float)

    if times.shape != MODEL.times.shape or not np.allclose(
        times,
        MODEL.times,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("times do not match the approved 31-sample contract")
    expected_locations = np.asarray(MODEL.receiver_locations, dtype=float)
    if locations.shape != expected_locations.shape or not np.allclose(
        locations,
        expected_locations,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("receiver_locations do not match the approved five-point contract")
    if components != COMPONENTS:
        raise ValueError(f"components must be exactly {COMPONENTS}")
    if values.shape != EXPECTED_SHAPE or values.size != 465:
        raise ValueError(f"values must have shape {EXPECTED_SHAPE} with 465 entries")
    if not np.all(np.isfinite(values)):
        raise ValueError("all 465 solver values must be finite")
    if method_name.lower() == "empymod":
        if "background_only_1d" not in payload or not _scalar_bool(
            payload["background_only_1d"]
        ):
            raise ValueError("empymod payload requires background_only_1d=true")


def empymod_background_payload(*, srcpts: int = 129) -> dict[str, Any]:
    values = run_empymod_reference(MODEL.times, srcpts=srcpts)
    payload = {
        "times": np.asarray(MODEL.times, dtype=float),
        "receiver_locations": np.asarray(MODEL.receiver_locations, dtype=float),
        "components": np.asarray(COMPONENTS),
        "values": np.asarray(values, dtype=float),
        "background_only_1d": np.asarray(True),
        "reference_role": np.asarray("layered_background_only"),
    }
    validate_result_payload("empymod", payload)
    return payload


def save_empymod_background(path: str | Path, *, srcpts: int = 129) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **empymod_background_payload(srcpts=srcpts))
    return output


def simpeg_payload_from_h5(path: str | Path) -> dict[str, Any]:
    payload = {
        "times": np.asarray(MODEL.times, dtype=float),
        "receiver_locations": np.asarray(MODEL.receiver_locations, dtype=float),
        "components": np.asarray(COMPONENTS),
        "values": load_simpeg_values(path, target_times=MODEL.times),
    }
    validate_result_payload("SimPEG", payload)
    return payload


def fenicsx_payload_from_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("FEniCSx predictions_5rx.csv is empty")
    required = {
        "receiver_id",
        "receiver_x_m",
        "receiver_y_m",
        "receiver_z_m",
        "time_obs",
        *COMPONENTS,
        "provenance",
    }
    if not required.issubset(rows[0]):
        raise ValueError("FEniCSx predictions_5rx.csv lacks required columns")
    values = np.full(EXPECTED_SHAPE, np.nan, dtype=float)
    expected_locations = np.asarray(MODEL.receiver_locations, dtype=float)
    for row in rows:
        if row["provenance"] != "explicit_full_domain":
            raise ValueError("every FEniCSx receiver row must be explicit_full_domain")
        receiver_index = int(row["receiver_id"].removeprefix("Rx")) - 1
        if not 0 <= receiver_index < 5:
            raise ValueError(f"invalid FEniCSx receiver_id: {row['receiver_id']}")
        physical_location = np.asarray(
            [
                float(row["receiver_x_m"]),
                float(row["receiver_y_m"]),
                -float(row["receiver_z_m"]),
            ]
        )
        if not np.allclose(
            physical_location,
            expected_locations[receiver_index],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("FEniCSx receiver coordinate adapter does not match contract")
        time_value = float(row["time_obs"])
        matches = np.flatnonzero(
            np.isclose(MODEL.times, time_value, rtol=1.0e-10, atol=1.0e-15)
        )
        if matches.size != 1:
            raise ValueError(f"FEniCSx time is outside the approved grid: {time_value}")
        time_index = int(matches[0])
        if np.all(np.isfinite(values[receiver_index, time_index])):
            raise ValueError("duplicate FEniCSx receiver/time row")
        values[receiver_index, time_index] = [float(row[name]) for name in COMPONENTS]
    payload = {
        "times": np.asarray(MODEL.times, dtype=float),
        "receiver_locations": expected_locations,
        "components": np.asarray(COMPONENTS),
        "values": values,
        "receiver_provenance": np.asarray(["explicit_full_domain"] * 5),
    }
    validate_result_payload("FEniCSx", payload)
    return payload


def _payload_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def aggregate_result_directory(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    empymod = _payload_from_npz(output / "empymod_background.npz")
    simpeg_background = _payload_from_npz(output / "simpeg_background.npz")
    simpeg_channel = _payload_from_npz(output / "simpeg_channel.npz")
    fenicsx_background = fenicsx_payload_from_csv(
        output / "fenicsx_background" / "predictions_5rx.csv"
    )
    fenicsx_channel = fenicsx_payload_from_csv(
        output / "fenicsx_channel" / "predictions_5rx.csv"
    )
    convergence_cases: dict[str, tuple[Any, Any]] = {}
    optional_cases = {
        "simpeg_spatial": (simpeg_channel, output / "simpeg_channel_refined.npz"),
        "simpeg_time": (simpeg_channel, output / "simpeg_channel_time_refined.npz"),
        "fenicsx_spatial": (fenicsx_channel, output / "fenicsx_channel_refined.npz"),
        "fenicsx_time": (fenicsx_channel, output / "fenicsx_channel_time_refined.npz"),
    }
    for name, (base, refined_path) in optional_cases.items():
        if refined_path.is_file():
            refined = _payload_from_npz(refined_path)
            validate_result_payload(name, refined)
            convergence_cases[name] = (base["values"], refined["values"])
    return aggregate_payloads(
        output,
        empymod_background=empymod,
        simpeg_background=simpeg_background,
        simpeg_channel=simpeg_channel,
        fenicsx_background=fenicsx_background,
        fenicsx_channel=fenicsx_channel,
        convergence_cases=convergence_cases,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "COMPONENTS",
    "EXPECTED_SHAPE",
    "aggregate_payloads",
    "aggregate_result_directory",
    "channel_delta",
    "empymod_background_payload",
    "fenicsx_payload_from_csv",
    "save_empymod_background",
    "sha256_file",
    "simpeg_payload_from_h5",
    "ordinary_relative_error",
    "strong_signal_mask",
    "summarize_convergence",
    "validate_result_payload",
]
