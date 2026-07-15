"""Validation and result I/O for the seepage-channel three-solver benchmark."""

from __future__ import annotations

import hashlib
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "COMPONENTS",
    "EXPECTED_SHAPE",
    "empymod_background_payload",
    "save_empymod_background",
    "sha256_file",
    "simpeg_payload_from_h5",
    "validate_result_payload",
]
