"""Contracts and adapters for the independent COMSOL thin-channel model."""

from __future__ import annotations

import csv
from dataclasses import replace
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .seepage_channel_model import ChannelBox, model_for_variant
from .seepage_verification import canonical_model_contract, model_fingerprint


COMSOL_CASES = ("background", "zero_contrast", "channel")
COMPONENT_EXPRESSIONS = {
    "mef.Ex": "Ex",
    "d(mef.Bz;t)": "dBzdt",
    "d(mef.Bz,t)": "dBzdt",
    "mef.Bz/mu0_const": "Hz",
}


def comsol_case_contract(case: str) -> dict[str, Any]:
    """Return the complete expected COMSOL contract for one independent run."""

    name = str(case)
    if name not in COMSOL_CASES:
        raise ValueError(f"unknown COMSOL seepage case: {name}")
    base = model_for_variant("thin_60x1x1")
    conductivity = 1.0 if name == "channel" else base.background_conductivity
    controlled = replace(
        base,
        channel=ChannelBox(
            center=base.channel.center,
            size=base.channel.size,
            conductivity=conductivity,
        ),
    )
    contract = canonical_model_contract(controlled)
    contract["base_model_fingerprint"] = model_fingerprint(base)
    contract["case_model_fingerprint"] = model_fingerprint(controlled)
    contract["case"] = name
    contract["background_conductivity_s_per_m"] = base.background_conductivity
    contract["channel"]["enabled"] = name != "background"
    return contract


def validate_distinct_model_paths(
    source_model: str | Path, output_model: str | Path
) -> None:
    """Fail before a batch job can overwrite the user's source MPH."""

    source = os.path.normcase(str(Path(source_model).resolve()))
    output = os.path.normcase(str(Path(output_model).resolve()))
    if source == output:
        raise ValueError("COMSOL output must not overwrite the source MPH")


def _header_component_and_time(header: str) -> tuple[str, float] | None:
    match = re.search(r"@\s*t=([0-9eE+.-]+)\s*$", header)
    if match is None:
        return None
    expression = header[: match.start()].strip()
    for prefix, component in COMPONENT_EXPRESSIONS.items():
        if expression.startswith(prefix):
            return component, float(match.group(1))
    return None


def read_comsol_wide_export(
    path: str | Path,
    *,
    time_rtol: float = 5.0e-5,
) -> dict[str, Any]:
    """Normalize COMSOL's point-evaluation wide CSV to (5, 31, 3)."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("% x,")),
        None,
    )
    if header_index is None:
        raise ValueError(f"{source}: missing COMSOL point-export header")
    header = next(csv.reader([lines[header_index][2:]]))
    data_rows = [
        next(csv.reader([line]))
        for line in lines[header_index + 1 :]
        if line.strip() and not line.startswith("%")
    ]
    if len(data_rows) != 5:
        raise ValueError(f"{source}: expected five receiver rows")

    available: dict[str, list[tuple[float, int]]] = {
        component: [] for component in ("Ex", "dBzdt", "Hz")
    }
    for column, name in enumerate(header):
        parsed = _header_component_and_time(name)
        if parsed is not None:
            component, time = parsed
            available[component].append((time, column))

    model = model_for_variant("thin_60x1x1")
    target_times = np.asarray(model.times, dtype=float)
    columns = np.empty((target_times.size, 3), dtype=int)
    for component_index, component in enumerate(("Ex", "dBzdt", "Hz")):
        if not available[component]:
            raise ValueError(f"{source}: missing COMSOL component {component}")
        times = np.asarray([item[0] for item in available[component]], dtype=float)
        indices = np.asarray([item[1] for item in available[component]], dtype=int)
        for time_index, target in enumerate(target_times):
            nearest = int(np.argmin(np.abs(times - target)))
            if not np.isclose(times[nearest], target, rtol=time_rtol, atol=0.0):
                raise ValueError(f"{source}: missing COMSOL output time {target:.17g}")
            columns[time_index, component_index] = indices[nearest]

    values = np.full((5, 31, 3), np.nan, dtype=float)
    expected_locations = np.asarray(model.receiver_locations, dtype=float)
    assigned: set[int] = set()
    for row in data_rows:
        numeric = np.asarray(row, dtype=float)
        location = numeric[:3]
        distances = np.linalg.norm(expected_locations - location[None, :], axis=1)
        receiver = int(np.argmin(distances))
        if distances[receiver] > 1.0e-9 or receiver in assigned:
            raise ValueError(f"{source}: receiver locations do not match the model contract")
        assigned.add(receiver)
        for time_index in range(31):
            values[receiver, time_index] = numeric[columns[time_index]]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{source}: COMSOL response contains NaN or infinite values")
    return {
        "times": target_times,
        "receiver_locations": expected_locations,
        "components": np.asarray(["Ex", "dBzdt", "Hz"]),
        "values": values,
        "model_fingerprint": np.asarray(model_fingerprint(model)),
    }


__all__ = [
    "COMSOL_CASES",
    "comsol_case_contract",
    "read_comsol_wide_export",
    "validate_distinct_model_paths",
]
