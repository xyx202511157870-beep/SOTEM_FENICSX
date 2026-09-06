"""Deterministic case registry for the ROADS layered falsification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates import CandidateConfig, CandidateSpec, freeze_candidate_registry, generate_candidates
from .io import canonical_json, sha256_hex, write_csv, write_json
from .protocol_constants import (
    AIR_RESISTIVITY,
    C_RANGE,
    CANONICAL_COLE_COLE_TAU,
    DISK_RADII,
    M_RANGE,
    N_TIMES,
    PILOT_WAVEFORMS,
    PRESSURE_C_RANGE,
    PRESSURE_M_RANGE,
    PRESSURE_TAU_RANGE,
    PRESSURE_WAVEFORMS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RECEIVER_Z,
    RHO0_RANGE,
    SOURCE_CURRENT,
    SOURCE_Z,
    SPLIT_COUNTS,
    SPLIT_PREFIX,
    SPLIT_SEEDS,
    TAU_RANGE,
    TEST_WAVEFORMS,
    TIME_WINDOW,
    TRAIN_VAL_WAVEFORMS,
    normalize_tilted_normal,
)


CASE_REGISTRY_COLUMNS: tuple[str, ...] = (
    "case_id",
    "split",
    "split_index",
    "split_seed",
    "family",
    "n_earth_layers",
    "polarizable_layer_index",
    "rho_air",
    "rho1",
    "rho2",
    "rho3",
    "thickness1",
    "thickness2",
    "m",
    "tau",
    "c",
    "src_x0",
    "src_y0",
    "src_z0",
    "src_x1",
    "src_y1",
    "src_z1",
    "src_azimuth_deg",
    "src_length",
    "src_current",
    "n_receivers",
    "rx1_x",
    "rx1_y",
    "rx1_z",
    "rx2_x",
    "rx2_y",
    "rx2_z",
    "sensor_frame",
    "sensor_normal_x",
    "sensor_normal_y",
    "sensor_normal_z",
    "waveform_set",
    "disk_radii",
    "t_min",
    "t_max",
    "n_times",
    "case_hash",
)

FAMILY_CYCLES = {
    "pilot_gap": (
        "2L_cover",
        "2L_basement",
        "3L_middle",
        "3L_basement",
        "2L_cover",
        "3L_cover",
        "3L_middle",
        "2L_basement",
    ),
    "train": (
        "2L_cover",
        "2L_basement",
        "3L_cover",
        "3L_middle",
        "3L_basement",
        "2L_cover",
        "2L_basement",
        "3L_middle",
        "3L_basement",
        "3L_cover",
        "2L_cover",
        "3L_middle",
    ),
    "validation": (
        "2L_cover",
        "2L_basement",
        "3L_middle",
        "3L_basement",
        "3L_cover",
        "2L_cover",
    ),
    "independent_test": (
        "2L_cover",
        "2L_basement",
        "3L_cover",
        "3L_middle",
        "3L_basement",
        "2L_cover",
        "3L_middle",
        "2L_basement",
        "3L_basement",
        "3L_cover",
    ),
    "layered_pressure": (
        "2L_cover",
        "2L_basement",
        "3L_middle",
        "3L_basement",
    ),
}


@dataclass(frozen=True)
class LayeredCase:
    """One frozen material/geometry case."""

    case_id: str
    split: str
    split_index: int
    split_seed: int
    family: str
    resistivities: tuple[float, ...]
    depths: tuple[float, ...]
    polarizable_layer_index: int
    m: float
    tau: float
    c: float
    source_start: tuple[float, float, float]
    source_end: tuple[float, float, float]
    receivers: tuple[tuple[float, float, float], ...]
    sensor_frame: str
    sensor_normal: tuple[float, float, float]
    waveform_ids: tuple[str, ...]
    disk_radii: tuple[float, ...] = DISK_RADII

    @property
    def n_earth_layers(self) -> int:
        return len(self.resistivities) - 1

    @property
    def earth_resistivities(self) -> tuple[float, ...]:
        return self.resistivities[1:]

    @property
    def thicknesses(self) -> tuple[float, ...]:
        if len(self.depths) < 2:
            return ()
        return tuple(float(right - left) for left, right in zip(self.depths[:-1], self.depths[1:]))

    @property
    def source_length(self) -> float:
        start = np.asarray(self.source_start[:2], dtype=float)
        end = np.asarray(self.source_end[:2], dtype=float)
        return float(np.linalg.norm(end - start))

    @property
    def source_azimuth_deg(self) -> float:
        start = np.asarray(self.source_start[:2], dtype=float)
        end = np.asarray(self.source_end[:2], dtype=float)
        delta = end - start
        return float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)

    def case_hash(self) -> str:
        return sha256_hex(canonical_json(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "split_index": self.split_index,
            "split_seed": self.split_seed,
            "family": self.family,
            "resistivities": list(self.resistivities),
            "depths": list(self.depths),
            "polarizable_layer_index": self.polarizable_layer_index,
            "m": self.m,
            "tau": self.tau,
            "c": self.c,
            "source_start": list(self.source_start),
            "source_end": list(self.source_end),
            "receivers": [list(item) for item in self.receivers],
            "sensor_frame": self.sensor_frame,
            "sensor_normal": list(self.sensor_normal),
            "waveform_ids": list(self.waveform_ids),
            "disk_radii": list(self.disk_radii),
        }

    def to_registry_row(self) -> dict[str, object]:
        rho = list(self.earth_resistivities) + ["", ""]
        thick = list(self.thicknesses) + ["", ""]
        receivers = list(self.receivers) + [( "", "", "")]
        return {
            "case_id": self.case_id,
            "split": self.split,
            "split_index": self.split_index,
            "split_seed": self.split_seed,
            "family": self.family,
            "n_earth_layers": self.n_earth_layers,
            "polarizable_layer_index": self.polarizable_layer_index,
            "rho_air": f"{AIR_RESISTIVITY:.6e}",
            "rho1": f"{float(rho[0]):.8g}",
            "rho2": "" if rho[1] == "" else f"{float(rho[1]):.8g}",
            "rho3": "" if rho[2] == "" else f"{float(rho[2]):.8g}",
            "thickness1": "" if thick[0] == "" else f"{float(thick[0]):.8g}",
            "thickness2": "" if thick[1] == "" else f"{float(thick[1]):.8g}",
            "m": f"{self.m:.8g}",
            "tau": f"{self.tau:.12e}",
            "c": f"{self.c:.8g}",
            "src_x0": f"{self.source_start[0]:.8g}",
            "src_y0": f"{self.source_start[1]:.8g}",
            "src_z0": f"{self.source_start[2]:.8g}",
            "src_x1": f"{self.source_end[0]:.8g}",
            "src_y1": f"{self.source_end[1]:.8g}",
            "src_z1": f"{self.source_end[2]:.8g}",
            "src_azimuth_deg": f"{self.source_azimuth_deg:.8g}",
            "src_length": f"{self.source_length:.8g}",
            "src_current": f"{SOURCE_CURRENT:.8g}",
            "n_receivers": len(self.receivers),
            "rx1_x": f"{self.receivers[0][0]:.8g}",
            "rx1_y": f"{self.receivers[0][1]:.8g}",
            "rx1_z": f"{self.receivers[0][2]:.8g}",
            "rx2_x": "" if len(self.receivers) < 2 else f"{self.receivers[1][0]:.8g}",
            "rx2_y": "" if len(self.receivers) < 2 else f"{self.receivers[1][1]:.8g}",
            "rx2_z": "" if len(self.receivers) < 2 else f"{self.receivers[1][2]:.8g}",
            "sensor_frame": self.sensor_frame,
            "sensor_normal_x": f"{self.sensor_normal[0]:.8g}",
            "sensor_normal_y": f"{self.sensor_normal[1]:.8g}",
            "sensor_normal_z": f"{self.sensor_normal[2]:.8g}",
            "waveform_set": ";".join(self.waveform_ids),
            "disk_radii": ";".join(f"{value:.1f}" for value in self.disk_radii),
            "t_min": f"{TIME_WINDOW[0]:.6e}",
            "t_max": f"{TIME_WINDOW[1]:.6e}",
            "n_times": N_TIMES,
            "case_hash": self.case_hash(),
        }


def protocol_candidate_config(cole_cole_tau: float = CANONICAL_COLE_COLE_TAU) -> CandidateConfig:
    """Return the frozen 36-per-K template generator."""

    return CandidateConfig(cole_cole_tau=float(cole_cole_tau))


def instantiate_candidates(case: LayeredCase) -> tuple[CandidateSpec, ...]:
    """Instantiate the frozen templates on one case's Cole-Cole tau."""

    return generate_candidates(protocol_candidate_config(case.tau))


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(low), np.log10(high)))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _layer_count(family: str) -> int:
    return 2 if family.startswith("2L") else 3


def _polarizable_index(family: str) -> int:
    if family.endswith("cover"):
        return 1
    if family.endswith("middle"):
        return 2
    if family.endswith("basement"):
        return _layer_count(family)
    raise ValueError(f"unknown family: {family}")


def _draw_resistivities(rng: np.random.Generator, n_earth: int) -> list[float]:
    values = [_log_uniform(rng, RHO0_RANGE[0], RHO0_RANGE[1])]
    for _ in range(n_earth - 1):
        for _attempt in range(64):
            ratio = 10.0 ** rng.uniform(-1.0, 1.0)
            candidate = values[-1] * ratio
            if RHO0_RANGE[0] <= candidate <= RHO0_RANGE[1] and abs(np.log10(ratio)) >= 0.3:
                values.append(float(candidate))
                break
        else:
            values.append(float(np.clip(values[-1] * 3.0, RHO0_RANGE[0], RHO0_RANGE[1])))
    return values


def _draw_thicknesses(rng: np.random.Generator, n_earth: int) -> list[float]:
    if n_earth == 2:
        return [_log_uniform(rng, 30.0, 200.0)]
    return [_log_uniform(rng, 20.0, 100.0), _log_uniform(rng, 40.0, 250.0)]


def _draw_source(
    rng: np.random.Generator,
    *,
    azimuth_deg: tuple[float, float] = (10.0, 80.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    length = _uniform(rng, 200.0, 500.0)
    psi = np.radians(_uniform(rng, azimuth_deg[0], azimuth_deg[1]))
    fraction = _uniform(rng, 0.30, 0.45)
    if rng.uniform() >= 0.5:
        fraction = 1.0 - fraction
    center = np.array([_uniform(rng, -50.0, 50.0), _uniform(rng, -50.0, 50.0)], dtype=float)
    direction = np.array([np.cos(psi), np.sin(psi)], dtype=float)
    start = center - fraction * length * direction
    end = center + (1.0 - fraction) * length * direction
    return (
        (float(start[0]), float(start[1]), SOURCE_Z),
        (float(end[0]), float(end[1]), SOURCE_Z),
    )


def _draw_receivers(
    rng: np.random.Generator,
    source_start: tuple[float, float, float],
    source_end: tuple[float, float, float],
    *,
    offset_range: tuple[float, float] = (400.0, 1000.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = 0.5 * (
        np.asarray(source_start[:2], dtype=float) + np.asarray(source_end[:2], dtype=float)
    )
    azimuth = np.radians(
        np.degrees(
            np.arctan2(source_end[1] - source_start[1], source_end[0] - source_start[0])
        )
    )
    points: list[tuple[float, float, float]] = []
    for _ in range(2):
        for _attempt in range(64):
            radius = _uniform(rng, offset_range[0], offset_range[1])
            phi = _uniform(rng, 0.0, 2.0 * np.pi)
            if abs(np.sin(phi - azimuth)) < 0.25 or abs(np.cos(phi - azimuth)) < 0.25:
                continue
            point = center + radius * np.array([np.cos(phi), np.sin(phi)])
            if points:
                previous = np.asarray(points[0][:2], dtype=float)
                if float(np.linalg.norm(point - previous)) < 150.0:
                    continue
            points.append((float(point[0]), float(point[1]), RECEIVER_Z))
            break
        else:
            fallback = center + np.array([offset_range[0], 0.6 * offset_range[0]])
            points.append((float(fallback[0]), float(fallback[1]), RECEIVER_Z))
    return points[0], points[1]


def _waveforms_for_split(split: str) -> tuple[str, ...]:
    if split in {"pilot_gap", "train", "validation"}:
        return TRAIN_VAL_WAVEFORMS if split != "pilot_gap" else PILOT_WAVEFORMS
    if split == "independent_test":
        return TEST_WAVEFORMS
    if split == "layered_pressure":
        return PRESSURE_WAVEFORMS
    raise ValueError(f"unknown split: {split}")


def _interpolate_train_value(train_values: list[float], rng: np.random.Generator, *, geometric: bool) -> float:
    ordered = sorted(float(value) for value in train_values)
    if len(ordered) < 2:
        raise ValueError("train interpolation requires at least two values")
    index = int(rng.integers(0, len(ordered) - 1))
    left, right = ordered[index], ordered[index + 1]
    if geometric:
        return float(np.sqrt(left * right))
    return float(0.5 * (left + right))


def generate_split(
    split: str,
    *,
    train_cases: tuple[LayeredCase, ...] = (),
) -> tuple[LayeredCase, ...]:
    """Draw one frozen split from its dedicated seed."""

    if split not in SPLIT_COUNTS:
        raise ValueError(f"unknown split: {split}")
    rng = np.random.default_rng(SPLIT_SEEDS[split])
    families = FAMILY_CYCLES[split]
    if len(families) != SPLIT_COUNTS[split]:
        raise ValueError(f"family cycle length does not match {split}")
    cases: list[LayeredCase] = []
    train_rho = [case.earth_resistivities[case.polarizable_layer_index - 1] for case in train_cases]
    train_m = [case.m for case in train_cases]
    train_tau = [case.tau for case in train_cases]
    train_c = [case.c for case in train_cases]
    for index, family in enumerate(families, start=1):
        n_earth = _layer_count(family)
        polarizable = _polarizable_index(family)
        resistivities = _draw_resistivities(rng, n_earth)
        thicknesses = _draw_thicknesses(rng, n_earth)
        if split == "layered_pressure":
            chargeability = _uniform(rng, PRESSURE_M_RANGE[0], PRESSURE_M_RANGE[1])
            tau = _log_uniform(rng, PRESSURE_TAU_RANGE[0], PRESSURE_TAU_RANGE[1])
            cole_c = _uniform(rng, PRESSURE_C_RANGE[0], PRESSURE_C_RANGE[1])
            source = _draw_source(rng)
            receivers = _draw_receivers(rng, source[0], source[1])
            sensor_frame = "geographic"
            normal = (0.0, 0.0, 1.0)
        elif split == "independent_test":
            if not train_cases:
                raise ValueError("independent_test interpolation requires train cases")
            resistivities[polarizable - 1] = _interpolate_train_value(train_rho, rng, geometric=True)
            chargeability = _interpolate_train_value(train_m, rng, geometric=False)
            tau = _interpolate_train_value(train_tau, rng, geometric=True)
            cole_c = _interpolate_train_value(train_c, rng, geometric=False)
            source = _draw_source(rng, azimuth_deg=(100.0, 170.0))
            receivers = _draw_receivers(rng, source[0], source[1], offset_range=(1100.0, 1500.0))
            sensor_frame = "tilted"
            normal = normalize_tilted_normal()
        else:
            chargeability = _uniform(rng, M_RANGE[0], M_RANGE[1])
            tau = _log_uniform(rng, TAU_RANGE[0], TAU_RANGE[1])
            cole_c = _uniform(rng, C_RANGE[0], C_RANGE[1])
            source = _draw_source(rng)
            receivers = _draw_receivers(rng, source[0], source[1])
            sensor_frame = "geographic"
            normal = (0.0, 0.0, 1.0)
        interfaces = np.concatenate(([0.0], np.cumsum(thicknesses)))
        depths = tuple(float(value) for value in interfaces)
        case_id = f"{SPLIT_PREFIX[split]}{index:02d}"
        cases.append(
            LayeredCase(
                case_id=case_id,
                split=split,
                split_index=index,
                split_seed=SPLIT_SEEDS[split],
                family=family,
                resistivities=(AIR_RESISTIVITY, *tuple(float(value) for value in resistivities)),
                depths=depths,
                polarizable_layer_index=polarizable,
                m=float(chargeability),
                tau=float(tau),
                c=float(cole_c),
                source_start=source[0],
                source_end=source[1],
                receivers=receivers,
                sensor_frame=sensor_frame,
                sensor_normal=normal,
                waveform_ids=_waveforms_for_split(split),
            )
        )
    return tuple(cases)


def generate_all_cases() -> tuple[LayeredCase, ...]:
    """Generate the frozen 40-case registry in split order."""

    train = generate_split("train")
    return (
        *generate_split("pilot_gap"),
        *train,
        *generate_split("validation"),
        *generate_split("independent_test", train_cases=train),
        *generate_split("layered_pressure"),
    )


def cases_for_split(split: str, cases: tuple[LayeredCase, ...] | None = None) -> tuple[LayeredCase, ...]:
    pool = generate_all_cases() if cases is None else cases
    return tuple(case for case in pool if case.split == split)


def freeze_case_registry(path) -> tuple[LayeredCase, ...]:
    """Write ``case_registry.csv`` and return the generated cases."""

    cases = generate_all_cases()
    write_csv(path, CASE_REGISTRY_COLUMNS, [case.to_registry_row() for case in cases])
    return cases


def load_case_registry(path) -> tuple[LayeredCase, ...]:
    """Rehydrate cases from the frozen CSV by regenerating and checking hashes."""

    from .io import read_csv

    rows = read_csv(path)
    generated = {case.case_id: case for case in generate_all_cases()}
    loaded: list[LayeredCase] = []
    for row in rows:
        case = generated[str(row["case_id"])]
        if case.case_hash() != str(row["case_hash"]):
            raise ValueError(f"case_hash mismatch for {case.case_id}")
        loaded.append(case)
    return tuple(loaded)


def freeze_protocol_artifacts(output_root: str | Path) -> dict[str, object]:
    """Write immutable Flow-1 registries and return their hashes."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    case_path = root / "case_registry.csv"
    candidate_path = root / "candidate_registry.csv"
    cases = freeze_case_registry(case_path)
    config = protocol_candidate_config()
    candidates = freeze_candidate_registry(config, candidate_path)
    payload = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "n_cases": len(cases),
        "n_candidates": len(candidates),
        "candidates_per_k": config.candidates_per_pole_count,
        "canonical_cole_cole_tau": CANONICAL_COLE_COLE_TAU,
        "case_registry_sha256": sha256_hex(case_path.read_text(encoding="utf-8")),
        "candidate_registry_sha256": sha256_hex(candidate_path.read_text(encoding="utf-8")),
        "candidate_config_hash": config.config_hash(),
        "split_counts": {split: sum(case.split == split for case in cases) for split in SPLIT_COUNTS},
    }
    write_json(root / "registry_manifest.json", payload)
    return payload
