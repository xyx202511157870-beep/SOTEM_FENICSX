"""Deterministic Debye pole-grid templates for the ROADS MVP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .io import (
    canonical_json,
    format_offsets,
    parse_offsets,
    read_candidate_registry_csv,
    sha256_hex,
    write_candidate_registry_csv,
)


FAMILY_COLE_COLE = "cole_cole_tau_centered"
FAMILY_TIME_WINDOW = "time_window_centered"
_FAMILY_SHORT = {
    FAMILY_COLE_COLE: "cc",
    FAMILY_TIME_WINDOW: "tw",
}
DEFAULT_POLE_COUNTS = (4, 6, 8, 10, 12)
DEFAULT_SPANS = (4.0, 6.0, 8.0)
DEFAULT_SHIFTS = (-0.5, 0.0, 0.5)
DEFAULT_DENSITY_EXPONENTS = (1.0, 1.25)
DEFAULT_TIME_WINDOW = (1.0e-5, 1.0e-2)
MIN_CANDIDATES_PER_K = 20
MAX_CANDIDATES_PER_K = 40
LOG10_ROUNDING_DECIMALS = 12


def _round_log10(values) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), LOG10_ROUNDING_DECIMALS)


def _as_unique_increasing_floats(values, name: str, *, positive: bool) -> tuple[float, ...]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    if array.size > 1 and not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class CandidateConfig:
    """Frozen template-generation settings."""

    cole_cole_tau: float
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW
    pole_counts: tuple[int, ...] = DEFAULT_POLE_COUNTS
    spans: tuple[float, ...] = DEFAULT_SPANS
    shifts: tuple[float, ...] = DEFAULT_SHIFTS
    density_exponents: tuple[float, ...] = DEFAULT_DENSITY_EXPONENTS
    families: tuple[str, ...] = (FAMILY_COLE_COLE, FAMILY_TIME_WINDOW)

    def __post_init__(self) -> None:
        tau = float(self.cole_cole_tau)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("cole_cole_tau must be positive")
        window = tuple(float(value) for value in self.time_window)
        if len(window) != 2 or not np.all(np.isfinite(window)) or not (0.0 < window[0] < window[1]):
            raise ValueError("time_window must satisfy 0 < t_min < t_max")
        poles = tuple(int(value) for value in self.pole_counts)
        if not poles or any(value < 1 for value in poles):
            raise ValueError("pole_counts must be positive integers")
        if len(set(poles)) != len(poles) or any(left >= right for left, right in zip(poles, poles[1:])):
            raise ValueError("pole_counts must be unique and strictly increasing")
        spans = _as_unique_increasing_floats(self.spans, "spans", positive=True)
        shifts = tuple(float(value) for value in self.shifts)
        if not shifts or not np.all(np.isfinite(shifts)):
            raise ValueError("shifts must be finite")
        if len(set(np.round(shifts, LOG10_ROUNDING_DECIMALS))) != len(shifts):
            raise ValueError("shifts must be unique")
        exponents = _as_unique_increasing_floats(self.density_exponents, "density_exponents", positive=True)
        families = tuple(str(value) for value in self.families)
        unknown = [name for name in families if name not in _FAMILY_SHORT]
        if unknown:
            raise ValueError(f"unknown template families: {unknown}")
        if not families or len(set(families)) != len(families):
            raise ValueError("families must be nonempty and unique")
        object.__setattr__(self, "cole_cole_tau", tau)
        object.__setattr__(self, "time_window", (window[0], window[1]))
        object.__setattr__(self, "pole_counts", poles)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "shifts", shifts)
        object.__setattr__(self, "density_exponents", exponents)
        object.__setattr__(self, "families", families)
        count = self.candidates_per_pole_count
        if not (MIN_CANDIDATES_PER_K <= count <= MAX_CANDIDATES_PER_K):
            raise ValueError(
                f"candidates per K must be in [{MIN_CANDIDATES_PER_K}, {MAX_CANDIDATES_PER_K}], got {count}"
            )

    @property
    def candidates_per_pole_count(self) -> int:
        return len(self.families) * len(self.spans) * len(self.shifts) * len(self.density_exponents)

    @property
    def time_window_anchor(self) -> float:
        t_min, t_max = self.time_window
        return float(np.sqrt(t_min * t_max))

    def config_hash(self) -> str:
        return sha256_hex(canonical_json(asdict(self)))

    def family_anchor(self, family: str) -> float:
        if family == FAMILY_COLE_COLE:
            return float(self.cole_cole_tau)
        if family == FAMILY_TIME_WINDOW:
            return float(self.time_window_anchor)
        raise ValueError(f"unknown template family: {family}")


@dataclass(frozen=True)
class CandidateSpec:
    """One deterministic pole-grid template."""

    candidate_id: str
    K: int
    template_family: str
    anchor_log10_tau: float
    offsets_log10_tau: tuple[float, ...]
    span: float
    shift: float
    density_exponent: float
    time_window_anchor: float
    log10_tau: tuple[float, ...]
    candidate_hash: str

    @property
    def tau_grid(self) -> np.ndarray:
        return np.power(10.0, np.asarray(self.log10_tau, dtype=float))

    def to_registry_row(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "K": str(int(self.K)),
            "template_family": self.template_family,
            "offsets_log10_tau": format_offsets(self.offsets_log10_tau),
            "span": f"{float(self.span):.6f}",
            "shift": f"{float(self.shift):.6f}",
            "time_window_anchor": f"{float(self.time_window_anchor):.12e}",
            "candidate_hash": self.candidate_hash,
        }


def template_offsets(K: int, span: float, density_exponent: float) -> np.ndarray:
    """Return strictly monotone log10 offsets centered on the template anchor."""

    poles = int(K)
    width = float(span)
    exponent = float(density_exponent)
    if poles < 1:
        raise ValueError("K must be a positive integer")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("span must be positive")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("density_exponent must be positive")
    if poles == 1:
        return _round_log10([0.0])
    units = np.linspace(-1.0, 1.0, poles)
    offsets = 0.5 * width * np.sign(units) * np.abs(units) ** exponent
    rounded = _round_log10(offsets)
    if poles > 1 and not np.all(np.diff(rounded) > 0.0):
        raise ValueError("template offsets are not strictly increasing")
    return rounded


def _candidate_hash_payload(spec_fields: dict) -> str:
    return sha256_hex(
        canonical_json(
            {
                "K": int(spec_fields["K"]),
                "template_family": spec_fields["template_family"],
                "anchor_log10_tau": spec_fields["anchor_log10_tau"],
                "span": spec_fields["span"],
                "shift": spec_fields["shift"],
                "offsets_log10_tau": list(spec_fields["offsets_log10_tau"]),
                "log10_tau": list(spec_fields["log10_tau"]),
            }
        )
    )


def build_candidate(
    config: CandidateConfig,
    K: int,
    family: str,
    span: float,
    shift: float,
    density_exponent: float,
) -> CandidateSpec:
    """Build one candidate from an enumerated template tuple."""

    poles = int(K)
    if poles < 1:
        raise ValueError("K must be a positive integer")
    if family not in _FAMILY_SHORT:
        raise ValueError(f"unknown template family: {family}")
    offsets = tuple(float(value) for value in template_offsets(poles, span, density_exponent))
    anchor = float(config.family_anchor(family))
    anchor_log10 = float(_round_log10([np.log10(anchor)])[0])
    shift_value = float(shift)
    log10_tau = tuple(float(value) for value in _round_log10(anchor_log10 + shift_value + np.asarray(offsets)))
    if poles > 1 and not np.all(np.diff(log10_tau) > 0.0):
        raise ValueError("candidate poles are not strictly increasing")
    payload = {
        "K": poles,
        "template_family": family,
        "anchor_log10_tau": anchor_log10,
        "span": float(span),
        "shift": shift_value,
        "offsets_log10_tau": offsets,
        "log10_tau": log10_tau,
    }
    candidate_id = (
        f"K{poles:02d}_{_FAMILY_SHORT[family]}_span{float(span):.1f}"
        f"_shift{shift_value:+.1f}_dens{float(density_exponent):.2f}"
    )
    return CandidateSpec(
        candidate_id=candidate_id,
        K=poles,
        template_family=family,
        anchor_log10_tau=anchor_log10,
        offsets_log10_tau=offsets,
        span=float(span),
        shift=shift_value,
        density_exponent=float(density_exponent),
        time_window_anchor=anchor,
        log10_tau=log10_tau,
        candidate_hash=_candidate_hash_payload(payload),
    )


def generate_candidates(config: CandidateConfig) -> tuple[CandidateSpec, ...]:
    """Enumerate the frozen template list for ``config``."""

    candidates: list[CandidateSpec] = []
    for poles in config.pole_counts:
        for family in config.families:
            for span in config.spans:
                for shift in config.shifts:
                    for exponent in config.density_exponents:
                        candidates.append(
                            build_candidate(config, poles, family, span, shift, exponent)
                        )
    ids = [spec.candidate_id for spec in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate identifiers are not unique")
    return tuple(candidates)


def registry_hash(candidates: Sequence[CandidateSpec]) -> str:
    """Hash the ordered list of candidate hashes."""

    return sha256_hex(canonical_json([spec.candidate_hash for spec in candidates]))


def freeze_candidate_registry(config: CandidateConfig, path) -> tuple[CandidateSpec, ...]:
    """Write ``candidate_registry.csv`` and return the generated specs."""

    candidates = generate_candidates(config)
    write_candidate_registry_csv(path, [spec.to_registry_row() for spec in candidates])
    return candidates


def _spec_from_registry_row(row: dict[str, str], *, density_exponent: float) -> CandidateSpec:
    offsets = parse_offsets(row["offsets_log10_tau"])
    poles = int(row["K"])
    if len(offsets) != poles:
        raise ValueError(f"offsets length {len(offsets)} does not match K={poles}")
    family = str(row["template_family"])
    shift = float(row["shift"])
    span = float(row["span"])
    anchor = float(row["time_window_anchor"])
    anchor_log10 = float(_round_log10([np.log10(anchor)])[0])
    log10_tau = tuple(float(value) for value in _round_log10(anchor_log10 + shift + np.asarray(offsets)))
    payload = {
        "K": poles,
        "template_family": family,
        "anchor_log10_tau": anchor_log10,
        "span": span,
        "shift": shift,
        "offsets_log10_tau": offsets,
        "log10_tau": log10_tau,
    }
    recomputed = _candidate_hash_payload(payload)
    stored = str(row["candidate_hash"])
    if recomputed != stored:
        raise ValueError("candidate_hash does not match reconstructed poles")
    return CandidateSpec(
        candidate_id=str(row["candidate_id"]),
        K=poles,
        template_family=family,
        anchor_log10_tau=anchor_log10,
        offsets_log10_tau=offsets,
        span=span,
        shift=shift,
        density_exponent=float(density_exponent),
        time_window_anchor=anchor,
        log10_tau=log10_tau,
        candidate_hash=stored,
    )


def _density_from_config(config: CandidateConfig, row: dict[str, str]) -> float:
    offsets = np.asarray(parse_offsets(row["offsets_log10_tau"]), dtype=float)
    span = float(row["span"])
    for exponent in config.density_exponents:
        expected = template_offsets(int(row["K"]), span, exponent)
        if np.allclose(offsets, expected, rtol=0.0, atol=5.0e-13):
            return float(exponent)
    raise ValueError("could not recover density_exponent from registry offsets")


def load_candidate_registry(path, config: CandidateConfig | None = None) -> tuple[CandidateSpec, ...]:
    """Rehydrate candidates from a frozen registry CSV."""

    rows = read_candidate_registry_csv(path)
    specs: list[CandidateSpec] = []
    for row in rows:
        if config is None:
            exponent = float("nan")
        else:
            exponent = _density_from_config(config, row)
        specs.append(_spec_from_registry_row(row, density_exponent=exponent))
    return tuple(specs)


def verify_candidate_registry(config: CandidateConfig, path) -> bool:
    """Return True when the CSV matches ``generate_candidates(config)`` exactly."""

    generated = generate_candidates(config)
    try:
        loaded = load_candidate_registry(path, config=config)
    except ValueError:
        return False
    if len(generated) != len(loaded):
        return False
    return all(
        left.candidate_hash == right.candidate_hash and left.candidate_id == right.candidate_id
        for left, right in zip(generated, loaded)
    )

