"""Frozen protocol constants for the ROADS-Debye-MVP layered falsification.

These numbers are part of the scientific contract. Do not change them after
Flow 1, and never retune them after seeing receiver results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROTOCOL_NAME = "ROADS-Debye-MVP-layered-v1"
PROTOCOL_VERSION = "1.0.0"

TIME_WINDOW = (1.0e-5, 1.0e-2)
N_TIMES = 31
TIMES = np.logspace(np.log10(TIME_WINDOW[0]), np.log10(TIME_WINDOW[1]), N_TIMES)

CHANNELS = ("Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt")
CHANNEL_UNITS = {
    "Hx": "A/m",
    "Hy": "A/m",
    "Hz": "A/m",
    "dBxdt": "T/s",
    "dBydt": "T/s",
    "dBzdt": "T/s",
}

K_PILOT = (4, 6, 8, 10, 12)
K_EXTENSION = (14, 16)
K_PRACTICAL = (6, 8, 10, 12)

PILOT_SEED = 202609111
TRAIN_SEED = 202609112
VALID_SEED = 202609113
TEST_SEED = 202609114
PRESSURE_SEED = 202609115
BOOTSTRAP_SEED = 202609116

SPLIT_COUNTS = {
    "pilot_gap": 8,
    "train": 12,
    "validation": 6,
    "independent_test": 10,
    "layered_pressure": 4,
}
SPLIT_SEEDS = {
    "pilot_gap": PILOT_SEED,
    "train": TRAIN_SEED,
    "validation": VALID_SEED,
    "independent_test": TEST_SEED,
    "layered_pressure": PRESSURE_SEED,
}
SPLIT_PREFIX = {
    "pilot_gap": "PG",
    "train": "TR",
    "validation": "VA",
    "independent_test": "TE",
    "layered_pressure": "LP",
}

RHO0_RANGE = (20.0, 500.0)
M_RANGE = (0.05, 0.40)
TAU_RANGE = (1.0e-4, 1.0e-1)
C_RANGE = (0.35, 0.90)
PRESSURE_M_RANGE = (0.45, 0.55)
PRESSURE_TAU_RANGE = (0.3, 3.0)
PRESSURE_C_RANGE = (0.20, 0.30)

AIR_RESISTIVITY = 1.0e8
SOURCE_CURRENT = 1.0
SOURCE_Z = 0.05
RECEIVER_Z = 0.0
DISK_RADII = (1.0, 4.0)
TILTED_NORMAL = (0.35, -0.20, 0.915)

CANONICAL_COLE_COLE_TAU = float(np.sqrt(TAU_RANGE[0] * TAU_RANGE[1]))
SPECTRAL_FREQUENCIES = np.logspace(-2.0, 4.0, 61)
SPECTRAL_F_LO = 1.0 / (2.0 * np.pi * TIME_WINDOW[1])
SPECTRAL_F_HI = 1.0 / (2.0 * np.pi * TIME_WINDOW[0])

EMPYMOD_SRCPTS = 9
EMPYMOD_RECPTS = 1
WAVEFORM_QUADRATURE_ORDER = 8
COORDINATE_SYSTEM = "depth_down"

TRAIN_ALLOWED_SPLITS = frozenset({"pilot_gap", "train", "validation"})
TEST_ONLY_SPLITS = frozenset({"independent_test", "layered_pressure"})

FINAL_STATUSES = (
    "STOP_LAYERED_NO_ACTIONABLE_GAP",
    "STOP_LAYERED_SELECTOR_FAILED",
    "3D_AUTHORIZED_PENDING_PREFLIGHT",
    "BLOCKED_BY_SOFTWARE_OR_RESOURCES",
)

FORBIDDEN_STATUSES = (
    "EVIDENCE_SUPPORTS_Q1_SUBMISSION",
    "EVIDENCE_SUPPORTS_SPECIALIST_JOURNAL",
    "STOP_3D_TRANSFER_FAILED",
    "Q1_READY",
    "GUARANTEED_Q1",
)


@dataclass(frozen=True)
class WaveformSpec:
    """Frozen transmitter waveform."""

    waveform_id: str
    kind: str
    duration_s: float | None
    times_s: tuple[float, ...] | None
    current_scales: tuple[float, ...] | None
    train_val_allowed: bool
    independent_test_only: bool


WAVEFORMS = (
    WaveformSpec("W0", "ideal_step_off", None, None, None, True, False),
    WaveformSpec("W1", "linear_ramp", 5.0e-6, None, None, True, False),
    WaveformSpec("W2", "linear_ramp", 20.0e-6, None, None, True, False),
    WaveformSpec(
        "W3",
        "tabulated",
        None,
        (-40.0e-6, -25.0e-6, -10.0e-6, 0.0),
        (1.0, 0.80, 0.25, 0.0),
        False,
        True,
    ),
    WaveformSpec("W4", "linear_ramp", 50.0e-6, None, None, False, False),
)
WAVEFORM_BY_ID = {item.waveform_id: item for item in WAVEFORMS}

PILOT_WAVEFORMS = ("W0", "W1", "W2")
TRAIN_VAL_WAVEFORMS = ("W0", "W1", "W2")
TEST_WAVEFORMS = ("W0", "W1", "W2", "W3")
PRESSURE_WAVEFORMS = ("W0", "W4")
PILOT_RECEIVERS = ("point", "disk_1.0", "disk_4.0")


def observation_times() -> np.ndarray:
    """Return the frozen 31-sample log time grid."""

    return TIMES.copy()


def spectral_weights(variant: str, frequencies=None) -> np.ndarray:
    """Return S0 uniform-log or S1 time-window-matched spectral weights."""

    freqs = SPECTRAL_FREQUENCIES if frequencies is None else np.asarray(frequencies, dtype=float)
    if variant in {"S0", "uniform_log_frequency"}:
        return np.ones(freqs.size, dtype=float)
    if variant not in {"S1", "time_window_matched_frequency_weight"}:
        raise ValueError(f"unknown spectral variant: {variant}")
    weights = np.ones(freqs.size, dtype=float)
    low = freqs < SPECTRAL_F_LO
    high = freqs > SPECTRAL_F_HI
    weights[low] = (freqs[low] / SPECTRAL_F_LO) ** 2
    weights[high] = (SPECTRAL_F_HI / freqs[high]) ** 2
    return weights


def normalize_tilted_normal() -> tuple[float, float, float]:
    vector = np.asarray(TILTED_NORMAL, dtype=float)
    vector = vector / float(np.linalg.norm(vector))
    return tuple(float(value) for value in vector)
