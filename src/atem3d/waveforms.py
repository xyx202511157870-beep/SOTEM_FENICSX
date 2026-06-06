"""Waveform interfaces and full turn-off time grids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Waveform(ABC):
    """Current waveform used by grounded-wire source time integration."""

    t_off: float
    min_steps_during_turnoff: int

    @abstractmethod
    def current(self, t: float) -> float:
        """Return source current at internal time ``t``."""

    def interval_average_didt(self, t0: float, t1: float) -> float:
        """Return interval-average dI/dt over ``[t0, t1]``."""

        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        return (self.current(t1) - self.current(t0)) / (t1 - t0)

    def required_internal_times(self, observation_times) -> np.ndarray:
        """Return internal times containing ramp history and observation times."""

        return build_internal_time_grid(observation_times, self)


@dataclass(frozen=True)
class StepOffWaveform(Waveform):
    """Ideal step-off current."""

    current_initial: float
    t_off: float = 0.0
    current_final: float = 0.0
    min_steps_during_turnoff: int = 1

    def __post_init__(self) -> None:
        _validate_turnoff(self.t_off, self.min_steps_during_turnoff)

    def current(self, t: float) -> float:
        return float(self.current_initial if float(t) <= self.t_off else self.current_final)

    def interval_average_didt(self, t0: float, t1: float) -> float:
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        if t0 < self.t_off <= t1:
            return (float(self.current_final) - float(self.current_initial)) / (t1 - t0)
        return 0.0


@dataclass(frozen=True)
class LinearRampOffWaveform(Waveform):
    """Linear ramp-off from initial to final current over ``0 <= t <= t_off``."""

    current_initial: float
    current_final: float
    t_off: float
    min_steps_during_turnoff: int = 10

    def __post_init__(self) -> None:
        _validate_turnoff(self.t_off, self.min_steps_during_turnoff)

    def current(self, t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return float(self.current_initial)
        if t >= self.t_off:
            return float(self.current_final)
        fraction = t / float(self.t_off)
        return float((1.0 - fraction) * self.current_initial + fraction * self.current_final)


@dataclass(frozen=True)
class TabulatedWaveform(Waveform):
    """Piecewise-linear waveform from tabulated ``time,current`` samples."""

    times: np.ndarray
    currents: np.ndarray
    min_steps_during_turnoff: int = 10

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        currents = np.asarray(self.currents, dtype=float)
        if times.ndim != 1 or currents.ndim != 1 or times.shape != currents.shape:
            raise ValueError("times and currents must be one-dimensional arrays with the same shape")
        if times.size < 2:
            raise ValueError("at least two waveform samples are required")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        _validate_turnoff(float(times[-1]), self.min_steps_during_turnoff)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "currents", currents)
        object.__setattr__(self, "t_off", float(times[-1]))

    @classmethod
    def from_csv(cls, path: str | Path, *, min_steps_during_turnoff: int = 10) -> "TabulatedWaveform":
        table = np.genfromtxt(path, delimiter=",", names=True)
        if "time" not in table.dtype.names or "current" not in table.dtype.names:
            raise ValueError("waveform CSV must contain time,current columns")
        return cls(table["time"], table["current"], min_steps_during_turnoff=min_steps_during_turnoff)

    def current(self, t: float) -> float:
        return float(np.interp(float(t), self.times, self.currents, left=self.currents[0], right=self.currents[-1]))


def build_internal_time_grid(observation_times, waveform: Waveform) -> np.ndarray:
    """Build an internal grid from turn-off start to ``t_off + t_obs`` samples."""

    return build_internal_time_grid_from_turnoff(
        observation_times,
        turnoff_time=float(waveform.t_off),
        turnoff_steps=int(waveform.min_steps_during_turnoff),
    )


def build_internal_time_grid_from_turnoff(
    observation_times,
    *,
    turnoff_time: float,
    turnoff_steps: int,
) -> np.ndarray:
    """Build an internal grid from turn-off start to ``turnoff_time + t_obs``."""

    observation_times = np.asarray(observation_times, dtype=float)
    if observation_times.ndim != 1 or observation_times.size == 0:
        raise ValueError("observation_times must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(observation_times)) or np.any(observation_times <= 0.0):
        raise ValueError("observation_times must be finite and positive")
    if np.any(np.diff(observation_times) <= 0.0):
        raise ValueError("observation_times must be strictly increasing")

    turnoff_time = float(turnoff_time)
    if not np.isfinite(turnoff_time) or turnoff_time < 0.0:
        raise ValueError("turnoff_time must be finite and nonnegative")
    turnoff_steps = int(turnoff_steps)
    if turnoff_steps < 1:
        raise ValueError("turnoff_steps must be positive")

    ramp = np.linspace(0.0, turnoff_time, turnoff_steps + 1)
    output_times = turnoff_time + observation_times
    return np.unique(np.r_[ramp, output_times])


def _validate_turnoff(t_off: float, min_steps: int) -> None:
    if not np.isfinite(float(t_off)) or float(t_off) < 0.0:
        raise ValueError("t_off must be finite and nonnegative")
    if int(min_steps) < 1:
        raise ValueError("min_steps_during_turnoff must be positive")
