"""Reusable forward-operator facade for inversion-ready workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForwardRequest:
    """Inputs passed to a concrete forward runner."""

    model: Any
    survey: Any
    waveform: Any
    times: np.ndarray


ForwardRunner = Callable[[ForwardRequest], np.ndarray]


@dataclass(frozen=True)
class ForwardOperator:
    """Small inversion-facing forward API with an injected solver backend."""

    runner: ForwardRunner

    def forward(self, model: Any, survey: Any, waveform: Any, times: Sequence[float]) -> np.ndarray:
        """Return predicted data for ``model``, ``survey``, ``waveform``, and ``times``."""

        time_array = _as_strictly_increasing_times(times)
        predicted_data = np.asarray(
            self.runner(
                ForwardRequest(
                    model=model,
                    survey=survey,
                    waveform=waveform,
                    times=time_array,
                )
            ),
            dtype=float,
        )
        if predicted_data.ndim != 2:
            raise ValueError("predicted_data must be a 2D array")
        if predicted_data.shape[0] != time_array.size:
            raise ValueError("predicted_data first dimension must match times")
        return predicted_data


def _as_strictly_increasing_times(times: Sequence[float]) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return values.copy()
