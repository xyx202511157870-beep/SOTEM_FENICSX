"""Empymod-backed primary provider skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import PrimaryFieldProvider, as_points


@dataclass(frozen=True)
class EmpymodPrimaryProvider(PrimaryFieldProvider):
    """Delayed-import skeleton for an empymod primary-field provider."""

    config: dict[str, Any]
    reference_runner: Callable[..., np.ndarray] | None = None
    empymod_kwargs: dict[str, Any] | None = None
    dc_runner: Callable[..., np.ndarray] | None = None
    dc_kwargs: dict[str, Any] | None = None

    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        points = as_points(V, "points")
        return self._receiver_components(t, points, ["Ex", "Ey", "Ez"])

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        points = as_points(V, "points")
        runner = self.dc_runner
        if runner is None:
            from .dc import analytic_halfspace_dc_runner

            runner = analytic_halfspace_dc_runner
        values = np.asarray(
            runner(points, config=self.config, **(self.dc_kwargs or {})),
            dtype=float,
        )
        if values.shape != points.shape:
            raise ValueError("dc_runner returned an unexpected DC field shape")
        return values.copy()

    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return self._receiver_components(t, points, ["Ex", "Ey", "Ez"])

    def get_receiver_H(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return self._receiver_components(t, points, ["Hx", "Hy", "Hz"])

    def get_receiver_dBdt(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return self._receiver_components(t, points, ["dBxdt", "dBydt", "dBzdt"])

    def _receiver_components(
        self,
        t: float,
        receivers: np.ndarray,
        components: list[str],
    ) -> np.ndarray:
        runner = self.reference_runner

        from atem3d.empymod_compare import EmpymodSurvey

        eval_times, weights = _ramp_average_quadrature(float(t), self.config)
        receiver_tuples = [tuple(float(value) for value in row) for row in receivers]
        flat = [
            (location, component)
            for location in receiver_tuples
            for component in components
        ]
        survey = EmpymodSurvey(
            source_start=self._source_start(),
            source_end=self._source_end(),
            receiver_locations=receiver_tuples,
            components=components,
            times=eval_times,
            depths=[float(value) for value in self.config["depths"]],
            resistivities=_resistivity_config(self.config["resistivities"]),
            strength=float(self.config.get("strength", self.config.get("current", 1.0))),
            signal=self.config.get("signal", -1),
            receiver_components=flat,
            coordinate_system=str(self.config.get("coordinate_system", "depth_down")),
        )
        if runner is None:
            from atem3d import empymod_compare

            values = np.asarray(
                empymod_compare.run_empymod_reference(survey, **(self.empymod_kwargs or {})),
                dtype=float,
            )
        else:
            values = np.asarray(runner(survey, **(self.empymod_kwargs or {})), dtype=float)
        expected_shape = (eval_times.size, receivers.shape[0] * len(components))
        if values.shape != expected_shape:
            raise ValueError("reference_runner returned an unexpected receiver table shape")
        averaged = np.asarray(weights, dtype=float) @ values
        return averaged.reshape(receivers.shape[0], len(components))

    def _source_start(self) -> tuple[float, float, float]:
        if "source_start" in self.config:
            return tuple(float(value) for value in self.config["source_start"])
        return tuple(float(value) for value in self.config["source"]["start"])

    def _source_end(self) -> tuple[float, float, float]:
        if "source_end" in self.config:
            return tuple(float(value) for value in self.config["source_end"])
        return tuple(float(value) for value in self.config["source"]["end"])


def _resistivity_config(values):
    if isinstance(values, dict):
        return dict(values)
    return list(values)


def _ramp_average_quadrature(t: float, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ramp_time = float(config.get("ramp_off_time", 0.0) or 0.0)
    if ramp_time <= 0.0:
        return np.asarray([float(t)], dtype=float), np.asarray([1.0], dtype=float)
    points = int(config.get("ramp_average_quadrature_points", 9) or 9)
    points = max(1, points)
    window = str(config.get("time_origin", "after_ramp")).strip().lower()
    if window == "after_ramp":
        a = float(t)
        b = float(t) + ramp_time
    elif window == "ramp_start":
        a = max(np.finfo(float).tiny, float(t) - ramp_time)
        b = max(a, float(t))
    else:
        raise ValueError("time_origin must be 'after_ramp' or 'ramp_start'")
    if b <= a or points == 1:
        return np.asarray([float(t)], dtype=float), np.asarray([1.0], dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(points)
    times = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    normalized_weights = 0.5 * weights
    return times.astype(float), normalized_weights.astype(float)
