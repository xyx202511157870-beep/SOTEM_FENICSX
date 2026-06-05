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

    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        points = as_points(V, "points")
        return self._receiver_components(t, points, ["Ex", "Ey", "Ez"])

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        as_points(V, "points")
        raise NotImplementedError("EmpymodPrimaryProvider DC sampling is not implemented yet")

    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return self._receiver_components(t, points, ["Ex", "Ey", "Ez"])

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
        if runner is None:
            raise NotImplementedError("EmpymodPrimaryProvider receiver sampling is not implemented yet")

        from atem3d.empymod_compare import EmpymodSurvey

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
            times=np.array([float(t)], dtype=float),
            depths=[float(value) for value in self.config["depths"]],
            resistivities=list(self.config["resistivities"]),
            strength=float(self.config.get("strength", self.config.get("current", 1.0))),
            signal=self.config.get("signal", -1),
            receiver_components=flat,
            coordinate_system=str(self.config.get("coordinate_system", "depth_down")),
        )
        values = np.asarray(runner(survey, **(self.empymod_kwargs or {})), dtype=float)
        if values.shape != (1, receivers.shape[0] * len(components)):
            raise ValueError("reference_runner returned an unexpected receiver table shape")
        return values.reshape(receivers.shape[0], len(components))

    def _source_start(self) -> tuple[float, float, float]:
        if "source_start" in self.config:
            return tuple(float(value) for value in self.config["source_start"])
        return tuple(float(value) for value in self.config["source"]["start"])

    def _source_end(self) -> tuple[float, float, float]:
        if "source_end" in self.config:
            return tuple(float(value) for value in self.config["source_end"])
        return tuple(float(value) for value in self.config["source"]["end"])
