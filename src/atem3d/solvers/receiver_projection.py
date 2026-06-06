"""Receiver projection adapters for primary-secondary secondary fields."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from atem3d.primary.base import as_points

from .tdem_secondary import SecondaryState


SecondaryFieldSampler = Callable[
    [SecondaryState, np.ndarray, float, float, np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class SecondaryReceiverProjection:
    """Project secondary receiver contributions into component tables.

    The injected samplers are the FEM-specific hooks. For DOLFINx they can
    evaluate the secondary electric field and secondary ``dB/dt`` at the
    configured receiver locations; this pure adapter handles component ordering.
    """

    receiver_locations: np.ndarray
    electric_sampler: SecondaryFieldSampler
    dbdt_sampler: SecondaryFieldSampler

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receiver_locations",
            as_points(self.receiver_locations, "receiver_locations"),
        )

    def __call__(
        self,
        state: SecondaryState,
        Ep_new,
        time_value: float,
        dt: float,
        components: Sequence[str],
    ) -> np.ndarray:
        electric = _as_receiver_vectors(
            self.electric_sampler(
                state,
                np.asarray(Ep_new, dtype=float),
                float(time_value),
                float(dt),
                self.receiver_locations.copy(),
            ),
            "secondary electric sampler output",
            self.receiver_locations.shape[0],
        )
        dbdt = _as_receiver_vectors(
            self.dbdt_sampler(
                state,
                np.asarray(Ep_new, dtype=float),
                float(time_value),
                float(dt),
                self.receiver_locations.copy(),
            ),
            "secondary dBdt sampler output",
            self.receiver_locations.shape[0],
        )
        columns = []
        for component in components:
            if component in _ELECTRIC_COMPONENTS:
                columns.append(electric[:, _ELECTRIC_COMPONENTS[component]])
            elif component in _DBDT_COMPONENTS:
                columns.append(dbdt[:, _DBDT_COMPONENTS[component]])
            else:
                raise ValueError(f"unsupported receiver component: {component}")
        return np.column_stack(columns)


_ELECTRIC_COMPONENTS = {"Ex": 0, "Ey": 1, "Ez": 2}
_DBDT_COMPONENTS = {"dBxdt": 0, "dBydt": 1, "dBzdt": 2}


def _as_receiver_vectors(values, name: str, receiver_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (receiver_count, 3):
        raise ValueError(f"{name} must have shape (n_receivers, 3)")
    return array
