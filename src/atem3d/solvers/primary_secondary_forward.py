"""Pure primary-secondary forward orchestration core."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

from atem3d.materials.prony import PronyConductivity
from atem3d.primary.base import PrimaryFieldProvider, as_points
from atem3d.primary.interpolation import PrimaryFEMInterpolator
from atem3d.waveforms import build_internal_time_grid_from_turnoff, summarize_internal_time_grid

from .dc_secondary import DCSecondaryInitialization, initialize_dc_secondary_from_primary
from .tdem_secondary import (
    SecondarySolver,
    SecondaryState,
    secondary_state_from_dc_initialization,
    secondary_step_ip,
    secondary_step_noip,
)


SecondaryReceiverProjector = Callable[
    [SecondaryState, np.ndarray, float, float, Sequence[str]],
    np.ndarray,
]

SecondaryStateStepper = Callable[
    [SecondaryState, np.ndarray, np.ndarray, PronyConductivity, float, float],
    SecondaryState,
]

SecondaryStateInitializer = Callable[
    [np.ndarray, PronyConductivity, float],
    DCSecondaryInitialization,
]


@dataclass(frozen=True)
class PrimarySecondaryForwardOperator:
    """Run a primary-secondary time sequence with injectable FEM pieces.

    This class is intentionally FEM-library agnostic. The DOLFINx layer is
    expected to provide ``secondary_field_solver``, ``secondary_step_solver``,
    and optionally ``secondary_receiver_projector``.
    """

    primary: PrimaryFieldProvider
    fem_points: np.ndarray
    receiver_locations: np.ndarray
    components: Sequence[str]
    material: PronyConductivity
    sigma_background: float
    secondary_field_solver: Callable[[np.ndarray], tuple[np.ndarray | None, np.ndarray]] | None = None
    secondary_step_solver: SecondarySolver | None = None
    secondary_receiver_projector: SecondaryReceiverProjector | None = None
    secondary_state_initializer: SecondaryStateInitializer | None = None
    secondary_state_stepper: SecondaryStateStepper | None = None
    contrast_atol: float = 0.0
    turnoff_time: float = 0.0
    turnoff_steps: int = 1
    diagnostics: MutableMapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fem_points", as_points(self.fem_points, "fem_points"))
        object.__setattr__(
            self,
            "receiver_locations",
            as_points(self.receiver_locations, "receiver_locations"),
        )
        components = tuple(str(component) for component in self.components)
        if not components:
            raise ValueError("components must not be empty")
        unsupported = [component for component in components if component not in _COMPONENT_KIND]
        if unsupported:
            raise ValueError(f"unsupported receiver components: {unsupported}")
        object.__setattr__(self, "components", components)
        sigma_background = float(self.sigma_background)
        if sigma_background <= 0.0:
            raise ValueError("sigma_background must be positive")
        object.__setattr__(self, "sigma_background", sigma_background)
        if self.contrast_atol < 0.0:
            raise ValueError("contrast_atol must be nonnegative")
        turnoff_time = float(self.turnoff_time)
        if not np.isfinite(turnoff_time) or turnoff_time < 0.0:
            raise ValueError("turnoff_time must be finite and nonnegative")
        object.__setattr__(self, "turnoff_time", turnoff_time)
        turnoff_steps = int(self.turnoff_steps)
        if turnoff_steps < 1:
            raise ValueError("turnoff_steps must be positive")
        object.__setattr__(self, "turnoff_steps", turnoff_steps)

    def forward(self, times: Sequence[float]) -> np.ndarray:
        """Return flattened receiver data for the requested observation times."""

        observation_times = _as_strictly_increasing_times(times)
        output_internal_times = observation_times + float(self.turnoff_time)
        internal_times = _internal_time_grid(
            observation_times,
            turnoff_time=float(self.turnoff_time),
            turnoff_steps=int(self.turnoff_steps),
        )
        self._record_internal_time_grid_diagnostics(observation_times, internal_times)
        primary_fem = PrimaryFEMInterpolator(provider=self.primary, points=self.fem_points)
        if self.secondary_state_initializer is None:
            initialization = initialize_dc_secondary_from_primary(
                primary=primary_fem,
                sigma0=self.material.sigma0,
                sigma_background=self.sigma_background,
                material=self.material,
                secondary_field_solver=self.secondary_field_solver,
                contrast_atol=self.contrast_atol,
            )
        else:
            initialization = self.secondary_state_initializer(
                primary_fem.sample_Ep_dc(),
                self.material,
                self.sigma_background,
            )
        state = secondary_state_from_dc_initialization(initialization)
        Ep_old = initialization.Ep0
        previous_time = 0.0
        rows: list[np.ndarray] = []
        output_index = 0

        for internal_time in internal_times:
            dt = float(internal_time - previous_time)
            if dt <= 0.0:
                raise ValueError("internal times must be greater than initial time 0")
            Ep_new = primary_fem.sample_Ep(float(internal_time))
            if self.secondary_state_stepper is not None:
                state = self.secondary_state_stepper(
                    state,
                    Ep_old,
                    Ep_new,
                    self.material,
                    self.sigma_background,
                    dt,
                )
            elif self.material.terms:
                state = secondary_step_ip(
                    state,
                    Ep_old=Ep_old,
                    Ep_new=Ep_new,
                    material=self.material,
                    sigma_background=self.sigma_background,
                    dt=dt,
                    secondary_solver=self.secondary_step_solver,
                    contrast_atol=self.contrast_atol,
                )
            else:
                state = secondary_step_noip(
                    state,
                    Ep_old=Ep_old,
                    Ep_new=Ep_new,
                    sigma=self.material.sigma_inf,
                    sigma_background=self.sigma_background,
                    dt=dt,
                    secondary_solver=self.secondary_step_solver,
                    contrast_atol=self.contrast_atol,
                )
            if output_index < output_internal_times.size and np.isclose(
                internal_time,
                output_internal_times[output_index],
                rtol=0.0,
                atol=1.0e-15,
            ):
                rows.append(self._receiver_row(state, Ep_new, float(internal_time), dt))
                output_index += 1
            Ep_old = Ep_new
            previous_time = float(internal_time)

        if len(rows) != observation_times.size:
            raise RuntimeError("internal time grid did not visit every observation output time")
        return np.vstack(rows)

    def _record_internal_time_grid_diagnostics(
        self,
        observation_times: np.ndarray,
        stepped_internal_times: np.ndarray,
    ) -> None:
        if self.diagnostics is None:
            return
        summary = dict(
            summarize_internal_time_grid(
                observation_times,
                turnoff_time=float(self.turnoff_time),
                turnoff_steps=int(self.turnoff_steps),
            )
        )
        summary["stepped_internal_points"] = int(stepped_internal_times.size)
        if stepped_internal_times.size:
            summary["first_stepped_internal_time_s"] = float(stepped_internal_times[0])
            summary["last_stepped_internal_time_s"] = float(stepped_internal_times[-1])
        self.diagnostics["primary_secondary_internal_time_grid"] = summary

    def _receiver_row(
        self,
        state: SecondaryState,
        Ep_new: np.ndarray,
        time_value: float,
        dt: float,
    ) -> np.ndarray:
        primary_E = self.primary.get_receiver_E(time_value, self.receiver_locations)
        primary_dbdt = self.primary.get_receiver_dBdt(time_value, self.receiver_locations)
        primary_row = _flatten_components(primary_E, primary_dbdt, self.components)
        if self.secondary_receiver_projector is None:
            return primary_row
        secondary = np.asarray(
            self.secondary_receiver_projector(state, Ep_new, time_value, dt, self.components),
            dtype=float,
        )
        if secondary.ndim == 2 and secondary.shape == (
            self.receiver_locations.shape[0],
            len(self.components),
        ):
            secondary = secondary.reshape(-1)
        if secondary.shape != primary_row.shape:
            raise ValueError("secondary_receiver_projector returned the wrong shape")
        return primary_row + secondary


_COMPONENT_KIND = {
    "Ex": ("E", 0),
    "Ey": ("E", 1),
    "Ez": ("E", 2),
    "dBxdt": ("dBdt", 0),
    "dBydt": ("dBdt", 1),
    "dBzdt": ("dBdt", 2),
}


def _flatten_components(
    receiver_E: np.ndarray,
    receiver_dbdt: np.ndarray,
    components: Sequence[str],
) -> np.ndarray:
    electric = _as_receiver_vectors(receiver_E, "receiver_E")
    dbdt = _as_receiver_vectors(receiver_dbdt, "receiver_dBdt")
    if dbdt.shape != electric.shape:
        raise ValueError("receiver_dBdt must have the same shape as receiver_E")
    columns = []
    for component in components:
        kind, index = _COMPONENT_KIND[component]
        source = electric if kind == "E" else dbdt
        columns.append(source[:, index])
    return np.column_stack(columns).reshape(-1)


def _as_receiver_vectors(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_receivers, 3)")
    return array


def _as_strictly_increasing_times(times: Sequence[float]) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return values.copy()


def _internal_time_grid(
    observation_times: Sequence[float],
    *,
    turnoff_time: float,
    turnoff_steps: int,
) -> np.ndarray:
    grid = build_internal_time_grid_from_turnoff(
        observation_times,
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
    )
    return grid[grid > 0.0]
