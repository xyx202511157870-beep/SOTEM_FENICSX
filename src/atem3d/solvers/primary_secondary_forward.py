"""Pure primary-secondary forward orchestration core."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
import inspect

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
    ...,
    SecondaryState,
]

SecondaryStateInitializer = Callable[
    [np.ndarray, PronyConductivity, float],
    DCSecondaryInitialization,
]

SecondaryStateLoader = Callable[[SecondaryState, np.ndarray], None]


@dataclass(frozen=True)
class PrimarySecondaryForwardStateResult:
    """Forward rows plus enough pure-array state to resume the sequence."""

    data: np.ndarray
    final_state: SecondaryState
    final_Ep_old: np.ndarray
    previous_time: float
    output_index: int


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
    secondary_state_loader: SecondaryStateLoader | None = None
    secondary_state_stepper: SecondaryStateStepper | None = None
    contrast_atol: float = 0.0
    turnoff_time: float = 0.0
    turnoff_steps: int = 1
    max_internal_dt: float = 0.0
    primary_time_floor: float = 0.0
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
        max_internal_dt = float(self.max_internal_dt)
        if not np.isfinite(max_internal_dt) or max_internal_dt < 0.0:
            raise ValueError("max_internal_dt must be finite and nonnegative")
        object.__setattr__(self, "max_internal_dt", max_internal_dt)
        primary_time_floor = float(self.primary_time_floor)
        if not np.isfinite(primary_time_floor) or primary_time_floor < 0.0:
            raise ValueError("primary_time_floor must be finite and nonnegative")
        object.__setattr__(self, "primary_time_floor", primary_time_floor)

    def forward(self, times: Sequence[float]) -> np.ndarray:
        """Return flattened receiver data for the requested observation times."""

        return self.forward_with_state(times).data

    def forward_with_state(
        self,
        times: Sequence[float],
        *,
        initial_state: SecondaryState | None = None,
        initial_Ep_old=None,
        previous_time: float = 0.0,
        output_index: int = 0,
        max_new_outputs: int = 0,
    ) -> PrimarySecondaryForwardStateResult:
        """Run or resume a primary-secondary sequence.

        ``output_index`` is the number of observation outputs already present in
        the caller's checkpoint. ``max_new_outputs`` limits the number of new
        rows returned by this call; zero means run to the end.
        """

        observation_times = _as_strictly_increasing_times(times)
        output_index = int(output_index)
        if output_index < 0 or output_index > observation_times.size:
            raise ValueError("output_index must be between 0 and the number of observation times")
        max_new_outputs = int(max_new_outputs)
        if max_new_outputs < 0:
            raise ValueError("max_new_outputs must be nonnegative")
        previous_time = float(previous_time)
        if not np.isfinite(previous_time) or previous_time < 0.0:
            raise ValueError("previous_time must be finite and nonnegative")
        output_internal_times = observation_times + float(self.turnoff_time)
        internal_times = _internal_time_grid(
            observation_times,
            turnoff_time=float(self.turnoff_time),
            turnoff_steps=int(self.turnoff_steps),
            max_internal_dt=float(self.max_internal_dt),
        )
        self._record_internal_time_grid_diagnostics(observation_times, internal_times)
        prefetch = getattr(self.primary, "prepare_receiver_reference_cache", None)
        if prefetch is not None:
            prefetch(observation_times, self.receiver_locations)
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
        if initial_state is None:
            state = secondary_state_from_dc_initialization(initialization)
            Ep_old = initialization.Ep0
        else:
            state = initial_state
            if initial_Ep_old is None:
                raise ValueError("initial_Ep_old is required when initial_state is provided")
            Ep_old = np.asarray(initial_Ep_old, dtype=float)
            if Ep_old.shape != state.Es.shape:
                raise ValueError("initial_Ep_old must have the same shape as initial_state.Es")
            if self.secondary_state_loader is not None:
                self.secondary_state_loader(state, Ep_old)
        rows: list[np.ndarray] = []
        new_output_count = 0
        first_primary_time = (
            float(self.primary_time_floor)
            if float(self.primary_time_floor) > 0.0
            else float(observation_times[0])
        )

        for internal_time in internal_times:
            if float(internal_time) <= previous_time + 1.0e-15:
                continue
            dt = float(internal_time - previous_time)
            if dt <= 0.0:
                raise ValueError("internal times must be greater than initial time 0")
            if float(self.turnoff_time) > 0.0 and float(internal_time) <= float(self.turnoff_time):
                primary_time = 0.0
                ramp_factor = max(0.0, 1.0 - float(internal_time) / float(self.turnoff_time))
                Ep_new = ramp_factor * initialization.Ep0
            else:
                primary_time = _primary_time_from_internal_time(
                    float(internal_time),
                    turnoff_time=float(self.turnoff_time),
                    first_primary_time=first_primary_time,
                )
                Ep_new = primary_fem.sample_Ep(primary_time)
            if self.secondary_state_stepper is not None:
                state = _call_secondary_state_stepper(
                    self.secondary_state_stepper,
                    state,
                    Ep_old,
                    Ep_new,
                    self.material,
                    self.sigma_background,
                    dt,
                    primary_time,
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
                rows.append(
                    self._receiver_row(
                        state,
                        Ep_new,
                        internal_time=float(internal_time),
                        primary_time=float(observation_times[output_index]),
                        dt=dt,
                    )
                )
                output_index += 1
                new_output_count += 1
                if max_new_outputs > 0 and new_output_count >= max_new_outputs:
                    Ep_old = Ep_new
                    previous_time = float(internal_time)
                    break
            Ep_old = Ep_new
            previous_time = float(internal_time)

        if max_new_outputs == 0 and output_index != observation_times.size:
            raise RuntimeError("internal time grid did not visit every observation output time")
        data = np.vstack(rows) if rows else np.empty((0, len(self.components)), dtype=float)
        return PrimarySecondaryForwardStateResult(
            data=data,
            final_state=state,
            final_Ep_old=np.asarray(Ep_old, dtype=float).copy(),
            previous_time=float(previous_time),
            output_index=int(output_index),
        )

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
        summary["max_internal_dt_s"] = float(self.max_internal_dt)
        if stepped_internal_times.size:
            summary["first_stepped_internal_time_s"] = float(stepped_internal_times[0])
            summary["last_stepped_internal_time_s"] = float(stepped_internal_times[-1])
        summary["primary_time_origin"] = "after_turnoff_observation_time"
        summary["primary_time_mapping"] = (
            "Ep=(1-t_internal/turnoff_time) Ep_dc for t_internal<=turnoff_time; "
            "otherwise t_primary=max(t_internal-turnoff_time, first_observation_time)"
        )
        summary["first_primary_time_s"] = float(observation_times[0])
        summary["primary_time_floor_s"] = (
            float(self.primary_time_floor)
            if float(self.primary_time_floor) > 0.0
            else float(observation_times[0])
        )
        summary["first_output_primary_time_s"] = float(observation_times[0])
        summary["last_output_primary_time_s"] = float(observation_times[-1])
        self.diagnostics["primary_secondary_internal_time_grid"] = summary

    def _receiver_row(
        self,
        state: SecondaryState,
        Ep_new: np.ndarray,
        internal_time: float,
        primary_time: float,
        dt: float,
    ) -> np.ndarray:
        primary_E = self.primary.get_receiver_E(primary_time, self.receiver_locations)
        primary_dbdt = self.primary.get_receiver_dBdt(primary_time, self.receiver_locations)
        primary_H = (
            self.primary.get_receiver_H(primary_time, self.receiver_locations)
            if _components_include_kind(self.components, "H")
            else np.zeros_like(primary_E, dtype=float)
        )
        primary_row = _flatten_components(primary_E, primary_H, primary_dbdt, self.components)
        if self.secondary_receiver_projector is None:
            self._record_receiver_decomposition(
                internal_time=internal_time,
                primary_time=primary_time,
                dt=dt,
                primary_row=primary_row,
                secondary_row=np.zeros_like(primary_row, dtype=float),
                total_row=primary_row,
            )
            return primary_row
        secondary = np.asarray(
            self.secondary_receiver_projector(state, Ep_new, internal_time, dt, self.components),
            dtype=float,
        )
        if secondary.ndim == 2 and secondary.shape == (
            self.receiver_locations.shape[0],
            len(self.components),
        ):
            secondary = secondary.reshape(-1)
        if secondary.shape != primary_row.shape:
            raise ValueError("secondary_receiver_projector returned the wrong shape")
        total_row = primary_row + secondary
        self._record_receiver_decomposition(
            internal_time=internal_time,
            primary_time=primary_time,
            dt=dt,
            primary_row=primary_row,
            secondary_row=secondary,
            total_row=total_row,
        )
        return total_row

    def _record_receiver_decomposition(
        self,
        *,
        internal_time: float,
        primary_time: float,
        dt: float,
        primary_row: np.ndarray,
        secondary_row: np.ndarray,
        total_row: np.ndarray,
    ) -> None:
        if self.diagnostics is None:
            return
        rows = self.diagnostics.setdefault("receiver_decomposition_rows", [])
        rows.append(
            {
                "time_value": float(internal_time),
                "primary_time": float(primary_time),
                "dt": float(dt),
                "components": list(self.components),
                "primary_row": np.asarray(primary_row, dtype=float).tolist(),
                "secondary_row": np.asarray(secondary_row, dtype=float).tolist(),
                "total_row": np.asarray(total_row, dtype=float).tolist(),
            }
        )


_COMPONENT_KIND = {
    "Ex": ("E", 0),
    "Ey": ("E", 1),
    "Ez": ("E", 2),
    "Hx": ("H", 0),
    "Hy": ("H", 1),
    "Hz": ("H", 2),
    "dBxdt": ("dBdt", 0),
    "dBydt": ("dBdt", 1),
    "dBzdt": ("dBdt", 2),
}


def _call_secondary_state_stepper(
    stepper: SecondaryStateStepper,
    state: SecondaryState,
    Ep_old: np.ndarray,
    Ep_new: np.ndarray,
    material: PronyConductivity,
    sigma_background: float,
    dt: float,
    primary_time: float,
) -> SecondaryState:
    signature = inspect.signature(stepper)
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    )
    if accepts_varargs or len(positional) >= 7:
        return stepper(state, Ep_old, Ep_new, material, sigma_background, dt, primary_time)
    return stepper(state, Ep_old, Ep_new, material, sigma_background, dt)


def _flatten_components(
    receiver_E: np.ndarray,
    receiver_H: np.ndarray,
    receiver_dbdt: np.ndarray,
    components: Sequence[str],
) -> np.ndarray:
    electric = _as_receiver_vectors(receiver_E, "receiver_E")
    magnetic = _as_receiver_vectors(receiver_H, "receiver_H")
    dbdt = _as_receiver_vectors(receiver_dbdt, "receiver_dBdt")
    if magnetic.shape != electric.shape:
        raise ValueError("receiver_H must have the same shape as receiver_E")
    if dbdt.shape != electric.shape:
        raise ValueError("receiver_dBdt must have the same shape as receiver_E")
    columns = []
    for component in components:
        kind, index = _COMPONENT_KIND[component]
        if kind == "E":
            source = electric
        elif kind == "H":
            source = magnetic
        else:
            source = dbdt
        columns.append(source[:, index])
    return np.column_stack(columns).reshape(-1)


def _components_include_kind(components: Sequence[str], kind: str) -> bool:
    return any(_COMPONENT_KIND[component][0] == kind for component in components)


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


def _primary_time_from_internal_time(
    internal_time: float,
    *,
    turnoff_time: float,
    first_primary_time: float,
) -> float:
    primary_time = float(internal_time) - float(turnoff_time)
    return max(float(first_primary_time), primary_time)


def _internal_time_grid(
    observation_times: Sequence[float],
    *,
    turnoff_time: float,
    turnoff_steps: int,
    max_internal_dt: float = 0.0,
) -> np.ndarray:
    grid = build_internal_time_grid_from_turnoff(
        observation_times,
        turnoff_time=turnoff_time,
        turnoff_steps=turnoff_steps,
    )
    max_dt = float(max_internal_dt)
    if max_dt > 0.0 and grid.size > 1:
        expanded = [float(grid[0])]
        for target in grid[1:]:
            previous = expanded[-1]
            interval = float(target) - previous
            if interval > max_dt:
                substeps = int(np.ceil(interval / max_dt))
                expanded.extend(float(value) for value in np.linspace(previous, float(target), substeps + 1)[1:])
            else:
                expanded.append(float(target))
        grid = np.unique(np.asarray(expanded, dtype=float))
    return grid[grid > 0.0]
