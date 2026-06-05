"""Primary-secondary TDEM time-step kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from atem3d.materials.prony import PronyConductivity


SecondarySolver = Callable[[np.ndarray, float, float], np.ndarray]


@dataclass(frozen=True)
class SecondaryState:
    """Secondary-field state at one time level."""

    Es: np.ndarray
    deltaJ: np.ndarray
    chi: list[np.ndarray]

    def __post_init__(self) -> None:
        Es = _as_vector_field(self.Es, "Es")
        deltaJ = _as_vector_field(self.deltaJ, "deltaJ")
        if deltaJ.shape != Es.shape:
            raise ValueError("deltaJ must have the same shape as Es")
        chi = [np.asarray(item, dtype=float) for item in self.chi]
        for item in chi:
            if item.shape != Es.shape:
                raise ValueError("each chi entry must have the same shape as Es")
        object.__setattr__(self, "Es", Es)
        object.__setattr__(self, "deltaJ", deltaJ)
        object.__setattr__(self, "chi", chi)


def secondary_step_noip(
    state: SecondaryState,
    *,
    Ep_old,
    Ep_new,
    sigma: float,
    sigma_background: float,
    dt: float,
    secondary_solver: SecondarySolver | None = None,
    contrast_atol: float = 0.0,
) -> SecondaryState:
    """Advance the no-IP primary-secondary equation by one BE step."""

    Ep_old_array, Ep_new_array = _validate_step_inputs(state, Ep_old, Ep_new, dt)
    sigma = _positive_scalar(sigma, "sigma")
    sigma_background = _positive_scalar(sigma_background, "sigma_background")
    contrast = sigma - sigma_background
    if abs(contrast) <= contrast_atol and _allclose_zero(state.Es):
        zero = np.zeros_like(state.Es)
        return SecondaryState(Es=zero, deltaJ=zero, chi=[])

    if secondary_solver is None:
        raise ValueError("secondary_solver is required for nonzero secondary response")
    c_new = contrast * Ep_new_array
    rhs = (state.deltaJ - c_new) / dt
    Es_new = _as_vector_field(secondary_solver(rhs, sigma, dt), "Es_new")
    if Es_new.shape != state.Es.shape:
        raise ValueError("secondary_solver returned an Es field with the wrong shape")
    deltaJ_new = sigma * (Ep_new_array + Es_new) - sigma_background * Ep_new_array
    return SecondaryState(Es=Es_new, deltaJ=deltaJ_new, chi=[])


def secondary_step_ip(
    state: SecondaryState,
    *,
    Ep_old,
    Ep_new,
    material: PronyConductivity,
    sigma_background: float,
    dt: float,
    secondary_solver: SecondarySolver | None = None,
    contrast_atol: float = 0.0,
) -> SecondaryState:
    """Advance the IP primary-secondary equation by one BE step."""

    _Ep_old_array, Ep_new_array = _validate_step_inputs(state, Ep_old, Ep_new, dt)
    sigma_background = _positive_scalar(sigma_background, "sigma_background")
    if len(state.chi) != len(material.terms):
        raise ValueError("state.chi must contain one memory field per Debye term")

    sigma_eff = material.sigma_eff(dt)
    contrast_is_zero = (
        abs(material.sigma_inf - sigma_background) <= contrast_atol
        and all(term.delta_sigma <= contrast_atol for term in material.terms)
    )
    if contrast_is_zero and _allclose_zero(state.Es):
        zero = np.zeros_like(state.Es)
        chi_new = material.update_memory(state.chi, Ep_new_array, dt)
        return SecondaryState(Es=zero, deltaJ=zero, chi=chi_new)

    if secondary_solver is None:
        raise ValueError("secondary_solver is required for nonzero secondary response")

    alpha = material.alpha(dt)
    c_new = (sigma_eff - sigma_background) * Ep_new_array
    for term, memory, alpha_i in zip(material.terms, state.chi, alpha):
        c_new = c_new - term.delta_sigma * alpha_i * memory
    rhs = (state.deltaJ - c_new) / dt
    Es_new = _as_vector_field(secondary_solver(rhs, sigma_eff, dt), "Es_new")
    if Es_new.shape != state.Es.shape:
        raise ValueError("secondary_solver returned an Es field with the wrong shape")
    Etotal_new = Ep_new_array + Es_new
    chi_new = material.update_memory(state.chi, Etotal_new, dt)
    deltaJ_new = material.current_density(Etotal_new, chi_new) - sigma_background * Ep_new_array
    return SecondaryState(Es=Es_new, deltaJ=deltaJ_new, chi=chi_new)


def _validate_step_inputs(
    state: SecondaryState,
    Ep_old,
    Ep_new,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    Ep_old_array = _as_vector_field(Ep_old, "Ep_old")
    Ep_new_array = _as_vector_field(Ep_new, "Ep_new")
    if Ep_old_array.shape != state.Es.shape:
        raise ValueError("Ep_old must have the same shape as Es")
    if Ep_new_array.shape != state.Es.shape:
        raise ValueError("Ep_new must have the same shape as Es")
    return Ep_old_array, Ep_new_array


def _as_vector_field(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return array


def _positive_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _allclose_zero(values: np.ndarray) -> bool:
    return bool(np.allclose(values, 0.0, rtol=0.0, atol=0.0))
