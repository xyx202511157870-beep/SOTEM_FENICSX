"""DC secondary initialization helpers for primary-secondary solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from atem3d.materials.prony import PronyConductivity


SecondaryFieldSolver = Callable[[np.ndarray], tuple[np.ndarray | None, np.ndarray]]


@dataclass(frozen=True)
class DCSecondaryInitialization:
    """Initial state for primary-secondary time stepping."""

    Ep0: np.ndarray
    Es0: np.ndarray
    Etotal0: np.ndarray
    chi0: list[np.ndarray]
    deltaJ0: np.ndarray
    phi_s: np.ndarray | None
    contrast_is_zero: bool


def initialize_dc_secondary(
    *,
    Ep0,
    sigma0: float,
    sigma_background: float,
    material: PronyConductivity,
    secondary_field_solver: SecondaryFieldSolver | None = None,
    contrast_atol: float = 0.0,
) -> DCSecondaryInitialization:
    """Initialize DC secondary fields and Debye memory states.

    The continuous scalar problem is
    ``int sigma0 grad(phi_s).grad(q) dx =
    int (sigma0 - sigma_b) Ep0.grad(q) dx``.  This pure core receives an
    already-discretized contrast current density and delegates the actual
    potential/field solve to ``secondary_field_solver``.
    """

    Ep0_array = _as_vector_field(Ep0, "Ep0")
    sigma0 = float(sigma0)
    sigma_background = float(sigma_background)
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    if sigma_background <= 0.0:
        raise ValueError("sigma_background must be positive")
    if contrast_atol < 0.0:
        raise ValueError("contrast_atol must be nonnegative")

    contrast = sigma0 - sigma_background
    contrast_is_zero = abs(contrast) <= contrast_atol
    if contrast_is_zero:
        Es0 = np.zeros_like(Ep0_array)
        phi_s = None
    else:
        if secondary_field_solver is None:
            raise ValueError("secondary_field_solver is required for nonzero contrast")
        phi_s, Es0 = secondary_field_solver(contrast * Ep0_array)
        Es0 = _as_vector_field(Es0, "Es0")
        if Es0.shape != Ep0_array.shape:
            raise ValueError("Es0 must have the same shape as Ep0")
        if phi_s is not None:
            phi_s = np.asarray(phi_s, dtype=float)

    Etotal0 = Ep0_array + Es0
    chi0 = material.initial_memory(Etotal0)
    deltaJ0 = material.current_density(Etotal0, chi0) - sigma_background * Ep0_array
    return DCSecondaryInitialization(
        Ep0=Ep0_array,
        Es0=Es0,
        Etotal0=Etotal0,
        chi0=chi0,
        deltaJ0=deltaJ0,
        phi_s=phi_s,
        contrast_is_zero=contrast_is_zero,
    )


def initialize_dc_secondary_from_primary(
    *,
    primary,
    sigma0: float,
    sigma_background: float,
    material: PronyConductivity,
    secondary_field_solver: SecondaryFieldSolver | None = None,
    contrast_atol: float = 0.0,
) -> DCSecondaryInitialization:
    """Initialize DC secondary state from a primary FEM interpolator."""

    if not hasattr(primary, "sample_Ep_dc"):
        raise TypeError("primary must provide sample_Ep_dc()")
    return initialize_dc_secondary(
        Ep0=primary.sample_Ep_dc(),
        sigma0=sigma0,
        sigma_background=sigma_background,
        material=material,
        secondary_field_solver=secondary_field_solver,
        contrast_atol=contrast_atol,
    )


def _as_vector_field(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return array
