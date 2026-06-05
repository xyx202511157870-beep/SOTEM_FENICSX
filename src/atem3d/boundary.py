"""Absorbing-boundary configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sponge import make_sponge_sigma


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary strategy for direct time-domain experiments."""

    kind: str = "none"
    thickness_cells: int = 0
    strength: float = 0.0
    power: float = 2.0
    disable_ip_in_shell: bool = True
    apply_to_initial: bool = True
    sides: Sequence[str] | None = None


def apply_boundary(
    mesh,
    sigma_infinity: np.ndarray,
    debye_terms: list[tuple[float, np.ndarray]],
    config: BoundaryConfig,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """Apply a boundary strategy to conductivity and IP properties."""

    if config.kind == "none":
        return sigma_infinity.copy(), [(tau, delta.copy()) for tau, delta in debye_terms]
    if config.kind == "cpml":
        raise ValueError(
            "boundary.kind='cpml' is a solver-level boundary; construct it through "
            "build_simulation so the CPML memory variables are passed to the time stepper."
        )
    if config.kind != "sponge":
        raise ValueError(f"unsupported boundary kind: {config.kind}")

    sigma = make_sponge_sigma(
        mesh,
        sigma_infinity,
        thickness_cells=config.thickness_cells,
        strength=config.strength,
        power=config.power,
        active_sides=config.sides,
    )
    shell = sigma > sigma_infinity
    terms: list[tuple[float, np.ndarray]] = []
    for tau, delta in debye_terms:
        updated = delta.copy()
        if config.disable_ip_in_shell:
            updated[shell] = 0.0
        terms.append((tau, updated))
    return sigma, terms
