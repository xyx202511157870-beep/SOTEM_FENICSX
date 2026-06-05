"""Scalar Debye/Prony conductivity material interface.

The time-domain convention is

    tau_k d chi_k / dt + chi_k = E
    J = sigma_inf E - sum(delta_sigma_k chi_k)

with backward-Euler memory elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DebyeTerm:
    """A single Debye relaxation pole in conductivity form."""

    delta_sigma: float
    tau: float

    def __post_init__(self) -> None:
        delta_sigma = float(self.delta_sigma)
        tau = float(self.tau)
        if delta_sigma < 0.0:
            raise ValueError("delta_sigma must be nonnegative")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        object.__setattr__(self, "delta_sigma", delta_sigma)
        object.__setattr__(self, "tau", tau)


@dataclass(frozen=True)
class PronyConductivity:
    """Debye/Prony IP conductivity for one homogeneous material."""

    sigma_inf: float
    terms: Iterable[DebyeTerm]

    def __post_init__(self) -> None:
        sigma_inf = float(self.sigma_inf)
        terms = tuple(self.terms)
        if sigma_inf <= 0.0:
            raise ValueError("sigma_inf must be positive")
        if sigma_inf - sum(term.delta_sigma for term in terms) <= 0.0:
            raise ValueError("sigma0 must be positive")
        object.__setattr__(self, "sigma_inf", sigma_inf)
        object.__setattr__(self, "terms", terms)

    @property
    def sigma0(self) -> float:
        """Low-frequency conductivity sigma(0)."""

        return self.sigma_inf - sum(term.delta_sigma for term in self.terms)

    @classmethod
    def no_ip(cls, sigma: float) -> "PronyConductivity":
        """Create a purely Ohmic material."""

        return cls(sigma_inf=sigma, terms=())

    def alpha(self, dt: float) -> np.ndarray:
        """Backward-Euler memory retention coefficients."""

        self._validate_dt(dt)
        return np.array([term.tau / (term.tau + dt) for term in self.terms], dtype=float)

    def beta(self, dt: float) -> np.ndarray:
        """Backward-Euler new-field memory coefficients."""

        self._validate_dt(dt)
        return np.array([dt / (term.tau + dt) for term in self.terms], dtype=float)

    def sigma_eff(self, dt: float) -> float:
        """Effective conductivity multiplying E at the new time level."""

        beta = self.beta(dt)
        return self.sigma_inf - sum(
            term.delta_sigma * beta_i for term, beta_i in zip(self.terms, beta)
        )

    def initial_memory(self, e0: np.ndarray | float) -> list[np.ndarray]:
        """Return chi_k(0)=E0 for long on-time step-off initial conditions."""

        e0_array = np.asarray(e0, dtype=float)
        return [e0_array.copy() for _ in self.terms]

    def update_memory(
        self,
        chi_old: list[np.ndarray],
        e_new: np.ndarray | float,
        dt: float,
    ) -> list[np.ndarray]:
        """Advance chi_k with backward Euler."""

        self._validate_memory_count(chi_old)
        e_new_array = np.asarray(e_new, dtype=float)
        alpha = self.alpha(dt)
        beta = self.beta(dt)
        updated: list[np.ndarray] = []
        for old, alpha_i, beta_i in zip(chi_old, alpha, beta):
            old_array = np.asarray(old, dtype=float)
            if old_array.shape != e_new_array.shape:
                raise ValueError("each chi_old entry must have the same shape as e_new")
            updated.append(alpha_i * old_array + beta_i * e_new_array)
        return updated

    def current_density(
        self,
        e: np.ndarray | float,
        chi: list[np.ndarray],
    ) -> np.ndarray:
        """Return J = sigma_inf E - sum(delta_sigma_k chi_k)."""

        self._validate_memory_count(chi)
        current = self.sigma_inf * np.asarray(e, dtype=float)
        for term, memory in zip(self.terms, chi):
            current = current - term.delta_sigma * np.asarray(memory, dtype=float)
        return current

    def lhs_effective_current_density(
        self,
        e_new: np.ndarray | float,
        dt: float,
    ) -> np.ndarray:
        """Return the matrix-side BE current term sigma_eff E_new."""

        return self.sigma_eff(dt) * np.asarray(e_new, dtype=float)

    def rhs_history_current_density(
        self,
        e_old: np.ndarray | float,
        chi_old: list[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Return the BE total-field RHS history current.

        Starting from
        ``J_new = sigma_eff E_new - sum(delta_sigma_k alpha_k chi_old_k)``,
        the total-field system moves the old-memory part to the RHS:
        ``sigma_eff E_new - [J_old + sum(delta_sigma_k alpha_k chi_old_k)]``.
        """

        self._validate_memory_count(chi_old)
        history = self.current_density(e_old, chi_old)
        alpha = self.alpha(dt)
        for term, memory, alpha_i in zip(self.terms, chi_old, alpha):
            history = history + term.delta_sigma * alpha_i * np.asarray(memory, dtype=float)
        return history

    def eliminated_current_density(
        self,
        e_new: np.ndarray | float,
        chi_old: list[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Return J_new after eliminating chi_new with backward Euler."""

        self._validate_memory_count(chi_old)
        current = self.lhs_effective_current_density(e_new, dt)
        alpha = self.alpha(dt)
        for term, memory, alpha_i in zip(self.terms, chi_old, alpha):
            current = current - term.delta_sigma * alpha_i * np.asarray(memory, dtype=float)
        return current

    def to_debye_ip_model(self):
        """Convert to the existing array-capable `atem3d.ip.DebyeIPModel`."""

        from atem3d.ip import DebyeIPModel, DebyeTerm as LegacyDebyeTerm

        return DebyeIPModel(
            sigma_infinity=np.array([self.sigma_inf], dtype=float),
            terms=[
                LegacyDebyeTerm(delta_sigma=np.array([term.delta_sigma], dtype=float), tau=term.tau)
                for term in self.terms
            ],
        )

    @classmethod
    def from_debye_ip_model(cls, model) -> "PronyConductivity":
        """Create a scalar material from a single-cell DebyeIPModel."""

        sigma = np.asarray(model.sigma_infinity, dtype=float)
        if sigma.size != 1:
            raise ValueError("only scalar/single-cell DebyeIPModel instances can be converted")
        terms = []
        for term in model.terms:
            delta = np.asarray(term.delta_sigma, dtype=float)
            if delta.size != 1:
                raise ValueError("only scalar/single-cell Debye terms can be converted")
            terms.append(DebyeTerm(delta_sigma=float(delta[0]), tau=float(term.tau)))
        return cls(sigma_inf=float(sigma[0]), terms=terms)

    @staticmethod
    def _validate_dt(dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

    def _validate_memory_count(self, chi: list[np.ndarray]) -> None:
        if len(chi) != len(self.terms):
            raise ValueError("one memory array is required for each Debye term")
