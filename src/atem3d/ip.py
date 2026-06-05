"""Induced-polarization constitutive models for direct time stepping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | float | list[float] | tuple[float, ...]


def _as_cell_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array")
    return array


@dataclass(frozen=True)
class DebyeTerm:
    """A single Debye relaxation pole in conductivity form.

    The conductivity is represented as

    ``sigma(s) = sigma_infinity - sum(delta_sigma / (1 + s * tau))``.
    """

    delta_sigma: ArrayLike
    tau: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_sigma", _as_cell_array(self.delta_sigma, "delta_sigma"))
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if np.any(self.delta_sigma < 0.0):
            raise ValueError("delta_sigma must be nonnegative")

    def coefficients(self, dt: float) -> tuple[float, float]:
        """Return backward-Euler memory coefficients ``alpha`` and ``beta``."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        alpha = self.tau / (self.tau + dt)
        beta = dt / (self.tau + dt)
        return alpha, beta


@dataclass(frozen=True)
class DebyeIPModel:
    """Finite-dimensional Debye approximation of IP memory."""

    sigma_infinity: ArrayLike
    terms: list[DebyeTerm]

    def __post_init__(self) -> None:
        sigma = _as_cell_array(self.sigma_infinity, "sigma_infinity")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma_infinity must be positive")
        object.__setattr__(self, "sigma_infinity", sigma)
        object.__setattr__(self, "terms", list(self.terms))
        for term in self.terms:
            self._validate_term_shape(term)
        if np.any(self.low_frequency_sigma() <= 0.0):
            raise ValueError("low-frequency conductivity must remain positive")

    @classmethod
    def no_ip(cls, sigma: ArrayLike) -> "DebyeIPModel":
        """Create a purely Ohmic model."""

        return cls(sigma_infinity=sigma, terms=[])

    @property
    def n_cells(self) -> int:
        return int(self.sigma_infinity.size)

    def expand_to_cells(self, n_cells: int) -> "DebyeIPModel":
        """Broadcast scalar properties to ``n_cells`` and return a new model."""

        sigma = _expand(self.sigma_infinity, n_cells, "sigma_infinity")
        terms = [
            DebyeTerm(delta_sigma=_expand(term.delta_sigma, n_cells, "delta_sigma"), tau=term.tau)
            for term in self.terms
        ]
        return DebyeIPModel(sigma, terms)

    def initial_memory(self, n_edges: int, initial_e: np.ndarray | None = None) -> list[np.ndarray]:
        """Return initial Debye memories.

        For a long on-time step-off response, the Debye states are in equilibrium
        with the DC electric field, so callers pass ``initial_e``. For zero-field
        starts, memory states are zero.
        """

        if initial_e is None:
            return [np.zeros(n_edges, dtype=float) for _ in self.terms]
        initial_e = np.asarray(initial_e, dtype=float)
        if initial_e.shape != (n_edges,):
            raise ValueError("initial_e must have shape (n_edges,)")
        return [initial_e.copy() for _ in self.terms]

    def update_memory(
        self, memories: list[np.ndarray], e_new: np.ndarray, dt: float
    ) -> list[np.ndarray]:
        """Advance Debye memory variables with backward Euler."""

        self._validate_memories(memories, e_new.size)
        updated: list[np.ndarray] = []
        for term, old in zip(self.terms, memories):
            alpha, beta = term.coefficients(dt)
            updated.append(alpha * old + beta * e_new)
        return updated

    def inverse_constitutive_update(
        self,
        current_density: np.ndarray,
        memories: list[np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Update H/J-form Debye memories and recover electric field from current.

        H/J formulations naturally know the conduction current
        ``j = curl h - s_e`` first.  For Debye IP this requires the inverse
        local constitutive relation
        ``e = rho_inf * (j + sum(delta_sigma_i * y_i))`` with all Debye
        memories coupled at the new time level.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        current = np.asarray(current_density, dtype=float)
        if current.ndim != 1:
            raise ValueError("current_density must be a 1D array")
        n_dofs = current.size
        rho_eff, electric_history = self.inverse_constitutive_coefficients(
            memories,
            dt,
            n_dofs=n_dofs,
        )
        electric = rho_eff * current + electric_history
        return electric, self.update_memory(memories, electric, dt)

    def inverse_constitutive_coefficients(
        self,
        memories: list[np.ndarray],
        dt: float,
        n_dofs: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return H/J-form ``rho_eff`` and electric history for one time step.

        After eliminating Debye memories at ``t^(n+1)``, the inverse
        constitutive law can be written locally as
        ``e^(n+1) = rho_eff * j^(n+1) + e_history``.  This coefficient form is
        the piece needed by an H/J magnetic-field system matrix.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if n_dofs is None:
            if memories:
                n_dofs = int(np.asarray(memories[0]).size)
            else:
                n_dofs = self.sigma_infinity.size
        n_dofs = int(n_dofs)
        if n_dofs <= 0:
            raise ValueError("n_dofs must be positive")
        self._validate_memories(memories, n_dofs)
        sigma_inf = _expand(self.sigma_infinity, n_dofs, "sigma_infinity")
        rho_inf = 1.0 / sigma_inf
        if not self.terms:
            return rho_inf, np.zeros(n_dofs, dtype=float)

        history = np.zeros(n_dofs, dtype=float)
        coupling = np.zeros(n_dofs, dtype=float)
        for term, memory in zip(self.terms, memories):
            delta = _expand(term.delta_sigma, n_dofs, "delta_sigma")
            denominator = term.tau + dt
            history += delta * term.tau * memory / denominator
            coupling += delta / denominator

        denominator = 1.0 - dt * rho_inf * coupling
        if np.any(denominator <= 0.0):
            raise ValueError("inverse constitutive denominator must be positive")
        rho_eff = rho_inf / denominator
        electric_history = rho_inf * history / denominator
        return rho_eff, electric_history

    def effective_sigma(self, dt: float) -> np.ndarray:
        """Conductivity multiplying ``e^(n+1)`` after eliminating memories."""

        sigma = self.sigma_infinity.copy()
        for term in self.terms:
            _, beta = term.coefficients(dt)
            sigma = sigma - beta * term.delta_sigma
        return sigma

    def low_frequency_sigma(self) -> np.ndarray:
        """Return ``sigma(0)``."""

        sigma = self.sigma_infinity.copy()
        for term in self.terms:
            sigma = sigma - term.delta_sigma
        return sigma

    def history_current(self, memories: list[np.ndarray]) -> np.ndarray:
        """Return a cell-space history current for shape-only no-IP cases.

        The full solver applies history terms through edge inner-product
        matrices because Debye conductivities live on cells while memories live
        on edges. For no-IP tests and diagnostics this helper returns zeros.
        """

        if self.terms:
            raise ValueError("history_current is only defined for no-IP diagnostics")
        return np.zeros(self.n_cells, dtype=float)

    def _validate_term_shape(self, term: DebyeTerm) -> None:
        if term.delta_sigma.size not in (1, self.sigma_infinity.size):
            raise ValueError("delta_sigma must be scalar or match sigma_infinity")
        if term.delta_sigma.size == 1 and self.sigma_infinity.size != 1:
            object.__setattr__(
                term,
                "delta_sigma",
                np.full(self.sigma_infinity.size, float(term.delta_sigma[0])),
            )

    def _validate_memories(self, memories: list[np.ndarray], n_edges: int) -> None:
        if len(memories) != len(self.terms):
            raise ValueError("one memory vector is required for each Debye term")
        for memory in memories:
            if np.asarray(memory).shape != (n_edges,):
                raise ValueError("each memory vector must have shape (n_edges,)")


def _expand(value: np.ndarray, n: int, name: str) -> np.ndarray:
    if value.size == n:
        return value.copy()
    if value.size == 1:
        return np.full(n, float(value[0]))
    raise ValueError(f"{name} must be scalar or length {n}")
