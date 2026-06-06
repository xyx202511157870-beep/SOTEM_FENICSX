"""Cole-Cole conductivity materials and Prony conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atem3d.fit import cole_cole_conductivity, fit_cole_cole_conductivity_debye
from atem3d.materials.prony import PronyConductivity


@dataclass(frozen=True)
class ColeColeConductivity:
    """Conductivity-form Cole-Cole material.

    The convention matches ``atem3d.fit.cole_cole_conductivity``:

        sigma(w) = sigma_inf * (1 - eta / (1 + (i w tau)^c))
    """

    sigma_inf: float
    eta: float
    tau: float
    c: float

    def __post_init__(self) -> None:
        sigma_inf = float(self.sigma_inf)
        eta = float(self.eta)
        tau = float(self.tau)
        c = float(self.c)
        if sigma_inf <= 0.0:
            raise ValueError("sigma_inf must be positive")
        if eta < 0.0 or eta >= 1.0:
            raise ValueError("eta must satisfy 0 <= eta < 1")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if c <= 0.0 or c > 1.0:
            raise ValueError("c must satisfy 0 < c <= 1")
        object.__setattr__(self, "sigma_inf", sigma_inf)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "c", c)

    @property
    def sigma0(self) -> float:
        """DC conductivity."""

        return self.sigma_inf * (1.0 - self.eta)

    def complex_conductivity(self, frequencies) -> np.ndarray:
        """Evaluate complex conductivity at positive frequencies."""

        return cole_cole_conductivity(
            frequencies,
            sigma_infinity=self.sigma_inf,
            eta=self.eta,
            tau=self.tau,
            c=self.c,
        )

    def to_prony_conductivity(
        self,
        *,
        frequencies,
        tau_grid=None,
        n_terms: int = 10,
    ) -> PronyConductivity:
        """Fit this Cole-Cole material with Debye poles and return Prony material."""

        fit = fit_cole_cole_conductivity_debye(
            sigma_infinity=self.sigma_inf,
            eta=self.eta,
            tau=self.tau,
            c=self.c,
            frequencies=frequencies,
            tau_grid=tau_grid,
            n_terms=n_terms,
        )
        return fit.to_prony_conductivity()
