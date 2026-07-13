"""Primary-field providers for primary-secondary forward solvers."""

from .base import PrimaryFieldProvider
from .cache import CachedPrimaryProvider
from .dc import (
    analytic_halfspace_dc_runner,
    analytic_halfspace_grounded_wire_dc_electric_field,
    empymod_quasistatic_dc_runner,
)
from .empymod_provider import EmpymodPrimaryProvider
from .interpolation import PrimaryFEMInterpolator, TabulatedVectorField, make_tabulated_vector_assembler
from .zero import ZeroPrimaryProvider

__all__ = [
    "CachedPrimaryProvider",
    "EmpymodPrimaryProvider",
    "PrimaryFEMInterpolator",
    "PrimaryFieldProvider",
    "TabulatedVectorField",
    "ZeroPrimaryProvider",
    "analytic_halfspace_dc_runner",
    "analytic_halfspace_grounded_wire_dc_electric_field",
    "empymod_quasistatic_dc_runner",
    "make_tabulated_vector_assembler",
]
