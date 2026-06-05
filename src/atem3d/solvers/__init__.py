"""Forward solver building blocks."""

from .dc_secondary import (
    DCSecondaryInitialization,
    initialize_dc_secondary,
    initialize_dc_secondary_from_primary,
)
from .tdem_secondary import (
    SecondaryState,
    secondary_state_from_dc_initialization,
    secondary_step_ip,
    secondary_step_noip,
)

__all__ = [
    "DCSecondaryInitialization",
    "SecondaryState",
    "initialize_dc_secondary",
    "initialize_dc_secondary_from_primary",
    "secondary_state_from_dc_initialization",
    "secondary_step_ip",
    "secondary_step_noip",
]
