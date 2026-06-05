"""Material constitutive models used by forward solvers."""

from .material_map import CellMaterialMap, apply_leakage_channel_marker, mark_leakage_channel
from .prony import DebyeTerm, PronyConductivity

__all__ = [
    "CellMaterialMap",
    "DebyeTerm",
    "PronyConductivity",
    "apply_leakage_channel_marker",
    "mark_leakage_channel",
]
