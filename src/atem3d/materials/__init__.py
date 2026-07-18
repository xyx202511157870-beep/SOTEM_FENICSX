"""Minimal constitutive model retained by the current forward solvers."""

from .prony import DebyeTerm, PronyConductivity

__all__ = [
    "DebyeTerm",
    "PronyConductivity",
]
