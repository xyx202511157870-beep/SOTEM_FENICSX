"""Material constitutive models used by forward solvers."""

from .prony import DebyeTerm, PronyConductivity

__all__ = ["DebyeTerm", "PronyConductivity"]
