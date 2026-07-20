"""Canonical component roles for the symmetric SOTEM benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CANONICAL_SOTEM_COMPONENT_ORDER = ("Ex", "Ey", "Hz", "dBzdt")
SOTEM_ACCEPTANCE_COMPONENTS = ("Ex", "Hz", "dBzdt")
SOTEM_DIAGNOSTIC_ONLY_COMPONENTS = ("Ey",)
SOTEM_ACCEPTANCE_PROFILE = "symmetric_sotem_ex_hz_dBzdt/v1"


def symmetric_sotem_component_contract(component_names: Sequence[str]) -> dict:
    """Return the fail-closed canonical component-role contract."""

    names = tuple(str(name) for name in component_names)
    if names != CANONICAL_SOTEM_COMPONENT_ORDER:
        raise ValueError(
            "symmetric SOTEM component order must be exactly "
            + ", ".join(CANONICAL_SOTEM_COMPONENT_ORDER)
        )
    roles = {
        name: (
            {"acceptance_status": "included", "diagnostic_role": None}
            if name in SOTEM_ACCEPTANCE_COMPONENTS
            else {
                "acceptance_status": "excluded_by_design",
                "diagnostic_role": "transverse_symmetry",
            }
        )
        for name in names
    }
    return {
        "acceptance_profile": SOTEM_ACCEPTANCE_PROFILE,
        "component_order": list(names),
        "acceptance_components": list(SOTEM_ACCEPTANCE_COMPONENTS),
        "diagnostic_only_components": list(SOTEM_DIAGNOSTIC_ONLY_COMPONENTS),
        "component_roles": roles,
    }


def ey_symmetry_diagnostics(
    prediction,
    reference,
    component_names: Sequence[str],
) -> dict:
    """Quantify Ey as a retained transverse-symmetry diagnostic."""

    contract = symmetric_sotem_component_contract(component_names)
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError("symmetry diagnostic arrays must be matching 2-D tables")
    if pred.shape[1] != len(CANONICAL_SOTEM_COMPONENT_ORDER):
        raise ValueError("symmetry diagnostic columns must match the SOTEM component order")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(ref)):
        raise ValueError("symmetry diagnostic arrays must contain only finite values")

    ex_index = CANONICAL_SOTEM_COMPONENT_ORDER.index("Ex")
    ey_index = CANONICAL_SOTEM_COMPONENT_ORDER.index("Ey")
    pred_ex_peak = float(np.max(np.abs(pred[:, ex_index])))
    ref_ex_peak = float(np.max(np.abs(ref[:, ex_index])))
    pred_ey_peak = float(np.max(np.abs(pred[:, ey_index])))
    ref_ey_peak = float(np.max(np.abs(ref[:, ey_index])))

    def ratio(numerator: float, denominator: float) -> tuple[float | None, bool]:
        if denominator > 0.0:
            return float(numerator / denominator), True
        if numerator == 0.0:
            return 0.0, True
        return None, False

    prediction_ratio, prediction_ratio_defined = ratio(pred_ey_peak, pred_ex_peak)
    reference_ratio, reference_ratio_defined = ratio(ref_ey_peak, ref_ex_peak)

    return {
        "Ey": {
            **contract["component_roles"]["Ey"],
            "prediction_peak_abs": pred_ey_peak,
            "reference_peak_abs": ref_ey_peak,
            "prediction_Ex_peak_abs": pred_ex_peak,
            "reference_Ex_peak_abs": ref_ex_peak,
            "prediction_to_Ex_peak_ratio": prediction_ratio,
            "prediction_to_Ex_peak_ratio_defined": prediction_ratio_defined,
            "reference_to_Ex_peak_ratio": reference_ratio,
            "reference_to_Ex_peak_ratio_defined": reference_ratio_defined,
        }
    }
