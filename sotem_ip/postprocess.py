"""Response post-processing utilities."""

from __future__ import annotations

import numpy as np


def relative_error(numerical, reference, *, floor_factor: float = 1.0e-3):
    """Return floor-denominator relative error."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if numerical.shape != reference.shape:
        raise ValueError("numerical and reference arrays must have the same shape")
    ref_max = float(np.max(np.abs(reference))) if reference.size else 0.0
    floor = max(floor_factor * ref_max, np.finfo(float).tiny)
    return np.abs(numerical - reference) / np.maximum(np.abs(reference), floor)


def ip_percent_effect(ip_response, no_ip_response, *, floor_factor: float = 1.0e-3):
    """Return 100*(IP-noIP)/noIP using a floor denominator."""

    ip_response = np.asarray(ip_response, dtype=float)
    no_ip_response = np.asarray(no_ip_response, dtype=float)
    if ip_response.shape != no_ip_response.shape:
        raise ValueError("responses must have the same shape")
    ref_max = float(np.max(np.abs(no_ip_response))) if no_ip_response.size else 0.0
    floor = max(floor_factor * ref_max, np.finfo(float).tiny)
    denom = np.maximum(np.abs(no_ip_response), floor)
    return 100.0 * (ip_response - no_ip_response) / denom

