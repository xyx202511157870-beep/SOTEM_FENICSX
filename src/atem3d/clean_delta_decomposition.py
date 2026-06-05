"""Clean-delta error decomposition for IP source-history diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import relative_l2


@dataclass(frozen=True)
class SampledReport:
    """Sampled empymod validation data loaded from a report JSON."""

    payload: dict
    names: list[str]
    times: np.ndarray
    numerical: np.ndarray
    reference: np.ndarray


def load_sampled_report(
    path: str | Path,
    *,
    component_names: Sequence[str] | None = None,
    component_prefix: str | None = None,
) -> SampledReport:
    """Load sampled validation data from ``path``.

    The returned arrays have shape ``(n_times, n_components)`` and are ordered by
    the requested component names or by sorted sample keys when no explicit list
    is supplied.
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"{path} does not contain sampled validation data")

    if component_names is not None:
        names = [str(name) for name in component_names]
    else:
        prefix = "" if component_prefix is None else str(component_prefix)
        names = [str(name) for name in samples if str(name).startswith(prefix)]
        names.sort()
    if not names:
        raise ValueError(f"{path} contains no selected sampled components")

    times = None
    numerical_columns = []
    reference_columns = []
    for name in names:
        if name not in samples:
            raise ValueError(f"{path} does not contain component {name!r}")
        rows = samples[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path} component {name!r} has no sample rows")
        column_times = np.asarray([row["time"] for row in rows], dtype=float)
        if times is None:
            times = column_times
        elif not np.allclose(times, column_times, rtol=0.0, atol=0.0):
            raise ValueError(f"sample times are inconsistent for {name}")
        numerical_columns.append([row["numerical"] for row in rows])
        reference_columns.append([row["reference"] for row in rows])

    return SampledReport(
        payload=payload,
        names=names,
        times=np.asarray(times, dtype=float),
        numerical=np.asarray(numerical_columns, dtype=float).T,
        reference=np.asarray(reference_columns, dtype=float).T,
    )


def clean_delta_decomposition_report(
    raw_ip: SampledReport,
    corrected_ip: SampledReport,
    noip: SampledReport,
) -> dict:
    """Return a clean-delta/no-IP baseline error decomposition report.

    Sign convention: ``error = numerical - reference``.  The ideal IP-only
    correction that reduces the raw IP error to the no-IP baseline error is
    ``noip_error - raw_ip_error``.  A runtime/postprocess candidate supplies
    ``corrected_ip_numerical - raw_ip_numerical``.
    """

    _validate_alignment(raw_ip, corrected_ip, label="raw and corrected IP")
    _validate_alignment(raw_ip, noip, label="IP and no-IP", require_same_reference=False)
    if not np.allclose(raw_ip.reference, corrected_ip.reference):
        raise ValueError("raw and corrected IP reports must use the same reference data")

    raw_ip_error = raw_ip.numerical - raw_ip.reference
    corrected_ip_error = corrected_ip.numerical - corrected_ip.reference
    noip_error = noip.numerical - noip.reference
    ideal_clean_delta = noip_error - raw_ip_error
    actual_correction = corrected_ip.numerical - raw_ip.numerical
    correction_mismatch = actual_correction - ideal_clean_delta
    identity_residual = corrected_ip_error - (noip_error + correction_mismatch)

    components = {}
    for column, name in enumerate(raw_ip.names):
        ip_ref = raw_ip.reference[:, column]
        components[name] = _component_report(
            raw_ip_error[:, column],
            corrected_ip_error[:, column],
            noip_error[:, column],
            ideal_clean_delta[:, column],
            actual_correction[:, column],
            correction_mismatch[:, column],
            identity_residual[:, column],
            ip_ref,
        )

    return {
        "diagnostic_only": True,
        "description": (
            "Decomposes corrected IP error into no-IP baseline error plus "
            "clean-delta correction mismatch using error = numerical - reference."
        ),
        "component_names": list(raw_ip.names),
        "n_times": int(raw_ip.times.size),
        "time_range": {
            "min": float(raw_ip.times[0]),
            "max": float(raw_ip.times[-1]),
        },
        "components": components,
        "aggregate": _component_report(
            raw_ip_error,
            corrected_ip_error,
            noip_error,
            ideal_clean_delta,
            actual_correction,
            correction_mismatch,
            identity_residual,
            raw_ip.reference,
        ),
        "identity": {
            "formula": "corrected_ip_error = noip_error + correction_mismatch",
            "max_abs_residual": float(np.max(np.abs(identity_residual))),
            "relative_l2_against_ip_reference": _norm_over(
                identity_residual,
                raw_ip.reference,
            ),
        },
    }


def _component_report(
    raw_ip_error: np.ndarray,
    corrected_ip_error: np.ndarray,
    noip_error: np.ndarray,
    ideal_clean_delta: np.ndarray,
    actual_correction: np.ndarray,
    correction_mismatch: np.ndarray,
    identity_residual: np.ndarray,
    ip_reference: np.ndarray,
) -> dict[str, float]:
    return {
        "raw_ip_error_relative_l2": _norm_over(raw_ip_error, ip_reference),
        "corrected_ip_error_relative_l2": _norm_over(
            corrected_ip_error,
            ip_reference,
        ),
        "noip_baseline_relative_l2_against_ip_reference": _norm_over(
            noip_error,
            ip_reference,
        ),
        "ideal_clean_delta_relative_l2_against_ip_reference": _norm_over(
            ideal_clean_delta,
            ip_reference,
        ),
        "actual_correction_relative_l2_against_ip_reference": _norm_over(
            actual_correction,
            ip_reference,
        ),
        "actual_correction_relative_l2_to_ideal": relative_l2(
            actual_correction,
            ideal_clean_delta,
        ),
        "correction_mismatch_relative_l2_against_ip_reference": _norm_over(
            correction_mismatch,
            ip_reference,
        ),
        "identity_max_abs_residual": float(np.max(np.abs(identity_residual))),
        "first_raw_ip_error": float(np.ravel(raw_ip_error)[0]),
        "first_noip_error": float(np.ravel(noip_error)[0]),
        "first_ideal_clean_delta": float(np.ravel(ideal_clean_delta)[0]),
        "first_actual_correction": float(np.ravel(actual_correction)[0]),
        "first_correction_mismatch": float(np.ravel(correction_mismatch)[0]),
    }


def _validate_alignment(
    first: SampledReport,
    second: SampledReport,
    *,
    label: str,
    require_same_reference: bool = True,
) -> None:
    if first.names != second.names:
        raise ValueError(f"{label} reports must contain the same sampled components")
    if first.times.shape != second.times.shape or not np.allclose(first.times, second.times):
        raise ValueError(f"{label} reports must contain the same sample times")
    if require_same_reference and not np.allclose(first.reference, second.reference):
        raise ValueError(f"{label} reports must contain the same reference data")


def _norm_over(values: np.ndarray, denominator_values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)
    denom = float(np.linalg.norm(denominator_values.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(values.ravel()))
    return float(np.linalg.norm(values.ravel()) / denom)
