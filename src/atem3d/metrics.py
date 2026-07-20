"""Validation error metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearResponseFit:
    """Least-squares fit of additive response components."""

    weights: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    rank: int
    singular_values: np.ndarray


def relative_l2(numerical, reference) -> float:
    """Return ``||numerical-reference||_2 / ||reference||_2``."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denom = np.linalg.norm(reference.ravel())
    if denom == 0.0:
        return float(np.linalg.norm((numerical - reference).ravel()))
    return float(np.linalg.norm((numerical - reference).ravel()) / denom)


def relative_linf(numerical, reference) -> float:
    """Return ``||numerical-reference||_inf / ||reference||_inf``."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denom = np.max(np.abs(reference))
    if denom == 0.0:
        return float(np.max(np.abs(numerical - reference)))
    return float(np.max(np.abs(numerical - reference)) / denom)


def absolute_linf(numerical, reference) -> float:
    """Return ``||numerical-reference||_inf``."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.max(np.abs(numerical - reference)))


def robust_relative_error(pred: float, ref: float, floor: float) -> float:
    """Return ``abs(pred-ref) / max(abs(ref), floor)``."""

    floor = float(floor)
    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError("floor must be finite and positive")
    return float(abs(float(pred) - float(ref)) / max(abs(float(ref)), floor))


def robust_component_errors(
    times,
    numerical,
    reference,
    component_names,
    *,
    threshold: float = 0.05,
    floor_overrides: dict[str, float] | None = None,
    acceptance_components=None,
    diagnostic_only_components: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return row-wise robust error table and summary for validation outputs."""

    times = np.asarray(times, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if numerical.shape != reference.shape:
        raise ValueError("numerical and reference arrays must have the same shape")
    if numerical.ndim != 2 or numerical.shape[0] != times.size:
        raise ValueError("arrays must have shape (n_times, n_components)")
    if len(component_names) != numerical.shape[1]:
        raise ValueError("component_names length must match component columns")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    component_names = [str(name) for name in component_names]
    if len(set(component_names)) != len(component_names):
        raise ValueError("component_names must be unique")
    if acceptance_components is None:
        acceptance_names = tuple(component_names)
        diagnostic_roles: dict[str, str] = {}
        explicit_acceptance_contract = False
    else:
        acceptance_names = tuple(str(name) for name in acceptance_components)
        if not acceptance_names or len(set(acceptance_names)) != len(acceptance_names):
            raise ValueError("acceptance_components must be nonempty and unique")
        missing_acceptance = [name for name in acceptance_names if name not in component_names]
        if missing_acceptance:
            raise ValueError(
                "acceptance_components are absent from component_names: "
                + ", ".join(missing_acceptance)
            )
        diagnostic_roles = {
            str(name): str(role)
            for name, role in dict(diagnostic_only_components or {}).items()
        }
        expected_diagnostics = [name for name in component_names if name not in acceptance_names]
        if set(diagnostic_roles) != set(expected_diagnostics):
            raise ValueError(
                "diagnostic_only_components must name every non-acceptance component"
            )
        if any(not role for role in diagnostic_roles.values()):
            raise ValueError("diagnostic-only component roles must be nonempty")
        explicit_acceptance_contract = True
    acceptance_name_set = set(acceptance_names)

    dtype = [
        ("time_obs", "f8"),
        ("component", "U32"),
        ("pred", "f8"),
        ("ref", "f8"),
        ("abs_error", "f8"),
        ("ordinary_relative_error", "f8"),
        ("relative_error_with_floor", "f8"),
        ("peak_normalized_error", "f8"),
        ("pass_5pct", "?"),
    ]
    rows = []
    summary: dict[str, object] = {
        "pass_all_components": True,
        "failed_times": [],
        "failed_components": [],
    }
    floor_overrides = floor_overrides or {}
    failed_components: set[str] = set()
    failed_times: set[float] = set()
    diagnostic_failed_components: set[str] = set()
    diagnostic_failed_times: set[float] = set()
    magnetic_component = None
    magnetic_components: list[str] = []

    for col, component in enumerate(component_names):
        component = str(component)
        ref_col = reference[:, col]
        num_col = numerical[:, col]
        max_abs_ref = float(np.max(np.abs(ref_col))) if ref_col.size else 0.0
        floor = float(floor_overrides.get(component, _default_component_floor(component, max_abs_ref)))
        robust_values = []
        peak_values = []
        for time, pred, ref in zip(times, num_col, ref_col):
            abs_error = float(abs(pred - ref))
            ordinary = float(abs_error / abs(ref)) if ref != 0.0 else float("inf")
            robust = robust_relative_error(float(pred), float(ref), floor)
            peak = float(abs_error / max(max_abs_ref, floor))
            passed = bool(robust <= threshold and peak <= threshold)
            if not passed:
                if component in acceptance_name_set:
                    failed_components.add(component)
                    failed_times.add(float(time))
                else:
                    diagnostic_failed_components.add(component)
                    diagnostic_failed_times.add(float(time))
            robust_values.append(robust)
            peak_values.append(peak)
            rows.append((float(time), component, float(pred), float(ref), abs_error, ordinary, robust, peak, passed))
        summary[f"max_error_{component}"] = float(np.max(robust_values))
        summary[f"rms_error_{component}"] = float(np.sqrt(np.mean(np.asarray(robust_values) ** 2)))
        summary[f"max_peak_normalized_error_{component}"] = float(np.max(peak_values))
        key = component if component in {"Ex", "Ey"} else "Hz_or_dBzdt"
        if component not in {"Ex", "Ey"}:
            magnetic_component = component
            magnetic_components.append(component)
        summary[f"max_error_{key}"] = float(np.max(robust_values))
        summary[f"rms_error_{key}"] = float(np.sqrt(np.mean(np.asarray(robust_values) ** 2)))
        summary[f"max_peak_normalized_error_{key}"] = float(np.max(peak_values))

    strict_failed_components = sorted(failed_components)
    weak_gate = _weak_horizontal_component_gate(
        numerical,
        reference,
        component_names,
        threshold=float(threshold),
    )
    if explicit_acceptance_contract:
        physical_failed_components = sorted(failed_components)
    else:
        weak_components = set(weak_gate["weak_components"])
        physical_failed_components = sorted(
            component for component in failed_components if component not in weak_components
        )
        if not weak_gate["passed"]:
            physical_failed_components.extend(
                component
                for component in weak_gate["weak_components"]
                if component not in physical_failed_components
            )

    summary["pass_all_components"] = len(failed_components) == 0
    summary["failed_components"] = strict_failed_components
    summary["failed_times"] = sorted(failed_times)
    summary["diagnostic_pass_all_components"] = len(diagnostic_failed_components) == 0
    summary["diagnostic_failed_components"] = sorted(diagnostic_failed_components)
    summary["diagnostic_failed_times"] = sorted(diagnostic_failed_times)
    summary["physical_pass_all_components"] = len(physical_failed_components) == 0
    summary["physical_failed_components"] = sorted(physical_failed_components)
    summary["weak_component_passed"] = bool(weak_gate["passed"])
    summary["weak_components"] = list(weak_gate["weak_components"])
    summary["weak_component_primary_scale"] = float(weak_gate["primary_scale"])
    summary["weak_component_scaled_abs_error_max"] = dict(weak_gate["maxima"])
    summary["weak_component_reference_max"] = dict(weak_gate["reference_maxima"])
    summary["acceptance_components"] = list(acceptance_names)
    summary["diagnostic_only_components"] = list(diagnostic_roles)
    summary["component_roles"] = {
        component: (
            {"acceptance_status": "included", "diagnostic_role": None}
            if component in acceptance_name_set
            else {
                "acceptance_status": "excluded_by_design",
                "diagnostic_role": diagnostic_roles[component],
            }
        )
        for component in component_names
    }
    if magnetic_component is not None:
        summary["magnetic_quantity"] = magnetic_component
    summary["magnetic_components"] = magnetic_components
    return np.asarray(rows, dtype=dtype), summary


def _weak_horizontal_component_gate(
    numerical,
    reference,
    component_names,
    *,
    threshold: float,
    weak_reference_fraction: float = 0.1,
) -> dict[str, object]:
    component_names = [str(name) for name in component_names]
    if "Ex" not in component_names or "Ey" not in component_names:
        return {
            "passed": True,
            "weak_components": [],
            "primary_scale": 0.0,
            "maxima": {},
            "reference_maxima": {},
        }

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ix = component_names.index("Ex")
    iy = component_names.index("Ey")
    ref_horizontal = np.sqrt(reference[:, ix] ** 2 + reference[:, iy] ** 2)
    primary_scale = float(np.max(np.abs(ref_horizontal))) if ref_horizontal.size else 0.0
    if primary_scale <= 0.0:
        primary_scale = float(np.max(np.abs(reference[:, [ix, iy]]))) if reference.size else 0.0
    if primary_scale <= 0.0:
        primary_scale = 1.0

    weak_components: list[str] = []
    maxima: dict[str, float] = {}
    reference_maxima: dict[str, float] = {}
    for component, index in (("Ex", ix), ("Ey", iy)):
        ref_max = float(np.max(np.abs(reference[:, index]))) if reference.shape[0] else 0.0
        if ref_max <= float(weak_reference_fraction) * primary_scale:
            weak_components.append(component)
            reference_maxima[component] = ref_max
            maxima[component] = float(np.max(np.abs(numerical[:, index] - reference[:, index])) / primary_scale)

    return {
        "passed": bool(all(value <= float(threshold) for value in maxima.values())),
        "weak_components": weak_components,
        "primary_scale": primary_scale,
        "maxima": maxima,
        "reference_maxima": reference_maxima,
    }


def _default_component_floor(component: str, max_abs_ref: float) -> float:
    if component.startswith("E"):
        return max(1.0e-14, 1.0e-6 * float(max_abs_ref))
    if component.startswith("H"):
        return max(1.0e-16, 1.0e-6 * float(max_abs_ref))
    if component.startswith("dB") or component.startswith("B"):
        return max(1.0e-18, 1.0e-6 * float(max_abs_ref))
    return max(np.finfo(float).tiny, 1.0e-6 * float(max_abs_ref))


def summarize_errors(numerical, reference, component_names) -> dict[str, dict[str, float]]:
    """Return per-column relative L2 and L-infinity errors."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if numerical.shape != reference.shape:
        raise ValueError("numerical and reference arrays must have the same shape")
    if numerical.ndim != 2:
        raise ValueError("expected arrays with shape (n_times, n_components)")
    if len(component_names) != numerical.shape[1]:
        raise ValueError("component_names length must match the number of columns")

    return {
        str(name): {
            "relative_l2": relative_l2(numerical[:, i], reference[:, i]),
            "relative_linf": relative_linf(numerical[:, i], reference[:, i]),
            "absolute_linf": absolute_linf(numerical[:, i], reference[:, i]),
        }
        for i, name in enumerate(component_names)
    }


def component_pass_report(
    summary: dict[str, dict[str, float]],
    *,
    relative_tolerance: float | None,
    absolute_tolerance: float | None,
) -> dict[str, dict[str, float | bool | str | None]]:
    """Attach per-component pass/fail flags to an error summary."""

    if relative_tolerance is not None and relative_tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if absolute_tolerance is not None and absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be nonnegative")

    report: dict[str, dict[str, float | bool | str | None]] = {}
    for name, values in summary.items():
        relative_passed = (
            False
            if relative_tolerance is None
            else values["relative_linf"] <= relative_tolerance
        )
        absolute_passed = (
            False
            if absolute_tolerance is None
            else values["absolute_linf"] <= absolute_tolerance
        )
        if relative_passed:
            passed_by = "relative"
        elif absolute_passed:
            passed_by = "absolute"
        elif relative_tolerance is None and absolute_tolerance is None:
            passed_by = "not_evaluated"
        else:
            passed_by = "none"
        item = {metric: float(value) for metric, value in values.items()}
        item["passed"] = (
            None
            if relative_tolerance is None and absolute_tolerance is None
            else bool(relative_passed or absolute_passed)
        )
        item["passed_by"] = passed_by
        report[name] = item
    return report


def component_group_summary(
    summary: dict[str, dict[str, float | bool | str | None]],
) -> dict[str, dict[str, float | int | bool | None | list[str]]]:
    """Return electric/magnetic/other max-error summaries."""

    groups: dict[str, dict[str, object]] = {}
    for name, values in summary.items():
        group_name = _component_group(name)
        group = groups.setdefault(
            group_name,
            {
                "components": [],
                "relative_l2_max": 0.0,
                "relative_linf_max": 0.0,
                "absolute_linf_max": 0.0,
                "passed_values": [],
            },
        )
        group["components"].append(str(name))
        group["relative_l2_max"] = max(
            float(group["relative_l2_max"]),
            float(values["relative_l2"]),
        )
        group["relative_linf_max"] = max(
            float(group["relative_linf_max"]),
            float(values["relative_linf"]),
        )
        group["absolute_linf_max"] = max(
            float(group["absolute_linf_max"]),
            float(values["absolute_linf"]),
        )
        if "passed" in values:
            group["passed_values"].append(values["passed"])

    report: dict[str, dict[str, float | int | bool | None | list[str]]] = {}
    for group_name, values in groups.items():
        passed_values = list(values["passed_values"])
        if not passed_values:
            passed = None
        elif any(value is None for value in passed_values):
            passed = None
        else:
            passed = all(bool(value) for value in passed_values)
        components = list(values["components"])
        report[group_name] = {
            "component_count": len(components),
            "components": components,
            "relative_l2_max": float(values["relative_l2_max"]),
            "relative_linf_max": float(values["relative_linf_max"]),
            "absolute_linf_max": float(values["absolute_linf_max"]),
            "passed": passed,
        }
    return report


def component_diagnostics(numerical, reference, component_names) -> dict[str, dict[str, float]]:
    """Return per-column scaling and endpoint diagnostics."""

    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if numerical.shape != reference.shape:
        raise ValueError("numerical and reference arrays must have the same shape")
    if numerical.ndim != 2:
        raise ValueError("expected arrays with shape (n_times, n_components)")
    if len(component_names) != numerical.shape[1]:
        raise ValueError("component_names length must match the number of columns")

    diagnostics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(component_names):
        n = numerical[:, i]
        r = reference[:, i]
        denom = float(np.dot(r, r))
        scale = float(np.dot(n, r) / denom) if denom != 0.0 else float("nan")
        scaled_reference = scale * r if np.isfinite(scale) else r
        first_ratio = _safe_ratio(n[0], r[0])
        last_ratio = _safe_ratio(n[-1], r[-1])
        diagnostics[str(name)] = {
            "least_squares_scale_numerical_over_reference": scale,
            "relative_l2_after_optimal_scale": relative_l2(n, scaled_reference),
            "first_numerical": float(n[0]),
            "first_reference": float(r[0]),
            "first_ratio_numerical_over_reference": first_ratio,
            "last_numerical": float(n[-1]),
            "last_reference": float(r[-1]),
            "last_ratio_numerical_over_reference": last_ratio,
        }
    return diagnostics


def _component_group(name: str) -> str:
    component = str(name).split("@", 1)[0]
    if component.startswith("E"):
        return "electric"
    if component.startswith(("H", "B", "dB")):
        return "magnetic"
    return "other"


def fit_linear_response_components(
    base,
    components,
    reference,
    *,
    signs=None,
    mask=None,
) -> LinearResponseFit:
    """Fit weights in ``base + sum(sign_i * weight_i * component_i)``.

    This is a diagnostic helper for linear receiver decompositions.  For the
    Debye magnetic-recovery scans, ``base`` is the Ohmic-current response,
    ``components`` are polarization-current basis responses, and ``signs`` is
    usually ``[-1, -1, -1]``.
    """

    base = np.asarray(base, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if base.shape != reference.shape:
        raise ValueError("base and reference arrays must have the same shape")
    if base.ndim != 2:
        raise ValueError("expected arrays with shape (n_times, n_components)")
    if not components:
        raise ValueError("at least one response component is required")

    component_arrays = [np.asarray(component, dtype=float) for component in components]
    for component in component_arrays:
        if component.shape != base.shape:
            raise ValueError("all response components must match base shape")

    if signs is None:
        sign_values = np.ones(len(component_arrays), dtype=float)
    else:
        sign_values = np.asarray(signs, dtype=float)
        if sign_values.shape != (len(component_arrays),):
            raise ValueError("signs length must match the number of components")

    if mask is None:
        fit_mask = np.ones(base.shape[0], dtype=bool)
    else:
        fit_mask = np.asarray(mask, dtype=bool)
        if fit_mask.shape != (base.shape[0],):
            raise ValueError("mask length must match the number of time samples")
        if not np.any(fit_mask):
            raise ValueError("mask selects no samples")

    design = np.column_stack(
        [
            (sign * component[fit_mask]).ravel()
            for sign, component in zip(sign_values, component_arrays)
        ]
    )
    target = (reference[fit_mask] - base[fit_mask]).ravel()
    weights, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    fitted = base.copy()
    for sign, weight, component in zip(sign_values, weights, component_arrays):
        fitted += sign * float(weight) * component
    return LinearResponseFit(
        weights=np.asarray(weights, dtype=float),
        fitted=fitted,
        residual=fitted - reference,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)
