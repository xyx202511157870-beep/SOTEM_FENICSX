"""empymod-only source-primary tau-transfer diagnostics."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .config import build_simulation
from .empymod_compare import (
    EmpymodSurvey,
    build_empymod_survey_from_config,
    make_debye_resistivity_model,
    run_empymod_reference,
)
from .source_primary import (
    fit_time_dependent_source_primary_kernel,
    normalized_source_primary_scale,
    scan_exponential_source_primary,
)


ReferenceRunner = Callable[..., np.ndarray]
SourceFieldRunner = Callable[
    [dict[str, Any], list[str], list[tuple[tuple[float, float, float], str]]],
    tuple[np.ndarray, float],
]


def run_empymod_source_primary_tau_scan(
    config: dict[str, Any],
    *,
    depths: Sequence[float],
    resistivities: Sequence[float],
    delta_sigma: Sequence[float],
    tau_values: Sequence[float],
    kernel_factors: Sequence[float],
    component_prefix: str = "Hz",
    signal: int | None = -1,
    positive_times_only: bool = True,
    skip_positive_times: int = 0,
    reference_runner: ReferenceRunner = run_empymod_reference,
    source_field_runner: SourceFieldRunner | None = None,
    empymod_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an empymod-only Debye tau scan and fit source-primary kernels.

    This diagnostic intentionally runs all empymod references before evaluating
    the static source field.  In the Windows/Conda environment used for this
    project, importing geoana before finite-source empymod time-domain calls can
    make those calls hang for larger ``srcpts`` values.
    """

    depths = [float(depth) for depth in depths]
    resistivities = np.asarray(resistivities, dtype=float)
    delta_sigma = np.asarray(delta_sigma, dtype=float)
    tau_values = np.asarray(tau_values, dtype=float)
    kernel_factors = np.asarray(kernel_factors, dtype=float)
    _validate_scan_inputs(depths, resistivities, delta_sigma, tau_values, kernel_factors)
    if skip_positive_times < 0:
        raise ValueError("skip_positive_times must be nonnegative")

    sim = build_simulation(config)
    times = np.asarray(sim.times, dtype=float)
    mask = np.ones(times.shape, dtype=bool)
    if positive_times_only:
        mask = times > 0.0
    if skip_positive_times:
        selected = np.flatnonzero(mask)
        mask[selected[:skip_positive_times]] = False
    if not np.any(mask):
        raise ValueError("time selection produced no samples")
    selected_times = times[mask]

    base_survey, all_names = build_empymod_survey_from_config(
        config,
        times=selected_times,
        depths=depths,
        resistivities=resistivities.tolist(),
        signal=signal,
    )
    receiver_components = list(base_survey.receiver_components or [])
    selected_indices = [
        index
        for index, name in enumerate(all_names)
        if str(name).startswith(component_prefix)
    ]
    if not selected_indices:
        raise ValueError(f"no receiver components match prefix {component_prefix!r}")
    names = [all_names[index] for index in selected_indices]
    selected_receiver_components = [receiver_components[index] for index in selected_indices]
    base_survey = _replace_survey_receivers(base_survey, selected_receiver_components)

    kwargs = dict(empymod_kwargs or {})
    noip = np.asarray(reference_runner(base_survey, **kwargs), dtype=float)
    sigma0 = 1.0 / resistivities
    sigma_infinity = sigma0 + delta_sigma
    targets: dict[float, np.ndarray] = {}
    for tau in tau_values:
        ip_resistivities = make_debye_resistivity_model(
            sigma_infinity,
            [{"delta_sigma": delta_sigma, "tau": float(tau)}],
        )
        ip_survey = EmpymodSurvey(
            source_start=base_survey.source_start,
            source_end=base_survey.source_end,
            receiver_locations=base_survey.receiver_locations,
            components=base_survey.components,
            times=base_survey.times,
            depths=base_survey.depths,
            resistivities=ip_resistivities,
            strength=base_survey.strength,
            signal=base_survey.signal,
            receiver_components=base_survey.receiver_components,
            coordinate_system=base_survey.coordinate_system,
        )
        targets[float(tau)] = np.asarray(reference_runner(ip_survey, **kwargs), dtype=float) - noip

    if source_field_runner is None:
        source_amplitudes, source_length = _source_field_amplitudes(
            config,
            names,
            selected_receiver_components,
        )
    else:
        source_amplitudes, source_length = source_field_runner(
            config,
            names,
            selected_receiver_components,
        )
    source_amplitudes = np.asarray(source_amplitudes, dtype=float)
    if source_amplitudes.shape != (len(names),):
        raise ValueError("source_field_runner must return one amplitude per selected receiver")

    normalization_delta = _first_positive(delta_sigma)
    report = {
        "description": (
            "empymod-only Debye tau source-primary scan; empymod references are "
            "computed before static source-field evaluation"
        ),
        "diagnostic_only": True,
        "times": {
            "start": float(selected_times[0]),
            "stop": float(selected_times[-1]),
            "n": int(selected_times.size),
            "positive_times_only": bool(positive_times_only),
            "skip_positive_times": int(skip_positive_times),
        },
        "receiver_names": names,
        "source_amplitudes": {
            name: float(value) for name, value in zip(names, source_amplitudes)
        },
        "normalization": {
            "delta_sigma": float(normalization_delta),
            "source_length": float(source_length),
        },
        "sigma0": sigma0.tolist(),
        "sigma_infinity": sigma_infinity.tolist(),
        "delta_sigma": delta_sigma.tolist(),
        "tau_values": {},
    }
    for tau, target in targets.items():
        scan = scan_exponential_source_primary(
            selected_times,
            target,
            source_amplitudes,
            kernel_taus=kernel_factors * tau,
            component_names=names,
        )
        report["tau_values"][f"{tau:g}"] = _tau_report(
            scan,
            empirical=fit_time_dependent_source_primary_kernel(
                selected_times,
                target,
                source_amplitudes,
                component_names=names,
            ),
            normalization_delta=normalization_delta,
            source_length=float(source_length),
        )
    return report


def _replace_survey_receivers(
    survey: EmpymodSurvey,
    receiver_components,
) -> EmpymodSurvey:
    return EmpymodSurvey(
        source_start=survey.source_start,
        source_end=survey.source_end,
        receiver_locations=survey.receiver_locations,
        components=survey.components,
        times=survey.times,
        depths=survey.depths,
        resistivities=survey.resistivities,
        strength=survey.strength,
        signal=survey.signal,
        receiver_components=receiver_components,
        coordinate_system=survey.coordinate_system,
    )


def _tau_report(
    scan,
    *,
    empirical,
    normalization_delta: float,
    source_length: float,
) -> dict[str, Any]:
    fits = {}
    for fit in scan.fits:
        fits[f"tau_kernel_{fit.kernel_tau:g}"] = {
            "scale": float(fit.scale),
            "relative_l2": float(fit.relative_l2),
            "scale_over_mu0_delta_l2": normalized_source_primary_scale(
                fit.scale,
                delta_sigma=normalization_delta,
                source_length=source_length,
            ),
            "components": {
                name: float(value) for name, value in fit.components.items()
            },
        }
    return {
        "best_key": f"tau_kernel_{scan.best.kernel_tau:g}",
        "best_tau_kernel": float(scan.best.kernel_tau),
        "best_relative_l2": float(scan.best.relative_l2),
        "fits": fits,
        "empirical_kernel": {
            "relative_l2": float(empirical.relative_l2),
            "kernel": [float(value) for value in empirical.kernel],
            "normalized_kernel_over_mu0_delta_l2": [
                normalized_source_primary_scale(
                    value,
                    delta_sigma=normalization_delta,
                    source_length=source_length,
                )
                for value in empirical.kernel
            ],
            "components": {
                name: float(value) for name, value in empirical.components.items()
            },
        },
    }


def _source_field_amplitudes(
    config: dict[str, Any],
    names: list[str],
    receiver_components: list[tuple[tuple[float, float, float], str]],
) -> tuple[np.ndarray, float]:
    # Keep this import local and late; see run_empymod_source_primary_tau_scan.
    from geoana.em.static import LineCurrentWholeSpace  # noqa: PLC0415

    sim = build_simulation(config)
    source = sim.sources[0]
    source_length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    fields = []
    for location, component in receiver_components:
        if component[0].upper() not in {"H", "B"}:
            fields.append(0.0)
            continue
        axis = {"x": 0, "y": 1, "z": 2}[component[-1].lower()]
        total = np.zeros(3, dtype=float)
        for source in sim.sources:
            current = float(source.current * source.waveform.initial_value())
            if current == 0.0:
                continue
            line_current = LineCurrentWholeSpace(
                source.locations,
                current=current,
                mu=sim.mu,
            )
            total += np.asarray(
                line_current.magnetic_field(
                    np.asarray(location, dtype=float).reshape(1, 3)
                )[0],
                dtype=float,
            )
        value = float(total[axis])
        if component[0].upper() == "B":
            value *= float(sim.mu)
        fields.append(value)
    return np.asarray(fields, dtype=float), source_length


def _validate_scan_inputs(
    depths: Sequence[float],
    resistivities: np.ndarray,
    delta_sigma: np.ndarray,
    tau_values: np.ndarray,
    kernel_factors: np.ndarray,
) -> None:
    if resistivities.shape != (len(depths) + 1,):
        raise ValueError("len(resistivities) must equal len(depths) + 1")
    if delta_sigma.shape != resistivities.shape:
        raise ValueError("delta_sigma must have one value per empymod layer")
    if np.any(resistivities <= 0.0):
        raise ValueError("resistivities must be positive")
    if np.any(delta_sigma < 0.0):
        raise ValueError("delta_sigma must be nonnegative")
    if tau_values.ndim != 1 or tau_values.size == 0 or np.any(tau_values <= 0.0):
        raise ValueError("tau_values must be a non-empty positive sequence")
    if (
        kernel_factors.ndim != 1
        or kernel_factors.size == 0
        or np.any(kernel_factors <= 0.0)
    ):
        raise ValueError("kernel_factors must be a non-empty positive sequence")
    if _first_positive(delta_sigma) <= 0.0:
        raise ValueError("at least one delta_sigma value must be positive")


def _first_positive(values: np.ndarray) -> float:
    positives = np.asarray(values, dtype=float)[np.asarray(values, dtype=float) > 0.0]
    if positives.size == 0:
        return 0.0
    return float(positives[0])
