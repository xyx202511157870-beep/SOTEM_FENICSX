"""CLI for projecting Ampere residuals onto the grounded-source vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .ampere_source_projection import ampere_source_projection, hj_ampere_source_projection
from .config import build_simulation
from .source_primary import (
    fit_source_history_kernel_discrete_debye_basis,
    normalized_source_primary_scale,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project FV Ampere residuals onto the initial grounded-source vector."
    )
    parser.add_argument("result", type=Path, help="ATEM3D HDF5 result with EB e/b or H/J e/h")
    parser.add_argument(
        "--subtract-baseline",
        type=Path,
        default=None,
        help="Optional baseline EB HDF5 result whose projection is subtracted",
    )
    parser.add_argument("--include-t0", action="store_true", help="Include the t=0 node")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Debye tau for optional discrete-basis fit; defaults to single config tau",
    )
    parser.add_argument(
        "--kernel-basis-discrete",
        action="store_true",
        help="Fit the projected source coefficient to BE Debye history bases",
    )
    parser.add_argument(
        "--kernel-basis-max-order",
        type=int,
        default=1,
        help="Maximum same-pole cascade order for --kernel-basis-discrete",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    sim, projection, config, formulation = _load_projection(
        args.result,
        include_t0=bool(args.include_t0),
    )
    normalization = _source_normalization(sim)
    report = _projection_report(projection, normalization=normalization)
    report["result_path"] = str(args.result)
    report["formulation"] = formulation
    report["include_t0"] = bool(args.include_t0)
    fit_times = projection.times
    fit_coefficients = projection.coefficients

    if args.subtract_baseline is not None:
        _, baseline, _, baseline_formulation = _load_projection(
            args.subtract_baseline,
            include_t0=bool(args.include_t0),
        )
        if formulation != baseline_formulation:
            raise ValueError("baseline formulation does not match result formulation")
        if projection.times.shape != baseline.times.shape or not np.allclose(
            projection.times,
            baseline.times,
        ):
            raise ValueError("baseline projection times do not match result projection times")
        delta = projection.coefficients - baseline.coefficients
        report["baseline_result_path"] = str(args.subtract_baseline)
        report["baseline_projection"] = _projection_values_report(
            baseline,
            normalization=normalization,
        )
        report["projection_delta"] = _coefficient_series_report(
            projection.times,
            delta,
            normalization=normalization,
        )
        fit_times = projection.times
        fit_coefficients = delta

    if args.kernel_basis_discrete:
        tau = args.tau
        if tau is None:
            tau = _single_config_tau(config)
        fit = fit_source_history_kernel_discrete_debye_basis(
            sim.time_steps,
            fit_times,
            fit_coefficients,
            tau=float(tau),
            max_order=int(args.kernel_basis_max_order),
        )
        report["discrete_basis_fit"] = _basis_fit_report(
            fit,
            normalization=normalization,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _load_projection(path: Path, *, include_t0: bool):
    with h5py.File(path, "r") as h5:
        if "e" not in h5:
            raise ValueError("result file must contain an e dataset")
        times = h5["times"][:]
        electric_fields = h5["e"][:]
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))
        formulation = str(
            h5.attrs.get("formulation", config.get("formulation", "eb"))
        ).lower()
        if formulation == "hj" or ("h" in h5 and "b" not in h5):
            if "h" not in h5:
                raise ValueError("H/J result file must contain an h dataset")
            magnetic_fields = h5["h"][:]
            formulation = "hj"
        else:
            if "b" not in h5:
                raise ValueError("EB result file must contain a b dataset")
            magnetic_fluxes = h5["b"][:]
            formulation = "eb"

    sim = build_simulation(config)
    if formulation == "hj":
        projection = hj_ampere_source_projection(
            sim,
            times,
            electric_fields,
            magnetic_fields,
            include_t0=include_t0,
        )
    else:
        projection = ampere_source_projection(
            sim,
            times,
            electric_fields,
            magnetic_fluxes,
            include_t0=include_t0,
        )
    return sim, projection, config, formulation


def _projection_report(projection, *, normalization: dict[str, float | None]) -> dict[str, Any]:
    report = {
        "diagnostic_only": True,
        "projection": _projection_values_report(
            projection,
            normalization=normalization,
        ),
        "normalization": normalization,
    }
    return report


def _projection_values_report(
    projection,
    *,
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    report = {
        "n": int(projection.times.size),
        "times": [float(value) for value in projection.times],
        "coefficients": [float(value) for value in projection.coefficients],
        "residual_norms": [float(value) for value in projection.residual_norms],
        "relative_residual_norms": [
            float(value) for value in projection.relative_residual_norms
        ],
        "source_norm": float(projection.source_norm),
    }
    if _has_normalization(normalization):
        report["coefficients_over_mu_delta_l2"] = _normalized_coefficients(
            projection.coefficients,
            normalization=normalization,
        )
    return report


def _coefficient_series_report(
    times,
    coefficients,
    *,
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    report = {
        "n": int(len(times)),
        "times": [float(value) for value in times],
        "coefficients": [float(value) for value in coefficients],
    }
    if _has_normalization(normalization):
        report["coefficients_over_mu_delta_l2"] = _normalized_coefficients(
            coefficients,
            normalization=normalization,
        )
    return report


def _basis_fit_report(fit, *, normalization: dict[str, float | None]) -> dict[str, Any]:
    report = {
        "basis_kind": "discrete_be_debye_history",
        "tau": float(fit.tau),
        "basis_labels": list(fit.basis_labels),
        "coefficients": [float(value) for value in fit.coefficients],
        "relative_l2": float(fit.relative_l2),
        "fitted": [float(value) for value in fit.fitted],
        "residual": [float(value) for value in fit.residual],
    }
    if _has_normalization(normalization):
        report["coefficients_over_mu_delta_l2"] = _normalized_coefficients(
            fit.coefficients,
            normalization=normalization,
        )
    return report


def _normalized_coefficients(coefficients, *, normalization: dict[str, float | None]):
    return [
        normalized_source_primary_scale(
            value,
            delta_sigma=float(normalization["delta_sigma"]),
            source_length=float(normalization["source_length"]),
            mu=float(normalization["mu"]),
        )
        for value in coefficients
    ]


def _source_normalization(sim) -> dict[str, float | None]:
    source_length = None
    delta_sigma = None
    if sim.sources:
        source = sim.sources[0]
        source_length = float(((source.locations[-1] - source.locations[0]) ** 2).sum() ** 0.5)
        if sim.ip_model.terms:
            cell_index = sim._source_midpoint_cell_index(source)
            delta_sigma = float(
                sum(term.delta_sigma[cell_index] for term in sim.ip_model.terms)
            )
    return {
        "delta_sigma": delta_sigma,
        "source_length": source_length,
        "mu": float(sim.mu),
    }


def _has_normalization(normalization: dict[str, float | None]) -> bool:
    return (
        normalization.get("delta_sigma") is not None
        and normalization.get("source_length") is not None
        and normalization.get("mu") is not None
    )


def _single_config_tau(config: dict[str, Any]) -> float:
    taus = []
    model = config.get("model", {})
    if "layers" in model:
        for layer in model["layers"]:
            for term in layer.get("debye_terms", []):
                taus.append(float(term["tau"]))
    else:
        for term in model.get("debye_terms", []):
            taus.append(float(term["tau"]))
    unique = sorted(set(taus))
    if len(unique) != 1:
        raise ValueError("--tau is required unless the config has exactly one Debye tau")
    return unique[0]


if __name__ == "__main__":
    raise SystemExit(main())
