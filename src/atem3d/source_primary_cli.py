"""Fit source-primary exponential kernels from sampled validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .config import build_simulation
from .metrics import summarize_errors
from .source_primary import (
    fit_source_history_kernel_discrete_debye_basis,
    fit_source_history_kernel_basis,
    fit_time_dependent_source_primary_kernel,
    normalized_source_primary_scale,
    scan_exponential_source_primary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit H_source*exp(-t/tau) source-primary kernels from two sampled "
            "validation JSON reports."
        )
    )
    parser.add_argument("ip_report", type=Path, help="Validation report with IP samples")
    parser.add_argument("noip_report", type=Path, help="Validation report with no-IP samples")
    parser.add_argument(
        "--target",
        choices=[
            "reference_delta",
            "numerical_delta",
            "ip_residual",
            "noip_residual",
            "delta_residual",
        ],
        default="reference_delta",
        help=(
            "Fit ip-noip reference difference, ip-noip numerical difference, "
            "reference-minus-numerical residual from one report, or the "
            "IP-specific residual difference"
        ),
    )
    parser.add_argument(
        "--component-prefix",
        default="Hz",
        help="Only fit sampled component names with this prefix",
    )
    parser.add_argument(
        "--source-basis",
        choices=["wire", "edge_current"],
        default="wire",
        help=(
            "Static source field basis used in the fit: analytic finite wire "
            "or FV edge-current Biot recovery"
        ),
    )
    parser.add_argument(
        "--tau-candidates",
        nargs="+",
        type=float,
        default=None,
        help="Candidate exponential kernel time constants in seconds",
    )
    parser.add_argument(
        "--tau-debye",
        type=float,
        default=None,
        help="Debye tau used with --tau-factors when --tau-candidates is omitted",
    )
    parser.add_argument(
        "--tau-factors",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        help="Kernel tau factors multiplied by --tau-debye",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="HDF5 result containing config_yaml; defaults to ip_report.result_path",
    )
    parser.add_argument(
        "--delta-sigma",
        type=float,
        default=None,
        help="Polarization contrast for scale/(mu*delta_sigma*L^2); default infers at source midpoint",
    )
    parser.add_argument(
        "--source-length",
        type=float,
        default=None,
        help="Source length for scale/(mu*delta_sigma*L^2); default infers from config",
    )
    parser.add_argument(
        "--kernel-basis-tau",
        type=float,
        default=None,
        help=(
            "Tau used to fit empirical K(t) to (t/tau)^p exp(-t/tau); "
            "defaults to the single Debye tau in the config when available"
        ),
    )
    parser.add_argument(
        "--kernel-basis-powers",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Integer powers p for the empirical K(t) basis fit",
    )
    parser.add_argument(
        "--kernel-basis-include-constant",
        action="store_true",
        help="Include a constant term in the empirical K(t) basis fit",
    )
    parser.add_argument(
        "--kernel-basis-discrete",
        action="store_true",
        help=(
            "Also fit empirical K(t) to backward-Euler Debye history bases "
            "computed from the config time_steps"
        ),
    )
    parser.add_argument(
        "--kernel-basis-max-order",
        type=int,
        default=1,
        help="Maximum same-pole cascade order for --kernel-basis-discrete",
    )
    parser.add_argument(
        "--kernel-basis-time-atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for matching sample times to config time nodes",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    ip = _load_sampled_report(args.ip_report, args.component_prefix)
    noip = _load_sampled_report(args.noip_report, args.component_prefix)
    _validate_sample_alignment(ip, noip)
    target = _target_data(ip, noip, args.target)

    result_path = args.result or _report_result_path(args.ip_report, ip.payload)
    config = _load_config_from_result(result_path)
    source_amplitudes, normalization = _source_amplitudes_and_normalization(
        config,
        ip.names,
        delta_sigma=args.delta_sigma,
        source_length=args.source_length,
        source_basis=args.source_basis,
    )
    kernel_taus = _kernel_taus(args, config)
    scan = scan_exponential_source_primary(
        ip.times,
        target,
        source_amplitudes,
        kernel_taus=kernel_taus,
        component_names=ip.names,
    )
    empirical = fit_time_dependent_source_primary_kernel(
        ip.times,
        target,
        source_amplitudes,
        component_names=ip.names,
    )
    basis_fit = _kernel_basis_fit(args, config, empirical.kernel, ip.times)
    discrete_basis_fit = _kernel_discrete_basis_fit(
        args,
        config,
        empirical.kernel,
        ip.times,
    )
    report = _scan_report(
        scan,
        empirical=empirical,
        basis_fit=basis_fit,
        discrete_basis_fit=discrete_basis_fit,
        target=args.target,
        component_names=ip.names,
        source_amplitudes=source_amplitudes,
        normalization=normalization,
        source_basis=args.source_basis,
        residual_base=_residual_base_report(ip, noip, args.target),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"best_tau={scan.best.kernel_tau:g} "
        f"scale={scan.best.scale:.12g} "
        f"relative_l2={scan.best.relative_l2:.6e}"
    )
    print(f"wrote {args.output}")
    return 0


class _SampledReport:
    def __init__(
        self,
        *,
        payload: dict[str, Any],
        names: list[str],
        times: np.ndarray,
        numerical: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        self.payload = payload
        self.names = names
        self.times = times
        self.numerical = numerical
        self.reference = reference


def _load_sampled_report(path: Path, component_prefix: str) -> _SampledReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"{path} does not contain sampled validation data")
    names = [
        str(name)
        for name in samples
        if str(name).startswith(component_prefix)
    ]
    names.sort()
    if not names:
        raise ValueError(f"{path} contains no samples with prefix {component_prefix!r}")

    times = None
    numerical_columns = []
    reference_columns = []
    for name in names:
        rows = samples[name]
        column_times = np.asarray([row["time"] for row in rows], dtype=float)
        if times is None:
            times = column_times
        elif not np.allclose(times, column_times, rtol=0.0, atol=0.0):
            raise ValueError(f"sample times are inconsistent for {name}")
        numerical_columns.append([row["numerical"] for row in rows])
        reference_columns.append([row["reference"] for row in rows])

    return _SampledReport(
        payload=payload,
        names=names,
        times=np.asarray(times, dtype=float),
        numerical=np.asarray(numerical_columns, dtype=float).T,
        reference=np.asarray(reference_columns, dtype=float).T,
    )


def _validate_sample_alignment(ip: _SampledReport, noip: _SampledReport) -> None:
    if ip.names != noip.names:
        raise ValueError("IP and no-IP reports must contain the same sampled components")
    if ip.times.shape != noip.times.shape or not np.allclose(ip.times, noip.times):
        raise ValueError("IP and no-IP reports must contain the same sample times")


def _target_data(ip: _SampledReport, noip: _SampledReport, target: str) -> np.ndarray:
    if target == "reference_delta":
        return ip.reference - noip.reference
    if target == "numerical_delta":
        return ip.numerical - noip.numerical
    if target == "ip_residual":
        return ip.reference - ip.numerical
    if target == "noip_residual":
        return noip.reference - noip.numerical
    if target == "delta_residual":
        return (ip.reference - noip.reference) - (ip.numerical - noip.numerical)
    raise ValueError(f"unsupported target: {target}")


def _report_result_path(report_path: Path, payload: dict[str, Any]) -> Path:
    raw = payload.get("result_path")
    if not raw:
        raise ValueError("--result is required when the report has no result_path")
    result = Path(raw)
    if result.exists():
        return result
    candidate = report_path.parent / result
    if candidate.exists():
        return candidate
    return result


def _load_config_from_result(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        raw = h5.attrs.get("config_yaml")
    if raw is None:
        raise ValueError(f"{path} does not contain config_yaml")
    return yaml.safe_load(raw)


def _source_amplitudes_and_normalization(
    config: dict[str, Any],
    names: list[str],
    *,
    delta_sigma: float | None,
    source_length: float | None,
    source_basis: str,
) -> tuple[np.ndarray, dict[str, float | None]]:
    sim = build_simulation(config)
    receiver_by_name = _receiver_name_map(config, sim.receivers)
    amplitudes = []
    for name in names:
        if name not in receiver_by_name:
            raise ValueError(f"receiver {name!r} was not found in config")
        receiver = receiver_by_name[name]
        amplitudes.append(
            _source_field_component(
                sim,
                receiver.location,
                receiver.component,
                source_basis=source_basis,
            )
        )

    inferred_length = source_length
    if inferred_length is None and sim.sources:
        source = sim.sources[0]
        inferred_length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))

    inferred_delta = delta_sigma
    if inferred_delta is None and sim.sources and sim.ip_model.terms:
        source = sim.sources[0]
        cell_index = sim._source_midpoint_cell_index(source)
        inferred_delta = float(
            sum(term.delta_sigma[cell_index] for term in sim.ip_model.terms)
        )

    normalization = {
        "delta_sigma": None if inferred_delta is None else float(inferred_delta),
        "source_length": None if inferred_length is None else float(inferred_length),
        "mu": float(sim.mu),
    }
    return np.asarray(amplitudes, dtype=float), normalization


def _receiver_name_map(config: dict[str, Any], receivers) -> dict[str, Any]:
    names: list[str] = []
    line = config.get("receiver_line")
    if line:
        components = [str(component) for component in line["components"]]
        for x in line["x"]:
            for component in components:
                names.append(f"{component}@x={float(x):g}")
    else:
        names = [f"{receiver.component}@{index}" for index, receiver in enumerate(receivers)]
    if len(names) != len(receivers):
        raise ValueError("generated receiver names do not match configured receivers")
    return dict(zip(names, receivers))


def _source_field_component(sim, location, component: str, *, source_basis: str) -> float:
    if component[0].upper() not in {"H", "B"}:
        return 0.0
    source_basis = str(source_basis).strip().lower()
    if source_basis not in {"wire", "edge_current"}:
        raise ValueError("source_basis must be 'wire' or 'edge_current'")
    axis = {"x": 0, "y": 1, "z": 2}[component[-1].lower()]
    if source_basis == "edge_current":
        from .magnetic_recovery import biot_savart_h_from_edge_current_moments  # noqa: PLC0415

        locations = np.asarray(location, dtype=float).reshape(1, 3)
        field = np.zeros((1, 3), dtype=float)
        for source in sim.sources:
            source_vector = source.initial_edge_vector(sim.mesh)
            if np.linalg.norm(source_vector) == 0.0:
                continue
            field += biot_savart_h_from_edge_current_moments(
                sim.mesh,
                source_vector,
                locations,
            )
        value = float(field[0, axis])
        if component[0].upper() == "B":
            value *= float(sim.mu)
        return value

    field = np.zeros(3, dtype=float)
    try:
        from geoana.em.static import LineCurrentWholeSpace  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError("source-primary fitting requires geoana") from err
    for source in sim.sources:
        current = float(source.current * source.waveform.initial_value())
        if current == 0.0:
            continue
        line_current = LineCurrentWholeSpace(source.locations, current=current, mu=sim.mu)
        field += np.asarray(
            line_current.magnetic_field(np.asarray(location, dtype=float).reshape(1, 3))[0],
            dtype=float,
        )
    value = float(field[axis])
    if component[0].upper() == "B":
        value *= float(sim.mu)
    return value


def _kernel_taus(args: argparse.Namespace, config: dict[str, Any]) -> np.ndarray:
    if args.tau_candidates is not None:
        taus = np.asarray(args.tau_candidates, dtype=float)
    else:
        tau_debye = args.tau_debye
        if tau_debye is None:
            tau_debye = _single_config_tau(config)
        factors = np.asarray(args.tau_factors, dtype=float)
        taus = float(tau_debye) * factors
    if taus.ndim != 1 or taus.size == 0 or np.any(taus <= 0.0):
        raise ValueError("candidate kernel taus must be positive")
    return taus


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
        raise ValueError("--tau-debye is required unless the config has exactly one Debye tau")
    return unique[0]


def _scan_report(
    scan,
    *,
    empirical,
    basis_fit,
    discrete_basis_fit,
    target: str,
    component_names: list[str],
    source_amplitudes: np.ndarray,
    normalization: dict[str, float | None],
    source_basis: str,
    residual_base: _SampledReport | None,
) -> dict[str, Any]:
    delta_sigma = normalization.get("delta_sigma")
    source_length = normalization.get("source_length")
    mu = normalization.get("mu")
    fits = {}
    for fit in scan.fits:
        item = {
            "scale": float(fit.scale),
            "relative_l2": float(fit.relative_l2),
            "components": {name: float(value) for name, value in fit.components.items()},
        }
        if delta_sigma is not None and source_length is not None and mu is not None:
            item["scale_over_mu_delta_l2"] = normalized_source_primary_scale(
                fit.scale,
                delta_sigma=float(delta_sigma),
                source_length=float(source_length),
                mu=float(mu),
            )
        fits[f"tau_{fit.kernel_tau:g}"] = item

    report = {
        "diagnostic_only": True,
        "target": target,
        "source_basis": source_basis,
        "component_names": component_names,
        "source_amplitudes": {
            name: float(value) for name, value in zip(component_names, source_amplitudes)
        },
        "normalization": normalization,
        "best_tau": float(scan.best.kernel_tau),
        "best_key": f"tau_{scan.best.kernel_tau:g}",
        "fits": fits,
        "empirical_kernel": _empirical_kernel_report(
            empirical,
            basis_fit=basis_fit,
            discrete_basis_fit=discrete_basis_fit,
            normalization=normalization,
        ),
    }
    if residual_base is not None:
        corrected = residual_base.numerical + empirical.fitted
        report["empirical_residual_correction"] = {
            "diagnostic_only": True,
            "formula": "corrected_numerical = numerical + K(t) * H_source",
            "components": summarize_errors(
                corrected,
                residual_base.reference,
                component_names,
            ),
        }
    return report


def _residual_base_report(
    ip: _SampledReport,
    noip: _SampledReport,
    target: str,
) -> _SampledReport | None:
    if target == "ip_residual":
        return ip
    if target == "noip_residual":
        return noip
    return None


def _kernel_basis_fit(
    args: argparse.Namespace,
    config: dict[str, Any],
    kernel: np.ndarray,
    times: np.ndarray,
):
    tau = _kernel_basis_tau(args, config)
    if tau is None:
        return None
    return fit_source_history_kernel_basis(
        times,
        kernel,
        tau=float(tau),
        powers=args.kernel_basis_powers,
        include_constant=bool(args.kernel_basis_include_constant),
    )


def _kernel_discrete_basis_fit(
    args: argparse.Namespace,
    config: dict[str, Any],
    kernel: np.ndarray,
    times: np.ndarray,
):
    if not args.kernel_basis_discrete:
        return None
    tau = _kernel_basis_tau(args, config)
    if tau is None:
        raise ValueError(
            "--kernel-basis-tau is required for --kernel-basis-discrete unless "
            "the config has exactly one Debye tau"
        )
    sim = build_simulation(config)
    return fit_source_history_kernel_discrete_debye_basis(
        sim.time_steps,
        times,
        kernel,
        tau=float(tau),
        max_order=int(args.kernel_basis_max_order),
        include_constant=bool(args.kernel_basis_include_constant),
        time_atol=float(args.kernel_basis_time_atol),
    )


def _kernel_basis_tau(args: argparse.Namespace, config: dict[str, Any]) -> float | None:
    if args.kernel_basis_tau is not None:
        return float(args.kernel_basis_tau)
    try:
        return _single_config_tau(config)
    except ValueError:
        return None


def _empirical_kernel_report(
    empirical,
    *,
    basis_fit,
    discrete_basis_fit,
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    delta_sigma = normalization.get("delta_sigma")
    source_length = normalization.get("source_length")
    mu = normalization.get("mu")
    report = {
        "relative_l2": float(empirical.relative_l2),
        "kernel": [float(value) for value in empirical.kernel],
        "components": {
            name: float(value) for name, value in empirical.components.items()
        },
    }
    if delta_sigma is not None and source_length is not None and mu is not None:
        report["normalized_kernel_over_mu_delta_l2"] = [
            normalized_source_primary_scale(
                value,
                delta_sigma=float(delta_sigma),
                source_length=float(source_length),
                mu=float(mu),
            )
            for value in empirical.kernel
        ]
    if basis_fit is not None:
        report["basis_fit"] = _basis_fit_report(
            basis_fit,
            basis_kind="continuous_exp_polynomial",
            normalization=normalization,
        )
    if discrete_basis_fit is not None:
        report["discrete_basis_fit"] = _basis_fit_report(
            discrete_basis_fit,
            basis_kind="discrete_be_debye_history",
            normalization=normalization,
        )
    return report


def _basis_fit_report(
    basis_fit,
    *,
    basis_kind: str,
    normalization: dict[str, float | None],
) -> dict[str, Any]:
    delta_sigma = normalization.get("delta_sigma")
    source_length = normalization.get("source_length")
    mu = normalization.get("mu")
    report = {
        "diagnostic_only": True,
        "basis_kind": basis_kind,
        "tau": float(basis_fit.tau),
        "powers": [int(value) for value in basis_fit.powers],
        "include_constant": bool(basis_fit.include_constant),
        "basis_labels": list(basis_fit.basis_labels),
        "coefficients": [float(value) for value in basis_fit.coefficients],
        "coefficient_by_label": {
            label: float(value)
            for label, value in zip(basis_fit.basis_labels, basis_fit.coefficients)
        },
        "relative_l2": float(basis_fit.relative_l2),
        "fitted": [float(value) for value in basis_fit.fitted],
        "residual": [float(value) for value in basis_fit.residual],
    }
    if delta_sigma is not None and source_length is not None and mu is not None:
        report["coefficients_over_mu_delta_l2"] = [
            normalized_source_primary_scale(
                value,
                delta_sigma=float(delta_sigma),
                source_length=float(source_length),
                mu=float(mu),
            )
            for value in basis_fit.coefficients
        ]
    return report


if __name__ == "__main__":
    raise SystemExit(main())
