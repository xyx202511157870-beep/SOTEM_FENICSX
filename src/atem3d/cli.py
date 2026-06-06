"""Command-line entry points for ATEM3D examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .materials.prony import DebyeTerm, PronyConductivity
from .validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    if argv and argv[0] == "run":
        return _main_run(argv[1:])
    if argv and argv[0] == "plot":
        return _main_plot(argv[1:])
    if argv and argv[0] == "validate-secondary":
        return _main_validate_secondary(argv[1:])
    if argv and argv[0] == "acceptance-report":
        return _main_acceptance_report(argv[1:])
    if argv and argv[0] == "corrected-model-spec":
        return _main_corrected_model_spec(argv[1:])
    if argv and argv[0] == "corrected-model-run":
        return _main_corrected_model_run(argv[1:])
    if argv and argv[0] in {"validate-noip-3comp", "validate-ip-3comp"}:
        return _main_validate(argv)
    return _main_run(argv)


def _main_run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a grounded-wire TDEM-IP simulation.")
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/result.h5"))
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Store receiver data without full field histories",
    )
    args = parser.parse_args(argv)

    from .config import build_simulation, load_config
    from .io import save_result_hdf5

    config = load_config(args.config)
    simulation = build_simulation(config)
    if args.data_only:
        if not hasattr(simulation, "run_data_only"):
            parser.error("--data-only is only available for simulations with run_data_only()")
        result = simulation.run_data_only()
    else:
        result = simulation.run()
    save_result_hdf5(args.output, result, config)
    print(f"wrote {args.output}")
    print(f"time nodes: {result.times.size}; receivers: {result.data.shape[1]}")
    if args.data_only:
        print("field histories: not saved")
    return 0


def _main_validate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write three-component validation artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-noip-3comp", "validate-ip-3comp"):
        sub = subparsers.add_parser(command)
        sub.add_argument("config", type=Path, help="YAML validation configuration")
    args = parser.parse_args(argv)

    config = _load_yaml(args.config)
    case_type = "ip" if args.command == "validate-ip-3comp" else "noip"
    case = _validation_case_from_config(config, case_type=case_type)
    summary = write_three_component_validation_artifacts(case)
    print(f"wrote {Path(case.output_dir)}")
    print(f"case_type: {summary['case_type']}; pass_all_components: {summary['pass_all_components']}")
    return 0


def _main_plot(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Regenerate three-component validation plots.")
    parser.add_argument("run_dir", type=Path, help="Directory containing validation CSV artifacts")
    args = parser.parse_args(argv)

    from .validation_3comp import _plot_comparison, _plot_errors

    run_dir = Path(args.run_dir)
    component_names = _response_component_names(run_dir / "predictions.csv")
    pred_times, predictions = _read_response_csv(run_dir / "predictions.csv", component_names)
    ref_times, reference = _read_response_csv(run_dir / "reference_empymod_or_1d.csv", component_names)
    if pred_times.shape != ref_times.shape or not np.allclose(pred_times, ref_times, rtol=0.0, atol=0.0):
        raise ValueError("prediction and reference CSV time_obs columns must match exactly")
    errors = _read_errors_csv(run_dir / "errors.csv")
    _plot_comparison(run_dir / "comparison_3comp.png", pred_times, predictions, reference, component_names)
    _plot_errors(run_dir / "error_curves_3comp.png", errors, component_names, threshold=0.05)
    print(f"wrote {run_dir / 'comparison_3comp.png'}")
    print(f"wrote {run_dir / 'error_curves_3comp.png'}")
    return 0


def _main_acceptance_report(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize no-IP/IP final validation acceptance.")
    parser.add_argument("config", type=Path, help="YAML acceptance-report configuration")
    args = parser.parse_args(argv)

    from .final_acceptance import write_final_acceptance_report

    config = _load_yaml(args.config)
    cfg = dict(config.get("acceptance", config))
    summary = write_final_acceptance_report(
        noip_summary_json=cfg["noip_summary_json"],
        ip_summary_json=cfg["ip_summary_json"],
        output_dir=cfg.get("output_dir", "outputs/final_acceptance"),
    )
    output_dir = Path(cfg.get("output_dir", "outputs/final_acceptance"))
    print(f"wrote {output_dir / 'final_acceptance_summary.json'}")
    print(f"final_acceptance_passed: {summary['final_acceptance_passed']}")
    return 0 if bool(summary["final_acceptance_passed"]) else 1


def _main_corrected_model_spec(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write corrected-model no-IP/IP validation case specs.")
    parser.add_argument("output_root", type=Path, help="Root directory for no-IP/IP validation outputs")
    parser.add_argument("--output", type=Path, default=Path("corrected_model_validation_spec.json"))
    args = parser.parse_args(argv)

    from .corrected_model import build_corrected_model_case_specs

    specs = build_corrected_model_case_specs(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _main_corrected_model_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run corrected-model no-IP/IP validation artifacts.")
    parser.add_argument("config", type=Path, help="JSON/YAML corrected-model case spec")
    parser.add_argument("--case", choices=("noip", "ip", "both"), default="both")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    from .corrected_model_runner import run_corrected_model_validation

    config = _load_yaml(args.config)
    specs = _selected_corrected_model_specs(config, case=args.case)
    summaries = []
    for case_name, spec in specs:
        case_spec = dict(spec)
        if args.output_root is not None:
            case_spec["output_dir"] = str(args.output_root / f"{case_name}_3comp")
            runner = dict(case_spec.get("runner", {}))
            runner["output_root"] = str(args.output_root)
            case_spec["runner"] = runner
        summary = run_corrected_model_validation(case_spec)
        summaries.append(summary)
        print(f"wrote {case_spec['output_dir']}")
        print(
            f"{case_name}: final_acceptance_passed={summary['final_acceptance_passed']}"
        )
    return 0 if all(bool(summary["final_acceptance_passed"]) for summary in summaries) else 1


def _main_validate_secondary(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate primary-secondary zero-contrast state.")
    parser.add_argument("config", type=Path, help="YAML secondary validation configuration")
    args = parser.parse_args(argv)

    from .materials.prony import PronyConductivity
    from .primary import CachedPrimaryProvider
    from .solvers import (
        PrimarySecondaryForwardOperator,
        initialize_dc_secondary,
        secondary_state_from_dc_initialization,
        secondary_step_ip,
        secondary_step_noip,
    )

    config = _load_yaml(args.config)
    cfg = dict(config.get("secondary", config))
    output_dir = Path(cfg.get("output_dir", "outputs/secondary_validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    Ep0 = np.asarray(cfg["Ep0"], dtype=float)
    sigma = float(cfg["sigma"])
    sigma_background = float(cfg.get("sigma_background", sigma))
    threshold = float(cfg.get("threshold", 1.0e-12))
    times = np.asarray(cfg.get("times", [1.0e-5]), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(times <= 0.0):
        raise ValueError("times must be positive observation times")
    material_config = cfg.get("material", config.get("material"))
    material = _material_from_config(material_config) or PronyConductivity.no_ip(sigma)
    material_model = "prony" if material_config is not None else "no_ip"

    init = initialize_dc_secondary(
        Ep0=Ep0,
        sigma0=material.sigma0,
        sigma_background=sigma_background,
        material=material,
        contrast_atol=threshold,
    )
    state = secondary_state_from_dc_initialization(init)
    max_abs_secondary_dbdt = 0.0
    previous = state
    previous_time = 0.0
    trace_rows: list[dict[str, float | bool]] = []
    for time in times:
        dt = float(time - previous_time)
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing")
        if material.terms:
            state = secondary_step_ip(
                previous,
                Ep_old=Ep0,
                Ep_new=Ep0,
                material=material,
                sigma_background=sigma_background,
                dt=dt,
                contrast_atol=threshold,
            )
        else:
            state = secondary_step_noip(
                previous,
                Ep_old=Ep0,
                Ep_new=Ep0,
                sigma=material.sigma_inf,
                sigma_background=sigma_background,
                dt=dt,
                contrast_atol=threshold,
            )
        max_abs_secondary_dbdt = max(
            max_abs_secondary_dbdt,
            float(np.max(np.abs((state.Es - previous.Es) / dt))),
        )
        trace_rows.append(
            {
                "time_obs": float(time),
                "max_abs_Es": float(np.max(np.abs(state.Es))) if state.Es.size else 0.0,
                "max_abs_secondary_dBdt": max_abs_secondary_dbdt,
                "total_response_equals_primary": bool(
                    np.allclose(Ep0 + state.Es, Ep0, rtol=0.0, atol=threshold)
                ),
            }
        )
        previous = state
        previous_time = float(time)

    max_abs_es = float(np.max(np.abs(state.Es))) if state.Es.size else 0.0
    total_equals_primary = bool(np.allclose(Ep0 + state.Es, Ep0, rtol=0.0, atol=threshold))
    forward_core_used = False
    max_abs_total_minus_primary = None
    if {"receiver_locations", "components", "receiver_E", "receiver_dBdt"}.issubset(cfg):
        receiver_locations = np.asarray(cfg["receiver_locations"], dtype=float)
        components = tuple(str(component) for component in cfg["components"])
        receiver_E = np.asarray(cfg["receiver_E"], dtype=float)
        receiver_dbdt = np.asarray(cfg["receiver_dBdt"], dtype=float)
        fem_points = np.asarray(
            cfg.get("fem_points", np.zeros((Ep0.shape[0], 3), dtype=float)),
            dtype=float,
        )
        provider = CachedPrimaryProvider(
            times=times,
            points=fem_points,
            receivers=receiver_locations,
            Ep_on_V=np.repeat(Ep0.reshape(1, *Ep0.shape), times.size, axis=0),
            receiver_E=receiver_E,
            receiver_dBdt=receiver_dbdt,
            Ep_dc_on_V=Ep0,
        )
        operator = PrimarySecondaryForwardOperator(
            primary=provider,
            fem_points=fem_points,
            receiver_locations=receiver_locations,
            components=components,
            material=material,
            sigma_background=sigma_background,
            contrast_atol=threshold,
        )
        predictions = operator.forward(times)
        primary_rows = _receiver_component_rows(receiver_E, receiver_dbdt, components)
        max_abs_total_minus_primary = float(np.max(np.abs(predictions - primary_rows)))
        _write_response_csv(output_dir / "primary_secondary_predictions.csv", times, predictions, components)
        forward_core_used = True
    summary = {
        "case_type": "secondary_zero_contrast",
        "sigma": sigma,
        "sigma0": material.sigma0,
        "sigma_inf": material.sigma_inf,
        "material_model": material_model,
        "delta_sigma_list": [term.delta_sigma for term in material.terms],
        "tau_list": [term.tau for term in material.terms],
        "sigma_background": sigma_background,
        "threshold": threshold,
        "time_count": int(times.size),
        "forward_core_used": forward_core_used,
        "max_abs_Es": max_abs_es,
        "max_abs_secondary_dBdt": max_abs_secondary_dbdt,
        "max_abs_total_minus_primary": max_abs_total_minus_primary,
        "total_response_equals_primary": total_equals_primary,
        "pass_zero_contrast": bool(
            max_abs_es <= threshold
            and max_abs_secondary_dbdt <= threshold
            and total_equals_primary
            and init.contrast_is_zero
            and (max_abs_total_minus_primary is None or max_abs_total_minus_primary <= threshold)
        ),
    }
    path = output_dir / "secondary_validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    trace_path = output_dir / "secondary_validation_trace.csv"
    trace_lines = [
        "time_obs,max_abs_Es,max_abs_secondary_dBdt,total_response_equals_primary",
        *[
            (
                f"{row['time_obs']:.17g},{row['max_abs_Es']:.17g},"
                f"{row['max_abs_secondary_dBdt']:.17g},{row['total_response_equals_primary']}"
            )
            for row in trace_rows
        ],
    ]
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    diagnostics = {
        "validation_type": "secondary_zero_contrast",
        "contrast_is_zero": bool(init.contrast_is_zero),
        "time_count": int(times.size),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "run_config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(f"pass_zero_contrast: {summary['pass_zero_contrast']}")
    return 0


def _receiver_component_rows(receiver_E, receiver_dbdt, components: tuple[str, ...]) -> np.ndarray:
    electric = np.asarray(receiver_E, dtype=float)
    dbdt = np.asarray(receiver_dbdt, dtype=float)
    if electric.ndim != 3 or electric.shape[2] != 3:
        raise ValueError("receiver_E must have shape (n_times, n_receivers, 3)")
    if dbdt.shape != electric.shape:
        raise ValueError("receiver_dBdt must have the same shape as receiver_E")
    rows = []
    for electric_t, dbdt_t in zip(electric, dbdt):
        columns = []
        for component in components:
            if component in {"Ex", "Ey", "Ez"}:
                columns.append(electric_t[:, {"Ex": 0, "Ey": 1, "Ez": 2}[component]])
            elif component in {"dBxdt", "dBydt", "dBzdt"}:
                columns.append(dbdt_t[:, {"dBxdt": 0, "dBydt": 1, "dBzdt": 2}[component]])
            else:
                raise ValueError(f"unsupported receiver component: {component}")
        rows.append(np.column_stack(columns).reshape(-1))
    return np.vstack(rows)


def _write_response_csv(
    path: Path,
    times: np.ndarray,
    values: np.ndarray,
    components: tuple[str, ...],
) -> None:
    lines = ["time_obs," + ",".join(components)]
    for time, row in zip(times, values):
        lines.append(",".join([f"{time:.17g}", *[f"{value:.17g}" for value in row]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _selected_corrected_model_specs(config: dict, *, case: str) -> list[tuple[str, dict]]:
    if "noip" in config or "ip" in config:
        if case == "both":
            names = ["noip", "ip"]
        else:
            names = [case]
        missing = [name for name in names if name not in config]
        if missing:
            raise ValueError(f"corrected-model spec missing cases: {missing}")
        return [(name, dict(config[name])) for name in names]
    case_name = str(config.get("case_type", case))
    if case == "both":
        return [(case_name, dict(config))]
    if case_name != case:
        raise ValueError(f"single corrected-model spec has case_type={case_name!r}, not {case!r}")
    return [(case_name, dict(config))]


def _response_component_names(path: Path) -> list[str]:
    table = np.genfromtxt(path, delimiter=",", names=True, max_rows=1, dtype=float, encoding="utf-8")
    names = list(table.dtype.names or [])
    if "time_obs" not in names:
        raise ValueError("response CSV must contain time_obs")
    return [name for name in names if name != "time_obs"]


def _read_errors_csv(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.ndim == 0:
        rows = np.asarray([rows], dtype=rows.dtype)
    required = {"time_obs", "component", "relative_error_with_floor"}
    missing = sorted(required.difference(rows.dtype.names or ()))
    if missing:
        raise ValueError(f"errors CSV missing columns: {missing}")
    return rows


def _validation_case_from_config(config: dict, *, case_type: str) -> ThreeComponentValidationInput:
    validation = dict(config.get("validation", config))
    component_names = [str(name) for name in validation["component_names"]]
    pred_times, predictions = _read_response_csv(validation["predictions_csv"], component_names)
    ref_times, reference = _read_response_csv(validation["reference_csv"], component_names)
    if pred_times.shape != ref_times.shape or not np.allclose(pred_times, ref_times, rtol=0.0, atol=0.0):
        raise ValueError("prediction and reference CSV time_obs columns must match exactly")
    material = _material_from_config(config.get("material")) if case_type == "ip" else None
    return ThreeComponentValidationInput(
        output_dir=validation["output_dir"],
        times=pred_times,
        predictions=predictions,
        reference=reference,
        component_names=component_names,
        case_type=case_type,
        reference_type=str(validation.get("reference_type", "empymod")),
        magnetic_quantity=str(validation.get("magnetic_quantity", component_names[-1])),
        threshold=float(validation.get("relative_error_threshold", 0.05)),
        diagnostics=dict(validation.get("diagnostics", {})),
        resolved_config=config,
        material=material,
        validation_scope=str(validation.get("validation_scope", config.get("validation_scope", "smoke"))),
    )


def _read_response_csv(path, component_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if table.ndim == 0:
        table = np.asarray([table], dtype=table.dtype)
    names = list(table.dtype.names or [])
    if "time_obs" not in names:
        raise ValueError("response CSV must contain time_obs")
    missing = [name for name in component_names if name not in names]
    if missing:
        raise ValueError(f"response CSV missing components: {missing}")
    times = np.asarray(table["time_obs"], dtype=float)
    values = np.column_stack([np.asarray(table[name], dtype=float) for name in component_names])
    return times, values


def _material_from_config(config) -> PronyConductivity | None:
    if config is None:
        return None
    cfg = dict(config)
    terms = [
        DebyeTerm(delta_sigma=float(term["delta_sigma"]), tau=float(term["tau"]))
        for term in cfg.get("terms", cfg.get("debye_terms", []))
    ]
    return PronyConductivity(sigma_inf=float(cfg["sigma_inf"]), terms=terms)


if __name__ == "__main__":
    raise SystemExit(main())
