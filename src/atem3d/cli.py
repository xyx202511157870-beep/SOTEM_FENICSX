"""Command-line entry points for ATEM3D examples."""

from __future__ import annotations

import argparse
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


def _load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


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
