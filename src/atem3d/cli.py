"""Command-line entry points for ATEM3D examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .materials.prony import DebyeTerm, PronyConductivity
from .validation_3comp import (
    ThreeComponentValidationInput,
    write_three_component_validation_artifacts,
)

_TOP_LEVEL_COMMANDS = (
    ("run", "Run a grounded-wire TDEM-IP simulation from a YAML config."),
    ("plot", "Regenerate three-component validation plots from CSV artifacts."),
    ("validate-noip-3comp", "Write no-IP Ex/Ey/Hz-or-dBzdt validation artifacts."),
    ("validate-ip-3comp", "Write IP Ex/Ey/Hz-or-dBzdt validation artifacts."),
    ("validate-secondary", "Run primary-secondary zero-contrast validation artifacts."),
    ("acceptance-report", "Combine no-IP/IP error summaries into a final acceptance report."),
    ("corrected-model-spec", "Write corrected source/receiver no-IP/IP case specs."),
    ("corrected-model-run", "Run corrected-model no-IP/IP validation artifacts."),
    ("dolfinx-backend-check", "Check required DOLFINx/FEniCSx backend imports."),
    ("corrected-model-convergence-run", "Run coarse-vs-refined corrected-model diagnostics."),
    ("convergence-sweep-report", "Summarize convergence diagnostic artifact directories."),
    ("leakage-marker-diagnostics", "Preflight leakage-channel cell marker coverage."),
    ("corrected-leakage-model-spec", "Write corrected leakage-channel diagnostic specs."),
    ("corrected-terrain-smoke-run", "Run a memory-safe Gmsh terrain/leakage artifact smoke."),
    ("model-schematic", "Write a corrected-model source/receiver schematic."),
    ("polarization-effect", "Write IP-minus-noIP response and error artifacts."),
    ("published-paper-model-spec", "Write the published-paper reproduction target spec."),
    ("published-paper-digitization-template", "Write paper-curve digitization templates."),
    ("published-paper-figure-pages", "Write/render published-paper target figure page assets."),
    ("published-paper-curve-artifacts", "Compare predictions against digitized paper curves."),
    ("published-paper-prony-materials", "Write Prony materials fitted from paper Cole-Cole models."),
)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in {"-h", "--help"}:
        return _main_help(argv)
    if argv and argv[0] == "run":
        return _main_run(argv[1:])
    if argv and argv[0] == "plot":
        return _main_plot(argv[1:])
    if argv and argv[0] == "model-schematic":
        return _main_model_schematic(argv[1:])
    if argv and argv[0] == "polarization-effect":
        return _main_polarization_effect(argv[1:])
    if argv and argv[0] == "validate-secondary":
        return _main_validate_secondary(argv[1:])
    if argv and argv[0] == "acceptance-report":
        return _main_acceptance_report(argv[1:])
    if argv and argv[0] == "corrected-model-spec":
        return _main_corrected_model_spec(argv[1:])
    if argv and argv[0] == "corrected-model-run":
        return _main_corrected_model_run(argv[1:])
    if argv and argv[0] == "dolfinx-backend-check":
        return _main_dolfinx_backend_check(argv[1:])
    if argv and argv[0] == "corrected-model-convergence-run":
        return _main_corrected_model_convergence_run(argv[1:])
    if argv and argv[0] == "convergence-sweep-report":
        return _main_convergence_sweep_report(argv[1:])
    if argv and argv[0] == "leakage-marker-diagnostics":
        return _main_leakage_marker_diagnostics(argv[1:])
    if argv and argv[0] == "corrected-leakage-model-spec":
        return _main_corrected_leakage_model_spec(argv[1:])
    if argv and argv[0] == "corrected-terrain-smoke-run":
        return _main_corrected_terrain_smoke_run(argv[1:])
    if argv and argv[0] == "published-paper-model-spec":
        return _main_published_paper_model_spec(argv[1:])
    if argv and argv[0] == "published-paper-digitization-template":
        return _main_published_paper_digitization_template(argv[1:])
    if argv and argv[0] == "published-paper-figure-pages":
        return _main_published_paper_figure_pages(argv[1:])
    if argv and argv[0] == "published-paper-curve-artifacts":
        return _main_published_paper_curve_artifacts(argv[1:])
    if argv and argv[0] == "published-paper-prony-materials":
        return _main_published_paper_prony_materials(argv[1:])
    if argv and argv[0] in {"validate-noip-3comp", "validate-ip-3comp"}:
        return _main_validate(argv)
    return _main_run(argv)


def _main_help(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="tdem-ip-forward",
        description="ATEM3D grounded-wire TDEM/IP forward modelling commands.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for command, help_text in _TOP_LEVEL_COMMANDS:
        subparsers.add_parser(command, help=help_text)
    parser.parse_args(argv)
    return 0


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


def _main_model_schematic(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write a corrected-model geometry schematic from a case spec.")
    parser.add_argument("config", type=Path, help="JSON/YAML corrected-model case spec")
    parser.add_argument("--case", choices=("noip", "ip"), default="noip")
    parser.add_argument("--output", type=Path, default=Path("model_schematic.png"))
    args = parser.parse_args(argv)

    from .model_schematic import write_model_schematic

    config = _load_yaml(args.config)
    case_spec = dict(config[args.case] if args.case in config else config)
    info = write_model_schematic(case_spec, args.output)
    print(f"wrote {args.output}")
    print(f"source_length_m: {info['source_length_m']}")
    print(f"parallel_offset_m: {info['parallel_offset_m']}")
    return 0


def _main_polarization_effect(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write IP-minus-noIP polarization-effect artifacts.")
    parser.add_argument("noip_dir", type=Path, help="Directory containing no-IP validation artifacts")
    parser.add_argument("ip_dir", type=Path, help="Directory containing IP validation artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("polarization_effect"))
    args = parser.parse_args(argv)

    from .polarization_effect import write_polarization_effect_artifacts

    summary = write_polarization_effect_artifacts(args.noip_dir, args.ip_dir, args.output_dir)
    print(f"wrote {args.output_dir}")
    print(f"pass_all_components: {summary['pass_all_components']}")
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
        noip_diagnostics_json=cfg.get("noip_diagnostics_json"),
        ip_diagnostics_json=cfg.get("ip_diagnostics_json"),
        polarization_effect_dir=cfg.get("polarization_effect_dir"),
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
    parser.add_argument("--n-observation-times", type=int, default=80)
    args = parser.parse_args(argv)

    from .corrected_model import CorrectedModelValidationConfig, build_corrected_model_case_specs

    config = CorrectedModelValidationConfig(n_observation_times=args.n_observation_times)
    specs = build_corrected_model_case_specs(args.output_root, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _main_published_paper_model_spec(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write published SOTEM paper reproduction target metadata.")
    parser.add_argument("output_root", type=Path, help="Root directory for paper-model reproduction outputs")
    parser.add_argument("--output", type=Path, default=Path("published_paper_model_target.json"))
    args = parser.parse_args(argv)

    from .corrected_model import build_published_paper_model_target_spec

    spec = build_published_paper_model_target_spec(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _main_published_paper_digitization_template(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write published paper response digitization templates.")
    parser.add_argument("config", type=Path, help="JSON/YAML published paper model target spec")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_digitization"))
    args = parser.parse_args(argv)

    from .paper_digitization import write_published_paper_digitization_template

    spec = _load_yaml(args.config)
    manifest = write_published_paper_digitization_template(spec, args.output_dir)
    print(f"wrote {args.output_dir / 'paper_curve_digitization_manifest.json'}")
    print(f"wrote {args.output_dir / manifest['template_csv']}")
    return 0


def _main_published_paper_figure_pages(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write/render target PDF pages for paper curve digitization.")
    parser.add_argument("config", type=Path, help="JSON/YAML published paper model target spec")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figure_pages"))
    parser.add_argument("--pdf", type=Path, help="Optional source PDF to render target pages with pdftoppm")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--renderer", default="pdftoppm")
    args = parser.parse_args(argv)

    from .paper_digitization import write_published_paper_figure_page_package

    spec = _load_yaml(args.config)
    manifest = write_published_paper_figure_page_package(
        spec,
        args.output_dir,
        pdf_path=args.pdf,
        dpi=args.dpi,
        render=args.pdf is not None,
        renderer=args.renderer,
    )
    print(f"wrote {args.output_dir / 'paper_figure_page_manifest.json'}")
    if manifest["rendered"]:
        print(f"rendered_pages: {len(manifest['pages'])}")
    else:
        print("rendered_pages: 0")
    return 0


def _main_published_paper_curve_artifacts(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write artifacts from digitized published response curves.")
    parser.add_argument("predictions_csv", type=Path, help="Prediction CSV with time_obs and component columns")
    parser.add_argument("digitized_csv", type=Path, help="Long-form digitized paper response CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_curve_artifacts"))
    parser.add_argument("--case-type", choices=("noip", "ip"), default="ip")
    parser.add_argument("--curve-label", default="paper_ip")
    parser.add_argument(
        "--component-figure",
        action="append",
        default=[],
        help="Component-to-figure mapping such as Ex=Fig. 12; may be repeated",
    )
    args = parser.parse_args(argv)

    from .paper_digitization import write_published_paper_curve_artifacts

    component_figures = _parse_component_figure_args(args.component_figure)
    summary = write_published_paper_curve_artifacts(
        predictions_csv=args.predictions_csv,
        digitized_csv=args.digitized_csv,
        output_dir=args.output_dir,
        case_type=args.case_type,
        curve_label=args.curve_label,
        component_figures=component_figures or None,
    )
    print(f"wrote {args.output_dir}")
    print(f"reference_type: {summary['reference_type']}")
    print(f"final_acceptance_passed: {summary['final_acceptance_passed']}")
    return 0


def _main_published_paper_prony_materials(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write Prony material fits for published paper models.")
    parser.add_argument("config", type=Path, help="JSON/YAML published paper model target spec")
    parser.add_argument("--output", type=Path, default=Path("paper_prony_materials.json"))
    parser.add_argument("--n-terms", type=int, default=10)
    args = parser.parse_args(argv)

    from .paper_materials import write_published_paper_prony_materials

    spec = _load_yaml(args.config)
    payload = write_published_paper_prony_materials(
        spec,
        args.output,
        n_terms=args.n_terms,
    )
    print(f"wrote {args.output}")
    print(f"materials: {len(payload['materials'])}")
    return 0


def _parse_component_figure_args(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--component-figure entries must use Component=Figure")
        component, figure = value.split("=", 1)
        component = component.strip()
        figure = figure.strip()
        if not component or not figure:
            raise ValueError("--component-figure entries must use Component=Figure")
        parsed[component] = figure
    return parsed


def _main_corrected_leakage_model_spec(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write corrected-scale leakage-channel validation case specs.")
    parser.add_argument("output_root", type=Path, help="Root directory for corrected leakage validation outputs")
    parser.add_argument("--output", type=Path, default=Path("corrected_leakage_model_spec.json"))
    parser.add_argument("--n-observation-times", type=int, default=80)
    args = parser.parse_args(argv)

    from .corrected_model import (
        CorrectedModelValidationConfig,
        build_corrected_leakage_channel_case_specs,
    )

    config = CorrectedModelValidationConfig(n_observation_times=args.n_observation_times)
    specs = build_corrected_leakage_channel_case_specs(args.output_root, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _main_corrected_terrain_smoke_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a Gmsh terrain/leakage diagnostic artifact smoke.")
    parser.add_argument("output_root", type=Path, help="Root directory for terrain smoke artifacts")
    parser.add_argument("--case", choices=("noip", "ip", "both"), default="noip")
    parser.add_argument("--spec-output", type=Path, help="Optional path to write the resolved smoke spec JSON")
    args = parser.parse_args(argv)

    from .corrected_model import build_corrected_terrain_smoke_case_specs
    from .corrected_model_runner import run_corrected_model_self_convergence_validation

    specs = build_corrected_terrain_smoke_case_specs(args.output_root)
    selected = specs.items() if args.case == "both" else [(args.case, specs[args.case])]
    if args.spec_output is not None:
        args.spec_output.parent.mkdir(parents=True, exist_ok=True)
        payload = specs if args.case == "both" else dict(selected[0][1])
        args.spec_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.spec_output}")
    for case_name, case_spec in selected:
        try:
            summary = run_corrected_model_self_convergence_validation(case_spec)
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"wrote {case_spec['output_dir']}")
        print(f"{case_name}: reference_type={summary['reference_type']}")
        print(f"{case_name}: final_acceptance_passed={summary['final_acceptance_passed']}")
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
    output_dirs: dict[str, Path] = {}
    for case_name, spec in specs:
        case_spec = dict(spec)
        if args.output_root is not None:
            case_spec["output_dir"] = str(args.output_root / f"{case_name}_3comp")
            runner = dict(case_spec.get("runner", {}))
            runner["output_root"] = str(args.output_root)
            case_spec["runner"] = runner
        try:
            summary = run_corrected_model_validation(case_spec)
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        summaries.append(summary)
        output_dirs[case_name] = Path(case_spec["output_dir"])
        print(f"wrote {case_spec['output_dir']}")
        print(
            f"{case_name}: final_acceptance_passed={summary['final_acceptance_passed']}"
        )
    if {"noip", "ip"}.issubset(output_dirs):
        acceptance_root = args.output_root or output_dirs["noip"].parent
        acceptance_path = _write_corrected_model_acceptance_config(acceptance_root, output_dirs)
        print(f"wrote {acceptance_path}")
        effect_path = _write_corrected_model_polarization_effect(acceptance_root, output_dirs)
        print(f"wrote {effect_path}")
    return 0 if all(bool(summary["final_acceptance_passed"]) for summary in summaries) else 1


def _main_dolfinx_backend_check(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check DOLFINx backend imports for corrected-model runs.")
    parser.add_argument("--output", type=Path, default=Path("dolfinx_backend_status.json"))
    args = parser.parse_args(argv)

    from . import corrected_model_runner as runner

    status = runner.dolfinx_backend_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"available: {status['available']}")
    if status["missing_modules"]:
        print("missing_modules: " + ", ".join(status["missing_modules"]))
    if status.get("missing_test_modules"):
        print("missing_test_modules: " + ", ".join(status["missing_test_modules"]))
    return 0 if bool(status["available"]) else 2


def _main_corrected_model_convergence_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run corrected-model coarse-vs-refined DOLFINx convergence diagnostics."
    )
    parser.add_argument("config", type=Path, help="JSON/YAML corrected-model case spec")
    parser.add_argument("--case", choices=("noip", "ip", "both"), default="both")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    from .corrected_model_runner import run_corrected_model_convergence_validation

    config = _load_yaml(args.config)
    specs = _selected_corrected_model_specs(config, case=args.case)
    for case_name, spec in specs:
        case_spec = dict(spec)
        if args.output_root is not None:
            case_spec["output_dir"] = str(args.output_root / f"{case_name}_convergence")
            runner = dict(case_spec.get("runner", {}))
            runner["output_root"] = str(args.output_root)
            runner["diagnostic"] = "dolfinx_refined_convergence"
            case_spec["runner"] = runner
        summary = run_corrected_model_convergence_validation(case_spec)
        print(f"wrote {case_spec['output_dir']}")
        print(
            f"{case_name}: physical_pass_all_components={summary.get('physical_pass_all_components', False)}"
        )
    return 0


def _main_convergence_sweep_report(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize multiple convergence validation run directories.")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Validation run directories")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/convergence_sweep"))
    parser.add_argument("--labels", help="Comma-separated labels matching run_dirs")
    args = parser.parse_args(argv)

    from .convergence_sweep import write_convergence_sweep_report

    labels = None
    if args.labels:
        labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    summary = write_convergence_sweep_report(args.run_dirs, args.output_dir, labels=labels)
    print(f"wrote {args.output_dir / 'convergence_sweep_summary.json'}")
    print(f"best_by_max_physical_error: {summary['best_by_max_physical_error']['label']}")
    return 0


def _main_leakage_marker_diagnostics(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Preflight leakage-channel cell marker diagnostics.")
    parser.add_argument("config", type=Path, help="JSON/YAML corrected leakage case spec")
    parser.add_argument("--case", choices=("noip", "ip"), default="noip")
    parser.add_argument("--output", type=Path, default=Path("leakage_marker_diagnostics.json"))
    args = parser.parse_args(argv)

    from .corrected_model_runner import _build_dolfinx_refined_reference_specs
    from .materials.material_map import leakage_channel_marker_diagnostics

    config = _load_yaml(args.config)
    case_spec = dict(config[args.case] if args.case in config else config)
    prediction_spec, reference_spec = _build_dolfinx_refined_reference_specs(case_spec)
    payload = {
        "case_type": str(case_spec.get("case_type", args.case)),
        "prediction": _leakage_marker_diagnostics_from_forward(
            prediction_spec["dolfinx_forward"],
            leakage_channel_marker_diagnostics,
        ),
        "reference": _leakage_marker_diagnostics_from_forward(
            reference_spec["dolfinx_forward"],
            leakage_channel_marker_diagnostics,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        "prediction_leakage_cell_count: "
        f"{payload['prediction']['leakage_cell_count']}; "
        f"reference_leakage_cell_count: {payload['reference']['leakage_cell_count']}"
    )
    return 0


def _leakage_marker_diagnostics_from_forward(forward_cfg: dict, diagnostics_func) -> dict:
    forward = dict(forward_cfg)
    leakage = dict(forward["leakage_channel"])
    return diagnostics_func(
        domain_min=forward["domain_min"],
        domain_max=forward["domain_max"],
        cells=forward["cells"],
        channel_points=leakage["points"],
        radius=float(leakage["radius"]),
        min_marked_cells=int(leakage.get("min_marked_cells", 0)),
    )


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
    from .yaml_io import safe_dump_yaml

    (output_dir / "run_config_resolved.yaml").write_text(
        safe_dump_yaml(cfg, sort_keys=True),
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
    from .yaml_io import safe_load_yaml

    text = Path(path).read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        config = safe_load_yaml(text)
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


def _write_corrected_model_acceptance_config(output_root: Path, output_dirs: dict[str, Path]) -> Path:
    from .yaml_io import safe_dump_yaml

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "acceptance": {
            "noip_summary_json": str(output_dirs["noip"] / "error_summary.json"),
            "ip_summary_json": str(output_dirs["ip"] / "error_summary.json"),
            "noip_diagnostics_json": str(output_dirs["noip"] / "diagnostics.json"),
            "ip_diagnostics_json": str(output_dirs["ip"] / "diagnostics.json"),
            "output_dir": str(output_root / "final_acceptance"),
            "polarization_effect_dir": str(output_root / "polarization_effect"),
        }
    }
    path = output_root / "acceptance.yaml"
    path.write_text(safe_dump_yaml(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_corrected_model_polarization_effect(output_root: Path, output_dirs: dict[str, Path]) -> Path:
    from .polarization_effect import write_polarization_effect_artifacts

    output_root = Path(output_root)
    effect_dir = output_root / "polarization_effect"
    write_polarization_effect_artifacts(output_dirs["noip"], output_dirs["ip"], effect_dir)
    return effect_dir


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
