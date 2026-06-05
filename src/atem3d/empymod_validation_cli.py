"""Command-line runner for in-memory empymod validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_config
from .empymod_validation import run_empymod_validation, run_empymod_validation_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an ATEM3D config and compare receiver data with empymod."
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--depths", nargs="+", type=float, required=True)
    parser.add_argument("--resistivities", nargs="+", type=float, required=True)
    parser.add_argument("--signal", type=int, default=-1)
    parser.add_argument("--srcpts", type=int, default=51)
    parser.add_argument("--recpts", type=int, default=1)
    parser.add_argument("--empymod-strength", type=float, default=None)
    parser.add_argument("--use-config-ip", action="store_true")
    parser.add_argument("--include-t0", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Do not store full field histories")
    parser.add_argument("--skip-positive-times", type=int, default=0)
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--absolute-tolerance", type=float, default=None)
    parser.add_argument(
        "--sweep-cases",
        type=Path,
        default=None,
        help="Optional YAML file containing named validation case overrides",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 1 unless the validation report has passed=true",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/empymod_validation.json"))
    args = parser.parse_args(argv)

    if args.skip_positive_times < 0:
        parser.error("--skip-positive-times must be nonnegative")
    if (
        args.time_min is not None
        and args.time_max is not None
        and args.time_min > args.time_max
    ):
        parser.error("--time-min must be <= --time-max")
    if args.tolerance is not None and args.tolerance < 0.0:
        parser.error("--tolerance must be nonnegative")
    if args.absolute_tolerance is not None and args.absolute_tolerance < 0.0:
        parser.error("--absolute-tolerance must be nonnegative")

    config = load_config(args.config)
    validation_kwargs = {
        "depths": list(args.depths),
        "resistivities": list(args.resistivities),
        "signal": args.signal,
        "use_config_ip": args.use_config_ip,
        "positive_times_only": not args.include_t0,
        "skip_positive_times": args.skip_positive_times,
        "time_min": args.time_min,
        "time_max": args.time_max,
        "data_only": args.data_only,
        "tolerance": args.tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "empymod_kwargs": {"srcpts": args.srcpts, "recpts": args.recpts},
        "empymod_strength": args.empymod_strength,
        "output_path": args.output,
    }
    if args.sweep_cases is not None:
        sweep = run_empymod_validation_sweep(
            config,
            _load_sweep_cases(args.sweep_cases),
            **validation_kwargs,
        )
        report = sweep.to_report()
        print(f"wrote {args.output}")
        print(f"passed={report['passed']}")
        for case_name, validation in sweep.cases.items():
            for name, values in validation.components.items():
                print(
                    f"{case_name}/{name}: relative_l2={values['relative_l2']:.6e}, "
                    f"relative_linf={values['relative_linf']:.6e}, "
                    f"passed={values.get('passed')}"
                )
        return _exit_status(args.require_pass, report)

    validation = run_empymod_validation(config, **validation_kwargs)
    report = validation.to_report()
    print(f"wrote {args.output}")
    print(f"passed={report['passed']}")
    for name, values in validation.components.items():
        print(
            f"{name}: relative_l2={values['relative_l2']:.6e}, "
            f"relative_linf={values['relative_linf']:.6e}, "
            f"passed={values.get('passed')}"
        )
    return _exit_status(args.require_pass, report)


def _load_sweep_cases(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if isinstance(payload, dict) and "cases" in payload:
        return payload["cases"]
    return payload


def _exit_status(require_pass: bool, report: dict) -> int:
    if not require_pass:
        return 0
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
