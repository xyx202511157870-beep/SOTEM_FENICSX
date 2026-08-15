"""Validate numerical Hx/Hy/Hz and dB/dt data against empymod.

The reference can use an ideal switch-off, a finite linear ramp, a waveform
read from CSV, or the waveform declared in the YAML source configuration.
Observation times are always interpreted relative to the end of turn-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .empymod_compare import make_debye_resistivity_model_from_config
from .empymod_magnetic6 import (
    MagneticSixNumericalData,
    build_magnetic6_survey_from_config,
    compare_magnetic6,
    load_magnetic6_numerical,
    run_empymod_magnetic6_reference,
    write_magnetic6_artifacts,
)
from .empymod_waveform import (
    PiecewiseLinearTurnOff,
    load_turnoff_csv,
    run_empymod_magnetic6_waveform_reference,
    turnoff_waveform_from_config,
)


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("values must be finite and non-empty")
    return values


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML source/model/receiver geometry")
    parser.add_argument(
        "--numerical",
        type=Path,
        required=True,
        help=(
            "Numerical CSV or NPZ. CSV requires time_obs/time_s plus "
            "Hx,Hy,Hz,dBxdt,dBydt,dBzdt; NPZ requires times,data and "
            "optional components."
        ),
    )
    parser.add_argument("--depths", type=_float_list, required=True)
    parser.add_argument(
        "--resistivities",
        type=_float_list,
        help="Comma-separated layered resistivities; length must be len(depths)+1.",
    )
    parser.add_argument(
        "--use-config-ip",
        action="store_true",
        help="Build empymod Debye dispersion directly from the YAML layer model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--srcpts", type=int, default=9)
    parser.add_argument("--recpts", type=int, default=1)
    parser.add_argument(
        "--dbdt-reference",
        choices=("auto", "native_b", "impulse_h"),
        default="auto",
        help=(
            "Primary dB/dt route. auto uses native mrec='b' with empymod>=2.6 "
            "and otherwise -mu0*H impulse."
        ),
    )
    parser.add_argument(
        "--no-impulse-audit",
        action="store_true",
        help="Disable native-B versus -mu0*H-impulse cross-check when available.",
    )
    parser.add_argument("--audit-tolerance", type=float, default=0.01)
    parser.add_argument("--comparison-tolerance", type=float, default=0.05)
    parser.add_argument("--floor-fraction", type=float, default=0.01)
    parser.add_argument(
        "--require-audit-pass",
        action="store_true",
        help="Require both empymod dB/dt routes to be available and agree.",
    )
    parser.add_argument("--srcpts-audit", type=int, default=0)

    waveform = parser.add_mutually_exclusive_group()
    waveform.add_argument(
        "--ideal-step-off",
        action="store_true",
        help="Ignore the YAML waveform and use an ideal instantaneous switch-off.",
    )
    waveform.add_argument(
        "--ramp-off-time",
        type=_positive_float,
        default=None,
        help=(
            "Finite linear turn-off duration in seconds. Example: 5e-6 for a "
            "5 microsecond ramp."
        ),
    )
    waveform.add_argument(
        "--waveform-csv",
        type=Path,
        default=None,
        help=(
            "Piecewise-linear waveform CSV with time_s,current_scale or "
            "time,current columns. The final sample is shifted to ramp-end time zero."
        ),
    )
    parser.add_argument(
        "--waveform-quadrature-order",
        type=int,
        default=8,
        help="Gauss-Legendre points per piecewise-linear waveform segment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    if args.waveform_quadrature_order < 2:
        raise ValueError("--waveform-quadrature-order must be at least 2")

    numerical = load_magnetic6_numerical(args.numerical)
    if args.use_config_ip:
        resistivity_model = make_debye_resistivity_model_from_config(
            config,
            depths=args.depths,
        )
    else:
        if args.resistivities is None:
            raise ValueError("--resistivities is required unless --use-config-ip is set")
        if len(args.resistivities) != len(args.depths) + 1:
            raise ValueError("len(resistivities) must equal len(depths)+1")
        resistivity_model = args.resistivities

    survey = build_magnetic6_survey_from_config(
        config,
        times=numerical.times,
        depths=args.depths,
        resistivities=resistivity_model,
        signal=-1,
    )
    if numerical.data.shape[1] != len(survey.receiver_locations):
        raise ValueError(
            "numerical location count does not match config receiver locations: "
            f"{numerical.data.shape[1]} versus {len(survey.receiver_locations)}"
        )

    selected_waveform = _select_waveform(args, config)
    reference = _run_reference(
        survey,
        args,
        waveform=selected_waveform,
        srcpts=args.srcpts,
    )
    comparison = compare_magnetic6(
        numerical,
        reference,
        tolerance=args.comparison_tolerance,
        floor_fraction=args.floor_fraction,
    )
    comparison["waveform"] = _waveform_metadata(reference, selected_waveform)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_magnetic6_artifacts(
        args.output_dir,
        numerical=numerical,
        reference=reference,
        comparison=comparison,
    )
    waveform_metadata = _waveform_metadata(reference, selected_waveform)
    convolution_metadata = getattr(reference, "convolution", {})
    np.savez_compressed(
        args.output_dir / "empymod_magnetic6_reference.npz",
        times=reference.times,
        data=reference.data,
        components=np.asarray(reference.components),
        units=np.asarray(reference.units),
        receiver_locations=np.asarray(reference.receiver_locations, dtype=float),
        primary_dbdt_reference=np.asarray(reference.primary_dbdt_reference),
        empymod_version=np.asarray(reference.empymod_version or ""),
        waveform_times=np.asarray(
            waveform_metadata.get("times_relative_to_ramp_end_s", []),
            dtype=float,
        ),
        waveform_current_scales=np.asarray(
            waveform_metadata.get("current_scales", []),
            dtype=float,
        ),
        waveform_name=np.asarray(waveform_metadata.get("name", "ideal_step_off")),
        waveform_time_origin=np.asarray("ramp_end"),
        waveform_quadrature_order=np.asarray(
            convolution_metadata.get("quadrature_order_per_segment", 0),
            dtype=int,
        ),
        dbdt_native=(
            np.asarray(reference.dbdt_native)
            if reference.dbdt_native is not None
            else np.empty((0, 0, 0), dtype=float)
        ),
        dbdt_impulse=(
            np.asarray(reference.dbdt_impulse)
            if reference.dbdt_impulse is not None
            else np.empty((0, 0, 0), dtype=float)
        ),
    )
    (args.output_dir / "empymod_waveform_reference.json").write_text(
        json.dumps(
            {
                "waveform": waveform_metadata,
                "convolution": convolution_metadata,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.srcpts_audit > 0:
        higher = _run_reference(
            survey,
            args,
            waveform=selected_waveform,
            srcpts=args.srcpts_audit,
        )
        source_audit = compare_magnetic6(
            MagneticSixNumericalData(
                times=reference.times,
                data=reference.data,
                receiver_locations=reference.receiver_locations,
            ),
            higher,
            tolerance=args.audit_tolerance,
            floor_fraction=args.floor_fraction,
        )
        source_audit["waveform"] = waveform_metadata
        (args.output_dir / "empymod_source_quadrature_audit.json").write_text(
            json.dumps(source_audit, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))
    audit_ok = bool(reference.audit.get("passed", True))
    return 0 if bool(comparison["passed"]) and audit_ok else 2


def _select_waveform(
    args: argparse.Namespace,
    config: dict,
) -> PiecewiseLinearTurnOff | None:
    if args.ideal_step_off:
        return None
    if args.ramp_off_time is not None:
        return PiecewiseLinearTurnOff.linear(args.ramp_off_time)
    if args.waveform_csv is not None:
        return load_turnoff_csv(args.waveform_csv)
    return turnoff_waveform_from_config(config, base_dir=args.config.parent)


def _run_reference(
    survey,
    args: argparse.Namespace,
    *,
    waveform: PiecewiseLinearTurnOff | None,
    srcpts: int,
):
    kwargs = {
        "srcpts": int(srcpts),
        "recpts": int(args.recpts),
        "dbdt_reference": args.dbdt_reference,
        "audit_impulse": not args.no_impulse_audit,
        "audit_tolerance": float(args.audit_tolerance),
        "audit_floor_fraction": float(args.floor_fraction),
        "require_audit_pass": bool(args.require_audit_pass),
    }
    if waveform is None:
        return run_empymod_magnetic6_reference(survey, **kwargs)
    return run_empymod_magnetic6_waveform_reference(
        survey,
        waveform,
        quadrature_order=int(args.waveform_quadrature_order),
        **kwargs,
    )


def _waveform_metadata(reference, waveform: PiecewiseLinearTurnOff | None) -> dict:
    if hasattr(reference, "waveform"):
        return dict(reference.waveform)
    if waveform is not None:
        return waveform.metadata()
    return {
        "name": "ideal_step_off",
        "time_origin": "switch_off",
        "times_relative_to_ramp_end_s": [0.0],
        "current_scales": [0.0],
        "duration_s": 0.0,
        "total_current_scale_drop": 1.0,
        "segment_count": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
