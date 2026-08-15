"""Validate numerical Hx/Hy/Hz and dB/dt three-component data against empymod."""

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


def _float_list(value: str) -> list[float]:
    try:
        values = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from exc
    if not values or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "values must be finite and non-empty"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="YAML source/model/receiver geometry",
    )
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
        help=(
            "Comma-separated layered resistivities; length must be "
            "len(depths)+1."
        ),
    )
    parser.add_argument(
        "--use-config-ip",
        action="store_true",
        help=(
            "Build empymod Debye dispersion directly from the YAML "
            "layer model."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--srcpts", type=int, default=9)
    parser.add_argument("--recpts", type=int, default=1)
    parser.add_argument(
        "--dbdt-reference",
        choices=("auto", "native_b", "impulse_h"),
        default="auto",
        help=(
            "Primary dB/dt route. auto uses native mrec='b' with "
            "empymod>=2.6 and otherwise -mu0*H impulse."
        ),
    )
    parser.add_argument(
        "--no-impulse-audit",
        action="store_true",
        help=(
            "Disable the native-B versus -mu0*H-impulse cross-check "
            "when the native route is available."
        ),
    )
    parser.add_argument("--audit-tolerance", type=float, default=0.01)
    parser.add_argument("--comparison-tolerance", type=float, default=0.05)
    parser.add_argument("--floor-fraction", type=float, default=0.01)
    parser.add_argument(
        "--require-audit-pass",
        action="store_true",
        help=(
            "Require the two empymod dB/dt constructions to be "
            "available and to agree."
        ),
    )
    parser.add_argument("--srcpts-audit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")

    numerical = load_magnetic6_numerical(args.numerical)
    if args.use_config_ip:
        resistivity_model = make_debye_resistivity_model_from_config(
            config,
            depths=args.depths,
        )
    else:
        if args.resistivities is None:
            raise ValueError(
                "--resistivities is required unless --use-config-ip is set"
            )
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

    reference = run_empymod_magnetic6_reference(
        survey,
        srcpts=args.srcpts,
        recpts=args.recpts,
        dbdt_reference=args.dbdt_reference,
        audit_impulse=not args.no_impulse_audit,
        audit_tolerance=args.audit_tolerance,
        audit_floor_fraction=args.floor_fraction,
        require_audit_pass=args.require_audit_pass,
    )
    comparison = compare_magnetic6(
        numerical,
        reference,
        tolerance=args.comparison_tolerance,
        floor_fraction=args.floor_fraction,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_magnetic6_artifacts(
        args.output_dir,
        numerical=numerical,
        reference=reference,
        comparison=comparison,
    )
    np.savez_compressed(
        args.output_dir / "empymod_magnetic6_reference.npz",
        times=reference.times,
        data=reference.data,
        components=np.asarray(reference.components),
        units=np.asarray(reference.units),
        receiver_locations=np.asarray(reference.receiver_locations, dtype=float),
        primary_dbdt_reference=np.asarray(reference.primary_dbdt_reference),
        empymod_version=np.asarray(reference.empymod_version or ""),
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

    if args.srcpts_audit > 0:
        higher = run_empymod_magnetic6_reference(
            survey,
            srcpts=args.srcpts_audit,
            recpts=args.recpts,
            dbdt_reference=args.dbdt_reference,
            audit_impulse=not args.no_impulse_audit,
            audit_tolerance=args.audit_tolerance,
            audit_floor_fraction=args.floor_fraction,
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
        (args.output_dir / "empymod_source_quadrature_audit.json").write_text(
            json.dumps(
                source_audit,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))
    audit_ok = bool(reference.audit.get("passed", True))
    return 0 if bool(comparison["passed"]) and audit_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
