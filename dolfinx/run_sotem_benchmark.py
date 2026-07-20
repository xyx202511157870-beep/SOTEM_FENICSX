"""Translate an approved SOTEM benchmark case into the DOLFINx pipeline CLI."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from atem3d.sotem_benchmark import BenchmarkCase, load_benchmark_case
import sotem_pipeline as PIPELINE_MODULE


PIPELINE_MAIN = PIPELINE_MODULE.main


_SOURCE_MESH_SIZE_M = {0: 40.0, 1: 20.0, 2: 10.0}
_RECEIVER_MESH_SIZE_M = {0: 20.0, 1: 10.0, 2: 5.0}
_OUTPUT_INTERVAL_SUBSTEPS = {0: 1, 1: 2, 2: 4}
_BOUNDARY_EXTENT_M = {0: 25_000.0, 1: 50_000.0, 2: 100_000.0}


def _float_flag(name: str, value: float) -> str:
    return f"--{name}={float(value)}"


def _source_geometry(case: BenchmarkCase) -> tuple[float, float]:
    start_x, start_y, start_z = case.source_start_down
    end_x, end_y, end_z = case.source_end_down
    source_length = math.dist((start_x, start_y, start_z), (end_x, end_y, end_z))
    horizontal_x = end_x - start_x
    horizontal_y = end_y - start_y
    horizontal_length = math.hypot(horizontal_x, horizontal_y)
    if horizontal_length == 0.0:
        raise ValueError("benchmark source must have a nonzero horizontal projection")
    receiver_x, receiver_y, _ = case.receiver_down
    relative_x = receiver_x - start_x
    relative_y = receiver_y - start_y
    parallel_offset = abs(
        (horizontal_x * relative_y - horizontal_y * relative_x) / horizontal_length
    )
    return source_length, parallel_offset


def _level_indices(level: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"S([012])T([012])B([012])", level)
    if match is None:
        raise ValueError("level must match S[0-2]T[0-2]B[0-2]")
    return tuple(int(value) for value in match.groups())


def build_pipeline_argv(
    case: BenchmarkCase,
    variant: str,
    level: str,
    workdir: str | Path,
) -> list[str]:
    """Translate a normalized benchmark model to explicit pipeline arguments."""

    if variant not in {"noip", "ip"}:
        raise ValueError("variant must be 'noip' or 'ip'")
    if variant == "ip" and case.polarization is None:
        raise ValueError("ip variant requires benchmark polarization parameters")

    source_level, time_level, boundary_level = _level_indices(level)
    source_length, parallel_offset = _source_geometry(case)
    boundary_extent = _BOUNDARY_EXTENT_M[boundary_level]

    start_x, start_y, start_down = case.source_start_down
    end_x, end_y, end_down = case.source_end_down
    receiver_x, receiver_y, receiver_down = case.receiver_down
    observation_times = ",".join(str(float(value)) for value in case.observation_times)

    argv = [
        f"--workdir={Path(workdir)}",
        _float_flag("source-start-x", start_x),
        _float_flag("source-start-y", start_y),
        _float_flag("source-start-z", -start_down),
        _float_flag("source-end-x", end_x),
        _float_flag("source-end-y", end_y),
        _float_flag("source-end-z", -end_down),
        _float_flag("source-current", case.current_a),
        "--ramp-off-time=0.0",
        _float_flag("receiver-x", receiver_x),
        _float_flag("receiver-y", receiver_y),
        _float_flag("receiver-z", -receiver_down),
        _float_flag("rho-air", case.rho_air_ohm_m),
        _float_flag("expected-source-length", source_length),
        _float_flag("expected-parallel-offset", parallel_offset),
        f"--observation-times={observation_times}",
        "--time-origin=after_ramp",
        "--time-method=theta",
        "--time-theta=1.0",
        "--initial-dc-mode=fem",
        "--magnetic-receiver-mode=faraday_integrated",
        "--magnetic-dbdt-mode=curl",
        _float_flag("source-mesh-size", _SOURCE_MESH_SIZE_M[source_level]),
        _float_flag("receiver-mesh-size", _RECEIVER_MESH_SIZE_M[source_level]),
        f"--output-interval-substeps={_OUTPUT_INTERVAL_SUBSTEPS[time_level]}",
        _float_flag("x-extent", boundary_extent),
        _float_flag("y-extent", boundary_extent),
        _float_flag("air-height", boundary_extent),
        _float_flag("earth-depth", boundary_extent),
        "--error-min-time=0.0",
        "--reference-audit-srcpts=9",
    ]

    if "rho_ohm_m" in case.earth:
        argv.append(_float_flag("rho-earth", case.earth["rho_ohm_m"]))
    else:
        layers = case.earth["layers"]
        depths = [layer["bottom_m"] for layer in layers[:-1]]
        resistivities = [layer["rho_ohm_m"] for layer in layers]
        argv.extend(
            [
                _float_flag("rho-earth", resistivities[0]),
                "--layer-depths=" + ",".join(str(float(value)) for value in depths),
                "--layer-resistivities=" + ",".join(
                    str(float(value)) for value in resistivities
                ),
            ]
        )

    argv.append(f"--polarization={'cole-cole' if variant == 'ip' else 'none'}")
    if variant == "ip":
        polarization = case.polarization
        assert polarization is not None
        argv.extend(
            [
                _float_flag("cole-rho0", polarization["rho0_ohm_m"]),
                _float_flag("cole-m", polarization["m"]),
                _float_flag("cole-tau", polarization["tau_s"]),
                _float_flag("cole-c", polarization["c"]),
                "--cole-n-terms=16",
                "--cole-f-min=0.001",
                "--cole-f-max=10000.0",
                "--cole-n-freq=81",
                "--cole-fit-tolerance=0.01",
                _float_flag("cole-layer-top", polarization["top_m"]),
                _float_flag("cole-layer-bottom", polarization["bottom_m"]),
            ]
        )
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--variant", choices=["noip", "ip"], required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--check-env-only", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args(argv)

    case = load_benchmark_case(args.case)
    pipeline_argv = build_pipeline_argv(case, args.variant, args.level, args.workdir)
    if args.check_env_only:
        pipeline_argv.append("--check-env-only")
    if args.no_install:
        pipeline_argv.append("--no-install")
    return PIPELINE_MAIN(pipeline_argv)


if __name__ == "__main__":
    raise SystemExit(main())
