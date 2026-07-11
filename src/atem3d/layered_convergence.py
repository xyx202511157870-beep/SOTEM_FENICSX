from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .publication_validation import (
    DBZDT_AMPLITUDE_FLOOR_FRACTION,
    LayeredRunProfile,
    build_layered_cases,
    build_pipeline_arguments,
)


@dataclass(frozen=True)
class ConvergenceLevel:
    axis: str
    level_id: str
    x_extent: float
    y_extent: float
    earth_depth: float
    air_height: float
    far_field_mesh_size: float
    source_mesh_size: float
    receiver_mesh_size: float
    max_internal_dt: float
    max_internal_dt_fraction: float
    workdir: Path
    existing_run_dir: Path | None = None
    reuse_mesh_path: Path | None = None


@dataclass(frozen=True)
class ConvergenceResponse:
    times: np.ndarray
    dbzdt: np.ndarray
    reference: np.ndarray


def build_convergence_levels(
    layered_root: Path,
    output_root: Path,
) -> dict[str, tuple[ConvergenceLevel, ...]]:
    layered_root = Path(layered_root)
    output_root = Path(output_root)
    case_id = "resistive_basement_rho1000_offset100"
    baseline = layered_root / "domain6000" / case_id
    large = layered_root / "domain12000" / case_id

    def level(axis: str, level_id: str, **overrides) -> ConvergenceLevel:
        values = {
            "axis": axis,
            "level_id": level_id,
            "x_extent": 6000.0,
            "y_extent": 6000.0,
            "earth_depth": 6000.0,
            "air_height": 600.0,
            "far_field_mesh_size": 750.0,
            "source_mesh_size": 8.0,
            "receiver_mesh_size": 6.0,
            "max_internal_dt": 2.5e-5,
            "max_internal_dt_fraction": 0.01,
            "workdir": output_root / axis / level_id,
        }
        values.update(overrides)
        return ConvergenceLevel(**values)

    return {
        "time": (
            level(
                "time",
                "coarse",
                max_internal_dt=5.0e-5,
                max_internal_dt_fraction=0.02,
                reuse_mesh_path=baseline / "verification_mesh.msh",
            ),
            level("time", "standard", existing_run_dir=baseline),
            level(
                "time",
                "fine",
                max_internal_dt=1.25e-5,
                max_internal_dt_fraction=0.005,
                reuse_mesh_path=baseline / "verification_mesh.msh",
            ),
        ),
        "mesh": (
            level(
                "mesh",
                "coarse",
                source_mesh_size=12.0,
                receiver_mesh_size=9.0,
            ),
            level("mesh", "standard", existing_run_dir=baseline),
            level(
                "mesh",
                "fine",
                source_mesh_size=6.0,
                receiver_mesh_size=4.5,
            ),
        ),
        "domain": (
            level(
                "domain",
                "small",
                x_extent=3000.0,
                y_extent=3000.0,
                earth_depth=3000.0,
                air_height=600.0,
            ),
            level("domain", "standard", existing_run_dir=baseline),
            level(
                "domain",
                "large",
                x_extent=12000.0,
                y_extent=12000.0,
                earth_depth=12000.0,
                air_height=1200.0,
                existing_run_dir=large,
            ),
        ),
    }


def _number(value: float) -> str:
    return format(float(value), ".12g")


def _replace_option(arguments: list[str], option: str, value: str) -> None:
    index = arguments.index(option)
    arguments[index + 1] = value


def build_pipeline_command_arguments(level: ConvergenceLevel) -> list[str]:
    case = build_layered_cases(
        offsets=(100.0,),
        basement_resistivities=(1000.0,),
    )[0]
    profile = LayeredRunProfile(
        profile_id=f"convergence_{level.axis}_{level.level_id}",
        x_extent=level.x_extent,
        y_extent=level.y_extent,
        air_height=level.air_height,
        earth_depth=level.earth_depth,
        far_field_mesh_size=level.far_field_mesh_size,
        max_internal_dt=level.max_internal_dt,
    )
    arguments = build_pipeline_arguments(case, profile, level.workdir)
    _replace_option(
        arguments,
        "--source-mesh-size",
        _number(level.source_mesh_size),
    )
    _replace_option(
        arguments,
        "--receiver-mesh-size",
        _number(level.receiver_mesh_size),
    )
    _replace_option(
        arguments,
        "--max-internal-dt-fraction",
        _number(level.max_internal_dt_fraction),
    )
    arguments.extend(("--stop-after-outputs", "25"))
    if level.reuse_mesh_path is not None:
        arguments.extend(("--reuse-mesh", str(level.reuse_mesh_path)))
    return arguments


def _validate_response(
    response: ConvergenceResponse,
    source: Path | str = "response",
) -> None:
    arrays = (response.times, response.dbzdt, response.reference)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(f"{source} arrays must be one-dimensional")
    if len({array.size for array in arrays}) != 1:
        raise ValueError(f"{source} arrays must have matching lengths")
    if response.times.size < 3:
        raise ValueError(f"{source} must contain at least three samples")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{source} contains nonfinite values")
    if np.any(response.times <= 0.0) or np.any(np.diff(response.times) <= 0.0):
        raise ValueError(f"{source} observation times must be strictly increasing")
    if float(np.min(np.diff(np.log(response.times)))) < 1.0e-6:
        raise ValueError(f"{source} observation grid contains near-duplicate times")


def load_response(run_dir: Path) -> ConvergenceResponse:
    path = Path(run_dir) / "verification_data.npz"
    with np.load(path, allow_pickle=False) as payload:
        required = {"times", "fem", "empymod", "components"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
        components = [str(value) for value in payload["components"].tolist()]
        if "dBzdt" not in components:
            raise ValueError(f"{path} components do not include dBzdt")
        component_index = components.index("dBzdt")
        fem = np.asarray(payload["fem"], dtype=float)
        reference = np.asarray(payload["empymod"], dtype=float)
        if fem.ndim != 2 or reference.ndim != 2:
            raise ValueError(f"{path} fem and empymod arrays must be two-dimensional")
        if component_index >= fem.shape[1] or component_index >= reference.shape[1]:
            raise ValueError(f"{path} component count does not match data columns")
        response = ConvergenceResponse(
            times=np.asarray(payload["times"], dtype=float),
            dbzdt=fem[:, component_index],
            reference=reference[:, component_index],
        )
    _validate_response(response, path)
    return response


def compare_responses(
    coarse: ConvergenceResponse,
    fine: ConvergenceResponse,
) -> dict:
    _validate_response(coarse)
    _validate_response(fine)
    if coarse.times.shape != fine.times.shape or not np.allclose(
        coarse.times,
        fine.times,
        rtol=1.0e-12,
        atol=1.0e-30,
    ):
        raise ValueError("responses use different observation grids")

    amplitude_floor = (
        float(np.max(np.abs(fine.reference)))
        * DBZDT_AMPLITUDE_FLOOR_FRACTION
    )
    gate_mask = np.abs(fine.reference) >= amplitude_floor
    sample_count = int(np.count_nonzero(gate_mask))
    if sample_count < 3:
        raise ValueError("fewer than three samples exceed the reference amplitude floor")
    denominator = np.abs(fine.dbzdt[gate_mask])
    if np.any(denominator == 0.0):
        raise ValueError("fine response is zero inside the effective-amplitude window")

    relative = (
        np.abs(coarse.dbzdt[gate_mask] - fine.dbzdt[gate_mask])
        / denominator
    )
    return {
        "sample_count": sample_count,
        "excluded_below_floor_count": int(gate_mask.size - sample_count),
        "amplitude_floor": amplitude_floor,
        "median_percent": 100.0 * float(statistics.median(relative.tolist())),
        "rms_percent": 100.0 * float(math.sqrt(np.mean(relative * relative))),
        "max_percent": 100.0 * float(np.max(relative)),
        "times": fine.times[gate_mask].copy(),
        "relative": relative,
    }
