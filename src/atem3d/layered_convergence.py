from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .publication_validation import (
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
