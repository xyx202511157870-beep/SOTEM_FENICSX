from __future__ import annotations

import csv
import hashlib
import json
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
    evaluate_errors_csv,
)


@dataclass(frozen=True)
class ConvergenceLevel:
    axis: str
    level_id: str
    run_id: str
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


@dataclass(frozen=True)
class PublicationMemoryContract:
    total_memory_gb: float = 20.0
    reserve_memory_gb: float = 6.0

    def __post_init__(self) -> None:
        total = float(self.total_memory_gb)
        reserve = float(self.reserve_memory_gb)
        if (
            not math.isfinite(total)
            or total <= 0.0
            or not math.isfinite(reserve)
            or reserve < 0.0
            or reserve >= total
        ):
            raise ValueError(
                "memory contract requires finite total > reserve >= 0"
            )

    @property
    def solver_memory_limit_gb(self) -> float:
        return float(self.total_memory_gb) - float(self.reserve_memory_gb)

    def as_dict(self) -> dict[str, float]:
        return {
            "total_memory_gb": float(self.total_memory_gb),
            "reserve_memory_gb": float(self.reserve_memory_gb),
            "solver_memory_limit_gb": self.solver_memory_limit_gb,
        }


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
            "run_id": f"{axis}_{level_id}",
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


def build_paper_baseline_convergence_levels(
    layered_root: Path,
    output_root: Path,
    prior_convergence_root: Path,
) -> dict[str, tuple[ConvergenceLevel, ...]]:
    layered_root = Path(layered_root)
    output_root = Path(output_root)
    prior_convergence_root = Path(prior_convergence_root)
    case_id = "resistive_basement_rho1000_offset100"
    time_coarse = layered_root / "domain12000" / case_id
    domain_small = prior_convergence_root / "time" / "fine"
    locked_mesh = time_coarse / "verification_mesh.msh"
    runs_root = output_root / "runs"

    def level(
        axis: str,
        level_id: str,
        run_id: str,
        **overrides,
    ) -> ConvergenceLevel:
        values = {
            "axis": axis,
            "level_id": level_id,
            "run_id": run_id,
            "x_extent": 12000.0,
            "y_extent": 12000.0,
            "earth_depth": 12000.0,
            "air_height": 1200.0,
            "far_field_mesh_size": 750.0,
            "source_mesh_size": 8.0,
            "receiver_mesh_size": 6.0,
            "max_internal_dt": 1.25e-5,
            "max_internal_dt_fraction": 0.005,
            "workdir": runs_root / run_id,
        }
        values.update(overrides)
        return ConvergenceLevel(**values)

    baseline = {
        "run_id": "baseline_12km_dt005_mesh8_6",
        "reuse_mesh_path": locked_mesh,
    }
    return {
        "time": (
            level(
                "time",
                "coarse",
                "existing_12km_dt01",
                existing_run_dir=time_coarse,
                max_internal_dt=2.5e-5,
                max_internal_dt_fraction=0.01,
            ),
            level("time", "standard", **baseline),
            level(
                "time",
                "fine",
                "time_fine_12km_dt0025_mesh8_6",
                max_internal_dt=6.25e-6,
                max_internal_dt_fraction=0.0025,
                reuse_mesh_path=locked_mesh,
            ),
        ),
        "mesh": (
            level(
                "mesh",
                "coarse",
                "mesh_coarse_12km_dt005_mesh12_9",
                source_mesh_size=12.0,
                receiver_mesh_size=9.0,
            ),
            level("mesh", "standard", **baseline),
            level(
                "mesh",
                "fine",
                "mesh_fine_12km_dt005_mesh6_4p5",
                source_mesh_size=6.0,
                receiver_mesh_size=4.5,
            ),
        ),
        "domain": (
            level(
                "domain",
                "small",
                "existing_6km_dt005",
                existing_run_dir=domain_small,
                x_extent=6000.0,
                y_extent=6000.0,
                earth_depth=6000.0,
                air_height=600.0,
            ),
            level("domain", "standard", **baseline),
            level(
                "domain",
                "large",
                "domain_large_18km_dt005_mesh8_6",
                x_extent=18000.0,
                y_extent=18000.0,
                earth_depth=18000.0,
                air_height=1800.0,
            ),
        ),
    }


def _number(value: float) -> str:
    return format(float(value), ".12g")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_publication_preflight(
    *,
    mesh_path: Path,
    diagnostics: dict,
    memory_limit_gb: float = 24.0,
) -> dict:
    mesh_path = Path(mesh_path)
    reasons: list[str] = []
    if not mesh_path.is_file():
        reasons.append("mesh missing")
    if not diagnostics.get("source_coverage_passed", False):
        reasons.append("source coverage failed")
    if not diagnostics.get("receiver_found", False):
        reasons.append("receiver location failed")
    if not diagnostics.get("source_divergence_passed", False):
        reasons.append("source divergence failed")
    estimated_memory_gb = float(diagnostics.get("estimated_memory_gb", math.inf))
    if (
        not math.isfinite(estimated_memory_gb)
        or estimated_memory_gb > float(memory_limit_gb)
    ):
        reasons.append(f"estimated memory exceeds {float(memory_limit_gb):g} GB")
    if reasons:
        raise ValueError("; ".join(reasons))
    return {
        "passed": True,
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "estimated_memory_gb": estimated_memory_gb,
        "memory_limit_gb": float(memory_limit_gb),
    }


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


def evaluate_axis_metrics(
    axis: str,
    comparisons: dict,
    *,
    large_external_passed: bool | None = None,
) -> dict:
    reasons: list[str] = []
    ordering_tolerance_percent = 0.1
    if axis in {"time", "mesh"}:
        coarse_to_standard = comparisons["coarse_to_standard"]
        standard_to_fine = comparisons["standard_to_fine"]
        if float(standard_to_fine["median_percent"]) > 1.0:
            reasons.append(f"{axis}_median_above_1pct")
        if float(standard_to_fine["rms_percent"]) > 2.0:
            reasons.append(f"{axis}_rms_above_2pct")
        if float(standard_to_fine["max_percent"]) > 5.0:
            reasons.append(f"{axis}_max_above_5pct")
        if (
            float(coarse_to_standard["rms_percent"])
            + ordering_tolerance_percent
            < float(standard_to_fine["rms_percent"])
        ):
            reasons.append(f"{axis}_rms_not_decreasing")
    elif axis == "domain":
        small_to_standard = comparisons["small_to_standard"]
        standard_to_large = comparisons["standard_to_large"]
        if (
            float(small_to_standard["rms_percent"])
            + ordering_tolerance_percent
            < float(standard_to_large["rms_percent"])
        ):
            reasons.append("domain_rms_not_decreasing")
        if large_external_passed is not True:
            reasons.append("domain_large_external_gate_failed")
    else:
        raise ValueError(f"unknown convergence axis: {axis}")
    return {
        "axis": axis,
        "passed": not reasons,
        "blocking_reasons": reasons,
    }


def read_run_metadata(run_dir: Path) -> dict:
    import meshio

    run_dir = Path(run_dir)
    mesh = meshio.read(run_dir / "verification_mesh.msh")
    tetrahedra = [block.data for block in mesh.cells if block.type == "tetra"]
    tetrahedron_count = sum(int(cells.shape[0]) for cells in tetrahedra)
    cell_block_count = sum(
        int(block.data.shape[0])
        for block in mesh.cells
        if block.type in {"tetra", "triangle", "line"}
    )
    if tetrahedron_count <= 0:
        raise ValueError(f"{run_dir} mesh contains no tetrahedra")
    tetra = np.vstack(tetrahedra).astype(np.int64, copy=False)
    edges = np.vstack(
        [
            tetra[:, [0, 1]],
            tetra[:, [0, 2]],
            tetra[:, [0, 3]],
            tetra[:, [1, 2]],
            tetra[:, [1, 3]],
            tetra[:, [2, 3]],
        ]
    )
    edges.sort(axis=1)
    nedelec_dofs = int(np.unique(edges, axis=0).shape[0])

    solver_data_path = run_dir / "forward_partial.npz"
    if not solver_data_path.is_file():
        solver_data_path = run_dir / "verification_data.npz"
    with np.load(solver_data_path, allow_pickle=False) as payload:
        if "internal_solver_steps" not in payload.files:
            raise ValueError(
                f"{solver_data_path} is missing internal_solver_steps"
            )
        internal_step_count = int(
            np.asarray(payload["internal_solver_steps"]).size
        )
        ksp_keys = {
            "solver_iterations",
            "solver_reasons",
            "solver_residuals",
        }
        present_ksp_keys = ksp_keys.intersection(payload.files)
        if present_ksp_keys and present_ksp_keys != ksp_keys:
            missing = sorted(ksp_keys.difference(present_ksp_keys))
            raise ValueError(f"KSP evidence is missing arrays: {missing}")
        if present_ksp_keys:
            iterations = np.asarray(payload["solver_iterations"], dtype=int).reshape(-1)
            reasons = np.asarray(payload["solver_reasons"], dtype=int).reshape(-1)
            residuals = np.asarray(payload["solver_residuals"], dtype=float).reshape(-1)
            if not (iterations.size == reasons.size == residuals.size):
                raise ValueError("KSP evidence arrays must have equal lengths")
            if iterations.size == 0:
                raise ValueError("KSP evidence contains no output solves")
            if np.any(iterations < 0):
                raise ValueError("KSP evidence contains negative iterations")
            if np.any(reasons <= 0):
                raise ValueError("KSP convergence reasons must all be positive")
            if not np.all(np.isfinite(residuals)):
                raise ValueError("KSP evidence must contain finite residuals")
            ksp_metadata = {
                "ksp_output_solve_count": int(iterations.size),
                "ksp_iterations_median": float(np.median(iterations)),
                "ksp_iterations_max": int(np.max(iterations)),
                "ksp_residual_max": float(np.max(residuals)),
                "ksp_all_converged": True,
            }
        else:
            ksp_metadata = {
                "ksp_output_solve_count": 0,
                "ksp_iterations_median": None,
                "ksp_iterations_max": None,
                "ksp_residual_max": None,
                "ksp_all_converged": None,
            }

    forward_runtime_seconds = 0.0
    timing_path = run_dir / "timing_events.jsonl"
    with timing_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {timing_path} line {line_number}"
                ) from exc
            seconds = event.get("seconds")
            if event.get("event") == "forward_done" and seconds is not None:
                seconds = float(seconds)
                if math.isfinite(seconds) and seconds >= 0.0:
                    forward_runtime_seconds += seconds

    node_count = int(mesh.points.shape[0])
    return {
        "mesh_sha256": sha256_file(run_dir / "verification_mesh.msh"),
        "nodes": node_count,
        "tetrahedra": tetrahedron_count,
        "cells_blocks": cell_block_count,
        "nedelec_dofs": nedelec_dofs,
        "internal_step_count": internal_step_count,
        **ksp_metadata,
        "forward_runtime_seconds": forward_runtime_seconds,
        "estimated_memory_gb": (
            cell_block_count * 2.85e-5 + node_count * 1.5e-6
        ),
    }


def _json_safe_summary(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe_summary(item)
            for key, item in value.items()
            if not str(key).startswith("_") and key not in {"times", "relative"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_summary(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _comparison_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for axis in summary.get("axes", []):
        for comparison in axis.get("comparisons", []):
            rows.append(
                {
                    "axis": str(axis["axis"]),
                    "comparison_id": str(comparison["comparison_id"]),
                    "sample_count": int(comparison["sample_count"]),
                    "excluded_below_floor_count": int(
                        comparison["excluded_below_floor_count"]
                    ),
                    "amplitude_floor": float(comparison["amplitude_floor"]),
                    "median_percent": float(comparison["median_percent"]),
                    "rms_percent": float(comparison["rms_percent"]),
                    "max_percent": float(comparison["max_percent"]),
                    "axis_passed": bool(axis["passed"]),
                    "blocking_reasons": ";".join(axis.get("blocking_reasons", [])),
                }
            )
    return rows


def _write_convergence_curves(path: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    axes_payload = list(summary.get("axes", []))
    row_count = max(1, len(axes_payload))
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(12.0, max(8.0, 4.0 * row_count)),
        squeeze=False,
    )
    if not axes_payload:
        axes[0, 0].text(0.5, 0.5, "No completed convergence axes", ha="center")
        axes[0, 0].set_axis_off()
    for row, axis_payload in enumerate(axes_payload):
        axis = axes[row, 0]
        responses = axis_payload.get("_responses", {})
        for level_id, response in responses.items():
            axis.loglog(
                response.times,
                np.abs(response.dbzdt),
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=str(level_id),
            )
        axis.set_title(f"{axis_payload['axis'].title()} convergence")
        axis.set_xlabel("Time after ramp (s)")
        axis.set_ylabel("abs(dBz/dt) (T/s)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, metadata={"Software": "atem3d"})
    plt.close(figure)


def _write_convergence_differences(path: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    axes_payload = list(summary.get("axes", []))
    row_count = max(1, len(axes_payload))
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(12.0, max(8.0, 4.0 * row_count)),
        squeeze=False,
    )
    if not axes_payload:
        axes[0, 0].text(0.5, 0.5, "No completed convergence axes", ha="center")
        axes[0, 0].set_axis_off()
    for row, axis_payload in enumerate(axes_payload):
        axis = axes[row, 0]
        for comparison in axis_payload.get("comparisons", []):
            axis.semilogx(
                comparison["times"],
                100.0 * np.asarray(comparison["relative"], dtype=float),
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=str(comparison["comparison_id"]),
            )
        for threshold, style in ((1.0, ":"), (2.0, "--"), (5.0, "-.")):
            axis.axhline(
                threshold,
                color="0.35",
                linestyle=style,
                linewidth=1.0,
                label=f"{threshold:g}%" if row == 0 else None,
            )
        axis.set_title(f"{axis_payload['axis'].title()} pairwise difference")
        axis.set_xlabel("Time after ramp (s)")
        axis.set_ylabel("Relative difference (%)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, metadata={"Software": "atem3d"})
    plt.close(figure)


def write_convergence_reports(output_dir: Path, summary: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_summary = _json_safe_summary(summary)
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(public_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if "candidate_baseline" in summary:
        acceptance = {
            "study_id": summary.get("study_id"),
            "coordinate_convention": summary.get("coordinate_convention"),
            "accepted_for_paper_figures": bool(summary.get("study_passed", False)),
            "candidate_baseline": summary["candidate_baseline"],
            "axis_gates": [
                {
                    "axis": axis["axis"],
                    "status": axis["status"],
                    "passed": bool(axis["passed"]),
                    "blocking_reasons": list(axis.get("blocking_reasons", [])),
                }
                for axis in summary.get("axes", [])
            ],
        }
        (output_dir / "baseline_acceptance.json").write_text(
            json.dumps(_json_safe_summary(acceptance), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    rows = _comparison_rows(summary)
    fields = [
        "axis",
        "comparison_id",
        "sample_count",
        "excluded_below_floor_count",
        "amplitude_floor",
        "median_percent",
        "rms_percent",
        "max_percent",
        "axis_passed",
        "blocking_reasons",
    ]
    with (output_dir / "convergence_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Layered FEniCSx Convergence Report",
        "",
        f"- Study: {summary.get('study_id', '')}",
        f"- Study passed: {bool(summary.get('study_passed', False))}",
        f"- Coordinates: {summary.get('coordinate_convention', '')}",
        "- Time/mesh gates: median <= 1%, RMS <= 2%, maximum <= 5%.",
        "- Domain gate: decreasing RMS response change and a passing large-domain empymod gate.",
        "- Effective samples satisfy abs(reference) >= 1e-6 * peak(abs(reference)).",
        "",
        "| Axis | Comparison | N | Median (%) | RMS (%) | Maximum (%) | Axis pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        markdown.append(
            "| {axis} | {comparison_id} | {sample_count} | "
            "{median_percent:.6g} | {rms_percent:.6g} | {max_percent:.6g} | "
            "{axis_passed} |".format(**row)
        )
    markdown.extend(("", "## Blocking Reasons", ""))
    for axis in summary.get("axes", []):
        reasons = axis.get("blocking_reasons", [])
        markdown.append(
            f"- {axis['axis']}: {', '.join(reasons) if reasons else 'none'}"
        )
    (output_dir / "convergence_report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    _write_convergence_curves(output_dir / "convergence_curves.png", summary)
    _write_convergence_differences(
        output_dir / "convergence_differences.png",
        summary,
    )


def convergence_level_manifest(level: ConvergenceLevel) -> dict:
    reuse_mesh = None
    if level.reuse_mesh_path is not None:
        reuse_mesh_path = Path(level.reuse_mesh_path)
        reuse_mesh = {
            "path": str(reuse_mesh_path),
            "sha256": (
                sha256_file(reuse_mesh_path) if reuse_mesh_path.is_file() else None
            ),
        }
    return {
        "axis": level.axis,
        "level_id": level.level_id,
        "run_id": level.run_id,
        "x_extent": level.x_extent,
        "y_extent": level.y_extent,
        "earth_depth": level.earth_depth,
        "air_height": level.air_height,
        "far_field_mesh_size": level.far_field_mesh_size,
        "source_mesh_size": level.source_mesh_size,
        "receiver_mesh_size": level.receiver_mesh_size,
        "max_internal_dt": level.max_internal_dt,
        "max_internal_dt_fraction": level.max_internal_dt_fraction,
        "workdir": str(level.workdir),
        "existing_run_dir": (
            str(level.existing_run_dir)
            if level.existing_run_dir is not None
            else None
        ),
        "reuse_mesh_path": (
            str(level.reuse_mesh_path)
            if level.reuse_mesh_path is not None
            else None
        ),
        "reuse_mesh": reuse_mesh,
        "effective_output_count": 25,
        "coordinate_convention": (
            "z=0 ground; underground positive; air negative"
        ),
    }


def resolved_run_dir(level: ConvergenceLevel) -> Path:
    return (
        Path(level.existing_run_dir)
        if level.existing_run_dir is not None
        else Path(level.workdir)
    )


def _effective_prefix(response: ConvergenceResponse, count: int = 25) -> ConvergenceResponse:
    if response.times.size < count:
        raise ValueError(
            f"response has {response.times.size} samples; {count} are required"
        )
    result = ConvergenceResponse(
        times=response.times[:count].copy(),
        dbzdt=response.dbzdt[:count].copy(),
        reference=response.reference[:count].copy(),
    )
    _validate_response(result)
    return result


def evaluate_convergence_study(
    levels: dict[str, tuple[ConvergenceLevel, ...]],
    *,
    selected_axes: tuple[str, ...] = ("time", "mesh", "domain"),
    study_id: str = "layered_resistive_offset100",
) -> dict:
    axis_summaries: list[dict] = []
    for axis_name in selected_axes:
        axis_levels = levels[axis_name]
        run_dirs = {
            level.level_id: resolved_run_dir(level) for level in axis_levels
        }
        missing = [
            level_id
            for level_id, run_dir in run_dirs.items()
            if not (run_dir / "verification_data.npz").is_file()
        ]
        if missing:
            axis_summaries.append(
                {
                    "axis": axis_name,
                    "status": "incomplete",
                    "passed": False,
                    "blocking_reasons": [
                        f"missing_level_{level_id}" for level_id in missing
                    ],
                    "levels": [
                        {
                            "level_id": level.level_id,
                            "run_dir": str(run_dirs[level.level_id]),
                            "status": (
                                "missing"
                                if level.level_id in missing
                                else "complete"
                            ),
                        }
                        for level in axis_levels
                    ],
                    "comparisons": [],
                    "_responses": {},
                }
            )
            continue

        responses = {
            level_id: _effective_prefix(load_response(run_dir))
            for level_id, run_dir in run_dirs.items()
        }
        if axis_name in {"time", "mesh"}:
            comparison_pairs = (
                ("coarse_to_standard", "coarse", "standard"),
                ("standard_to_fine", "standard", "fine"),
            )
        else:
            comparison_pairs = (
                ("small_to_standard", "small", "standard"),
                ("standard_to_large", "standard", "large"),
            )
        comparisons: list[dict] = []
        comparison_map: dict[str, dict] = {}
        for comparison_id, coarse_id, fine_id in comparison_pairs:
            metrics = compare_responses(
                responses[coarse_id],
                responses[fine_id],
            )
            comparison = {"comparison_id": comparison_id, **metrics}
            comparisons.append(comparison)
            comparison_map[comparison_id] = metrics

        large_external_passed = None
        external_reference_gate = None
        if axis_name == "domain":
            external_reference_gate = evaluate_errors_csv(
                run_dirs["large"] / "errors.csv"
            )
            large_external_passed = bool(
                external_reference_gate["publication_gate_passed"]
            )
        gate = evaluate_axis_metrics(
            axis_name,
            comparison_map,
            large_external_passed=large_external_passed,
        )
        level_rows = []
        for level in axis_levels:
            run_dir = run_dirs[level.level_id]
            level_rows.append(
                {
                    "level_id": level.level_id,
                    "run_dir": str(run_dir),
                    "status": "complete",
                    **read_run_metadata(run_dir),
                }
            )
        axis_summary = {
            "axis": axis_name,
            "status": "complete",
            "passed": gate["passed"],
            "blocking_reasons": gate["blocking_reasons"],
            "levels": level_rows,
            "comparisons": comparisons,
            "_responses": responses,
        }
        if external_reference_gate is not None:
            axis_summary["large_external_reference_gate"] = (
                external_reference_gate
            )
        axis_summaries.append(axis_summary)

    complete_count = sum(
        axis["status"] == "complete" for axis in axis_summaries
    )
    passed_count = sum(bool(axis["passed"]) for axis in axis_summaries)
    study_passed = (
            complete_count == len(axis_summaries)
            and passed_count == len(axis_summaries)
    )
    if study_id == "layered_resistive_offset100_stage2":
        study_passed = study_passed and set(selected_axes) == {
            "time",
            "mesh",
            "domain",
        }
    result = {
        "study_id": study_id,
        "study_passed": study_passed,
        "coordinate_convention": (
            "z=0 ground; underground positive; air negative"
        ),
        "complete_axis_count": complete_count,
        "passed_axis_count": passed_count,
        "axes": axis_summaries,
    }
    if study_id == "layered_resistive_offset100_stage2":
        baseline = levels["time"][1]
        baseline_dir = resolved_run_dir(baseline)
        baseline_complete = (
            (baseline_dir / "verification_data.npz").is_file()
            and (baseline_dir / "errors.csv").is_file()
        )
        candidate_baseline = {
            "run_id": baseline.run_id,
            "run_dir": str(baseline_dir),
            "status": "complete" if baseline_complete else "incomplete",
            "accepted_for_paper_figures": bool(study_passed),
            "external_reference_gate": (
                evaluate_errors_csv(baseline_dir / "errors.csv")
                if baseline_complete
                else None
            ),
        }
        if baseline_complete:
            candidate_baseline.update(read_run_metadata(baseline_dir))
        result["candidate_baseline"] = candidate_baseline
    return result
