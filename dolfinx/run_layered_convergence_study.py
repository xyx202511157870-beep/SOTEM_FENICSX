#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.layered_convergence import (  # noqa: E402
    PublicationMemoryContract,
    build_convergence_levels,
    build_paper_baseline_convergence_levels,
    build_pipeline_command_arguments,
    convergence_level_manifest,
    evaluate_convergence_study,
    sha256_file,
    validate_publication_preflight,
    write_convergence_reports,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _option_integer(arguments: list[str], option: str) -> int:
    if option not in arguments:
        return 0
    return int(arguments[arguments.index(option) + 1])


def _set_option_integer(arguments: list[str], option: str, value: int) -> None:
    index = arguments.index(option)
    arguments[index + 1] = str(int(value))


def _remove_option_with_value(arguments: list[str], option: str) -> None:
    index = arguments.index(option)
    del arguments[index : index + 2]


def _checkpoint_output_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as checkpoint:
        rows = np.asarray(checkpoint["rows"])
    if rows.ndim != 2:
        raise ValueError(f"checkpoint rows must be two-dimensional: {path}")
    return int(rows.shape[0])


def _publication_preflight_diagnostics(run_dir: Path) -> dict:
    import math

    import meshio

    run_dir = Path(run_dir)
    source = _read_json(run_dir / "source_diagnostics.json")
    if not isinstance(source, dict):
        raise FileNotFoundError(run_dir / "source_diagnostics.json")
    local = source.get("source_local_projection") or {}
    projection = source.get("source_projection") or {}
    receiver = source.get("receiver_location") or {}
    quadrature_points = int(local.get("quadrature_points", 0))
    missed_points = int(local.get("missed_points", quadrature_points))
    unique_hit_cells = int(local.get("unique_hit_cells", 0))
    endpoint_norm = float(projection.get("endpoint_norm", math.nan))
    after_residual = float(projection.get("after_residual", math.nan))

    mesh = meshio.read(run_dir / "verification_mesh.msh")
    cell_block_count = sum(
        int(block.data.shape[0])
        for block in mesh.cells
        if block.type in {"tetra", "triangle", "line"}
    )
    node_count = int(mesh.points.shape[0])
    return {
        "source_coverage_passed": bool(
            quadrature_points > 0
            and missed_points == 0
            and unique_hit_cells > 0
        ),
        "receiver_found": bool(receiver.get("found", False)),
        "source_divergence_passed": bool(
            math.isfinite(endpoint_norm)
            and endpoint_norm > 0.0
            and math.isfinite(after_residual)
            and after_residual <= max(1.0e-10, 1.0e-8 * endpoint_norm)
        ),
        "estimated_memory_gb": (
            cell_block_count * 2.85e-5 + node_count * 1.5e-6
        ),
        "nodes": node_count,
        "cells_blocks": cell_block_count,
        "source_quadrature_points": quadrature_points,
        "source_missed_points": missed_points,
        "source_unique_hit_cells": unique_hit_cells,
        "source_projection_after_residual": after_residual,
        "source_projection_endpoint_norm": endpoint_norm,
    }


def _write_publication_preflight(
    level,
    memory_contract: PublicationMemoryContract | None = None,
) -> dict:
    diagnostics = _publication_preflight_diagnostics(level.workdir)
    memory_limit_gb = (
        memory_contract.solver_memory_limit_gb
        if memory_contract is not None
        else 24.0
    )
    evidence = validate_publication_preflight(
        mesh_path=level.workdir / "verification_mesh.msh",
        diagnostics=diagnostics,
        memory_limit_gb=memory_limit_gb,
    )
    if level.reuse_mesh_path is not None:
        locked_mesh_path = Path(level.reuse_mesh_path)
        if not locked_mesh_path.is_file():
            raise FileNotFoundError(f"locked mesh is missing: {locked_mesh_path}")
        locked_sha256 = sha256_file(locked_mesh_path)
        if evidence["mesh_sha256"] != locked_sha256:
            raise ValueError(
                "locked_mesh_hash_mismatch: "
                f"run={evidence['mesh_sha256']}, locked={locked_sha256}"
            )
        evidence["locked_mesh_path"] = str(locked_mesh_path)
        evidence["locked_mesh_sha256"] = locked_sha256
    payload = {
        "run_id": level.run_id,
        **diagnostics,
        **evidence,
        **(memory_contract.as_dict() if memory_contract is not None else {}),
    }
    _write_json(level.workdir / "preflight.json", payload)
    return payload


def _require_publication_preflight(
    level,
    memory_contract: PublicationMemoryContract,
) -> dict:
    preflight_path = level.workdir / "preflight.json"
    preflight = _read_json(preflight_path)
    if not isinstance(preflight, dict) or not preflight.get("passed", False):
        raise ValueError(f"passing publication preflight is required: {preflight_path}")
    for key, expected_value in memory_contract.as_dict().items():
        try:
            actual_value = float(preflight[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "publication preflight memory contract changed"
            ) from exc
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("publication preflight memory contract changed")
    if not math.isclose(
        float(preflight.get("memory_limit_gb", math.nan)),
        memory_contract.solver_memory_limit_gb,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("publication preflight memory contract changed")
    mesh_path = level.workdir / "verification_mesh.msh"
    if sha256_file(mesh_path) != preflight.get("mesh_sha256"):
        raise ValueError(f"publication preflight mesh hash changed: {mesh_path}")
    if level.reuse_mesh_path is not None:
        locked_sha256 = sha256_file(Path(level.reuse_mesh_path))
        if locked_sha256 != preflight.get("locked_mesh_sha256"):
            raise ValueError("locked_mesh_hash_mismatch")
    return preflight


def _wsl_path(path: Path) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _convert_path_options_for_wsl(arguments: list[str]) -> list[str]:
    converted = list(arguments)
    for option in ("--workdir", "--reuse-mesh"):
        if option in converted:
            index = converted.index(option)
            converted[index + 1] = _wsl_path(Path(converted[index + 1]))
    return converted


def _execution_command(arguments: list[str], fenicsx_python: str) -> list[str]:
    pipeline = ROOT / "dolfinx" / "sotem_pipeline.py"
    if os.name != "nt":
        return [fenicsx_python, str(pipeline), *arguments]
    wsl_command = [
        fenicsx_python,
        _wsl_path(pipeline),
        *_convert_path_options_for_wsl(arguments),
    ]
    return [
        "wsl",
        "-d",
        "Ubuntu",
        "--",
        "bash",
        "-lc",
        shlex.join(wsl_command),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the approved layered FEniCSx convergence study."
    )
    parser.add_argument(
        "--study",
        choices=("stage1", "paper-baseline"),
        default="stage1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "output"
            / "publication_validation"
            / "convergence"
            / "layered_resistive_offset100"
        ),
    )
    parser.add_argument(
        "--layered-root",
        type=Path,
        default=ROOT / "output" / "publication_validation" / "layered",
    )
    parser.add_argument(
        "--prior-convergence-root",
        type=Path,
        default=(
            ROOT
            / "output"
            / "publication_validation"
            / "convergence"
            / "layered_resistive_offset100"
        ),
    )
    parser.add_argument(
        "--axis",
        action="append",
        choices=("time", "mesh", "domain"),
        default=[],
    )
    parser.add_argument("--level", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("mesh", "full", "evaluate"), default="mesh"
    )
    parser.add_argument("--force-mesh", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--total-memory-gb", type=float, default=20.0)
    parser.add_argument("--memory-reserve-gb", type=float, default=6.0)
    parser.add_argument(
        "--fenicsx-python",
        default="/home/paidaxin/miniconda3/envs/fenicsx/bin/python",
    )
    args = parser.parse_args(argv)
    memory_contract = (
        PublicationMemoryContract(
            args.total_memory_gb,
            args.memory_reserve_gb,
        )
        if args.study == "paper-baseline"
        else None
    )

    if args.study == "paper-baseline":
        levels = build_paper_baseline_convergence_levels(
            args.layered_root,
            args.output_root,
            args.prior_convergence_root,
        )
    else:
        levels = build_convergence_levels(args.layered_root, args.output_root)
    selected_axes = tuple(args.axis or ("time", "mesh", "domain"))
    if args.mode == "evaluate":
        summary = evaluate_convergence_study(
            levels,
            selected_axes=selected_axes,
            study_id=(
                "layered_resistive_offset100_stage2"
                if args.study == "paper-baseline"
                else "layered_resistive_offset100"
            ),
        )
        write_convergence_reports(args.output_root, summary)
        print(f"CONVERGENCE_COMPLETE={summary['complete_axis_count']}")
        print(f"CONVERGENCE_PASSED={summary['passed_axis_count']}")
        return 0 if summary["study_passed"] else 1

    known_levels = {
        level.level_id
        for axis_name in selected_axes
        for level in levels[axis_name]
    }
    unknown_levels = set(args.level).difference(known_levels)
    if unknown_levels:
        parser.error("unknown levels: " + ", ".join(sorted(unknown_levels)))

    processed_run_ids: set[str] = set()
    for axis_name in selected_axes:
        for level in levels[axis_name]:
            if args.level and level.level_id not in args.level:
                continue
            level_manifest = convergence_level_manifest(level)
            if args.study == "paper-baseline":
                _write_json(
                    args.output_root
                    / "axis_levels"
                    / axis_name
                    / level.level_id
                    / "level_pointer.json",
                    {
                        "axis": axis_name,
                        "level_id": level.level_id,
                        "run_id": level.run_id,
                        "resolved_run_dir": str(
                            level.existing_run_dir or level.workdir
                        ),
                        "source_type": (
                            "existing"
                            if level.existing_run_dir is not None
                            else "generated"
                        ),
                        "level_manifest_sha256": _manifest_sha256(level_manifest),
                    },
                )
            if level.run_id in processed_run_ids:
                print(f"SHARED_RUN={level.run_id}")
                continue
            processed_run_ids.add(level.run_id)
            level.workdir.mkdir(parents=True, exist_ok=True)
            if level.existing_run_dir is not None:
                _write_json(
                    level.workdir / "existing_run.json",
                    {"existing_run_dir": str(level.existing_run_dir)},
                )
                print(f"REUSE_AXIS={axis_name}")
                print(f"REUSE_LEVEL={level.level_id}")
                print(f"REUSE_RUN_ID={level.run_id}")
                continue

            manifest_path = level.workdir / "case_spec.json"
            manifest = level_manifest
            prior_manifest = _read_json(manifest_path)
            checkpoint_matches = (
                prior_manifest == manifest
                and (level.workdir / "forward_checkpoint.npz").is_file()
            )
            _write_json(manifest_path, manifest)
            pipeline_arguments = build_pipeline_command_arguments(
                level,
                memory_limit_gb=(
                    memory_contract.solver_memory_limit_gb
                    if memory_contract is not None
                    else None
                ),
            )
            if args.mode == "mesh":
                pipeline_arguments.append("--source-only")
                if args.force_mesh:
                    pipeline_arguments.append("--force-mesh")
            else:
                if args.study == "paper-baseline" and not args.dry_run:
                    if memory_contract is None:
                        raise AssertionError("paper-baseline memory contract missing")
                    _require_publication_preflight(level, memory_contract)
                if (
                    (level.workdir / "verification_data.npz").is_file()
                    and not args.rerun
                ):
                    print(f"SKIP_COMPLETE={axis_name}/{level.level_id}")
                    continue
                postprocess_partial = False
                if checkpoint_matches:
                    target_outputs = _option_integer(
                        pipeline_arguments,
                        "--stop-after-outputs",
                    )
                    completed_outputs = _checkpoint_output_count(
                        level.workdir / "forward_checkpoint.npz"
                    )
                    remaining_outputs = max(target_outputs - completed_outputs, 0)
                    if target_outputs > 0 and remaining_outputs == 0:
                        partial_path = level.workdir / "forward_partial.npz"
                        if not partial_path.is_file():
                            raise FileNotFoundError(
                                "completed checkpoint requires forward_partial.npz for postprocessing: "
                                + str(partial_path)
                            )
                        _remove_option_with_value(
                            pipeline_arguments,
                            "--stop-after-outputs",
                        )
                        pipeline_arguments.append("--postprocess-partial")
                        postprocess_partial = True
                    elif target_outputs > 0:
                        _set_option_integer(
                            pipeline_arguments,
                            "--stop-after-outputs",
                            remaining_outputs,
                        )
                if not postprocess_partial:
                    pipeline_arguments.extend(
                        ("--checkpoint-forward", "--checkpoint-interval-steps", "10")
                    )
                    if checkpoint_matches:
                        pipeline_arguments.append("--resume-forward")

            command = _execution_command(
                pipeline_arguments,
                args.fenicsx_python,
            )
            (level.workdir / "command.txt").write_text(
                shlex.join(command) + "\n",
                encoding="utf-8",
            )
            print(f"RUN_AXIS={axis_name}")
            print(f"RUN_LEVEL={level.level_id}")
            print(f"RUN_ID={level.run_id}")
            print(f"RUN_COMMAND={shlex.join(command)}")
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
            if args.mode == "mesh" and (
                not args.dry_run
                or (
                    (level.workdir / "verification_mesh.msh").is_file()
                    and (level.workdir / "source_diagnostics.json").is_file()
                )
            ):
                _write_publication_preflight(level, memory_contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
