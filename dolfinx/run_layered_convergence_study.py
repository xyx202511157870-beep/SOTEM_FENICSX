#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.layered_convergence import (  # noqa: E402
    build_convergence_levels,
    build_pipeline_command_arguments,
    convergence_level_manifest,
    evaluate_convergence_study,
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
    parser.add_argument(
        "--fenicsx-python",
        default="/home/paidaxin/miniconda3/envs/fenicsx/bin/python",
    )
    args = parser.parse_args(argv)

    levels = build_convergence_levels(args.layered_root, args.output_root)
    selected_axes = tuple(args.axis or ("time", "mesh", "domain"))
    if args.mode == "evaluate":
        summary = evaluate_convergence_study(
            levels,
            selected_axes=selected_axes,
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

    for axis_name in selected_axes:
        for level in levels[axis_name]:
            if args.level and level.level_id not in args.level:
                continue
            level.workdir.mkdir(parents=True, exist_ok=True)
            if level.existing_run_dir is not None:
                _write_json(
                    level.workdir / "existing_run.json",
                    {"existing_run_dir": str(level.existing_run_dir)},
                )
                print(f"REUSE_AXIS={axis_name}")
                print(f"REUSE_LEVEL={level.level_id}")
                continue

            manifest_path = level.workdir / "case_spec.json"
            manifest = convergence_level_manifest(level)
            prior_manifest = _read_json(manifest_path)
            checkpoint_matches = (
                prior_manifest == manifest
                and (level.workdir / "forward_checkpoint.npz").is_file()
            )
            _write_json(manifest_path, manifest)
            pipeline_arguments = build_pipeline_command_arguments(level)
            if args.mode == "mesh":
                pipeline_arguments.append("--mesh-only")
                if args.force_mesh:
                    pipeline_arguments.append("--force-mesh")
            else:
                if (
                    (level.workdir / "verification_data.npz").is_file()
                    and not args.rerun
                ):
                    print(f"SKIP_COMPLETE={axis_name}/{level.level_id}")
                    continue
                pipeline_arguments.append("--checkpoint-forward")
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
            print(f"RUN_COMMAND={shlex.join(command)}")
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
