#!/usr/bin/env python3
"""Prepare and execute the approved seepage-channel verification matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_case_matrix import (  # noqa: E402
    VerificationCase,
    build_case_matrix,
    case_model,
    write_case_manifest,
)
from atem3d.seepage_channel_validation import (  # noqa: E402
    fenicsx_payload_from_csv,
    simpeg_payload_from_h5,
)
from atem3d.seepage_verification import (  # noqa: E402
    verify_fenicsx_run_contract,
    verify_simpeg_config_contract,
)


def _case_dir(output_root: str | Path, case: VerificationCase) -> Path:
    return Path(output_root) / "verification_runs" / case.solver / case.case_id


def _replace_local_segment(
    segments: list[list[Any]], *, mesh_size: float, width: float
) -> None:
    count = int(round(width / mesh_size))
    if count < 2 or not np.isclose(count * mesh_size, width):
        raise ValueError("channel cross-section must align to the requested mesh size")
    matches = [index for index, segment in enumerate(segments) if segment[:2] == [0.25, 4]]
    if len(matches) != 1:
        raise ValueError("expected one canonical 0.25 m x 4 local mesh segment")
    segments[matches[0]][:2] = [float(mesh_size), count]


def write_simpeg_case_config(
    case: VerificationCase, output_root: str | Path
) -> Path:
    """Generate one SimPEG YAML while changing only declared case controls."""

    if case.solver != "simpeg":
        raise ValueError("SimPEG config requested for a non-SimPEG case")
    base_name = (
        "seepage_channel_100m_5rx_simpeg_thin_background.yaml"
        if case.role == "background"
        else "seepage_channel_100m_5rx_simpeg_thin_channel.yaml"
    )
    with (ROOT / "examples" / base_name).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    cross = float(case.cross_section_m)
    mesh_size = float(case.local_mesh_size_m)
    for axis in ("hy", "hz"):
        _replace_local_segment(
            config["mesh"][axis], mesh_size=mesh_size, width=cross
        )
    expansion = cross - 1.0
    config["mesh"]["origin"][1] = float(config["mesh"]["origin"][1]) - expansion / 2.0
    config["mesh"]["origin"][2] = float(config["mesh"]["origin"][2]) - expansion / 2.0

    if case.role == "channel":
        half = cross / 2.0
        box = config["model"]["conductivity_boxes"][0]
        box["bounds"] = [
            [-30.0, 30.0],
            [-half, half],
            [-20.0 - half, -20.0 + half],
        ]
        box["sigma_infinity"] = float(case.conductivity_s_per_m)
        box["minimum_cells_per_cross_section"] = int(round(cross / mesh_size))

    factor = float(case.time_step_factor)
    config["time_steps"] = [
        [float(step) * factor, int(round(int(count) / factor)), *rest]
        for step, count, *rest in config["time_steps"]
    ]

    case_dir = _case_dir(output_root, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "simpeg_config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix()[2:]
    return f"/mnt/{drive}{remainder}"


def build_case_command(
    case: VerificationCase, output_root: str | Path
) -> tuple[str, ...]:
    """Return the exact subprocess command for one verification case."""

    case_dir = _case_dir(output_root, case)
    if case.solver == "simpeg":
        config = write_simpeg_case_config(case, output_root)
        return (
            sys.executable,
            "-m",
            "atem3d.cli",
            "run",
            str(config),
            "--output",
            str(case_dir / "result.h5"),
            "--data-only",
        )

    script = (
        "tools/run_fenicsx_seepage_thin_background.sh"
        if case.role == "background"
        else "tools/run_fenicsx_seepage_thin_channel.sh"
    )
    half = case.cross_section_m / 2.0
    bounds = f"-30,30;{-half:g},{half:g};{-20.0-half:g},{-20.0+half:g}"
    overrides = [
        "--workdir",
        _windows_to_wsl(case_dir),
        f"--conductivity-box-bounds={bounds}",
        "--conductivity-box-sigma",
        f"{case.conductivity_s_per_m:g}",
        "--conductivity-box-mesh-size",
        f"{case.local_mesh_size_m:g}",
        "--max-internal-dt",
        f"{1e-4 * case.time_step_factor:g}",
        "--max-internal-dt-fraction",
        f"{0.05 * case.time_step_factor:g}",
        "--magnetic-receiver-mode",
        "biot_current",
        "--magnetic-dbdt-mode",
        "biot_rate",
        "--biot-current-integration",
        "tetra4",
    ]
    shell_command = (
        f"cd {shlex.quote(_windows_to_wsl(ROOT))} && bash {shlex.quote(script)} "
        + " ".join(shlex.quote(item) for item in overrides)
    )
    return ("wsl", "-d", "Ubuntu", "--", "bash", "-lc", shell_command)


def _normalized_output(output_root: str | Path, case: VerificationCase) -> Path:
    return _case_dir(output_root, case) / "normalized.npz"


def _output_is_current(path: Path, case: VerificationCase) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as stored:
            return str(np.asarray(stored["case_fingerprint"]).item()) == case.case_fingerprint
    except (KeyError, OSError, ValueError):
        return False


def reuse_case_output(
    case: VerificationCase, output_root: str | Path
) -> Path | None:
    """Reuse an identical solver execution while retaining study provenance."""

    target = _normalized_output(output_root, case)
    search_root = Path(output_root) / "verification_runs" / case.solver
    for candidate in sorted(search_root.glob("*/normalized.npz")):
        if candidate == target:
            continue
        try:
            with np.load(candidate, allow_pickle=False) as stored:
                execution = str(np.asarray(stored["execution_fingerprint"]).item())
                if execution != case.execution_fingerprint:
                    continue
                payload = {name: np.asarray(stored[name]) for name in stored.files}
        except (KeyError, OSError, ValueError):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        payload["case_fingerprint"] = np.asarray(case.case_fingerprint)
        payload["execution_fingerprint"] = np.asarray(case.execution_fingerprint)
        payload["study"] = np.asarray(case.study)
        payload["role"] = np.asarray(case.role)
        payload["reused_from"] = np.asarray(candidate.parent.name)
        np.savez_compressed(target, **payload)
        return target
    return None


def run_case(
    case: VerificationCase,
    output_root: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Execute and normalize one case, resuming only fingerprint-matching output."""

    normalized = _normalized_output(output_root, case)
    if not force and _output_is_current(normalized, case):
        return normalized
    if not force:
        reused = reuse_case_output(case, output_root)
        if reused is not None:
            return reused
    case_dir = _case_dir(output_root, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps(case.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = build_case_command(case, output_root)
    with (case_dir / "run_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    model = case_model(case)
    if case.solver == "simpeg":
        config_path = case_dir / "simpeg_config.yaml"
        verify_simpeg_config_contract(
            config_path,
            model=model,
            case=case.role,
            expected_local_mesh_size=case.local_mesh_size_m,
        )
        payload = simpeg_payload_from_h5(case_dir / "result.h5", model=model)
    else:
        verify_fenicsx_run_contract(
            case_dir,
            model=model,
            case=case.role,
            expected_local_mesh_size=case.local_mesh_size_m,
        )
        payload = fenicsx_payload_from_csv(
            case_dir / "predictions_5rx.csv", model=model
        )
    payload = dict(payload)
    payload["base_model_fingerprint"] = np.asarray(case.model_fingerprint)
    payload["case_fingerprint"] = np.asarray(case.case_fingerprint)
    payload["execution_fingerprint"] = np.asarray(case.execution_fingerprint)
    payload["study"] = np.asarray(case.study)
    payload["role"] = np.asarray(case.role)
    np.savez_compressed(normalized, **payload)
    return normalized


def _selected_cases(solver: str | None, case_ids: list[str]) -> list[VerificationCase]:
    cases = list(build_case_matrix())
    if solver:
        cases = [case for case in cases if case.solver == solver]
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError("unknown case ids: " + ", ".join(sorted(missing)))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--solver", choices=("simpeg", "fenicsx"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cases = _selected_cases(args.solver, args.case_id)
    manifest = write_case_manifest(args.output_root, tuple(cases))
    if args.action == "manifest":
        print(manifest.resolve())
        return 0
    if args.action == "prepare":
        for case in cases:
            command = build_case_command(case, args.output_root)
            print(case.case_id + "\t" + " ".join(command))
        return 0
    for case in cases:
        print(run_case(case, args.output_root, force=args.force).resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
