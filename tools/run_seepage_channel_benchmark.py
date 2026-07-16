#!/usr/bin/env python3
"""Generate solver inputs/results for the seepage-channel benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.seepage_channel_validation import (  # noqa: E402
    aggregate_result_directory,
    save_empymod_background,
    sha256_file,
    simpeg_payload_from_h5,
    validate_result_payload,
)
from atem3d.seepage_channel_model import benchmark_model  # noqa: E402
from atem3d.seepage_verification import verify_simpeg_config_contract  # noqa: E402


VARIANT_CONFIGS = {
    "baseline_60x10x10": {
        "simpeg": {
            "background": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_background.yaml",
            "channel": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_channel.yaml",
        },
        "fenicsx": {
            "background": ROOT / "tools" / "run_fenicsx_seepage_background.sh",
            "channel": ROOT / "tools" / "run_fenicsx_seepage_channel.sh",
        },
    },
    "thin_60x1x1": {
        "simpeg": {
            "background": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_thin_background.yaml",
            "channel": ROOT / "examples" / "seepage_channel_100m_5rx_simpeg_thin_channel.yaml",
        },
        "fenicsx": {
            "background": ROOT / "tools" / "run_fenicsx_seepage_thin_background.sh",
            "channel": ROOT / "tools" / "run_fenicsx_seepage_thin_channel.sh",
        },
    },
}
SIMPEG_CONFIGS = VARIANT_CONFIGS["baseline_60x10x10"]["simpeg"]
VARIANT_CHOICES = tuple(VARIANT_CONFIGS)


@dataclass(frozen=True)
class RunJob:
    name: str
    command: tuple[str, ...]
    required_inputs: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]


def _wsl_project_path() -> str:
    drive = ROOT.drive.rstrip(":").lower()
    remainder = ROOT.as_posix()[2:]
    return f"/mnt/{drive}{remainder}"


def build_run_plan(
    output_root: str | Path,
    *,
    variant: str = "baseline_60x10x10",
) -> list[RunJob]:
    output = Path(output_root).resolve()
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"unknown benchmark variant: {variant}")
    variant_config = VARIANT_CONFIGS[variant]
    simpeg_configs = variant_config["simpeg"]
    fenicsx_scripts = variant_config["fenicsx"]
    runner = str(Path(__file__).resolve())
    plotter = str(ROOT / "tools" / "plot_seepage_channel_benchmark.py")
    wsl_root = _wsl_project_path()

    def python_job(name: str, arguments: list[str], outputs: list[Path], inputs=None):
        return RunJob(
            name=name,
            command=(sys.executable, runner, *arguments),
            required_inputs=tuple(inputs or ()),
            expected_outputs=tuple(outputs),
        )

    return [
        python_job(
            "empymod_background",
            ["empymod", "--variant", variant, "--output-root", str(output)],
            [output / "empymod_background.npz"],
        ),
        python_job(
            "simpeg_background",
            ["simpeg", "--case", "background", "--variant", variant, "--output-root", str(output)],
            [output / "simpeg_background.h5", output / "simpeg_background.npz", output / "simpeg_background_provenance.json"],
            [simpeg_configs["background"]],
        ),
        python_job(
            "simpeg_channel",
            ["simpeg", "--case", "channel", "--variant", variant, "--output-root", str(output)],
            [output / "simpeg_channel.h5", output / "simpeg_channel.npz", output / "simpeg_channel_provenance.json"],
            [simpeg_configs["channel"]],
        ),
        RunJob(
            name="fenicsx_background",
            command=("wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"cd '{wsl_root}' && bash tools/{fenicsx_scripts['background'].name}"),
            required_inputs=(fenicsx_scripts["background"],),
            expected_outputs=(output / "fenicsx_background" / "predictions_5rx.csv",),
        ),
        RunJob(
            name="fenicsx_channel",
            command=("wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"cd '{wsl_root}' && bash tools/{fenicsx_scripts['channel'].name}"),
            required_inputs=(fenicsx_scripts["channel"],),
            expected_outputs=(output / "fenicsx_channel" / "predictions_5rx.csv",),
        ),
        python_job(
            "aggregate",
            ["aggregate", "--variant", variant, "--output-root", str(output)],
            [output / "benchmark_results.npz", output / "benchmark_summary.json"],
            [
                output / "empymod_background.npz",
                output / "simpeg_background.npz",
                output / "simpeg_channel.npz",
                output / "fenicsx_background" / "predictions_5rx.csv",
                output / "fenicsx_channel" / "predictions_5rx.csv",
            ],
        ),
        RunJob(
            name="plot",
            command=(sys.executable, plotter, str(output)),
            required_inputs=(output / "benchmark_results.npz", output / "convergence_summary.json"),
            expected_outputs=(output / "model_geometry.png", output / "channel_delta.png"),
        ),
        RunJob(
            name="manifest",
            command=(sys.executable, plotter, str(output), "--manifest-only"),
            required_inputs=(output / "benchmark_results.npz",),
            expected_outputs=(output / "benchmark_manifest.json",),
        ),
    ]


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def _job_outputs_valid(job: RunJob) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in job.expected_outputs):
        return False
    try:
        if job.name == "empymod_background":
            validate_result_payload("empymod", _load_npz_payload(job.expected_outputs[0]))
        elif job.name.startswith("simpeg_"):
            validate_result_payload("SimPEG", _load_npz_payload(job.expected_outputs[1]))
            provenance = json.loads(job.expected_outputs[2].read_text(encoding="utf-8"))
            config = ROOT / provenance["config_path"]
            if provenance["config_sha256"] != sha256_file(config):
                return False
        elif job.name.startswith("fenicsx_"):
            from atem3d.seepage_channel_validation import fenicsx_payload_from_csv

            fenicsx_payload_from_csv(job.expected_outputs[0])
        elif job.name == "aggregate":
            payload = _load_npz_payload(job.expected_outputs[0])
            if payload["simpeg_delta"].shape != (5, 31, 3):
                return False
        elif job.name == "manifest":
            manifest = json.loads(job.expected_outputs[0].read_text(encoding="utf-8"))
            if "benchmark_manifest.json" in manifest.get("inventory", {}):
                return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def execute_run_plan(
    output_root: str | Path,
    *,
    force: bool = False,
    variant: str = "baseline_60x10x10",
) -> None:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "run_events.jsonl"
    for job in build_run_plan(output, variant=variant):
        missing_inputs = [path for path in job.required_inputs if not path.exists()]
        if missing_inputs:
            raise FileNotFoundError(f"{job.name} missing inputs: {missing_inputs}")
        if not force and _job_outputs_valid(job):
            event = {
                "job": job.name,
                "event": "reuse",
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "validated": True,
            }
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            continue
        started = datetime.now(timezone.utc)
        start_clock = time.perf_counter()
        result = subprocess.run(job.command, cwd=ROOT, check=False)
        event = {
            "job": job.name,
            "event": "run",
            "start_utc": started.isoformat(),
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - start_clock,
            "exit_code": int(result.returncode),
            "peak_memory_evidence": "external solver memory preflight and run logs",
            "outputs_valid": bool(result.returncode == 0 and _job_outputs_valid(job)),
        }
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if result.returncode != 0 or not event["outputs_valid"]:
            raise RuntimeError(f"job {job.name} failed contract validation: {event}")


def _run_empymod(output_root: Path, *, srcpts: int, variant: str) -> Path:
    output = output_root / "empymod_background.npz"
    save_empymod_background(
        output, srcpts=srcpts, model=benchmark_model(variant)
    )
    with np.load(output, allow_pickle=False) as stored:
        validate_result_payload("empymod", stored)
    return output


def _run_simpeg(
    output_root: Path,
    *,
    case: str,
    variant: str = "baseline_60x10x10",
) -> Path:
    config_path = VARIANT_CONFIGS[variant]["simpeg"][case]
    output_h5 = output_root / f"simpeg_{case}.h5"
    output_npz = output_root / f"simpeg_{case}.npz"
    output_root.mkdir(parents=True, exist_ok=True)
    model = benchmark_model(variant)
    contract_provenance = verify_simpeg_config_contract(
        config_path, model=model, case=case
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "atem3d.cli",
            "run",
            str(config_path),
            "--output",
            str(output_h5),
            "--data-only",
        ],
        cwd=ROOT,
        check=True,
    )
    payload = simpeg_payload_from_h5(
        output_h5, model=model
    )
    np.savez_compressed(output_npz, **payload)
    provenance = {
        **contract_provenance,
        "method": "SimPEG",
        "case": case,
        "variant": variant,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "result_h5": output_h5.name,
        "result_h5_sha256": sha256_file(output_h5),
        "normalized_npz": output_npz.name,
        "value_shape": list(np.asarray(payload["values"]).shape),
        "coordinate_adapter": "physical_z_down_to_internal_z_up",
        "model_fingerprint": str(np.asarray(payload["model_fingerprint"]).item()),
    }
    (output_root / f"simpeg_{case}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="solver", required=True)

    empymod_parser = subparsers.add_parser("empymod")
    empymod_parser.add_argument("--output-root", type=Path, required=True)
    empymod_parser.add_argument("--srcpts", type=int, default=129)
    empymod_parser.add_argument("--variant", choices=VARIANT_CHOICES, default="baseline_60x10x10")

    simpeg_parser = subparsers.add_parser("simpeg")
    simpeg_parser.add_argument("--case", choices=tuple(SIMPEG_CONFIGS), required=True)
    simpeg_parser.add_argument("--variant", choices=VARIANT_CHOICES, default="baseline_60x10x10")
    simpeg_parser.add_argument("--output-root", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--variant", choices=VARIANT_CHOICES, default="baseline_60x10x10")

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--output-root", type=Path, required=True)
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--variant", choices=VARIANT_CHOICES, default="baseline_60x10x10")

    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if args.solver == "empymod":
        output = _run_empymod(
            output_root, srcpts=args.srcpts, variant=args.variant
        )
    elif args.solver == "simpeg":
        output = _run_simpeg(output_root, case=args.case, variant=args.variant)
    elif args.solver == "aggregate":
        aggregate_result_directory(
            output_root,
            model=benchmark_model(args.variant),
            variant=args.variant,
        )
        output = output_root / "benchmark_results.npz"
    else:
        execute_run_plan(output_root, force=args.force, variant=args.variant)
        output = output_root / "benchmark_manifest.json"
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
