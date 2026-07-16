#!/usr/bin/env python3
"""Prepare or run independent COMSOL background/thin-channel validation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.comsol_seepage_channel_3d import (  # noqa: E402
    COMSOL_CASES,
    comsol_case_contract,
    read_comsol_wide_export,
    validate_distinct_model_paths,
)


JAVA_SOURCE = (
    ROOT
    / "COMSOL"
    / "seepage_channel_3d"
    / "ConfigureAndRunSeepageChannel3D.java"
)
JAVA_CLASS = JAVA_SOURCE.with_suffix(".class")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_case_paths(
    output_root: str | Path, case: str, source_model: str | Path
) -> dict[str, Path]:
    case_dir = Path(output_root) / "comsol_3d" / str(case)
    paths = {
        "case_dir": case_dir,
        "source_model": Path(source_model),
        "output_model": case_dir / f"seepage_channel_3d_{case}.mph",
        "output_csv": case_dir / "comsol_point_export.csv",
        "normalized": case_dir / "normalized.npz",
        "log": case_dir / "comsol_batch.log",
        "contract": case_dir / "model_contract.json",
        "provenance": case_dir / "provenance.json",
    }
    validate_distinct_model_paths(paths["source_model"], paths["output_model"])
    return paths


def build_commands(comsol_bin: str | Path, paths: dict[str, Path]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    binary_dir = Path(comsol_bin)
    compile_command = (
        str(binary_dir / "comsolcompile.exe"),
        str(JAVA_SOURCE),
    )
    batch_command = (
        str(binary_dir / "comsolbatch.exe"),
        "-np",
        "4",
        "-inputfile",
        str(JAVA_CLASS),
        "-batchlog",
        str(paths["log"]),
    )
    return compile_command, batch_command


def _write_contract(paths: dict[str, Path], case: str) -> dict:
    paths["case_dir"].mkdir(parents=True, exist_ok=True)
    contract = comsol_case_contract(case)
    paths["contract"].write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def _save_normalized(paths: dict[str, Path], case: str, *, method: str) -> Path:
    payload = dict(read_comsol_wide_export(paths["output_csv"]))
    contract = comsol_case_contract(case)
    payload["base_model_fingerprint"] = np.asarray(contract["base_model_fingerprint"])
    payload["case_model_fingerprint"] = np.asarray(contract["case_model_fingerprint"])
    payload["case"] = np.asarray(case)
    np.savez_compressed(paths["normalized"], **payload)
    provenance = {
        "method": method,
        "case": case,
        "base_model_fingerprint": contract["base_model_fingerprint"],
        "case_model_fingerprint": contract["case_model_fingerprint"],
        "source_model": str(paths["source_model"]),
        "source_model_sha256": _sha256(paths["source_model"]),
        "output_csv_sha256": _sha256(paths["output_csv"]),
        "normalized_sha256": _sha256(paths["normalized"]),
        "java_source_sha256": _sha256(JAVA_SOURCE),
    }
    if paths["output_model"].is_file():
        provenance["output_model_sha256"] = _sha256(paths["output_model"])
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths["normalized"]


def import_background(
    *,
    output_root: str | Path,
    source_model: str | Path,
    background_export: str | Path,
) -> Path:
    paths = build_case_paths(output_root, "uniform_background_reference", source_model)
    _write_contract(paths, "background")
    paths["output_csv"] = Path(background_export)
    return _save_normalized(paths, "background", method="verified_retained_uniform_run")


def run_case(
    *,
    case: str,
    output_root: str | Path,
    source_model: str | Path,
    comsol_bin: str | Path,
    prepare_only: bool,
) -> Path:
    paths = build_case_paths(output_root, case, source_model)
    _write_contract(paths, case)
    compile_command, batch_command = build_commands(comsol_bin, paths)
    command_manifest = {
        "compile": list(compile_command),
        "batch": list(batch_command),
        "java_source": str(JAVA_SOURCE),
    }
    manifest_path = paths["case_dir"] / "commands.json"
    manifest_path.write_text(
        json.dumps(command_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if prepare_only:
        return manifest_path

    subprocess.run(compile_command, cwd=JAVA_SOURCE.parent, check=True)
    environment = os.environ.copy()
    environment.update(
        {
            "ATEM3D_COMSOL_INPUT_MODEL": str(paths["source_model"].resolve()),
            "ATEM3D_COMSOL_OUTPUT_MODEL": str(paths["output_model"].resolve()),
            "ATEM3D_COMSOL_OUTPUT_CSV": str(paths["output_csv"].resolve()),
            "ATEM3D_COMSOL_CASE": case,
            "ATEM3D_COMSOL_CHANNEL_SIGMA": "1.0" if case == "channel" else "0.01",
        }
    )
    subprocess.run(
        batch_command,
        cwd=JAVA_SOURCE.parent,
        env=environment,
        check=True,
    )
    if not paths["output_model"].is_file() or not paths["output_csv"].is_file():
        raise RuntimeError("COMSOL batch completed without the required MPH/CSV outputs")
    return _save_normalized(paths, case, method="independent_comsol_3d_batch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "import-background"))
    parser.add_argument("--case", choices=COMSOL_CASES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--background-export", type=Path)
    parser.add_argument(
        "--comsol-bin",
        type=Path,
        default=Path(r"D:\APP\Comsol_64\bin\win64"),
    )
    args = parser.parse_args(argv)

    if args.action == "import-background":
        if args.background_export is None:
            parser.error("import-background requires --background-export")
        result = import_background(
            output_root=args.output_root,
            source_model=args.source_model,
            background_export=args.background_export,
        )
    else:
        if args.case is None:
            parser.error("prepare/run requires --case")
        result = run_case(
            case=args.case,
            output_root=args.output_root,
            source_model=args.source_model,
            comsol_bin=args.comsol_bin,
            prepare_only=args.action == "prepare",
        )
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
