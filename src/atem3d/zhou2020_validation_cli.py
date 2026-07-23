"""Fail-closed orchestration for the strict Zhou 2020 validation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np

from .run_lock import run_lock
from .zhou2020_metrics import (
    COMPONENTS,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_TIME_BOUNDS,
    TOTAL_FIELD_L2_GATE,
    compare_zhou_responses,
)


SCHEMA = "atem3d.zhou2020.validation-run/v1"
NOIP_GATE_SCHEMA = "atem3d.zhou2020.noip-gate/v1"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def prepare_run(
    root: str | Path,
    *,
    case_path: str | Path,
    provenance_path: str | Path,
    run_id: str | None = None,
) -> Path:
    """Create an immutable run envelope containing input snapshots."""

    identifier = run_id or _default_run_id()
    if not SAFE_RUN_ID.fullmatch(identifier):
        raise ValueError("run_id must be a safe single path component")
    root_path = Path(root).expanduser().resolve()
    run_dir = root_path / identifier
    case = Path(case_path).expanduser().resolve()
    provenance = Path(provenance_path).expanduser().resolve()
    for source in (case, provenance):
        if not source.is_file():
            raise FileNotFoundError(source)
    if run_dir.exists():
        raise FileExistsError(run_dir)

    root_path.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}.", dir=root_path))
    try:
        snapshot_dir = temporary / "snapshots"
        snapshot_dir.mkdir()
        case_target = snapshot_dir / f"case{case.suffix or '.yaml'}"
        provenance_target = snapshot_dir / f"provenance{provenance.suffix or '.json'}"
        shutil.copy2(case, case_target)
        shutil.copy2(provenance, provenance_target)
        manifest = {
            "schema": SCHEMA,
            "run_id": identifier,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "prepared",
            "snapshots": {
                "case": _artifact_record(temporary, case_target),
                "provenance": _artifact_record(temporary, provenance_target),
            },
            "stages": {},
            "comparisons": {},
        }
        _atomic_write_json(temporary / "run_manifest.json", manifest)
        os.replace(temporary, run_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return run_dir


def publish_reference(run_dir: str | Path, source_dir: str | Path) -> dict[str, Any]:
    """Atomically publish a verified empymod reference stage."""

    run = _validated_run_dir(run_dir)
    source = Path(source_dir).expanduser().resolve()
    manifest_path = source / "reference_manifest.json"
    reference_manifest = _load_json(manifest_path)
    if reference_manifest.get("status") != "reference_verified":
        raise ValueError("reference manifest status must be reference_verified")
    _verify_reference_manifest(source, reference_manifest)

    with run_lock(run):
        manifest = _load_run_manifest(run)
        if "reference" in manifest["stages"]:
            raise FileExistsError("reference stage is already published")
        target = run / "reference"
        _atomic_copy_tree(source, target, run)
        files = _tree_records(run, target)
        manifest["stages"]["reference"] = {
            "status": "reference_verified",
            "path": "reference",
            "files": files,
        }
        manifest["status"] = "reference_verified"
        _atomic_write_json(run / "run_manifest.json", manifest)
        return manifest["stages"]["reference"]


def register_fenicsx(
    run_dir: str | Path,
    *,
    variant: str,
    level: str,
    source_dir: str | Path,
) -> dict[str, Any]:
    """Register one immutable FEniCSx result after checking prerequisites."""

    if variant not in {"noip", "ip"}:
        raise ValueError("variant must be noip or ip")
    _validate_level(level)
    run = _validated_run_dir(run_dir)
    source = Path(source_dir).expanduser().resolve()
    for required in ("predictions.csv", "run_config_resolved.yaml"):
        if not (source / required).is_file():
            raise FileNotFoundError(source / required)

    with run_lock(run):
        manifest = _load_run_manifest(run)
        if "reference" not in manifest["stages"]:
            raise RuntimeError("verified reference stage is required")
        if variant == "ip":
            gate_path = run / "comparisons" / level / "noip_gate.json"
            if not gate_path.is_file() or not _load_json(gate_path).get("passed"):
                raise RuntimeError("a passed no-IP gate is required before IP registration")
        key = f"fenicsx/{variant}/{level}"
        if key in manifest["stages"]:
            raise FileExistsError(f"{key} is already registered")
        target = run / "fenicsx" / variant / level
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy_tree(source, target, run)
        record = {
            "status": "registered",
            "variant": variant,
            "level": level,
            "path": target.relative_to(run).as_posix(),
            "files": _tree_records(run, target),
        }
        manifest["stages"][key] = record
        manifest["status"] = f"fenicsx_{variant}_registered"
        _atomic_write_json(run / "run_manifest.json", manifest)
        return record


def compare_registered(
    run_dir: str | Path,
    *,
    level: str,
    mode: str,
) -> dict[str, Any]:
    """Compare registered results while preserving every failed record."""

    if mode not in {"noip", "full"}:
        raise ValueError("mode must be noip or full")
    _validate_level(level)
    run = _validated_run_dir(run_dir)
    with run_lock(run):
        manifest = _load_run_manifest(run)
        reference = _load_reference_arrays(run)
        noip = _load_fenicsx_arrays(run, manifest, "noip", level)
        comparison_dir = run / "comparisons" / level
        comparison_dir.mkdir(parents=True, exist_ok=True)
        if mode == "noip":
            result = _compare_noip(noip, reference["noip"])
            output = comparison_dir / "noip_gate.json"
        else:
            ip = _load_fenicsx_arrays(run, manifest, "ip", level)
            if not _matching_full_time_grid(noip[0], reference["noip"][0]) or not (
                _matching_full_time_grid(ip[0], reference["ip"][0])
            ):
                result = {
                    "schema": "atem3d.zhou2020.strict-comparison/v1",
                    "status": "incomplete_time_window",
                    "numerical_gates_passed": False,
                    "full_time_window": {
                        "expected_bounds_s": list(EXPECTED_TIME_BOUNDS),
                        "expected_count": EXPECTED_SAMPLE_COUNT,
                        "passed": False,
                    },
                    "failed_components": list(COMPONENTS),
                    "failed_times_s": [],
                    "point_errors": [],
                }
            else:
                result = compare_zhou_responses(
                    times=reference["noip"][0],
                    prediction_noip=noip[1],
                    reference_noip=reference["noip"][1],
                    prediction_ip=ip[1],
                    reference_ip=reference["ip"][1],
                )
            output = comparison_dir / "strict_comparison.json"
        _atomic_write_json(output, result)
        relative = output.relative_to(run).as_posix()
        manifest["comparisons"][f"{mode}/{level}"] = {
            "path": relative,
            "sha256": _sha256_file(output),
            "status": result["status"],
        }
        manifest["status"] = result["status"]
        _atomic_write_json(run / "run_manifest.json", manifest)
        return result


def record_convergence(
    run_dir: str | Path, source_path: str | Path
) -> dict[str, Any]:
    """Record a machine-readable mesh/time/boundary convergence decision."""

    run = _validated_run_dir(run_dir)
    source = Path(source_path).expanduser().resolve()
    payload = _load_json(source)
    passed = bool(
        payload.get("all_gates_passed")
        and payload.get("status") in {"passed", "convergence_verified"}
    )
    with run_lock(run):
        target = run / "convergence" / "convergence.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        manifest = _load_run_manifest(run)
        manifest["stages"]["convergence"] = {
            "status": "convergence_verified" if passed else "failed",
            **_artifact_record(run, target),
        }
        _atomic_write_json(run / "run_manifest.json", manifest)
        return manifest["stages"]["convergence"]


def finalize_run(run_dir: str | Path, *, level: str) -> dict[str, Any]:
    """Finalize only when reference, no-IP, IP, and convergence gates pass."""

    _validate_level(level)
    run = _validated_run_dir(run_dir)
    with run_lock(run):
        manifest = _load_run_manifest(run)
        noip_path = run / "comparisons" / level / "noip_gate.json"
        full_path = run / "comparisons" / level / "strict_comparison.json"
        convergence_path = run / "convergence" / "convergence.json"
        if not noip_path.is_file() or not _load_json(noip_path).get("passed"):
            raise RuntimeError("passed no-IP gate is required")
        full = _load_json(full_path)
        if full.get("status") != "ip_internally_validated":
            raise RuntimeError("passed strict comparison is required")
        convergence = _load_json(convergence_path)
        if not (
            convergence.get("all_gates_passed")
            and convergence.get("status") in {"passed", "convergence_verified"}
        ):
            raise RuntimeError("passed convergence evidence is required")
        verification = verify_run(run)
        if not verification["verified"]:
            raise RuntimeError("artifact hash verification failed")
        manifest = _load_run_manifest(run)
        manifest["status"] = "ip_internally_validated"
        manifest["final_level"] = level
        manifest["finalized_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(run / "run_manifest.json", manifest)
        return manifest


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify all manifest-addressed artifacts and reject unsafe paths."""

    run = _validated_run_dir(run_dir)
    manifest = _load_run_manifest(run)
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = list(manifest.get("snapshots", {}).values())
    for stage in manifest.get("stages", {}).values():
        if isinstance(stage.get("files"), dict):
            records.extend(stage["files"].values())
        elif "path" in stage and "sha256" in stage:
            records.append(stage)
    records.extend(manifest.get("comparisons", {}).values())
    for record in records:
        relative = _safe_relative_path(record["path"])
        target = run / Path(*relative.parts)
        if not target.is_file():
            failures.append({"path": relative.as_posix(), "reason": "missing"})
            continue
        actual = _sha256_file(target)
        if actual != record["sha256"]:
            failures.append({"path": relative.as_posix(), "reason": "sha256"})
    return {
        "schema": "atem3d.zhou2020.artifact-verification/v1",
        "verified": not failures,
        "hash_failures": failures,
    }


def _compare_noip(
    prediction: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    times, values = prediction
    reference_times, expected = reference
    full = _matching_full_time_grid(times, reference_times)
    metrics: dict[str, Any] = {}
    passed = full
    if full:
        for index, component in enumerate(COMPONENTS):
            residual = values[:, index] - expected[:, index]
            denominator = float(np.linalg.norm(expected[:, index]))
            relative_l2 = (
                float(np.linalg.norm(residual) / denominator)
                if denominator > 0.0
                else float("inf")
            )
            metrics[component] = {
                "relative_l2": relative_l2,
                "gate": TOTAL_FIELD_L2_GATE,
                "passed": relative_l2 <= TOTAL_FIELD_L2_GATE,
            }
        passed = all(item["passed"] for item in metrics.values())
    return {
        "schema": NOIP_GATE_SCHEMA,
        "status": (
            "noip_internally_validated"
            if passed
            else "failed_with_reproducible_evidence"
            if full
            else "incomplete_time_window"
        ),
        "passed": bool(passed),
        "full_time_window": {
            "expected_bounds_s": list(EXPECTED_TIME_BOUNDS),
            "actual_bounds_s": [float(times[0]), float(times[-1])],
            "expected_count": EXPECTED_SAMPLE_COUNT,
            "actual_count": int(times.size),
            "passed": bool(full),
        },
        "components": metrics,
    }


def _load_reference_arrays(
    run: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        variant: _read_response_csv(
            run / "reference" / f"empymod_{variant}.csv",
            time_column="time_s",
            component_columns=("Ex_V_per_m", "Hz_A_per_m", "dBzdt_T_per_s"),
        )
        for variant in ("noip", "ip")
    }


def _load_fenicsx_arrays(
    run: Path,
    manifest: dict[str, Any],
    variant: str,
    level: str,
) -> tuple[np.ndarray, np.ndarray]:
    key = f"fenicsx/{variant}/{level}"
    if key not in manifest["stages"]:
        raise RuntimeError(f"registered stage {key} is required")
    relative = _safe_relative_path(manifest["stages"][key]["path"])
    return _read_response_csv(
        run / Path(*relative.parts) / "predictions.csv",
        time_column="time_obs",
        component_columns=COMPONENTS,
    )


def _read_response_csv(
    path: Path,
    *,
    time_column: str,
    component_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = {time_column, *component_columns}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)} in {path}")
        rows = list(reader)
    times = np.asarray([float(row[time_column]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[name]) for name in component_columns] for row in rows],
        dtype=float,
    )
    if (
        times.ndim != 1
        or times.size < 2
        or values.shape != (times.size, len(component_columns))
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or np.any(times <= 0.0)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(f"invalid response data: {path}")
    return times, values


def _matching_full_time_grid(actual: np.ndarray, expected: np.ndarray) -> bool:
    return bool(
        actual.size == EXPECTED_SAMPLE_COUNT
        and expected.size == EXPECTED_SAMPLE_COUNT
        and np.isclose(actual[0], EXPECTED_TIME_BOUNDS[0], rtol=0, atol=1.0e-15)
        and np.isclose(actual[-1], EXPECTED_TIME_BOUNDS[1], rtol=0, atol=1.0e-12)
        and np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-15)
    )


def _verify_reference_manifest(source: Path, manifest: dict[str, Any]) -> None:
    if isinstance(manifest.get("file_sha256"), dict):
        records = {
            name: {"path": name, "sha256": digest}
            for name, digest in manifest["file_sha256"].items()
        }
    elif isinstance(manifest.get("artifacts"), dict):
        records = manifest["artifacts"]
    else:
        raise ValueError("reference manifest has no artifact hashes")
    required = {
        "empymod_noip.csv",
        "empymod_ip.csv",
        "empymod_srcpts_convergence.json",
        "empymod_metadata.json",
    }
    addressed = {record["path"] for record in records.values()}
    if not required.issubset(addressed):
        raise ValueError("reference manifest misses required artifacts")
    for record in records.values():
        relative = _safe_relative_path(record["path"])
        target = source / Path(*relative.parts)
        if not target.is_file() or _sha256_file(target) != record["sha256"]:
            raise ValueError(f"reference artifact hash mismatch: {relative}")
    metadata = _load_json(source / "empymod_metadata.json")
    convention = metadata.get("component_conventions", {}).get("dBzdt", {})
    expected_convention = {
        "empymod_receiver": "H",
        "empymod_signal": 0,
        "scale": "-mu0",
        "source_waveform": "ideal_step_off",
    }
    if (
        metadata.get("schema") != "atem3d.zhou2020.empymod-metadata/v2"
        or convention != expected_convention
    ):
        raise ValueError(
            "reference dBzdt convention must use impulse H with signal=0 "
            "and scale=-mu0 for an ideal step-off source"
        )


def _atomic_copy_tree(source: Path, target: Path, run: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(source)
    resolved_run = run.resolve()
    resolved_target_parent = target.parent.resolve()
    if resolved_run != resolved_target_parent and resolved_run not in resolved_target_parent.parents:
        raise ValueError("copy target must remain inside run directory")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        for item in source.iterdir():
            destination = temporary / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            elif item.is_file():
                shutil.copy2(item, destination)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _tree_records(run: Path, root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(run).as_posix()
        result[relative] = {"path": relative, "sha256": _sha256_file(path)}
    return result


def _artifact_record(run: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(run).as_posix()
    return {"path": relative, "sha256": _sha256_file(path)}


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("artifact path must be a POSIX relative path")
    result = PurePosixPath(value)
    if result.is_absolute() or not result.parts or ".." in result.parts:
        raise ValueError("artifact path escapes the run directory")
    return result


def _validated_run_dir(value: str | Path) -> Path:
    result = Path(value).expanduser().resolve()
    if not result.is_dir() or not (result / "run_manifest.json").is_file():
        raise FileNotFoundError(f"not a prepared Zhou validation run: {result}")
    return result


def _load_run_manifest(run: Path) -> dict[str, Any]:
    result = _load_json(run / "run_manifest.json")
    if result.get("schema") != SCHEMA:
        raise ValueError("unexpected Zhou validation manifest schema")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_level(level: str) -> None:
    if not SAFE_RUN_ID.fullmatch(level):
        raise ValueError("level must be a safe single path component")


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Zhou 2020 FEniCSx/empymod validation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--provenance", required=True)
    prepare.add_argument("--run-id")
    reference = subparsers.add_parser("reference")
    reference.add_argument("--run-dir", required=True)
    reference.add_argument("--source-dir", required=True)
    register = subparsers.add_parser("register-fenicsx")
    register.add_argument("--run-dir", required=True)
    register.add_argument("--variant", choices=("noip", "ip"), required=True)
    register.add_argument("--level", required=True)
    register.add_argument("--source-dir", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--run-dir", required=True)
    compare.add_argument("--level", required=True)
    compare.add_argument("--mode", choices=("noip", "full"), required=True)
    convergence = subparsers.add_parser("record-convergence")
    convergence.add_argument("--run-dir", required=True)
    convergence.add_argument("--source", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--level", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result: Any = {
            "run_dir": str(
                prepare_run(
                    args.root,
                    case_path=args.case,
                    provenance_path=args.provenance,
                    run_id=args.run_id,
                )
            )
        }
    elif args.command == "reference":
        result = publish_reference(args.run_dir, args.source_dir)
    elif args.command == "register-fenicsx":
        result = register_fenicsx(
            args.run_dir,
            variant=args.variant,
            level=args.level,
            source_dir=args.source_dir,
        )
    elif args.command == "compare":
        result = compare_registered(args.run_dir, level=args.level, mode=args.mode)
    elif args.command == "record-convergence":
        result = record_convergence(args.run_dir, args.source)
    elif args.command == "finalize":
        result = finalize_run(args.run_dir, level=args.level)
    else:
        result = verify_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "verify" and not result["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
