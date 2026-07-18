"""Reproducible, fail-closed orchestration for the canonical SOTEM suite."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any
import uuid

import numpy as np

from .polarization_effect import write_polarization_effect_artifacts
from .sotem_benchmark import BenchmarkCase, load_benchmark_case
from .sotem_gate import summarize_sotem_gates
from .sotem_observables import CanonicalResponse, canonical_response
from .sotem_simpeg_adapter import run_simpeg_benchmark


MANIFEST_SCHEMA = "atem3d.sotem.validation-manifest"
MANIFEST_SCHEMA_VERSION = 1
SIMPEG_SOLVER_ID = "atem3d_simpeg_discretize_debye"
EFFECT_SOLVER_ID = "polarization_effect"
GATE_SOLVER_ID = "sotem_gate"
_LEVELS = tuple(
    f"S{spatial}T{temporal}B{boundary}"
    for spatial in range(3)
    for temporal in range(3)
    for boundary in range(3)
)
_SUBSTEPS = {"T0": 1, "T1": 2, "T2": 4}
_CRITICAL_LIBRARIES = (
    "numpy",
    "scipy",
    "discretize",
    "simpeg",
    "empymod",
    "pymatsolver",
    "PyYAML",
)
_PIPELINE_MODULE: ModuleType | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime | None = None) -> str:
    instant = _utc_now() if value is None else value
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_run_id() -> str:
    return f"{_utc_now().strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"


def _nonempty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON evidence must contain only finite floating-point values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON evidence mappings must use string keys")
            result[key] = _json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported JSON evidence type: {type(value).__qualname__}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = _json_value(payload)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _case_payload(case: BenchmarkCase) -> dict[str, Any]:
    earth: dict[str, Any]
    if "rho_ohm_m" in case.earth:
        earth = {"rho_ohm_m": float(case.earth["rho_ohm_m"])}
    else:
        earth = {
            "layers": [
                {
                    "top_m": float(layer["top_m"]),
                    "bottom_m": (
                        None if layer["bottom_m"] is None else float(layer["bottom_m"])
                    ),
                    "rho_ohm_m": float(layer["rho_ohm_m"]),
                }
                for layer in case.earth["layers"]
            ]
        }
    polarization = (
        None
        if case.polarization is None
        else {key: float(value) for key, value in case.polarization.items()}
    )
    return {
        "case_id": case.case_id,
        "coordinates": case.coordinates,
        "source_start_down": list(case.source_start_down),
        "source_end_down": list(case.source_end_down),
        "receiver_down": list(case.receiver_down),
        "current_a": float(case.current_a),
        "rho_air_ohm_m": float(case.rho_air_ohm_m),
        "earth": earth,
        "polarization": polarization,
        "components": list(case.components),
        "observation_times": np.asarray(case.observation_times, dtype=float).tolist(),
    }


def _case_identity(path: str | Path) -> tuple[Path, BenchmarkCase, dict[str, str]]:
    normalized = Path(path).expanduser().resolve(strict=True)
    if not normalized.is_file():
        raise FileNotFoundError(f"benchmark case is not a file: {normalized}")
    case = load_benchmark_case(normalized)
    canonical = json.dumps(
        _case_payload(case),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return normalized, case, {
        "case_file_sha256": _sha256_file(normalized),
        "case_hash": _sha256_bytes(canonical),
    }


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    commit_text = commit.stdout.strip() if commit.returncode == 0 else "unavailable"
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return commit_text, dirty


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _CRITICAL_LIBRARIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _manifest_payload(
    *,
    run_id: str,
    case_path: Path,
    case: BenchmarkCase,
    identity: Mapping[str, str],
    solver_id: str,
    level: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    git_commit, git_dirty = _git_metadata(repo_root)
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case.case_id,
        "case_path": str(case_path),
        **dict(identity),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "solver_id": solver_id,
        "level": level,
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
        "created_at": _utc_text(),
        "status": "prepared",
        "stages": {},
    }


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"run manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("run manifest must contain a JSON object")
    return payload


def _verify_manifest(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    case_path: Path,
    case: BenchmarkCase,
    identity: Mapping[str, str],
    solver_id: str,
    level: str | None = None,
) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("run manifest schema mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest schema version mismatch")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("run manifest run_id does not match its directory")
    if manifest.get("case_id") != case.case_id:
        raise ValueError("run manifest case_id mismatch")
    if manifest.get("case_path") != str(case_path):
        raise ValueError("run manifest normalized case path mismatch")
    for name in ("case_file_sha256", "case_hash"):
        if manifest.get(name) != identity[name]:
            raise ValueError(f"run manifest case hash mismatch ({name})")
    if manifest.get("solver_id") != solver_id:
        raise ValueError("run manifest solver_id mismatch")
    manifest_level = manifest.get("level")
    if manifest_level not in _LEVELS:
        raise ValueError("run manifest level is invalid")
    if level is not None and manifest_level != level:
        raise ValueError("run manifest level mismatch")


def _prepare(args: argparse.Namespace) -> int:
    case_path, case, identity = _case_identity(args.case)
    if args.run_dir is None:
        run_id = _new_run_id()
        run_dir = Path(args.output_root).expanduser().resolve() / case.case_id / run_id
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_id = run_dir.name
    if not run_id:
        raise ValueError("run directory must have a non-empty final path component")

    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"refusing existing non-empty run directory without --resume: {run_dir}"
            )
        manifest = _load_manifest(run_dir)
        _verify_manifest(
            run_dir,
            manifest,
            case_path=case_path,
            case=case,
            identity=identity,
            solver_id=args.solver,
            level=args.level,
        )
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(
        run_id=run_id,
        case_path=case_path,
        case=case,
        identity=identity,
        solver_id=args.solver,
        level=args.level,
    )
    _atomic_write_json(run_dir / "manifest.json", manifest)
    return 0


def _open_run(
    args: argparse.Namespace, *, expected_solver: str
) -> tuple[Path, BenchmarkCase, dict[str, Any]]:
    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory is not a directory: {run_dir}")
    if not args.resume:
        raise FileExistsError(
            f"refusing existing non-empty run directory without --resume: {run_dir}"
        )
    case_path, case, identity = _case_identity(args.case)
    manifest = _load_manifest(run_dir)
    _verify_manifest(
        run_dir,
        manifest,
        case_path=case_path,
        case=case,
        identity=identity,
        solver_id=expected_solver,
    )
    return run_dir, case, manifest


def _files_hashes(run_dir: Path, paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"stage output is missing: {path}")
        try:
            relative = path.relative_to(run_dir).as_posix()
        except ValueError as exc:
            raise ValueError("stage output must be inside the run directory") from exc
        result[relative] = _sha256_file(path)
    return result


def _completed_stage_is_intact(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stage_name: str,
    inputs: Mapping[str, Any],
) -> bool:
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        raise ValueError("run manifest stages must be a mapping")
    record = stages.get(stage_name)
    if record is None:
        return False
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        raise ValueError(f"run manifest has an invalid {stage_name} stage record")
    if record.get("inputs") != _json_value(inputs):
        raise ValueError(f"completed {stage_name} stage inputs do not match")
    hashes = record.get("file_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError(f"completed {stage_name} stage lacks evidence hashes")
    for relative, expected in hashes.items():
        if type(relative) is not str or type(expected) is not str:
            raise ValueError(f"completed {stage_name} stage has invalid evidence hashes")
        path = run_dir / Path(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"completed {stage_name} stage evidence hash mismatch")
    return True


def _refuse_existing_outputs(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite untracked stage evidence: {joined}")


def _record_stage(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    stage_name: str,
    status: str,
    inputs: Mapping[str, Any],
    output_files: Sequence[Path],
) -> None:
    stages = manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("run manifest stages must be a JSON object")
    stages[stage_name] = {
        "status": "complete",
        "completed_at": _utc_text(),
        "inputs": _json_value(inputs),
        "file_sha256": _files_hashes(run_dir, output_files),
    }
    manifest["status"] = status
    _atomic_write_json(run_dir / "manifest.json", manifest)


def _atomic_write_canonical(path: Path, response: CanonicalResponse) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        response.write_csv(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _compatibility_csv_bytes(response: CanonicalResponse) -> bytes:
    rows: list[list[str]] = [["time_obs", "Ex", "Ey", "Hz", "dBzdt"]]
    selected = response.values[:, [0, 1, 2, 4]]
    for time_value, values in zip(response.times, selected):
        rows.append(
            [format(float(time_value), ".17g")]
            + [format(float(value), ".17g") for value in values]
        )
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _load_pipeline_module() -> ModuleType:
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is not None:
        return _PIPELINE_MODULE
    path = Path(__file__).resolve().parents[2] / "dolfinx" / "sotem_pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(f"SOTEM pipeline module is missing: {path}")
    module_name = "_atem3d_validation_sotem_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SOTEM pipeline module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _PIPELINE_MODULE = module
    return module


def get_empymod_reference(times, config, *, mode: str):
    """Monkeypatchable adapter to the existing independent reference API."""

    return _load_pipeline_module().get_empymod_reference(times, config, mode=mode)


def _parallel_offset(case: BenchmarkCase) -> float:
    start = np.asarray(case.source_start_down[:2], dtype=float)
    end = np.asarray(case.source_end_down[:2], dtype=float)
    receiver = np.asarray(case.receiver_down[:2], dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    return abs(float(axis[0] * (receiver[1] - start[1]) - axis[1] * (receiver[0] - start[0]))) / length


def _pipeline_config_for_case(case: BenchmarkCase, variant: str):
    pipeline = _load_pipeline_module()
    if variant == "cole-cole-exact" and case.polarization is None:
        raise ValueError("cole-cole-exact reference requires a polarizable benchmark case")
    if "rho_ohm_m" in case.earth:
        rho_earth = float(case.earth["rho_ohm_m"])
        layer_depths: tuple[float, ...] = ()
        layer_resistivities: tuple[float, ...] = ()
    else:
        layers = list(case.earth["layers"])
        rho_earth = float(layers[0]["rho_ohm_m"])
        layer_depths = tuple(
            float(layer["bottom_m"])
            for layer in layers[:-1]
            if layer["bottom_m"] is not None
        )
        layer_resistivities = tuple(float(layer["rho_ohm_m"]) for layer in layers)
    values: dict[str, Any] = {
        "source_start": (
            float(case.source_start_down[0]),
            float(case.source_start_down[1]),
            -float(case.source_start_down[2]),
        ),
        "source_end": (
            float(case.source_end_down[0]),
            float(case.source_end_down[1]),
            -float(case.source_end_down[2]),
        ),
        "receiver": (
            float(case.receiver_down[0]),
            float(case.receiver_down[1]),
            -float(case.receiver_down[2]),
        ),
        "source_current": float(case.current_a),
        "ramp_off_time": 0.0,
        "rho_air": float(case.rho_air_ohm_m),
        "rho_earth": rho_earth,
        "layer_depths": layer_depths,
        "layer_resistivities": layer_resistivities,
        "observation_times": tuple(float(value) for value in case.observation_times),
        "expected_source_length": float(
            np.linalg.norm(
                np.asarray(case.source_end_down) - np.asarray(case.source_start_down)
            )
        ),
        "expected_parallel_offset": _parallel_offset(case),
        "magnetic_receiver_mode": "faraday_integrated",
        "time_origin": "after_ramp",
        "polarization": "cole-cole" if variant == "cole-cole-exact" else "none",
    }
    if variant == "cole-cole-exact":
        polarization = case.polarization
        assert polarization is not None
        values.update(
            cole_rho0=float(polarization["rho0_ohm_m"]),
            cole_m=float(polarization["m"]),
            cole_tau=float(polarization["tau_s"]),
            cole_c=float(polarization["c"]),
            cole_layer_top=float(polarization["top_m"]),
            cole_layer_bottom=float(polarization["bottom_m"]),
        )
    return pipeline.PipelineConfig(**values)


def _reference(args: argparse.Namespace) -> int:
    run_dir, case, manifest = _open_run(args, expected_solver="empymod")
    inputs = {"variant": args.variant}
    outputs = [
        run_dir / "empymod.csv",
        run_dir / "reference_empymod_or_1d.csv",
        run_dir / "empymod_metadata.json",
    ]
    if _completed_stage_is_intact(run_dir, manifest, "reference", inputs):
        return 0
    _refuse_existing_outputs(outputs)

    config = _pipeline_config_for_case(case, args.variant)
    result = get_empymod_reference(
        np.asarray(case.observation_times, dtype=float),
        config,
        mode=args.variant,
    )
    if not isinstance(result, Mapping):
        raise ValueError("empymod reference API must return a mapping")
    if result.get("reference_mode") != args.variant:
        raise ValueError("empymod reference API reported a different reference mode")
    response = canonical_response(result["times"], result["data"], result["components"])
    if not np.array_equal(response.times, case.observation_times):
        raise ValueError("empymod reference times must match the benchmark exactly")
    provenance = (
        "empymod_noip_layered"
        if args.variant == "noip"
        else "empymod_exact_layered"
    )
    metadata = {
        "solver_id": "empymod",
        "case_id": case.case_id,
        "reference_mode": args.variant,
        "reference_provenance": provenance,
        "coordinate_system": "z_down",
        "columns": list(response.columns),
    }
    _atomic_write_canonical(outputs[0], response)
    _atomic_write_bytes(outputs[1], _compatibility_csv_bytes(response))
    _atomic_write_json(outputs[2], metadata)
    _record_stage(
        run_dir,
        manifest,
        stage_name="reference",
        status="reference_complete",
        inputs=inputs,
        output_files=outputs,
    )
    return 0


def _parse_manifest_level(manifest: Mapping[str, Any]) -> tuple[str, str, int]:
    level = manifest.get("level")
    if level not in _LEVELS:
        raise ValueError("run manifest level is invalid")
    return level[0:2], level[4:6], _SUBSTEPS[level[2:4]]


def _simpeg(args: argparse.Namespace) -> int:
    run_dir, case, manifest = _open_run(args, expected_solver=SIMPEG_SOLVER_ID)
    spatial_level, boundary_level, substeps = _parse_manifest_level(manifest)
    inputs = {"variant": args.variant, "level": manifest["level"]}
    outputs = [
        run_dir / "simpeg.csv",
        run_dir / "predictions.csv",
        run_dir / "simpeg_metadata.json",
    ]
    if _completed_stage_is_intact(run_dir, manifest, "simpeg", inputs):
        return 0
    _refuse_existing_outputs(outputs)

    result = run_simpeg_benchmark(
        case,
        variant=args.variant,
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=substeps,
    )
    if not isinstance(result, Mapping):
        raise ValueError("SimPEG benchmark API must return a mapping")
    if result.get("solver_id") != SIMPEG_SOLVER_ID:
        raise ValueError("SimPEG benchmark reported an unexpected solver_id")
    if result.get("variant") != args.variant:
        raise ValueError("SimPEG benchmark reported a different variant")
    response = canonical_response(result["times"], result["data"], result["components"])
    if not np.array_equal(response.times, case.observation_times):
        raise ValueError("SimPEG result times must match the benchmark exactly")
    metadata = {
        "solver_id": SIMPEG_SOLVER_ID,
        "case_id": case.case_id,
        "variant": args.variant,
        "level": manifest["level"],
        "coordinate_system": "z_down",
        "columns": list(response.columns),
        "mesh_hash": result.get("mesh_hash"),
        "time_hash": result.get("time_hash"),
        "mesh_stats": result.get("mesh_stats"),
        "material_fit": result.get("material_fit"),
    }
    normalized_metadata = _json_value(metadata)
    _atomic_write_canonical(outputs[0], response)
    _atomic_write_bytes(outputs[1], _compatibility_csv_bytes(response))
    _atomic_write_json(outputs[2], normalized_metadata)
    _record_stage(
        run_dir,
        manifest,
        stage_name="simpeg",
        status="simpeg_complete",
        inputs=inputs,
        output_files=outputs,
    )
    return 0


def _source_run_evidence(
    run_dir: Path,
    *,
    effect_manifest: Mapping[str, Any],
    role: str,
    solver_id: str,
    stage_name: str,
    variant: str,
    evidence_name: str,
) -> tuple[Path, dict[str, Any]]:
    manifest = _load_manifest(run_dir)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{role} manifest schema mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{role} manifest schema version mismatch")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError(f"{role} manifest run_id mismatch")
    if manifest.get("case_id") != effect_manifest.get("case_id"):
        raise ValueError(f"{role} case_id does not match the effect run")
    if manifest.get("case_hash") != effect_manifest.get("case_hash"):
        raise ValueError(f"{role} case_hash does not match the effect run")
    if manifest.get("level") != effect_manifest.get("level"):
        raise ValueError(f"{role} level does not match the effect run")
    if manifest.get("solver_id") != solver_id:
        raise ValueError(f"{role} solver_id mismatch")
    expected_inputs = {"variant": variant}
    if stage_name == "simpeg":
        expected_inputs["level"] = manifest.get("level")
    if not _completed_stage_is_intact(
        run_dir,
        manifest,
        stage_name,
        expected_inputs,
    ):
        raise ValueError(f"{role} lacks a completed {stage_name} stage")
    stage = manifest["stages"][stage_name]
    evidence_path = run_dir / evidence_name
    expected_hash = stage["file_sha256"].get(evidence_name)
    if type(expected_hash) is not str:
        raise ValueError(f"{role} stage does not hash {evidence_name}")
    actual_hash = _sha256_file(evidence_path)
    if actual_hash != expected_hash:
        raise ValueError(f"{role} {evidence_name} evidence hash mismatch")
    return evidence_path, {
        "run_dir": str(run_dir),
        "run_id": manifest["run_id"],
        "case_id": manifest["case_id"],
        "case_hash": manifest["case_hash"],
        "level": manifest["level"],
        "solver_id": solver_id,
        "stage": stage_name,
        "variant": variant,
        "evidence_file": evidence_name,
        "evidence_file_sha256": actual_hash,
    }


def _validated_effect_sources(
    args: argparse.Namespace,
    *,
    effect_run_dir: Path,
    effect_case: BenchmarkCase,
    effect_manifest: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    if (
        effect_case.case_id != "song2025_layered_pair"
        or effect_case.polarization is None
    ):
        raise ValueError("polarization effect requires a polarizable Song benchmark case")
    raw_paths = {
        "noip_simpeg": args.noip_simpeg_run,
        "noip_reference": args.noip_reference_run,
        "ip_simpeg": args.ip_simpeg_run,
        "ip_reference": args.ip_reference_run,
    }
    resolved: dict[str, Path] = {}
    for role, value in raw_paths.items():
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise FileNotFoundError(f"{role} source run is not a directory: {path}")
        resolved[role] = path
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("the four polarization-effect source runs must be distinct")
    if effect_run_dir in resolved.values():
        raise ValueError("the effect output run cannot also be a source run")

    specifications = {
        "noip_simpeg": (
            SIMPEG_SOLVER_ID,
            "simpeg",
            "noip",
            "predictions.csv",
        ),
        "noip_reference": (
            "empymod",
            "reference",
            "noip",
            "reference_empymod_or_1d.csv",
        ),
        "ip_simpeg": (
            SIMPEG_SOLVER_ID,
            "simpeg",
            "ip",
            "predictions.csv",
        ),
        "ip_reference": (
            "empymod",
            "reference",
            "cole-cole-exact",
            "reference_empymod_or_1d.csv",
        ),
    }
    evidence_paths: dict[str, Path] = {}
    identities: dict[str, Any] = {}
    for role, (solver_id, stage_name, variant, evidence_name) in specifications.items():
        evidence_path, identity = _source_run_evidence(
            resolved[role],
            effect_manifest=effect_manifest,
            role=role,
            solver_id=solver_id,
            stage_name=stage_name,
            variant=variant,
            evidence_name=evidence_name,
        )
        evidence_paths[role] = evidence_path
        identities[role] = identity
    return evidence_paths, {"source_runs": identities, "threshold": 0.10}


def _effect(args: argparse.Namespace) -> int:
    run_dir, case, manifest = _open_run(args, expected_solver=EFFECT_SOLVER_ID)
    evidence, inputs = _validated_effect_sources(
        args,
        effect_run_dir=run_dir,
        effect_case=case,
        effect_manifest=manifest,
    )
    output_dir = run_dir / "effect"
    if _completed_stage_is_intact(run_dir, manifest, "effect", inputs):
        return 0
    _refuse_existing_outputs([output_dir])

    with tempfile.TemporaryDirectory(
        prefix=".effect-staging-", dir=run_dir
    ) as temporary_text:
        temporary = Path(temporary_text)
        noip_staging = temporary / "noip"
        ip_staging = temporary / "ip"
        staged_output = temporary / "output"
        noip_staging.mkdir()
        ip_staging.mkdir()
        shutil.copyfile(evidence["noip_simpeg"], noip_staging / "predictions.csv")
        shutil.copyfile(
            evidence["noip_reference"],
            noip_staging / "reference_empymod_or_1d.csv",
        )
        shutil.copyfile(evidence["ip_simpeg"], ip_staging / "predictions.csv")
        shutil.copyfile(
            evidence["ip_reference"],
            ip_staging / "reference_empymod_or_1d.csv",
        )
        summary = write_polarization_effect_artifacts(
            noip_staging,
            ip_staging,
            staged_output,
            threshold=0.10,
        )
        _json_value(summary)
        if not staged_output.is_dir() or not any(staged_output.iterdir()):
            raise RuntimeError("polarization effect API produced no evidence files")
        os.replace(staged_output, output_dir)
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    _record_stage(
        run_dir,
        manifest,
        stage_name="effect",
        status="effect_complete",
        inputs=inputs,
        output_files=output_files,
    )
    return 0


def _gate_inputs(path_value: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path_value is None:
        return {}, {"gates_path": None, "gates_file_sha256": None}
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"gates evidence is not a file: {path}")
    try:
        gates = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"gates evidence is not valid JSON: {path}") from exc
    if not isinstance(gates, dict):
        raise ValueError("gates evidence must contain a JSON object")
    return gates, {"gates_path": str(path), "gates_file_sha256": _sha256_file(path)}


def _finalize(args: argparse.Namespace) -> int:
    run_dir, _case, manifest = _open_run(args, expected_solver=GATE_SOLVER_ID)
    gates, inputs = _gate_inputs(args.gates)
    output = run_dir / "final_gate_summary.json"
    if _completed_stage_is_intact(run_dir, manifest, "finalize", inputs):
        return 0
    _refuse_existing_outputs([output])

    summary = summarize_sotem_gates(gates)
    if summary.get("state") not in {
        "plumbing_pass",
        "noip_internally_validated",
        "ip_internally_validated",
        "failed_with_reproducible_evidence",
    }:
        raise ValueError("SOTEM gate API returned an invalid state")
    _atomic_write_json(output, summary)
    _record_stage(
        run_dir,
        manifest,
        stage_name="finalize",
        status=str(summary["state"]),
        inputs=inputs,
        output_files=[output],
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create an immutable run manifest")
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--solver", required=True, type=_nonempty_text)
    prepare.add_argument("--level", choices=_LEVELS, default="S0T0B0")
    prepare.add_argument("--output-root", default="generated/validation")
    prepare.add_argument("--run-dir")
    prepare.add_argument("--resume", action="store_true")
    prepare.set_defaults(handler=_prepare)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-dir", required=True)
        command.add_argument("--case", required=True)
        command.add_argument("--resume", action="store_true")

    reference = subparsers.add_parser("reference", help="compute independent empymod evidence")
    add_run_arguments(reference)
    reference.add_argument(
        "--variant", choices=("noip", "cole-cole-exact"), default="noip"
    )
    reference.set_defaults(handler=_reference)

    simpeg = subparsers.add_parser("simpeg", help="run the ATEM3D SimPEG-Debye adapter")
    add_run_arguments(simpeg)
    simpeg.add_argument("--variant", choices=("noip", "ip"), required=True)
    simpeg.set_defaults(handler=_simpeg)

    effect = subparsers.add_parser("effect", help="compare signed IP-minus-noIP evidence")
    add_run_arguments(effect)
    effect.add_argument("--noip-simpeg-run", required=True)
    effect.add_argument("--noip-reference-run", required=True)
    effect.add_argument("--ip-simpeg-run", required=True)
    effect.add_argument("--ip-reference-run", required=True)
    effect.set_defaults(handler=_effect)

    finalize = subparsers.add_parser("finalize", help="apply the fail-closed state machine")
    add_run_arguments(finalize)
    finalize.add_argument("--gates")
    finalize.set_defaults(handler=_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
