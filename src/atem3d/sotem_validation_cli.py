"""Reproducible, fail-closed orchestration for the canonical SOTEM suite."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any
import uuid

import numpy as np

from .polarization_effect import write_polarization_effect_artifacts
from .sotem_benchmark import BenchmarkCase, load_benchmark_case
from .sotem_gate import summarize_sotem_gates
from .sotem_observables import CanonicalResponse, canonical_response
from .sotem_simpeg_adapter import build_benchmark_config, run_simpeg_benchmark


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
_TRANSACTION_SCHEMA = "atem3d.sotem.stage-transaction"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


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


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


@contextmanager
def _run_lock(run_dir: str | Path):
    """Hold a non-blocking cross-process writer lock for one run directory."""

    run_path = Path(run_dir).expanduser().resolve()
    lock_path = run_path.parent / f".{run_path.name}.writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another writer holds the run lock: {run_path}") from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _process_peak_rss_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _environment_metadata() -> dict[str, Any]:
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_DISTRO_NAME"))
    return {
        "platform": platform.platform(),
        "os_name": os.name,
        "machine": platform.machine(),
        "sys_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "cwd": str(Path.cwd().resolve()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "is_wsl": bool(is_wsl),
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME", ""),
    }


def _timing_record(started_at: str, started_monotonic: float) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "completed_at": _utc_text(),
        "elapsed_seconds": max(0.0, float(time.perf_counter() - started_monotonic)),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "peak_rss_scope": "process_global_peak",
        "environment": _environment_metadata(),
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
    }


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


def _finite_number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"SimPEG provenance {name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"SimPEG provenance {name} must be a finite number")
    return numeric


def _positive_int(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"SimPEG provenance {name} must be a positive integer")
    return value


def _validate_simpeg_provenance(
    result: Mapping[str, Any],
    variant: str,
    expected_config: Mapping[str, Any],
) -> None:
    for name in ("mesh_hash", "time_hash"):
        value = result.get(name)
        if type(value) is not str or not _HEX64.fullmatch(value):
            raise ValueError(f"SimPEG provenance {name} must be a SHA256 hash")
        if value != expected_config.get(name):
            raise ValueError(f"SimPEG provenance {name} does not match the requested case")
    if result.get("coordinate_system") != "z_down":
        raise ValueError("SimPEG provenance coordinate_system must be z_down")
    expected_transform = expected_config["adapter_metadata"]["coordinate_transform"]
    if _json_value(result.get("coordinate_transform")) != _json_value(
        expected_transform
    ):
        raise ValueError("SimPEG provenance coordinate transform mismatch")

    mesh_stats = result.get("mesh_stats")
    if not isinstance(mesh_stats, Mapping):
        raise ValueError("SimPEG provenance mesh_stats must be a mapping")
    _positive_int(mesh_stats, "n_cells")
    _positive_int(mesh_stats, "n_edges")
    if mesh_stats.get("mesh_hash") != result["mesh_hash"]:
        raise ValueError("SimPEG provenance mesh_stats mesh_hash mismatch")
    expected_mesh = expected_config["mesh"]["metadata"]
    expected_mesh_stats = {
        "n_cells": expected_mesh["n_cells"],
        "n_edges": expected_mesh["n_edges"],
        "axis_cell_counts": expected_mesh["axis_cell_counts"],
        "bounds_m": expected_mesh["public_bounds_z_down_m"],
        "internal_bounds_z_up_m": expected_mesh["bounds_m"],
        "spatial_level": expected_mesh["spatial_level"],
        "boundary_level": expected_mesh["boundary_level"],
        "mesh_hash": expected_config["mesh_hash"],
    }
    for name, expected in expected_mesh_stats.items():
        if _json_value(mesh_stats.get(name)) != _json_value(expected):
            raise ValueError(f"SimPEG mesh provenance {name} mismatch")

    material_fit = result.get("material_fit")
    if variant == "noip":
        if material_fit is not None:
            raise ValueError("no-IP SimPEG provenance material_fit must be null")
        return
    if not isinstance(material_fit, Mapping):
        raise ValueError("IP SimPEG provenance material_fit must be a mapping")
    if material_fit.get("material_gate_pass") is not True:
        raise ValueError("IP SimPEG material provenance gate did not pass")
    if material_fit.get("fit_term_count") != 16 or type(
        material_fit.get("fit_term_count")
    ) is not int:
        raise ValueError("IP SimPEG material provenance requires exactly 16 terms")

    relative_l2 = _finite_number(material_fit, "relative_l2")
    relative_l2_limit = _finite_number(material_fit, "relative_l2_limit")
    if relative_l2_limit <= 0.0 or relative_l2_limit > 0.01:
        raise ValueError("IP SimPEG material relative-L2 limit exceeds 1 percent")
    if relative_l2 < 0.0 or relative_l2 > relative_l2_limit:
        raise ValueError("IP SimPEG material relative-L2 gate failed")

    dc_residual = _finite_number(material_fit, "dc_residual")
    dc_tolerance = _finite_number(material_fit, "dc_absolute_tolerance")
    if dc_tolerance < 0.0 or abs(dc_residual) > dc_tolerance:
        raise ValueError("IP SimPEG material DC constraint failed")
    if _finite_number(material_fit, "minimum_delta_sigma") <= 0.0:
        raise ValueError("IP SimPEG material terms must be strictly positive")

    sigma_dc = _finite_number(material_fit, "sigma_dc")
    sigma_infinity = _finite_number(material_fit, "sigma_infinity")
    delta_sum = _finite_number(material_fit, "delta_sum")
    if sigma_dc <= 0.0 or sigma_infinity <= sigma_dc or delta_sum <= 0.0:
        raise ValueError("IP SimPEG material conductivity provenance is invalid")

    terms = material_fit.get("debye_terms")
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise ValueError("IP SimPEG material debye_terms must be a sequence")
    if len(terms) != 16:
        raise ValueError("IP SimPEG material provenance requires 16 Debye terms")
    term_deltas: list[float] = []
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("IP SimPEG Debye term provenance must be a mapping")
        if _finite_number(term, "tau") <= 0.0:
            raise ValueError("IP SimPEG Debye relaxation times must be positive")
        term_delta = _finite_number(term, "delta_sigma")
        if term_delta <= 0.0:
            raise ValueError("IP SimPEG Debye amplitudes must be positive")
        term_deltas.append(term_delta)
    consistency_tolerance = max(dc_tolerance, 1.0e-14)
    if not math.isclose(
        sigma_infinity - sigma_dc,
        delta_sum,
        rel_tol=1.0e-12,
        abs_tol=consistency_tolerance,
    ):
        raise ValueError("IP SimPEG material conductivity balance is inconsistent")
    if not math.isclose(
        sum(term_deltas),
        delta_sum,
        rel_tol=1.0e-12,
        abs_tol=consistency_tolerance,
    ):
        raise ValueError("IP SimPEG material delta_sum does not match Debye terms")
    if not math.isclose(
        min(term_deltas),
        _finite_number(material_fit, "minimum_delta_sigma"),
        rel_tol=1.0e-12,
        abs_tol=consistency_tolerance,
    ):
        raise ValueError("IP SimPEG minimum_delta_sigma provenance mismatch")
    expected_material = expected_config["adapter_metadata"].get("material_fit")
    if _json_value(material_fit) != _json_value(expected_material):
        raise ValueError("IP SimPEG material provenance does not match the requested case")


def _fsync_directory(path: Path) -> None:
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = create_file(
        str(directory),
        0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def _fsync_file(path: Path) -> None:
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def _is_linklike(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _ensure_safe_directory(run_dir: Path, directory: Path) -> Path:
    run_root = run_dir.resolve(strict=True)
    try:
        relative = directory.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("output directory must be inside the run") from exc
    current = run_dir
    for part in relative.parts:
        current = current / part
        if current.exists() or _is_linklike(current):
            if _is_linklike(current):
                raise ValueError("output directory cannot contain symlinks or junctions")
            if not current.is_dir():
                raise ValueError("output directory path contains a non-directory")
            try:
                current.resolve(strict=True).relative_to(run_root)
            except ValueError as exc:
                raise ValueError("output directory resolves outside the run") from exc
            continue
        current.mkdir()
        _fsync_directory(current.parent)
    return current


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if _is_linklike(path):
            raise ValueError("stage bundle cannot contain symlinks or junctions")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
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


def _case_identity(
    path: str | Path,
) -> tuple[Path, BenchmarkCase, dict[str, str], bytes]:
    normalized = Path(path).expanduser().resolve(strict=True)
    if not normalized.is_file():
        raise FileNotFoundError(f"benchmark case is not a file: {normalized}")
    source_bytes = normalized.read_bytes()
    with tempfile.TemporaryDirectory(prefix="atem3d-case-snapshot-") as temporary_text:
        snapshot = Path(temporary_text) / normalized.name
        _atomic_write_bytes(snapshot, source_bytes)
        case = load_benchmark_case(snapshot)
    canonical = json.dumps(
        _case_payload(case),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return (
        normalized,
        case,
        {
            "case_file_sha256": _sha256_bytes(source_bytes),
            "case_hash": _sha256_bytes(canonical),
            "model_file_sha256": _sha256_bytes(_json_bytes(_case_payload(case))),
        },
        source_bytes,
    )


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
    prepare_record: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    git_commit, git_dirty = _git_metadata(repo_root)
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case.case_id,
        "case_path": str(case_path),
        "case_snapshot_path": "case_snapshot.yaml",
        "model_path": "model.json",
        **dict(identity),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "solver_id": solver_id,
        "level": level,
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
        "environment": _environment_metadata(),
        "created_at": _utc_text(),
        "prepare": _json_value(prepare_record),
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
    for name in ("case_file_sha256", "case_hash", "model_file_sha256"):
        if manifest.get(name) != identity[name]:
            raise ValueError(f"run manifest case hash mismatch ({name})")
    evidence = {
        "case_snapshot_path": "case_file_sha256",
        "model_path": "model_file_sha256",
    }
    for path_name, hash_name in evidence.items():
        relative = manifest.get(path_name)
        if type(relative) is not str:
            raise ValueError(f"run manifest {path_name} is invalid")
        evidence_path = _safe_evidence_path(run_dir, relative)
        if _sha256_file(evidence_path) != manifest.get(hash_name):
            raise ValueError(f"run manifest {path_name} hash mismatch")
    if manifest.get("solver_id") != solver_id:
        raise ValueError("run manifest solver_id mismatch")
    manifest_level = manifest.get("level")
    if manifest_level not in _LEVELS:
        raise ValueError("run manifest level is invalid")
    if level is not None and manifest_level != level:
        raise ValueError("run manifest level mismatch")


def _case_identity_for_manifest(
    run_dir: Path,
    manifest: Mapping[str, Any],
    requested_case: str | Path,
) -> tuple[Path, BenchmarkCase, dict[str, str], bytes]:
    requested_path = Path(requested_case).expanduser().resolve(strict=False)
    if requested_path.is_file():
        return _case_identity(requested_path)
    if manifest.get("case_path") != str(requested_path):
        raise FileNotFoundError(f"benchmark case is not a file: {requested_path}")
    snapshot_name = manifest.get("case_snapshot_path")
    if type(snapshot_name) is not str:
        raise ValueError("run manifest case snapshot path is invalid")
    snapshot = _safe_evidence_path(run_dir, snapshot_name)
    _snapshot_path, case, identity, source_bytes = _case_identity(snapshot)
    return requested_path, case, identity, source_bytes


def _prepare(args: argparse.Namespace) -> int:
    started_at = _utc_text()
    started_monotonic = time.perf_counter()
    case_details = None
    if args.run_dir is None:
        case_details = _case_identity(args.case)
        _case_path, case, _identity, _source_bytes = case_details
        run_id = _new_run_id()
        run_dir = Path(args.output_root).expanduser().resolve() / case.case_id / run_id
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_id = run_dir.name
    if not run_id:
        raise ValueError("run directory must have a non-empty final path component")

    with _run_lock(run_dir):
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(
                    f"refusing existing non-empty run directory without --resume: {run_dir}"
                )
            manifest = _load_manifest(run_dir)
            case_path, case, identity, _source_bytes = _case_identity_for_manifest(
                run_dir, manifest, args.case
            )
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
        if case_details is None:
            case_details = _case_identity(args.case)
        case_path, case, identity, source_bytes = case_details
        run_dir.mkdir(parents=True, exist_ok=True)
        _fsync_directory(run_dir.parent)
        _atomic_write_bytes(run_dir / "case_snapshot.yaml", source_bytes)
        _atomic_write_json(run_dir / "model.json", _case_payload(case))
        manifest = _manifest_payload(
            run_id=run_id,
            case_path=case_path,
            case=case,
            identity=identity,
            solver_id=args.solver,
            level=args.level,
            prepare_record=_timing_record(started_at, started_monotonic),
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
    manifest = _load_manifest(run_dir)
    case_path, case, identity, _source_bytes = _case_identity_for_manifest(
        run_dir, manifest, args.case
    )
    _verify_manifest(
        run_dir,
        manifest,
        case_path=case_path,
        case=case,
        identity=identity,
        solver_id=expected_solver,
    )
    return run_dir, case, manifest


def _safe_evidence_path(run_dir: Path, relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError("manifest evidence path must be a non-empty relative string")
    windows = PureWindowsPath(relative)
    posix = PurePosixPath(relative)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise ValueError("manifest evidence path must be relative")
    parts = windows.parts if "\\" in relative else posix.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("manifest evidence path cannot contain dot segments")
    candidate = run_dir.joinpath(*parts)
    current = run_dir
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("manifest evidence path cannot contain symlinks")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_dir.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("manifest evidence path resolves outside the run") from exc
    if not resolved.is_file():
        raise ValueError("manifest evidence path is not a file")
    return resolved


def _files_hashes(run_dir: Path, paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(paths):
        try:
            relative = path.relative_to(run_dir).as_posix()
        except ValueError as exc:
            raise ValueError("stage output must be inside the run directory") from exc
        safe_path = _safe_evidence_path(run_dir, relative)
        result[relative] = _sha256_file(safe_path)
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
        path = _safe_evidence_path(run_dir, relative)
        if _sha256_file(path) != expected:
            raise ValueError(f"completed {stage_name} stage evidence hash mismatch")
    return True


def _refuse_existing_outputs(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists() or _is_linklike(path)]
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
    timing: Mapping[str, Any] | None = None,
) -> None:
    stages = manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("run manifest stages must be a JSON object")
    timing_payload = dict(timing or {})
    if not timing_payload:
        now = _utc_text()
        timing_payload = {
            "started_at": now,
            "completed_at": now,
            "elapsed_seconds": 0.0,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "peak_rss_scope": "process_global_peak",
            "environment": _environment_metadata(),
        }
    stages[stage_name] = {
        "status": "complete",
        **_json_value(timing_payload),
        "inputs": _json_value(inputs),
        "file_sha256": _files_hashes(run_dir, output_files),
    }
    manifest["status"] = status
    _atomic_write_json(run_dir / "manifest.json", manifest)


def _bundle_dir(run_dir: Path, stage_name: str) -> Path:
    return run_dir / "artifacts" / stage_name


def _transaction_payload(
    run_dir: Path,
    bundle: Path,
    *,
    stage_name: str,
    status: str,
    inputs: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    final_bundle = _bundle_dir(run_dir, stage_name)
    hashes: dict[str, str] = {}
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        if path.name == "_transaction.json":
            continue
        relative_in_bundle = path.relative_to(bundle)
        final_relative = (final_bundle / relative_in_bundle).relative_to(run_dir).as_posix()
        hashes[final_relative] = _sha256_file(path)
    if not hashes:
        raise RuntimeError(f"{stage_name} stage produced no evidence files")
    return {
        "schema": _TRANSACTION_SCHEMA,
        "stage": stage_name,
        "manifest_status": status,
        "inputs": _json_value(inputs),
        "timing": _json_value(timing),
        "file_sha256": hashes,
    }


def _verify_transaction_bundle(
    run_dir: Path,
    stage_name: str,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    bundle = _bundle_dir(run_dir, stage_name)
    journal_path = bundle / "_transaction.json"
    if not journal_path.is_file() or journal_path.is_symlink():
        raise ValueError(f"published {stage_name} bundle lacks a safe transaction journal")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if not isinstance(journal, dict) or journal.get("schema") != _TRANSACTION_SCHEMA:
        raise ValueError(f"published {stage_name} bundle transaction schema mismatch")
    if journal.get("stage") != stage_name or journal.get("inputs") != _json_value(inputs):
        raise ValueError(f"published {stage_name} bundle identity mismatch")
    hashes = journal.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"published {stage_name} bundle lacks evidence hashes")
    files: list[Path] = []
    for relative, expected in hashes.items():
        if type(expected) is not str or not _HEX64.fullmatch(expected):
            raise ValueError(f"published {stage_name} bundle has invalid hashes")
        path = _safe_evidence_path(run_dir, relative)
        if _sha256_file(path) != expected:
            raise ValueError(f"published {stage_name} bundle evidence hash mismatch")
        files.append(path)
    return journal, files


def _transaction_export_hashes(
    stage_name: str, journal: Mapping[str, Any]
) -> dict[str, str]:
    hashes = journal.get("file_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError(f"published {stage_name} bundle lacks evidence hashes")
    prefix = PurePosixPath("artifacts", stage_name)
    exports: dict[str, str] = {}
    for relative, expected in sorted(hashes.items()):
        if type(relative) is not str or type(expected) is not str:
            raise ValueError(f"published {stage_name} bundle has invalid exports")
        bundle_relative = PurePosixPath(relative)
        if tuple(bundle_relative.parts[:2]) != tuple(prefix.parts):
            raise ValueError(f"published {stage_name} bundle path mismatch")
        export_relative = PurePosixPath(*bundle_relative.parts[2:])
        if not export_relative.parts:
            raise ValueError(f"published {stage_name} bundle has an empty export path")
        exports[export_relative.as_posix()] = expected
    return exports


def _materialize_bundle_exports(
    run_dir: Path,
    stage_name: str,
    journal: Mapping[str, Any],
) -> list[Path]:
    """Restore the specification-level run outputs from an immutable bundle."""

    exports: list[Path] = []
    for export_text, expected in _transaction_export_hashes(
        stage_name, journal
    ).items():
        export_relative = PurePosixPath(export_text)
        source_relative = PurePosixPath("artifacts", stage_name, *export_relative.parts)
        source = _safe_evidence_path(run_dir, source_relative.as_posix())
        target = run_dir.joinpath(*export_relative.parts)
        _ensure_safe_directory(run_dir, target.parent)
        if target.exists() or _is_linklike(target):
            if _is_linklike(target) or not target.is_file():
                raise ValueError(f"published {stage_name} export is not a safe file")
            if _sha256_file(target) != expected:
                raise ValueError(f"published {stage_name} export hash mismatch")
            exports.append(target)
            continue
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if _sha256_file(temporary) != expected:
                raise ValueError(f"published {stage_name} export copy hash mismatch")
            _fsync_file(temporary)
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        exports.append(target)
    return exports


def _recover_or_complete_stage(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    stage_name: str,
    status: str,
    inputs: Mapping[str, Any],
) -> bool:
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        raise ValueError("run manifest stages must be a mapping")
    record = stages.get(stage_name)
    if record is not None:
        if not isinstance(record, Mapping) or record.get("status") != "complete":
            raise ValueError(f"run manifest has an invalid {stage_name} stage record")
        if record.get("inputs") != _json_value(inputs):
            raise ValueError(f"completed {stage_name} stage inputs do not match")
        stage_hashes = record.get("file_sha256")
        if not isinstance(stage_hashes, Mapping) or not stage_hashes:
            raise ValueError(f"completed {stage_name} stage lacks evidence hashes")
        bundle = _bundle_dir(run_dir, stage_name)
        if not bundle.exists() or _is_linklike(bundle):
            raise ValueError(f"completed {stage_name} stage lacks its transaction bundle")
        journal, _bundle_files = _verify_transaction_bundle(
            run_dir, stage_name, inputs
        )
        if journal.get("manifest_status") != status:
            raise ValueError(f"published {stage_name} bundle status mismatch")
        if _transaction_export_hashes(stage_name, journal) != stage_hashes:
            raise ValueError(f"completed {stage_name} manifest and bundle hashes differ")
        _materialize_bundle_exports(run_dir, stage_name, journal)
        if not _completed_stage_is_intact(run_dir, manifest, stage_name, inputs):
            raise RuntimeError(f"completed {stage_name} stage could not be recovered")
        return True
    bundle = _bundle_dir(run_dir, stage_name)
    if not bundle.exists():
        return False
    journal, _bundle_files = _verify_transaction_bundle(run_dir, stage_name, inputs)
    if journal.get("manifest_status") != status:
        raise ValueError(f"published {stage_name} bundle status mismatch")
    files = _materialize_bundle_exports(run_dir, stage_name, journal)
    _record_stage(
        run_dir,
        manifest,
        stage_name=stage_name,
        status=status,
        inputs=inputs,
        output_files=files,
        timing=journal.get("timing"),
    )
    return True


def _publish_stage_bundle(
    run_dir: Path,
    *,
    stage_name: str,
    status: str,
    inputs: Mapping[str, Any],
    started_at: str,
    started_monotonic: float,
    build,
) -> tuple[list[Path], dict[str, Any]]:
    artifacts = run_dir / "artifacts"
    _ensure_safe_directory(run_dir, artifacts)
    final_bundle = _bundle_dir(run_dir, stage_name)
    if final_bundle.exists() or _is_linklike(final_bundle):
        raise FileExistsError(f"refusing to overwrite published {stage_name} bundle")
    staging = artifacts / f".{stage_name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    _fsync_directory(artifacts)
    try:
        build(staging)
        timing = _timing_record(started_at, started_monotonic)
        transaction = _transaction_payload(
            run_dir,
            staging,
            stage_name=stage_name,
            status=status,
            inputs=inputs,
            timing=timing,
        )
        _atomic_write_json(staging / "_transaction.json", transaction)
        _fsync_tree(staging)
        os.replace(staging, final_bundle)
        _fsync_directory(artifacts)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    journal, _bundle_files = _verify_transaction_bundle(run_dir, stage_name, inputs)
    files = _materialize_bundle_exports(run_dir, stage_name, journal)
    return files, timing


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


def get_empymod_reference(times, config, *, mode: str, srcpts: int):
    """Monkeypatchable adapter to the existing independent reference API."""

    return _load_pipeline_module().get_empymod_reference(
        times, config, mode=mode, srcpts=srcpts
    )


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
    started_at = _utc_text()
    started_monotonic = time.perf_counter()
    run_dir, case, manifest = _open_run(args, expected_solver="empymod")
    inputs = {"variant": args.variant, "srcpts": args.srcpts}
    outputs = [
        run_dir / "empymod.csv",
        run_dir / "reference_empymod_or_1d.csv",
        run_dir / "empymod_metadata.json",
    ]
    if _recover_or_complete_stage(
        run_dir,
        manifest,
        stage_name="reference",
        status="reference_complete",
        inputs=inputs,
    ):
        return 0
    _refuse_existing_outputs(outputs)
    _ensure_safe_directory(run_dir, run_dir / "artifacts")

    config = _pipeline_config_for_case(case, args.variant)
    config.empymod_srcpts = args.srcpts
    result = get_empymod_reference(
        np.asarray(case.observation_times, dtype=float),
        config,
        mode=args.variant,
        srcpts=args.srcpts,
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
        "srcpts": args.srcpts,
        "coordinate_system": "z_down",
        "columns": list(response.columns),
    }
    def build(staging: Path) -> None:
        _atomic_write_canonical(staging / "empymod.csv", response)
        _atomic_write_bytes(
            staging / "reference_empymod_or_1d.csv",
            _compatibility_csv_bytes(response),
        )
        _atomic_write_json(staging / "empymod_metadata.json", metadata)

    output_files, timing = _publish_stage_bundle(
        run_dir,
        stage_name="reference",
        status="reference_complete",
        inputs=inputs,
        started_at=started_at,
        started_monotonic=started_monotonic,
        build=build,
    )
    _record_stage(
        run_dir,
        manifest,
        stage_name="reference",
        status="reference_complete",
        inputs=inputs,
        output_files=output_files,
        timing=timing,
    )
    return 0


def _parse_manifest_level(manifest: Mapping[str, Any]) -> tuple[str, str, int]:
    level = manifest.get("level")
    if level not in _LEVELS:
        raise ValueError("run manifest level is invalid")
    return level[0:2], level[4:6], _SUBSTEPS[level[2:4]]


def _simpeg(args: argparse.Namespace) -> int:
    started_at = _utc_text()
    started_monotonic = time.perf_counter()
    run_dir, case, manifest = _open_run(args, expected_solver=SIMPEG_SOLVER_ID)
    spatial_level, boundary_level, substeps = _parse_manifest_level(manifest)
    inputs = {"variant": args.variant, "level": manifest["level"]}
    outputs = [
        run_dir / "simpeg.csv",
        run_dir / "predictions.csv",
        run_dir / "simpeg_metadata.json",
    ]
    if _recover_or_complete_stage(
        run_dir,
        manifest,
        stage_name="simpeg",
        status="simpeg_complete",
        inputs=inputs,
    ):
        return 0
    _refuse_existing_outputs(outputs)
    _ensure_safe_directory(run_dir, run_dir / "artifacts")

    expected_config = build_benchmark_config(
        case,
        variant=args.variant,
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=substeps,
    )
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
    _validate_simpeg_provenance(result, args.variant, expected_config)
    expected_metadata = expected_config["adapter_metadata"]
    metadata = {
        "solver_id": SIMPEG_SOLVER_ID,
        "case_id": case.case_id,
        "variant": args.variant,
        "level": manifest["level"],
        "coordinate_system": "z_down",
        "internal_coordinate_system": expected_config["coordinate_system"],
        "coordinate_transform": result["coordinate_transform"],
        "columns": list(response.columns),
        "mesh_hash": result.get("mesh_hash"),
        "time_hash": result.get("time_hash"),
        "mesh_stats": result.get("mesh_stats"),
        "material_fit": result.get("material_fit"),
        "time_schedule": {
            "time_steps_s": expected_config["time_steps"],
            "output_indices": expected_metadata["output_indices"],
            "observation_times_s": expected_metadata["observation_times"],
            "time_hash": expected_config["time_hash"],
        },
    }
    normalized_metadata = _json_value(metadata)
    def build(staging: Path) -> None:
        _atomic_write_canonical(staging / "simpeg.csv", response)
        _atomic_write_bytes(
            staging / "predictions.csv", _compatibility_csv_bytes(response)
        )
        _atomic_write_json(staging / "simpeg_metadata.json", normalized_metadata)

    output_files, timing = _publish_stage_bundle(
        run_dir,
        stage_name="simpeg",
        status="simpeg_complete",
        inputs=inputs,
        started_at=started_at,
        started_monotonic=started_monotonic,
        build=build,
    )
    _record_stage(
        run_dir,
        manifest,
        stage_name="simpeg",
        status="simpeg_complete",
        inputs=inputs,
        output_files=output_files,
        timing=timing,
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
    expected_status = f"{stage_name}_complete"
    if manifest.get("status") != expected_status:
        raise ValueError(f"{role} manifest status mismatch")
    source_case_path = manifest.get("case_path")
    if type(source_case_path) is not str:
        raise ValueError(f"{role} manifest case_path is invalid")
    snapshot_name = manifest.get("case_snapshot_path")
    if type(snapshot_name) is not str:
        raise ValueError(f"{role} manifest case snapshot path is invalid")
    snapshot = _safe_evidence_path(run_dir, snapshot_name)
    _snapshot_path, source_case, source_identity, _source_bytes = _case_identity(snapshot)
    _verify_manifest(
        run_dir,
        manifest,
        case_path=Path(source_case_path),
        case=source_case,
        identity=source_identity,
        solver_id=solver_id,
        level=str(effect_manifest.get("level")),
    )
    if stage_name == "reference":
        stage_record = manifest.get("stages", {}).get(stage_name, {})
        recorded_inputs = stage_record.get("inputs", {})
        if not isinstance(recorded_inputs, Mapping):
            raise ValueError(f"{role} reference stage inputs are invalid")
        if recorded_inputs.get("variant") != variant:
            raise ValueError(f"{role} reference variant mismatch")
        srcpts = recorded_inputs.get("srcpts")
        if type(srcpts) is not int or srcpts <= 0:
            raise ValueError(f"{role} reference srcpts provenance is invalid")
        expected_inputs = {"variant": variant, "srcpts": srcpts}
    else:
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
    bundle = _bundle_dir(run_dir, stage_name)
    if not bundle.exists() or _is_linklike(bundle):
        raise ValueError(f"{role} lacks a safe {stage_name} transaction bundle")
    journal, _bundle_files = _verify_transaction_bundle(
        run_dir, stage_name, expected_inputs
    )
    if journal.get("manifest_status") != expected_status:
        raise ValueError(f"{role} {stage_name} transaction status mismatch")
    if _transaction_export_hashes(stage_name, journal) != stage.get("file_sha256"):
        raise ValueError(f"{role} {stage_name} manifest and bundle hashes differ")
    evidence_path = run_dir / evidence_name
    expected_hash = stage["file_sha256"].get(evidence_name)
    if type(expected_hash) is not str:
        raise ValueError(f"{role} stage does not hash {evidence_name}")
    actual_hash = _sha256_file(evidence_path)
    if actual_hash != expected_hash:
        raise ValueError(f"{role} {evidence_name} evidence hash mismatch")
    identity = {
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
    if stage_name == "reference":
        identity["srcpts"] = srcpts
    return evidence_path, identity


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
    started_at = _utc_text()
    started_monotonic = time.perf_counter()
    run_dir, case, manifest = _open_run(args, expected_solver=EFFECT_SOLVER_ID)
    evidence, inputs = _validated_effect_sources(
        args,
        effect_run_dir=run_dir,
        effect_case=case,
        effect_manifest=manifest,
    )
    output_dir = run_dir / "effect"
    if _recover_or_complete_stage(
        run_dir,
        manifest,
        stage_name="effect",
        status="effect_complete",
        inputs=inputs,
    ):
        return 0
    _refuse_existing_outputs([output_dir])
    _ensure_safe_directory(run_dir, run_dir / "artifacts")

    with tempfile.TemporaryDirectory(
        prefix=".effect-staging-", dir=run_dir
    ) as temporary_text:
        temporary = Path(temporary_text)
        noip_staging = temporary / "noip"
        ip_staging = temporary / "ip"
        noip_staging.mkdir()
        ip_staging.mkdir()
        staged_sources = {
            "noip_simpeg": noip_staging / "predictions.csv",
            "noip_reference": noip_staging / "reference_empymod_or_1d.csv",
            "ip_simpeg": ip_staging / "predictions.csv",
            "ip_reference": ip_staging / "reference_empymod_or_1d.csv",
        }
        for role, target in staged_sources.items():
            shutil.copyfile(evidence[role], target)
            expected_hash = inputs["source_runs"][role]["evidence_file_sha256"]
            if _sha256_file(target) != expected_hash:
                raise ValueError(f"{role} source changed while it was being copied")

        def build(staging: Path) -> None:
            staged_output = staging / "effect"
            summary = write_polarization_effect_artifacts(
                noip_staging,
                ip_staging,
                staged_output,
                threshold=0.10,
            )
            _json_value(summary)
            if not staged_output.is_dir() or not any(staged_output.iterdir()):
                raise RuntimeError("polarization effect API produced no evidence files")

        output_files, timing = _publish_stage_bundle(
            run_dir,
            stage_name="effect",
            status="effect_complete",
            inputs=inputs,
            started_at=started_at,
            started_monotonic=started_monotonic,
            build=build,
        )
    _record_stage(
        run_dir,
        manifest,
        stage_name="effect",
        status="effect_complete",
        inputs=inputs,
        output_files=output_files,
        timing=timing,
    )
    return 0


def _decode_gates(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        gates = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"gates evidence is not valid JSON: {path}") from exc
    if not isinstance(gates, dict):
        raise ValueError("gates evidence must contain a JSON object")
    return gates


def _gate_inputs(
    path_value: str | None,
) -> tuple[dict[str, Any], dict[str, Any], bytes | None]:
    if path_value is None:
        return {}, {"gates_path": None, "gates_file_sha256": None}, None
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"gates evidence is not a file: {path}")
    payload = path.read_bytes()
    gates = _decode_gates(payload, path)
    return (
        gates,
        {"gates_path": str(path), "gates_file_sha256": _sha256_bytes(payload)},
        payload,
    )


def _resume_gate_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
    path_value: str,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    requested = Path(path_value).expanduser().resolve(strict=False)
    stages = manifest.get("stages", {})
    record = stages.get("finalize") if isinstance(stages, Mapping) else None
    recorded_inputs = record.get("inputs") if isinstance(record, Mapping) else None
    if not isinstance(recorded_inputs, Mapping):
        journal_path = _safe_evidence_path(
            run_dir, "artifacts/finalize/_transaction.json"
        )
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if not isinstance(journal, Mapping) or journal.get("stage") != "finalize":
            raise ValueError("finalize transaction journal is invalid")
        recorded_inputs = journal.get("inputs")
    if not isinstance(recorded_inputs, Mapping):
        raise ValueError("finalize transaction lacks gate input provenance")
    inputs = dict(recorded_inputs)
    if inputs.get("gates_path") != str(requested):
        raise ValueError("finalize gate snapshot path does not match the request")
    expected_hash = inputs.get("gates_file_sha256")
    if type(expected_hash) is not str or not _HEX64.fullmatch(expected_hash):
        raise ValueError("finalize gate snapshot hash is invalid")
    snapshot = _safe_evidence_path(
        run_dir, "artifacts/finalize/inputs/gates.json"
    )
    payload = snapshot.read_bytes()
    if _sha256_bytes(payload) != expected_hash:
        raise ValueError("finalize gate snapshot hash mismatch")
    return _decode_gates(payload, snapshot), inputs, payload


def _finalize(args: argparse.Namespace) -> int:
    started_at = _utc_text()
    started_monotonic = time.perf_counter()
    run_dir, _case, manifest = _open_run(args, expected_solver=GATE_SOLVER_ID)
    try:
        gates, inputs, gates_payload = _gate_inputs(args.gates)
    except FileNotFoundError:
        if args.gates is None:
            raise
        gates, inputs, gates_payload = _resume_gate_snapshot(
            run_dir, manifest, args.gates
        )
    output = run_dir / "final_gate_summary.json"
    summary = summarize_sotem_gates(gates)
    if summary.get("state") not in {
        "plumbing_pass",
        "noip_internally_validated",
        "ip_internally_validated",
        "failed_with_reproducible_evidence",
    }:
        raise ValueError("SOTEM gate API returned an invalid state")
    if summary["state"] != "failed_with_reproducible_evidence":
        summary["claimed_state"] = summary["state"]
        summary["state"] = "failed_with_reproducible_evidence"
        summary["noip_internally_validated"] = False
        summary["ip_internally_validated"] = False
        summary["reference_independent"] = False
        summary["noip_reference_independent"] = False
        summary["ip_reference_independent"] = False
        summary["reason_codes"] = list(summary.get("reason_codes", [])) + [
            "unverified_external_gate_evidence"
        ]
    status = str(summary["state"])
    if _recover_or_complete_stage(
        run_dir,
        manifest,
        stage_name="finalize",
        status=status,
        inputs=inputs,
    ):
        return 0
    _refuse_existing_outputs([output])
    _ensure_safe_directory(run_dir, run_dir / "artifacts")

    def build(staging: Path) -> None:
        if gates_payload is not None:
            _atomic_write_bytes(staging / "inputs" / "gates.json", gates_payload)
        _atomic_write_json(staging / "final_gate_summary.json", summary)

    output_files, timing = _publish_stage_bundle(
        run_dir,
        stage_name="finalize",
        status=status,
        inputs=inputs,
        started_at=started_at,
        started_monotonic=started_monotonic,
        build=build,
    )
    _record_stage(
        run_dir,
        manifest,
        stage_name="finalize",
        status=status,
        inputs=inputs,
        output_files=output_files,
        timing=timing,
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
    reference.add_argument("--srcpts", type=_positive_cli_int, default=5)
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
    if args.command == "prepare":
        return int(args.handler(args))
    with _run_lock(args.run_dir):
        return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
