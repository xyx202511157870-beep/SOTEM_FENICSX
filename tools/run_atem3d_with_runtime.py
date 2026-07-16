#!/usr/bin/env python3
"""Bootstrap native DLLs and verify PARDISO before starting an ATEM3D run."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


def _add_native_dll_directories() -> list[object]:
    handles: list[object] = []
    configured = os.environ.get("ATEM3D_DLL_DIRS", "")
    for raw_path in configured.split(os.pathsep):
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_dir():
            raise RuntimeError(f"ATEM3D native DLL directory does not exist: {path}")
        if hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(path)))
    return handles


def _out_of_core_enabled() -> bool:
    return os.environ.get("ATEM3D_PARDISO_OUT_OF_CORE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pardiso_threads() -> int | None:
    raw = os.environ.get("ATEM3D_PARDISO_THREADS", "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("ATEM3D_PARDISO_THREADS must be a positive integer")
    return value


def _preflight_pardiso() -> None:
    import numpy as np
    from pymatsolver import Pardiso
    from scipy.sparse import eye

    solver = Pardiso(eye(2, format="csc"), n_threads=_pardiso_threads())
    if _out_of_core_enabled():
        solver.solver.set_iparm(59, 2)
    result = np.asarray(solver * np.array([1.0, 2.0]), dtype=float)
    solver.clean()
    if not np.allclose(result, [1.0, 2.0]):
        raise RuntimeError("PARDISO preflight returned an incorrect solution")


def runtime_audit_path(arguments: list[str]) -> Path | None:
    if "--output" not in arguments:
        return None
    index = arguments.index("--output")
    if index + 1 >= len(arguments):
        raise ValueError("--output requires a path")
    return Path(arguments[index + 1]).parent / "runtime_environment.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_runtime_audit(arguments: list[str]) -> Path | None:
    path = runtime_audit_path(arguments)
    if path is None:
        return None
    from pydiso.mkl_solver import get_mkl_version

    packages = {}
    for name in ("atem3d", "numpy", "scipy", "discretize", "simpeg", "pymatsolver", "pydiso"):
        packages[name] = importlib.metadata.version(name)
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "mkl": _json_safe(get_mkl_version()),
        "native_dll_directories": [
            item for item in os.environ.get("ATEM3D_DLL_DIRS", "").split(os.pathsep) if item
        ],
        "pardiso_out_of_core": _out_of_core_enabled(),
        "pardiso_threads": _pardiso_threads(),
        "pardiso_preflight": "passed",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    _dll_handles = _add_native_dll_directories()
    _preflight_pardiso()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--preflight-pardiso"]:
        print("PARDISO preflight passed")
        return 0
    _write_runtime_audit(arguments)
    from atem3d.cli import main as atem3d_main

    return int(atem3d_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
