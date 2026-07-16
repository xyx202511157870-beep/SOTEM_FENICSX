#!/usr/bin/env python3
"""Bootstrap native DLLs and verify PARDISO before starting an ATEM3D run."""

from __future__ import annotations

import os
from pathlib import Path
import sys


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


def _preflight_pardiso() -> None:
    import numpy as np
    from pymatsolver import Pardiso
    from scipy.sparse import eye

    solver = Pardiso(eye(2, format="csc"))
    result = np.asarray(solver * np.array([1.0, 2.0]), dtype=float)
    solver.clean()
    if not np.allclose(result, [1.0, 2.0]):
        raise RuntimeError("PARDISO preflight returned an incorrect solution")


def main(argv: list[str] | None = None) -> int:
    _dll_handles = _add_native_dll_directories()
    _preflight_pardiso()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--preflight-pardiso"]:
        print("PARDISO preflight passed")
        return 0
    from atem3d.cli import main as atem3d_main

    return int(atem3d_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
