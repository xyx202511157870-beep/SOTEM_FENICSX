#!/usr/bin/env python3
"""Flow 0: baseline freeze. Do not modify solver code."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp" / "flow0_baseline"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> dict:
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "status", "--porcelain=v1"]),
        "workspace": str(REPO_ROOT),
        "describe": _run(["git", "log", "-1", "--oneline"]),
    }


def _python_info() -> dict:
    modules = {}
    for name in ("empymod", "numpy", "scipy", "dolfinx", "petsc4py", "mpi4py", "gmsh", "simpeg"):
        try:
            module = __import__(name)
            modules[name] = getattr(module, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            modules[name] = f"missing: {type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "modules": modules,
    }


def _memory_info() -> dict:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return {}
    values = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            key, raw, _unit = line.split()
            values[key[:-1]] = int(raw) * 1024
    return values


def _run_pytest(log_path: Path) -> int:
    command = [sys.executable, "-m", "pytest", "-q"]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _layered_smoke() -> dict:
    import numpy as np

    from atem3d.adaptive_debye_mvp.layered_forward import (
        noip_resistivities,
        pelton_layers,
        run_case_models,
    )
    from atem3d.adaptive_debye_mvp.registry import generate_split
    from atem3d.adaptive_debye_mvp.protocol_constants import observation_times

    case = generate_split("pilot_gap")[0]
    times = observation_times()[:7]
    noip = run_case_models(
        case,
        resistivities=noip_resistivities(case),
        model_id="flow0_noip",
        waveform_ids=("W0",),
        times=times,
        include_disks=False,
    )["W0"]
    ip = run_case_models(
        case,
        resistivities=pelton_layers(case),
        model_id="flow0_ip",
        waveform_ids=("W0",),
        times=times,
        include_disks=False,
    )["W0"]
    delta = np.asarray(ip.data) - np.asarray(noip.data)
    return {
        "case_id": case.case_id,
        "n_times": int(times.size),
        "n_locations": len(noip.locations),
        "noip_peak_abs": float(np.max(np.abs(noip.data))),
        "ip_peak_abs": float(np.max(np.abs(ip.data))),
        "ip_increment_peak_abs": float(np.max(np.abs(delta))),
        "ip_increment_nonzero": bool(np.max(np.abs(delta)) > 0.0),
        "script": "paper_algorithm/run_ip_debye_sweep.py skipped (3-D FEniCSx)",
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    git = _git_info()
    environment = {**_python_info(), "memory": _memory_info(), "recorded_at": datetime.now(timezone.utc).isoformat()}
    (OUTPUT / "baseline_git.json").write_text(json.dumps(git, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "baseline_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    test_log = OUTPUT / "baseline_test.log"
    test_code = _run_pytest(test_log)
    smoke = _layered_smoke()
    manifest = {
        "pytest_exit_code": test_code,
        "layered_smoke": smoke,
        "skipped_3d": True,
        "solver_edits": False,
    }
    (OUTPUT / "baseline_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Flow 0 baseline report",
        "",
        f"- git: `{git['branch']}` `{git['commit']}`",
        f"- python: `{environment['python'].splitlines()[0]}`",
        f"- empymod: `{environment['modules'].get('empymod')}`",
        f"- numpy: `{environment['modules'].get('numpy')}`",
        f"- scipy: `{environment['modules'].get('scipy')}`",
        f"- pytest exit code: `{test_code}`",
        f"- layered smoke case: `{smoke['case_id']}` IP increment peak `{smoke['ip_increment_peak_abs']:.3e}`",
        "- 3-D sweep not run.",
        "- Solver code was not modified in Flow 0.",
        "",
    ]
    (OUTPUT / "BASELINE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    hash_lines = []
    for path in sorted(OUTPUT.iterdir()):
        if path.name == "hashes.sha256" or not path.is_file():
            continue
        hash_lines.append(f"{_sha256_file(path)}  {path.name}")
    (OUTPUT / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} pytest_exit={test_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
