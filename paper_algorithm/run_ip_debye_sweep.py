#!/usr/bin/env python3
"""Run the Zhou-2020 no-IP/IP benchmark with a Debye-term sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
DOLFINX_DIR = REPO_ROOT / "dolfinx"
for path in (SRC, REPO_ROOT, DOLFINX_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atem3d.sotem_benchmark import load_benchmark_case
from run_sotem_benchmark import build_pipeline_argv


def _replace_flag(argv: list[str], name: str, value: str) -> list[str]:
    prefix = f"--{name}="
    filtered = [item for item in argv if not item.startswith(prefix)]
    filtered.append(f"{prefix}{value}")
    return filtered


def _run_case(
    *,
    case,
    variant: str,
    level: str,
    workdir: Path,
    terms: int | None,
    memory_limit_gb: float,
    force: bool,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    if (workdir / "verification_report.txt").is_file() and not force:
        print(f"[skip] {workdir}")
        return

    argv = build_pipeline_argv(case, variant, level, workdir)
    argv.extend(["--no-install", "--checkpoint-forward"])
    argv = _replace_flag(argv, "memory-limit-gb", f"{memory_limit_gb:g}")
    if terms is not None:
        argv = _replace_flag(argv, "cole-n-terms", str(int(terms)))
    if (workdir / "forward_checkpoint.npz").is_file():
        argv.append("--resume-forward")

    command = [sys.executable, str(REPO_ROOT / "dolfinx" / "sotem_pipeline.py"), *argv]
    time_executable = Path("/usr/bin/time")
    if time_executable.is_file():
        command = [str(time_executable), "-v", *command]

    print("[run]", " ".join(command))
    with (workdir / "run.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _parse_terms(value: str) -> tuple[int, ...]:
    terms = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not terms or any(item <= 0 for item in terms):
        raise argparse.ArgumentTypeError("terms must be comma-separated positive integers")
    return terms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--level", default="S1T1B1")
    parser.add_argument("--terms", type=_parse_terms, default=(8, 12, 16, 20))
    parser.add_argument("--memory-limit-gb", type=float, default=32.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    case = load_benchmark_case(
        REPO_ROOT / "benchmarks" / "sotem" / "zhou2020_grounded_wire.yaml"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    _run_case(
        case=case,
        variant="noip",
        level=args.level,
        workdir=args.output_root / f"noip_{args.level}",
        terms=None,
        memory_limit_gb=args.memory_limit_gb,
        force=args.force,
    )
    for n_terms in args.terms:
        _run_case(
            case=case,
            variant="ip",
            level=args.level,
            workdir=args.output_root / f"ip_K{n_terms}_{args.level}",
            terms=n_terms,
            memory_limit_gb=args.memory_limit_gb,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
