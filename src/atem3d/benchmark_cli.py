"""Command-line runner for boundary convergence benchmarks."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .benchmark import BoundaryBenchmarkCase, run_boundary_benchmark
from .config import build_simulation


@dataclass(frozen=True)
class BenchmarkSpec:
    """Boundary benchmark YAML specification."""

    cases: list[BoundaryBenchmarkCase]
    component_names: list[str]
    tolerance: float
    reference: str | None = None
    time_min: float | None = None
    time_max: float | None = None
    data_only: bool = False
    absolute_tolerance: float | None = None


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load a boundary benchmark spec from YAML."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    base_config = payload.get("base_config")
    cases = [_benchmark_case_from_payload(case, base_config) for case in payload["cases"]]
    return BenchmarkSpec(
        cases=cases,
        component_names=[str(name) for name in payload["component_names"]],
        tolerance=float(payload["tolerance"]),
        reference=payload.get("reference"),
        time_min=None if payload.get("time_min") is None else float(payload["time_min"]),
        time_max=None if payload.get("time_max") is None else float(payload["time_max"]),
        data_only=bool(payload.get("data_only", False)),
        absolute_tolerance=(
            None
            if payload.get("absolute_tolerance") is None
            else float(payload["absolute_tolerance"])
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ATEM3D boundary convergence benchmark.")
    parser.add_argument("spec", type=Path, help="Benchmark YAML file")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/boundary_benchmark.json"))
    args = parser.parse_args(argv)

    spec = load_benchmark_spec(args.spec)
    report = run_boundary_benchmark(
        spec.cases,
        runner=_runner(data_only=spec.data_only),
        component_names=spec.component_names,
        tolerance=spec.tolerance,
        output_path=args.output,
        reference_name=spec.reference,
        time_min=spec.time_min,
        time_max=spec.time_max,
        absolute_tolerance=spec.absolute_tolerance,
    )
    print(f"wrote {args.output}")
    print(
        f"reference: {report['reference']}; tolerance: {report['tolerance']}; "
        f"absolute_tolerance: {report['absolute_tolerance']}"
    )
    for case in report["cases"]:
        print(
            f"{case['name']}: relative_linf_max={case['relative_linf_max']:.6e}; "
            f"passed={case['passed']}"
        )
    return 0


def _benchmark_case_from_payload(
    case: Mapping[str, Any],
    base_config: Mapping[str, Any] | None,
) -> BoundaryBenchmarkCase:
    if "config" in case:
        return BoundaryBenchmarkCase(name=str(case["name"]), config=case["config"])
    if "overrides" not in case:
        raise ValueError("each benchmark case must define either config or overrides")
    if base_config is None:
        raise ValueError("benchmark cases with overrides require base_config")
    config = deepcopy(dict(base_config))
    _merge_into(config, case["overrides"])
    return BoundaryBenchmarkCase(name=str(case["name"]), config=config)


def _runner(*, data_only: bool):
    def run(config):
        simulation = build_simulation(config)
        if data_only:
            return simulation.run_data_only()
        return simulation.run()

    return run


def _merge_into(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _merge_into(target[key], value)
        else:
            target[key] = deepcopy(value)


if __name__ == "__main__":
    raise SystemExit(main())
