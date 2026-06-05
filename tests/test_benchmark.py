import json

import numpy as np

from atem3d.benchmark import BoundaryBenchmarkCase, run_boundary_benchmark


class FakeResult:
    def __init__(self, data, times=None):
        self.data = np.asarray(data, dtype=float)
        if times is not None:
            self.times = np.asarray(times, dtype=float)


def test_boundary_benchmark_uses_last_case_as_reference_and_writes_report(tmp_path):
    cases = [
        BoundaryBenchmarkCase(name="thin", config={"value": 1.2}),
        BoundaryBenchmarkCase(name="medium", config={"value": 1.05}),
        BoundaryBenchmarkCase(name="reference", config={"value": 1.0}),
    ]

    def runner(config):
        return FakeResult([[config["value"], 2.0 * config["value"]]])

    report_path = tmp_path / "boundary.json"
    report = run_boundary_benchmark(
        cases,
        runner=runner,
        component_names=["Ex", "Hz"],
        tolerance=0.1,
        output_path=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload == report
    assert payload["reference"] == "reference"
    assert payload["cases"][0]["name"] == "thin"
    assert payload["cases"][0]["passed"] is False
    assert payload["cases"][1]["passed"] is True
    assert "Ex" in payload["cases"][0]["components"]


def test_boundary_benchmark_can_use_named_reference_case():
    cases = [
        BoundaryBenchmarkCase(name="reference", config={"value": 2.0}),
        BoundaryBenchmarkCase(name="candidate", config={"value": 2.1}),
    ]

    report = run_boundary_benchmark(
        cases,
        runner=lambda config: FakeResult([[config["value"]]]),
        component_names=["Ex"],
        reference_name="reference",
        tolerance=0.2,
    )

    assert report["reference"] == "reference"
    assert report["cases"][1]["name"] == "candidate"
    assert report["cases"][1]["passed"] is True


def test_boundary_benchmark_can_limit_comparison_to_time_window():
    cases = [
        BoundaryBenchmarkCase(name="candidate", config={"middle": 10.0}),
        BoundaryBenchmarkCase(name="reference", config={"middle": 1.0}),
    ]

    def runner(config):
        return FakeResult(
            [[1.0], [config["middle"]], [3.0]],
            times=[0.0, 1.0, 2.0],
        )

    report = run_boundary_benchmark(
        cases,
        runner=runner,
        component_names=["Hz"],
        reference_name="reference",
        tolerance=0.0,
        time_min=2.0,
        time_max=2.0,
    )

    assert report["time_window"] == {"min": 2.0, "max": 2.0}
    assert report["cases"][0]["n_times"] == 1
    assert report["cases"][0]["passed"] is True


def test_boundary_benchmark_can_pass_near_zero_components_by_absolute_tolerance():
    cases = [
        BoundaryBenchmarkCase(name="candidate", config={"near_zero": 1.0e-8, "regular": 1.05}),
        BoundaryBenchmarkCase(name="reference", config={"near_zero": 1.0e-15, "regular": 1.0}),
    ]

    def runner(config):
        return FakeResult([[config["near_zero"], config["regular"]]])

    report = run_boundary_benchmark(
        cases,
        runner=runner,
        component_names=["Ey_center", "Hz_side"],
        reference_name="reference",
        tolerance=0.1,
        absolute_tolerance=1.0e-7,
    )

    candidate = report["cases"][0]
    assert candidate["relative_linf_max"] > 1.0e6
    assert candidate["passed"] is True
    assert candidate["components"]["Ey_center"]["passed"] is True
    assert candidate["components"]["Ey_center"]["passed_by"] == "absolute"
    assert candidate["components"]["Hz_side"]["passed"] is True
    assert candidate["components"]["Hz_side"]["passed_by"] == "relative"
