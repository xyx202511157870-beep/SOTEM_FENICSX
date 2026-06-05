import json

import h5py
import numpy as np

from atem3d.validation import ValidationCase, write_validation_report


def test_write_validation_report_stores_reference_and_error_summary(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0]))
        h5.create_dataset("data", data=np.array([[1.0, 2.0], [2.0, 4.0]]))

    case = ValidationCase(
        result_path=result_path,
        reference=np.array([[1.0, 1.0], [1.0, 2.0]]),
        component_names=["Ex", "Hz"],
    )
    report_path = tmp_path / "report.json"

    report = write_validation_report(case, report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["components"]["Ex"]["relative_linf"] == 1.0
    assert payload["components"]["Hz"]["relative_linf"] == 1.0
    assert payload["relative_linf_max"] == 1.0
    assert payload["diagnostics"]["Hz"]["least_squares_scale_numerical_over_reference"] == 2.0
    assert payload["n_times"] == 2
    assert report["components"] == payload["components"]


def test_write_validation_report_stores_metadata(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0]))
        h5.create_dataset("data", data=np.array([[0.0], [1.0]]))

    case = ValidationCase(
        result_path=result_path,
        reference=np.array([[1.0], [2.0]]),
        component_names=["Ex"],
        metadata={"empymod": {"srcpts": 5, "signal": -1}},
    )

    report = write_validation_report(case, tmp_path / "report.json")

    assert report["metadata"]["empymod"]["srcpts"] == 5


def test_write_validation_report_can_compare_only_positive_times(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0, 2.0]))
        h5.create_dataset("data", data=np.array([[99.0], [1.0], [2.0]]))

    case = ValidationCase(
        result_path=result_path,
        reference=np.array([[1.0], [2.0]]),
        component_names=["Ex"],
        positive_times_only=True,
    )
    report_path = tmp_path / "report.json"

    report = write_validation_report(case, report_path)

    assert report["n_times"] == 2
    assert report["components"]["Ex"]["relative_linf"] == 0.0


def test_write_validation_report_can_compare_component_subset(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0]))
        h5.create_dataset("data", data=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]))

    case = ValidationCase(
        result_path=result_path,
        reference=np.array([[3.0], [3.5]]),
        component_names=["Hz"],
        component_indices=[2],
    )

    report = write_validation_report(case, tmp_path / "report.json")

    assert report["n_components"] == 1
    assert set(report["components"]) == {"Hz"}
    assert report["components"]["Hz"]["relative_linf"] == 0.0


def test_write_validation_report_can_store_per_time_samples(tmp_path):
    result_path = tmp_path / "result.h5"
    with h5py.File(result_path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0, 2.0]))
        h5.create_dataset("data", data=np.array([[99.0], [2.0], [6.0]]))

    case = ValidationCase(
        result_path=result_path,
        reference=np.array([[1.0], [3.0]]),
        component_names=["Ex"],
        positive_times_only=True,
        include_samples=True,
    )

    report = write_validation_report(case, tmp_path / "report.json")

    assert report["samples"]["Ex"] == [
        {
            "time": 1.0,
            "numerical": 2.0,
            "reference": 1.0,
            "difference": 1.0,
            "ratio_numerical_over_reference": 2.0,
        },
        {
            "time": 2.0,
            "numerical": 6.0,
            "reference": 3.0,
            "difference": 3.0,
            "ratio_numerical_over_reference": 2.0,
        },
    ]
