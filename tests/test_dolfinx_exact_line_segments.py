from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_exact_line_segments_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TETRA = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.0, 1.0)),
        ((2.0, 0.1, 0.1), (3.0, 0.1, 0.1), None),
        ((-0.5, 0.1, 0.1), (0.5, 0.1, 0.1), (0.5, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0)),
    ],
    ids=["inside", "outside", "cross", "grazing-vertex", "along-edge"],
)
def test_line_tetra_interval_uses_exact_barycentric_boundaries(start, end, expected):
    sp = _load_pipeline_module()

    interval = sp._line_tetra_positive_interval(TETRA, start, end)

    if expected is None:
        assert interval is None
    else:
        assert interval == pytest.approx(expected, abs=1.0e-15)


def test_shared_face_intervals_are_both_positive_before_deterministic_assignment():
    sp = _load_pipeline_module()
    reflected = TETRA.copy()
    reflected[3] = [0.0, 0.0, -1.0]
    start = (0.1, 0.1, 0.0)
    end = (0.7, 0.1, 0.0)

    upper = sp._line_tetra_positive_interval(TETRA, start, end)
    lower = sp._line_tetra_positive_interval(reflected, start, end)

    assert upper == pytest.approx((0.0, 1.0))
    assert lower == pytest.approx((0.0, 1.0))


def test_atomic_intervals_cluster_endpoints_and_choose_lowest_global_cell_id():
    sp = _load_pipeline_module()
    intervals = [
        {"s_start": 0.0, "s_end": 0.5 + 2.0e-14, "cell": 4, "global_cell": 40},
        {"s_start": 0.5, "s_end": 1.0, "cell": 8, "global_cell": 80},
        {"s_start": 0.0, "s_end": 1.0, "cell": 2, "global_cell": 20},
    ]

    result = sp._atomic_line_cell_intervals(intervals, parameter_tolerance=1.0e-12)

    np.testing.assert_allclose(
        [(item["s_start"], item["s_end"]) for item in result["intervals"]],
        [(0.0, 0.5), (0.5, 1.0)],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert [item["global_cell"] for item in result["intervals"]] == [20, 20]
    assert result["diagnostics"]["candidate_positive_interval_count"] == 3
    assert result["diagnostics"]["atomic_interval_count"] == 2
    assert result["diagnostics"]["multi_candidate_atomic_interval_count"] == 2
    assert result["diagnostics"]["candidate_overlap_parameter_length"] > 0.0
    assert result["diagnostics"]["union_coverage_fraction"] == pytest.approx(1.0)
    assert result["diagnostics"]["gap_parameter_length"] == pytest.approx(0.0)
    assert result["diagnostics"]["assigned_overlap_parameter_length"] == pytest.approx(0.0)


def test_atomic_intervals_fail_closed_on_gap_and_missing_endpoints():
    sp = _load_pipeline_module()
    intervals = [
        {"s_start": 0.1, "s_end": 0.4, "cell": 1, "global_cell": 1},
        {"s_start": 0.6, "s_end": 0.9, "cell": 2, "global_cell": 2},
    ]

    result = sp._atomic_line_cell_intervals(intervals, parameter_tolerance=1.0e-12)

    assert result["diagnostics"]["union_coverage_fraction"] == pytest.approx(0.6)
    assert result["diagnostics"]["gap_parameter_length"] == pytest.approx(0.4)
    assert result["diagnostics"]["start_endpoint_covered"] is False
    assert result["diagnostics"]["end_endpoint_covered"] is False
    assert result["diagnostics"]["passed"] is False


class _GatherComm:
    def __init__(self, *, rank, payloads):
        self.rank = int(rank)
        self.size = len(payloads)
        self._payloads = payloads

    def allgather(self, _local_payload):
        return self._payloads


def test_collective_atomic_intervals_assign_each_global_atom_to_one_rank():
    sp = _load_pipeline_module()
    payloads = [
        [
            {
                "s_start": 0.0,
                "s_end": 0.5,
                "cell": 4,
                "global_cell": 40,
                "owner_rank": 0,
            }
        ],
        [
            {
                "s_start": 0.5,
                "s_end": 1.0,
                "cell": 8,
                "global_cell": 80,
                "owner_rank": 1,
            }
        ],
    ]

    rank0 = sp._collective_atomic_line_intervals(
        _GatherComm(rank=0, payloads=payloads), payloads[0]
    )
    rank1 = sp._collective_atomic_line_intervals(
        _GatherComm(rank=1, payloads=payloads), payloads[1]
    )

    assert [(item["s_start"], item["s_end"]) for item in rank0["intervals"]] == [
        (0.0, 0.5)
    ]
    assert [(item["s_start"], item["s_end"]) for item in rank1["intervals"]] == [
        (0.5, 1.0)
    ]
    assert rank0["diagnostics"] == rank1["diagnostics"]
    assert rank0["diagnostics"]["distributed_ownership_complete"] is True
    assert rank0["diagnostics"]["global_atomic_interval_count"] == 2
    assert rank0["diagnostics"]["union_coverage_fraction"] == pytest.approx(1.0)
    assert rank0["diagnostics"]["passed"] is True


def test_collective_atomic_intervals_choose_global_cell_before_owner_rank():
    sp = _load_pipeline_module()
    payloads = [
        [
            {
                "s_start": 0.0,
                "s_end": 1.0,
                "cell": 9,
                "global_cell": 90,
                "owner_rank": 0,
            }
        ],
        [
            {
                "s_start": 0.0,
                "s_end": 1.0,
                "cell": 2,
                "global_cell": 20,
                "owner_rank": 1,
            }
        ],
    ]

    rank0 = sp._collective_atomic_line_intervals(
        _GatherComm(rank=0, payloads=payloads), payloads[0]
    )
    rank1 = sp._collective_atomic_line_intervals(
        _GatherComm(rank=1, payloads=payloads), payloads[1]
    )

    assert rank0["intervals"] == []
    assert len(rank1["intervals"]) == 1
    assert rank1["intervals"][0]["global_cell"] == 20
    assert rank1["intervals"][0]["owner_rank"] == 1


def test_collective_point_cell_owner_is_unique_across_partition_boundary():
    sp = _load_pipeline_module()
    payloads = [
        [{"global_cell": 40, "cell": 4, "owner_rank": 0}],
        [{"global_cell": 20, "cell": 8, "owner_rank": 1}],
    ]

    owner0 = sp._collective_point_cell_owner(
        _GatherComm(rank=0, payloads=payloads), payloads[0]
    )
    owner1 = sp._collective_point_cell_owner(
        _GatherComm(rank=1, payloads=payloads), payloads[1]
    )

    assert owner0 == owner1 == {"global_cell": 20, "cell": 8, "owner_rank": 1}
