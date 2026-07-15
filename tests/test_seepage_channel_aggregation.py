import numpy as np
from pathlib import Path

from atem3d.seepage_channel_validation import (
    aggregate_payloads,
    channel_delta,
    strong_signal_mask,
    summarize_convergence,
)


def test_channel_delta_preserves_sign() -> None:
    background = np.array([[[2.0, -3.0, 4.0]]])
    channel = np.array([[[1.0, -1.0, 8.0]]])
    np.testing.assert_allclose(
        channel_delta(channel, background),
        [[[-1.0, 2.0, 4.0]]],
    )


def test_strong_signal_mask_uses_component_peak_fraction() -> None:
    values = np.array([1.0, 0.04, 0.0, -0.2])
    np.testing.assert_array_equal(
        strong_signal_mask(values, 0.05),
        [True, False, False, True],
    )


def test_convergence_median_is_reported_without_dropping_raw_values() -> None:
    coarse = np.array([1.0, 2.0, 0.0])
    refined = np.array([1.02, 1.90, 0.01])
    summary = summarize_convergence(
        coarse,
        refined,
        strong_mask=np.array([True, True, False]),
    )
    assert summary["raw_count"] == 3
    assert summary["strong_count"] == 2
    assert summary["median_relative_change"] > 0.0


def test_aggregate_writes_canonical_artifacts_without_channel_empymod_error(
    tmp_path: Path,
) -> None:
    times = np.logspace(-5, -2, 31)
    locations = np.asarray([(0.0, y, -0.1) for y in (-20, -10, 0, 10, 20)])

    def payload(value: float, *, empymod: bool = False):
        result = {
            "times": times,
            "receiver_locations": locations,
            "components": np.asarray(("Ex", "dBzdt", "Hz")),
            "values": np.full((5, 31, 3), value),
        }
        if empymod:
            result["background_only_1d"] = True
        return result

    aggregate_payloads(
        tmp_path,
        empymod_background=payload(1.0, empymod=True),
        simpeg_background=payload(1.1),
        simpeg_channel=payload(1.3),
        fenicsx_background=payload(0.9),
        fenicsx_channel=payload(1.25),
    )
    required = {
        "benchmark_values.csv",
        "background_errors.csv",
        "channel_delta_values.csv",
        "channel_delta_errors.csv",
        "model_audit.json",
        "convergence_summary.json",
        "benchmark_summary.json",
        "benchmark_results.npz",
    }
    assert required.issubset(path.name for path in tmp_path.iterdir())
    background_error_text = (tmp_path / "background_errors.csv").read_text(
        encoding="utf-8"
    )
    assert "channel" not in background_error_text.lower()
