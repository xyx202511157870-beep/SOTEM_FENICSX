from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from atem3d.comsol_seepage_channel_3d import (
    comsol_case_contract,
    read_comsol_wide_export,
    validate_distinct_model_paths,
)
from atem3d.seepage_channel_model import model_for_variant
from atem3d.seepage_verification import model_fingerprint
from tools.run_comsol_seepage_channel_3d import build_case_paths, build_commands


def test_comsol_channel_contract_matches_the_canonical_thin_3d_model() -> None:
    model = model_for_variant("thin_60x1x1")

    contract = comsol_case_contract("channel")

    assert contract["base_model_fingerprint"] == model_fingerprint(model)
    assert contract["coordinate_convention"] == "z_down"
    assert contract["source"]["endpoints_m"] == [[-50.0, 0.0, 0.1], [50.0, 0.0, 0.1]]
    assert contract["receivers_m"] == [list(point) for point in model.receiver_locations]
    assert contract["components"] == ["Ex", "dBzdt", "Hz"]
    assert len(contract["times_s"]) == 31
    assert contract["channel"]["size_m"] == [60.0, 1.0, 1.0]
    assert contract["channel"]["bounds_m"] == [
        [-30.0, 30.0],
        [-0.5, 0.5],
        [19.5, 20.5],
    ]
    assert contract["channel"]["conductivity_s_per_m"] == 1.0


def test_comsol_zero_contrast_keeps_geometry_but_uses_background_sigma() -> None:
    contract = comsol_case_contract("zero_contrast")

    assert contract["channel"]["enabled"] is True
    assert contract["channel"]["conductivity_s_per_m"] == 0.01
    assert contract["background_conductivity_s_per_m"] == 0.01


def test_comsol_adapter_never_overwrites_the_source_mph(tmp_path: Path) -> None:
    source = tmp_path / "source.mph"
    output = tmp_path / "output.mph"

    validate_distinct_model_paths(source, output)
    with pytest.raises(ValueError, match="must not overwrite"):
        validate_distinct_model_paths(source, source)


def test_comsol_wide_export_normalizes_five_receivers_and_31_times(
    tmp_path: Path,
) -> None:
    model = model_for_variant("thin_60x1x1")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    header = ["x", "y", "z"]
    for time in model.times:
        header.extend(
            [
                f"mef.Ex (V/m) @ t={time:.17g}",
                f"d(mef.Bz;t) (T/s) @ t={time:.17g}",
                f"mef.Bz/mu0_const (A/m) @ t={time:.17g}",
            ]
        )
    writer.writerow(header)
    for receiver, point in enumerate(model.receiver_locations):
        row = [*point]
        for time_index, _time in enumerate(model.times):
            row.extend([receiver + time_index, 100 + receiver + time_index, 200 + receiver + time_index])
        writer.writerow(row)
    path = tmp_path / "comsol.csv"
    path.write_text("% Model,test.mph\n% " + stream.getvalue(), encoding="utf-8")

    payload = read_comsol_wide_export(path)

    assert payload["values"].shape == (5, 31, 3)
    assert np.array_equal(payload["times"], np.asarray(model.times))
    assert payload["components"].tolist() == ["Ex", "dBzdt", "Hz"]
    assert payload["values"][4, 30].tolist() == [34.0, 134.0, 234.0]


def test_comsol_batch_spec_uses_distinct_case_output_and_read_only_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "uniform.mph"
    paths = build_case_paths(tmp_path / "results", "channel", source)
    compile_command, batch_command = build_commands(tmp_path / "bin", paths)

    assert paths["source_model"] == source
    assert paths["output_model"] != source
    assert paths["output_model"].name == "seepage_channel_3d_channel.mph"
    assert compile_command[0].endswith("comsolcompile.exe")
    assert "-inputfile" in batch_command
    assert paths["log"] == Path(batch_command[batch_command.index("-batchlog") + 1])
