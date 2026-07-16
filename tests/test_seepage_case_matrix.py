from __future__ import annotations

import json
from pathlib import Path

import pytest
from atem3d.seepage_case_matrix import build_case_matrix, write_case_manifest
from atem3d.seepage_channel_model import model_for_variant
from atem3d.seepage_verification import (
    model_fingerprint,
    verify_simpeg_config_contract,
)
from tools.run_seepage_verification_matrix import (
    _output_is_current,
    _simpeg_h5_matches_config,
    build_case_command,
    reuse_case_output,
    write_simpeg_case_config,
)

import h5py
import yaml
import numpy as np


def test_case_matrix_covers_all_approved_control_values() -> None:
    cases = build_case_matrix()

    conductivity = {
        case.conductivity_s_per_m
        for case in cases
        if case.study == "conductivity" and case.role == "channel"
    }
    cross_sections = {
        case.cross_section_m
        for case in cases
        if case.study == "volume" and case.role == "channel"
    }
    spatial = {
        case.local_mesh_size_m
        for case in cases
        if case.study == "spatial" and case.role == "channel"
    }
    temporal = {
        case.time_step_factor
        for case in cases
        if case.study == "temporal" and case.role == "channel"
    }

    assert conductivity == {0.01, 0.02, 0.1, 1.0}
    assert cross_sections == {1.0, 2.0, 10.0}
    assert spatial == {0.5, 0.25, 0.125}
    assert temporal == {1.0, 0.5, 0.25}
    assert {case.solver for case in cases} == {"simpeg", "fenicsx"}


def test_every_case_inherits_canonical_contract_and_one_controlled_study() -> None:
    cases = build_case_matrix()
    expected = model_fingerprint(model_for_variant("thin_60x1x1"))

    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.model_fingerprint == expected for case in cases)
    assert all(case.role in {"background", "channel"} for case in cases)
    assert all(case.study in {"conductivity", "volume", "spatial", "temporal"} for case in cases)
    assert all(case.receiver_count == 5 and case.output_time_count == 31 for case in cases)

    conductivity_base = _case("simpeg-conductivity-channel-sigma-1")
    volume_base = _case("simpeg-volume-channel-cross-1")
    assert conductivity_base.case_fingerprint != volume_base.case_fingerprint
    assert conductivity_base.execution_fingerprint == volume_base.execution_fingerprint


def test_case_manifest_is_deterministic_and_records_expected_artifacts(
    tmp_path: Path,
) -> None:
    first = write_case_manifest(tmp_path, build_case_matrix())
    first_text = first.read_text(encoding="utf-8")
    second = write_case_manifest(tmp_path, build_case_matrix())

    assert second.read_text(encoding="utf-8") == first_text
    manifest = json.loads(first_text)
    assert manifest["model_fingerprint"] == model_fingerprint(
        model_for_variant("thin_60x1x1")
    )
    assert manifest["cases"]
    assert all(item["expected_output"] for item in manifest["cases"])


def _case(case_id: str):
    return next(case for case in build_case_matrix() if case.case_id == case_id)


def test_simpeg_generated_config_changes_only_the_declared_case_controls(
    tmp_path: Path,
) -> None:
    zero = _case("simpeg-conductivity-channel-sigma-0p01")
    path = write_simpeg_case_config(zero, tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    box = config["model"]["conductivity_boxes"][0]
    assert box["bounds"] == [[-30.0, 30.0], [-0.5, 0.5], [-20.5, -19.5]]
    assert box["sigma_infinity"] == 0.01
    assert min(segment[0] for segment in config["mesh"]["hy"]) == 0.25
    assert sum(
        step * count for step, count, *rest in config["time_steps"]
    ) == pytest.approx(0.01)

    refined = _case("simpeg-spatial-channel-h-0p125")
    refined_config = yaml.safe_load(
        write_simpeg_case_config(refined, tmp_path).read_text(encoding="utf-8")
    )
    assert [0.125, 8] in refined_config["mesh"]["hy"]
    assert [0.125, 8] in refined_config["mesh"]["hz"]

    coarse = _case("simpeg-spatial-channel-h-0p5")
    coarse_path = write_simpeg_case_config(coarse, tmp_path)
    verify_simpeg_config_contract(
        coarse_path,
        model=model_for_variant("thin_60x1x1"),
        case="channel",
        expected_local_mesh_size=0.5,
    )


def test_fenicsx_case_command_overrides_geometry_time_and_stable_magnetic_mode(
    tmp_path: Path,
) -> None:
    case = _case("fenicsx-temporal-channel-dt-0p25")
    command = " ".join(build_case_command(case, tmp_path))

    assert "run_fenicsx_seepage_thin_channel.sh" in command
    assert "--max-internal-dt 2.5e-05" in command
    assert "--max-internal-dt-fraction 0.0125" in command
    assert "--magnetic-dbdt-mode biot_rate" in command
    assert "--biot-current-integration tetra4" in command
    assert "--conductivity-box-sigma 1" in command


def test_duplicate_execution_reuses_normalized_arrays_with_new_case_provenance(
    tmp_path: Path,
) -> None:
    source = _case("simpeg-conductivity-channel-sigma-1")
    target = _case("simpeg-volume-channel-cross-1")
    source_path = tmp_path / source.expected_output
    source_path.parent.mkdir(parents=True)
    np.savez_compressed(
        source_path,
        values=np.ones((5, 31, 3)),
        case_fingerprint=np.asarray(source.case_fingerprint),
        execution_fingerprint=np.asarray(source.execution_fingerprint),
    )

    reused = reuse_case_output(target, tmp_path)

    assert reused == tmp_path / target.expected_output
    with np.load(reused, allow_pickle=False) as stored:
        assert str(stored["case_fingerprint"].item()) == target.case_fingerprint
        assert str(stored["execution_fingerprint"].item()) == target.execution_fingerprint
        assert str(stored["reused_from"].item()) == source.case_id


def test_matching_normalized_output_is_current(tmp_path: Path) -> None:
    case = _case("simpeg-conductivity-background-reference")
    path = tmp_path / case.expected_output
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        values=np.ones((5, 31, 3)),
        case_fingerprint=np.asarray(case.case_fingerprint),
        base_model_fingerprint=np.asarray(case.model_fingerprint),
        material_relative_volume_error=np.asarray(0.0),
    )

    assert _output_is_current(path, case)

    different = _case("simpeg-conductivity-channel-sigma-0p01")
    assert not _output_is_current(path, different)

    np.savez_compressed(
        path,
        values=np.ones((5, 31, 3)),
        case_fingerprint=np.asarray(case.case_fingerprint),
    )
    assert not _output_is_current(path, case)


def test_legacy_simpeg_h5_is_reused_only_for_semantically_identical_config(
    tmp_path: Path,
) -> None:
    case = _case("simpeg-conductivity-channel-sigma-1")
    config_path = write_simpeg_case_config(case, tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=False)

    assert _simpeg_h5_matches_config(h5_path, config_path)

    config["source"]["current"] = 2.0
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=False)
    assert not _simpeg_h5_matches_config(h5_path, config_path)
