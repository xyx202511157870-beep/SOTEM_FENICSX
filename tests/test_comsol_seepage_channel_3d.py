from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

import tools.run_comsol_seepage_channel_3d as comsol_runner
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


def test_comsol_channel_mesh_size_precedes_the_free_tetrahedral_feature() -> None:
    source = (
        Path(__file__).parents[1]
        / "COMSOL"
        / "seepage_channel_3d"
        / "ConfigureAndRunSeepageChannel3D.java"
    ).read_text(encoding="utf-8")

    create = source.index('feature().create("size_channel_3d", "Size")')
    move = source.index('feature().move("size_channel_3d", 4)')
    configure_call = source.index("createChannelMeshControl(model);")
    mesh_run = source.index('mesh("mesh1").run()', configure_call)
    assert create < move
    assert configure_call < mesh_run


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


def test_comsol_runtime_routes_out_of_core_files_to_the_case_drive(
    tmp_path: Path,
) -> None:
    paths = build_case_paths(tmp_path / "results", "channel", tmp_path / "source.mph")
    ooc_root = tmp_path / "ascii_ooc"

    environment = comsol_runner.build_runtime_environment(
        paths,
        "channel",
        base_environment={
            "KEEP": "yes",
            "ATEM3D_COMSOL_OOC_ROOT": str(ooc_root),
        },
    )

    expected = str((ooc_root / "seepage_channel_3d_channel").resolve())
    assert environment["KEEP"] == "yes"
    assert environment["MKL_PARDISO_OOC_PATH"] == expected
    assert environment["TEMP"] == expected
    assert environment["TMP"] == expected
    expected.encode("iso-8859-1")


def test_isolated_comsol_prefs_enable_external_runtime_without_modifying_template(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template" / "comsol.prefs"
    template.parent.mkdir()
    original = (
        "keep.before=true\n"
        "security.external.runtimepermission=off\n"
        "keep.after=false\n"
    )
    template.write_text(original, encoding="utf-8")
    prefs_dir = tmp_path / "case" / "isolated_prefs"

    result = comsol_runner.prepare_isolated_prefs(template, prefs_dir)

    assert result == prefs_dir
    assert template.read_text(encoding="utf-8") == original
    assert (prefs_dir / "comsol.prefs").read_text(encoding="utf-8") == (
        "keep.before=true\n"
        "security.external.runtimepermission=on\n"
        "keep.after=false\n"
    )


def test_isolated_comsol_prefs_accept_an_already_enabled_runtime_setting(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template" / "comsol.prefs"
    template.parent.mkdir()
    original = (
        "keep.before=true\n"
        "security.external.runtimepermission=on\n"
        "keep.after=false\n"
    )
    template.write_text(original, encoding="utf-8")

    prefs_dir = comsol_runner.prepare_isolated_prefs(
        template, tmp_path / "case" / "isolated_prefs"
    )

    assert template.read_text(encoding="utf-8") == original
    assert (prefs_dir / "comsol.prefs").read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "content",
    [
        "keep=true\n",
        (
            "security.external.runtimepermission=off\n"
            "security.external.runtimepermission=on\n"
        ),
        "# security.external.runtimepermission=off\n",
    ],
)
def test_isolated_comsol_prefs_reject_missing_or_ambiguous_runtime_settings(
    tmp_path: Path, content: str
) -> None:
    template = tmp_path / "template" / "comsol.prefs"
    template.parent.mkdir()
    template.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        comsol_runner.prepare_isolated_prefs(
            template, tmp_path / "case" / "isolated_prefs"
        )

    assert template.read_text(encoding="utf-8") == content


def test_clear_generated_outputs_preserves_run_inputs_and_manifests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mph"
    source.write_text("source", encoding="utf-8")
    paths = build_case_paths(tmp_path / "results", "channel", source)
    paths["case_dir"].mkdir(parents=True)
    commands = paths["case_dir"] / "commands.json"
    paths["contract"].write_text("contract", encoding="utf-8")
    commands.write_text("commands", encoding="utf-8")
    for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
        paths[key].write_text(f"stale {key}", encoding="utf-8")

    comsol_runner.clear_generated_outputs(paths)

    assert source.read_text(encoding="utf-8") == "source"
    assert paths["contract"].read_text(encoding="utf-8") == "contract"
    assert commands.read_text(encoding="utf-8") == "commands"
    for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
        assert not paths[key].exists()


def test_run_invalidates_stale_outputs_before_prefs_validation_can_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mph"
    source.write_text("source", encoding="utf-8")
    invalid_template = tmp_path / "template" / "comsol.prefs"
    invalid_template.parent.mkdir()
    invalid_template.write_text("keep=true\n", encoding="utf-8")
    paths = build_case_paths(tmp_path / "results", "channel", source)
    paths["case_dir"].mkdir(parents=True)
    commands = paths["case_dir"] / "commands.json"
    paths["contract"].write_text("old contract", encoding="utf-8")
    commands.write_text("old commands", encoding="utf-8")
    for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
        paths[key].write_text(f"stale {key}", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid prefs must fail before subprocess")

    monkeypatch.setattr(comsol_runner.subprocess, "run", fail_if_called)

    with pytest.raises(ValueError, match="exactly one"):
        comsol_runner.run_case(
            case="channel",
            output_root=tmp_path / "results",
            source_model=source,
            comsol_bin=tmp_path / "bin",
            prefs_template=invalid_template,
            prepare_only=False,
        )

    assert source.read_text(encoding="utf-8") == "source"
    assert paths["contract"].is_file()
    assert commands.is_file()
    for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
        assert not paths[key].exists()


def test_run_clears_stale_outputs_and_prepares_prefs_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mph"
    source.write_text("source", encoding="utf-8")
    template = tmp_path / "template" / "comsol.prefs"
    template.parent.mkdir()
    template.write_text(
        "security.external.runtimepermission=off\n", encoding="utf-8"
    )
    output_root = tmp_path / "results"
    paths = build_case_paths(output_root, "channel", source)
    paths["case_dir"].mkdir(parents=True)
    for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
        paths[key].write_text(f"stale {key}", encoding="utf-8")

    def stop_at_first_subprocess(*_args, **_kwargs):
        for key in ("output_model", "output_csv", "normalized", "provenance", "log"):
            assert not paths[key].exists()
        assert source.is_file()
        assert paths["contract"].is_file()
        assert (paths["case_dir"] / "commands.json").is_file()
        assert (paths["prefs_dir"] / "comsol.prefs").read_text(
            encoding="utf-8"
        ) == "security.external.runtimepermission=on\n"
        raise RuntimeError("stop before external process")

    monkeypatch.setattr(comsol_runner.subprocess, "run", stop_at_first_subprocess)

    with pytest.raises(RuntimeError, match="stop before external process"):
        comsol_runner.run_case(
            case="channel",
            output_root=output_root,
            source_model=source,
            comsol_bin=tmp_path / "bin",
            prefs_template=template,
            prepare_only=False,
        )


def test_prepare_case_writes_isolated_prefs_without_launching_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template" / "comsol.prefs"
    template.parent.mkdir()
    template.write_text(
        "security.external.runtimepermission=off\n", encoding="utf-8"
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("prepare-only must not launch an external process")

    monkeypatch.setattr(comsol_runner.subprocess, "run", fail_if_called)
    result = comsol_runner.run_case(
        case="channel",
        output_root=tmp_path / "results",
        source_model=tmp_path / "source.mph",
        comsol_bin=tmp_path / "bin",
        prefs_template=template,
        prepare_only=True,
    )

    prefs = result.parent / "isolated_prefs" / "comsol.prefs"
    assert prefs.read_text(encoding="utf-8") == (
        "security.external.runtimepermission=on\n"
    )


def test_cli_accepts_a_custom_comsol_prefs_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_case(**kwargs):
        captured.update(kwargs)
        return tmp_path / "commands.json"

    monkeypatch.setattr(comsol_runner, "run_case", fake_run_case)
    template = tmp_path / "comsol.prefs"

    assert (
        comsol_runner.main(
            [
                "prepare",
                "--case",
                "channel",
                "--output-root",
                str(tmp_path / "results"),
                "--source-model",
                str(tmp_path / "source.mph"),
                "--prefs-template",
                str(template),
            ]
        )
        == 0
    )
    assert captured["prefs_template"] == template


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
    assert paths["prefs_dir"] == Path(
        batch_command[batch_command.index("-prefsdir") + 1]
    )
