from pathlib import Path

from tools.run_seepage_channel_benchmark import build_run_plan


def test_run_plan_contains_no_mirrored_fenicsx_jobs() -> None:
    plan = build_run_plan(output_root="output/seepage_channel_100m_5rx")
    assert [job.name for job in plan] == [
        "empymod_background",
        "simpeg_background",
        "simpeg_channel",
        "fenicsx_background",
        "fenicsx_channel",
        "aggregate",
        "plot",
        "manifest",
    ]
    assert all("mirror" not in " ".join(job.command).lower() for job in plan)


def test_thin_run_plan_selects_thin_configs_scripts_and_variant_argument() -> None:
    plan = build_run_plan(
        output_root="output/seepage_channel_100m_5rx_60x1x1",
        variant="thin_60x1x1",
    )
    commands = {job.name: " ".join(job.command) for job in plan}
    inputs = {job.name: " ".join(str(path) for path in job.required_inputs) for job in plan}
    assert "simpeg_thin_background.yaml" in inputs["simpeg_background"]
    assert "simpeg_thin_channel.yaml" in inputs["simpeg_channel"]
    assert "run_fenicsx_seepage_thin_background.sh" in commands["fenicsx_background"]
    assert "run_fenicsx_seepage_thin_channel.sh" in commands["fenicsx_channel"]
    assert "--variant thin_60x1x1" in commands["empymod_background"]
    assert "--variant thin_60x1x1" in commands["aggregate"]
    assert all("mirror" not in command.lower() for command in commands.values())


def test_fenicsx_shell_entrypoints_use_lf_line_endings() -> None:
    for name in (
        "run_fenicsx_seepage_thin_background.sh",
        "run_fenicsx_seepage_thin_channel.sh",
    ):
        payload = (Path("tools") / name).read_bytes()
        assert b"\r\n" not in payload, f"{name} must stay LF-only for WSL bash"
