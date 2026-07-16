from pathlib import Path


BACKGROUND = Path("tools/run_fenicsx_seepage_background.sh")
CHANNEL = Path("tools/run_fenicsx_seepage_channel.sh")
MAGNETIC_SHORT = Path("tools/run_fenicsx_magnetic_background_short.sh")
MAGNETIC_CHANNEL = Path("tools/run_fenicsx_magnetic_channel_full.sh")


def _command_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    command = text[text.index("exec ") :]
    return command.replace("\\\n", " ").split()


def test_background_and_channel_commands_share_full_domain_contract() -> None:
    background = BACKGROUND.read_text(encoding="utf-8")
    channel = CHANNEL.read_text(encoding="utf-8")
    for forbidden in ("mirror", "y10_baseline", "y20_baseline"):
        assert forbidden not in background.lower()
        assert forbidden not in channel.lower()

    receivers = (
        "0,-20,0.1",
        "0,-10,0.1",
        "0,0,0.1",
        "0,10,0.1",
        "0,20,0.1",
    )
    for receiver in receivers:
        assert f"--receiver-location {receiver}" in background
        assert f"--receiver-location {receiver}" in channel

    channel_only = (
        "--conductivity-box-name seepage_channel",
        '"--conductivity-box-bounds=-30,30;-5,5;-25,-15"',
        "--conductivity-box-sigma 1.0",
        "--conductivity-box-mesh-size 2.5",
    )
    for item in channel_only:
        assert item not in background
        assert item in channel

    background_tokens = _command_tokens(BACKGROUND)
    channel_tokens = _command_tokens(CHANNEL)
    for token in (
        "--x-extent",
        "--source-start-x",
        "--time-growth",
        "--outer-boundary-mode",
        "--ksp-type",
        "--checkpoint-forward",
        '"$@"',
    ):
        assert token in background_tokens
        assert token in channel_tokens


def test_staged_magnetic_commands_use_stable_full_domain_receivers() -> None:
    for path in (MAGNETIC_SHORT, MAGNETIC_CHANNEL):
        source = path.read_text(encoding="utf-8")
        for receiver in ("0,-20,0.1", "0,-10,0.1", "0,0,0.1", "0,10,0.1", "0,20,0.1"):
            assert f"--receiver-location {receiver}" in source
        for contract in (
            "--magnetic-dbdt-mode faraday_loop",
            "--biot-current-integration tetra4",
            "--faraday-loop-radius 2",
            "--faraday-loop-quadrature-points 32",
            "--magnetic-diagnostic-methods curl,biot_rate,faraday_loop,biot_center,biot_tetra4",
        ):
            assert contract in source
        assert "y10_baseline" not in source
        assert "copy_receiver" not in source
        assert "mirror" not in source.lower()
    assert "--stop-after-outputs 10" in MAGNETIC_SHORT.read_text(encoding="utf-8")
    assert "--stop-after-outputs" not in MAGNETIC_CHANNEL.read_text(encoding="utf-8")
