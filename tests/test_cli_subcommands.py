from atem3d import cli


def test_cli_run_subcommand_dispatches_to_main_run(monkeypatch):
    seen = {}

    def fake_main_run(argv):
        seen["argv"] = list(argv)
        return 17

    monkeypatch.setattr(cli, "_main_run", fake_main_run)

    exit_code = cli.main(["run", "config.yaml", "--data-only"])

    assert exit_code == 17
    assert seen["argv"] == ["config.yaml", "--data-only"]
