import importlib.util
from pathlib import Path

import pytest

from atem3d.sotem_benchmark import load_benchmark_case


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "dolfinx/run_sotem_benchmark.py"


def _load_runner():
    if not RUNNER_PATH.exists():
        pytest.fail(f"benchmark runner is missing: {RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("run_sotem_benchmark_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _flag_values(argv):
    result = {}
    for item in argv:
        if item.startswith("--") and "=" in item:
            name, value = item.split("=", 1)
            result[name] = value
    return result


def _run_through_real_pipeline(monkeypatch, runner, argv):
    pipeline = runner.PIPELINE_MODULE
    captured = {}
    validate_model_consistency = pipeline.validate_model_consistency

    def capture_config(config, reference_mode=None):
        model = validate_model_consistency(config, reference_mode=reference_mode)
        captured["config"] = config
        captured["model"] = model
        return model

    monkeypatch.setattr(pipeline, "validate_model_consistency", capture_config)
    monkeypatch.setattr(pipeline, "check_environment", lambda **_kwargs: {})

    assert runner.PIPELINE_MAIN is pipeline.main
    assert runner.main(argv) == 0
    return captured


def test_main_propagates_song_ip_case_to_real_pipeline_contract(monkeypatch, tmp_path):
    runner = _load_runner()
    captured = {}

    def pipeline_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(runner, "PIPELINE_MAIN", pipeline_main)

    result = runner.main(
        [
            "--case",
            "benchmarks/sotem/song2025_layered_pair.yaml",
            "--variant",
            "ip",
            "--level",
            "S0T0B0",
            "--workdir",
            str(tmp_path),
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    argv = captured["argv"]
    flags = _flag_values(argv)
    assert "--source-current=10.0" in argv
    assert "--ramp-off-time=0.0" in argv
    assert "--rho-air=1000000.0" in argv
    assert "--cole-layer-bottom=300.0" in argv
    assert "--cole-m=0.3" in argv
    assert len(flags["--observation-times"].split(",")) == 51
    assert argv[-2:] == ["--check-env-only", "--no-install"]


def test_main_rejects_invalid_level_without_traceback(capsys, tmp_path):
    runner = _load_runner()

    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                "--case",
                "benchmarks/sotem/song2025_layered_pair.yaml",
                "--variant",
                "ip",
                "--level",
                "S3T0B0",
                "--workdir",
                str(tmp_path),
                "--check-env-only",
                "--no-install",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr
    assert "Traceback" not in stderr


def test_main_passes_source_only_and_observation_override_to_real_pipeline(monkeypatch, tmp_path):
    runner = _load_runner()
    times = "1e-5,1e-4,1e-3,1e-2,1e-1"
    captured = _run_through_real_pipeline(
        monkeypatch,
        runner,
        [
            "--case",
            "benchmarks/sotem/lei2023_noip.yaml",
            "--variant",
            "noip",
            "--level",
            "S0T0B0",
            "--workdir",
            str(tmp_path),
            "--observation-times",
            times,
            "--source-only",
            "--check-env-only",
            "--no-install",
        ],
    )

    assert captured["config"].source_only is True
    assert captured["config"].observation_times == pytest.approx(
        (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    )


@pytest.mark.parametrize(
    "times",
    ["", "0.0,1e-3", "1e-3,nan", "1e-3,inf", "1e-3,1e-3", "1e-3,1e-4"],
)
def test_main_rejects_invalid_observation_override(capsys, tmp_path, times):
    runner = _load_runner()
    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                "--case",
                "benchmarks/sotem/lei2023_noip.yaml",
                "--variant",
                "noip",
                "--level",
                "S0T0B0",
                "--workdir",
                str(tmp_path),
                "--observation-times",
                times,
            ]
        )
    assert exc_info.value.code == 2
    assert "observation" in capsys.readouterr().err.lower()


def test_song_ip_runner_reaches_real_pipeline_config_and_validation(monkeypatch, tmp_path):
    runner = _load_runner()

    captured = _run_through_real_pipeline(
        monkeypatch,
        runner,
        [
            "--case",
            "benchmarks/sotem/song2025_layered_pair.yaml",
            "--variant",
            "ip",
            "--level",
            "S2T0B1",
            "--workdir",
            str(tmp_path),
            "--check-env-only",
            "--no-install",
        ],
    )

    config = captured["config"]
    assert config.source_start == pytest.approx((-500.0, 0.0, -0.1))
    assert config.source_end == pytest.approx((500.0, 0.0, -0.1))
    assert config.receiver == pytest.approx((0.0, -500.0, -0.1))
    assert config.source_current == pytest.approx(10.0)
    assert config.ramp_off_time == pytest.approx(0.0)
    assert config.rho_air == pytest.approx(1.0e6)
    assert config.layer_depths == pytest.approx((300.0,))
    assert config.layer_resistivities == pytest.approx((100.0, 100.0))
    assert config.expected_source_length == pytest.approx(1000.0)
    assert config.expected_parallel_offset == pytest.approx(500.0)
    assert config.output_interval_substeps == 1
    assert config.polarization == "cole-cole"
    assert captured["model"]["reference_mode"] == "cole-cole-exact"


def test_lei_noip_runner_reaches_real_pipeline_config_and_validation(monkeypatch, tmp_path):
    runner = _load_runner()

    captured = _run_through_real_pipeline(
        monkeypatch,
        runner,
        [
            "--case",
            "benchmarks/sotem/lei2023_noip.yaml",
            "--variant",
            "noip",
            "--level",
            "S0T2B0",
            "--workdir",
            str(tmp_path),
            "--check-env-only",
            "--no-install",
        ],
    )

    config = captured["config"]
    assert config.receiver == pytest.approx((0.0, 800.0, -0.1))
    assert config.source_current == pytest.approx(1.0)
    assert config.rho_air == pytest.approx(1.0e8)
    assert config.rho_earth == pytest.approx(100.0)
    assert config.expected_source_length == pytest.approx(1000.0)
    assert config.expected_parallel_offset == pytest.approx(800.0)
    assert config.output_interval_substeps == 4
    assert config.polarization == "none"
    assert captured["model"]["reference_mode"] == "noip"


@pytest.mark.parametrize(
    ("level", "source_size", "receiver_size", "substeps", "boundary_extent"),
    [
        ("S0T0B0", "40.0", "20.0", "1", "25000.0"),
        ("S1T1B1", "20.0", "10.0", "2", "50000.0"),
        ("S2T2B2", "10.0", "5.0", "4", "100000.0"),
        ("S2T0B1", "10.0", "5.0", "1", "50000.0"),
        ("S0T2B0", "40.0", "20.0", "4", "25000.0"),
        ("S1T1B2", "20.0", "10.0", "2", "100000.0"),
    ],
)
def test_build_pipeline_argv_maps_approved_stb_levels(
    tmp_path, level, source_size, receiver_size, substeps, boundary_extent
):
    runner = _load_runner()
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")

    flags = _flag_values(runner.build_pipeline_argv(case, "noip", level, tmp_path))

    assert flags["--source-mesh-size"] == source_size
    assert flags["--receiver-mesh-size"] == receiver_size
    assert flags["--output-interval-substeps"] == substeps
    assert flags["--x-extent"] == boundary_extent
    assert flags["--y-extent"] == boundary_extent
    assert flags["--air-height"] == boundary_extent
    assert flags["--earth-depth"] == boundary_extent


def test_song_noip_translation_is_explicit_and_has_no_cole_cole_fit(tmp_path):
    runner = _load_runner()
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")

    argv = runner.build_pipeline_argv(case, "noip", "S0T0B0", tmp_path)
    flags = _flag_values(argv)

    assert flags["--polarization"] == "none"
    assert flags["--layer-depths"] == "300.0"
    assert flags["--layer-resistivities"] == "100.0,100.0"
    assert not any(item.startswith("--cole-") for item in argv)


def test_song_ip_translation_sets_approved_cole_cole_contract(tmp_path):
    runner = _load_runner()
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")

    flags = _flag_values(runner.build_pipeline_argv(case, "ip", "S0T0B0", tmp_path))

    expected = {
        "--polarization": "cole-cole",
        "--cole-rho0": "100.0",
        "--cole-m": "0.3",
        "--cole-tau": "1.0",
        "--cole-c": "0.3",
        "--cole-n-terms": "16",
        "--cole-f-min": "0.001",
        "--cole-f-max": "10000.0",
        "--cole-n-freq": "81",
        "--cole-fit-tolerance": "0.01",
        "--cole-layer-top": "0.0",
        "--cole-layer-bottom": "300.0",
    }
    assert expected.items() <= flags.items()


def test_translation_emits_all_approved_geometry_and_fixed_solver_flags(tmp_path):
    runner = _load_runner()
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")

    flags = _flag_values(runner.build_pipeline_argv(case, "ip", "S0T0B0", tmp_path))

    expected = {
        "--workdir": str(tmp_path),
        "--source-start-x": "-500.0",
        "--source-start-y": "0.0",
        "--source-start-z": "-0.1",
        "--source-end-x": "500.0",
        "--source-end-y": "0.0",
        "--source-end-z": "-0.1",
        "--source-current": "10.0",
        "--ramp-off-time": "0.0",
        "--receiver-x": "0.0",
        "--receiver-y": "-500.0",
        "--receiver-z": "-0.1",
        "--rho-air": "1000000.0",
        "--time-method": "theta",
        "--time-theta": "1.0",
        "--initial-dc-mode": "fem",
        "--magnetic-receiver-mode": "faraday_integrated",
        "--magnetic-dbdt-mode": "curl",
        "--error-min-time": "0.0",
        "--reference-audit-srcpts": "9",
    }
    assert expected.items() <= flags.items()


def test_halfspace_case_translates_earth_resistivity_without_layer_flags(tmp_path):
    runner = _load_runner()
    case = load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")

    argv = runner.build_pipeline_argv(case, "noip", "S0T0B0", tmp_path)
    flags = _flag_values(argv)

    assert flags["--rho-earth"] == "100.0"
    assert "--layer-depths" not in flags
    assert "--layer-resistivities" not in flags
