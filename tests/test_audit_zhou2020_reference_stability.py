import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_zhou2020_reference_stability.py"
CASE = ROOT / "benchmarks/sotem/zhou2020_grounded_wire.yaml"
FORMAL_RUN = (
    ROOT
    / "generated/validation/zhou2020_grounded_wire/runs"
    / "20260723T062004Z_zhou_strict_v2"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "zhou_reference_audit_under_test",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _publisher_inputs(tmp_path):
    run = tmp_path / "run"
    comparison = run / "comparisons/S1T1B1"
    comparison.mkdir(parents=True)
    strict = comparison / "strict_comparison.json"
    strict.write_text(
        '{"status":"failed_with_reproducible_evidence"}\n',
        encoding="utf-8",
    )
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.geomspace(1.0e-12, 1.0e-9, 8)
    return strict, {
        "run": run,
        "output": tmp_path / "audit",
        "times": times,
        "default_dlf": direct * 1.02,
        "separate_total_qwe": direct * 1.01,
        "direct_frequency_qwe": direct,
        "direct_qwe_converged": False,
        "fenicsx_increment": direct * 1.05,
        "consecutive": 2,
        "method_metadata": {"srcpts": 17, "qwe": {"ftarg": {"rtol": 1.0e-8}}},
    }


def _assert_no_transaction_debris(output):
    assert not output.with_name(f".{output.name}.lock").exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_script_direct_execution_imports_worktree_package():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--srcpts" in completed.stdout


def test_publish_audit_writes_json_and_npz_without_modifying_strict_json(tmp_path):
    module = _load_script()
    strict, arguments = _publisher_inputs(tmp_path)
    before = strict.read_bytes()

    module.publish_audit(**arguments)

    assert strict.read_bytes() == before
    output = arguments["output"]
    payload = json.loads(
        (output / "reference_stability.json").read_text("utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    with np.load(output / "reference_stability.npz") as arrays:
        assert arrays["time_s"].shape == (8,)
        assert set(arrays.files) == {
            "time_s",
            "default_dlf",
            "separate_total_qwe",
            "direct_frequency_qwe",
            "fenicsx_increment",
        }
    assert payload["status"] == "inconclusive"
    assert payload["input_sha256"]["strict_comparison.json"] == hashlib.sha256(
        before
    ).hexdigest()
    assert manifest["schema"] == "atem3d.zhou2020.reference-stability-manifest/v1"
    assert manifest["status"] == "inconclusive"
    assert manifest["input_sha256"] == payload["input_sha256"]
    assert manifest["code_sha256"] == payload["code_sha256"]
    assert manifest["methods"] == arguments["method_metadata"]
    for relative_path, expected_hash in manifest["code_sha256"].items():
        assert expected_hash == module.sha256_file(ROOT / relative_path)
    for name in ("reference_stability.json", "reference_stability.npz"):
        assert manifest["artifacts"][name]["sha256"] == module.sha256_file(
            output / name
        )
    validated = module.load_validated_audit(output)
    assert validated["audit"] == payload
    assert validated["manifest"] == manifest
    _assert_no_transaction_debris(output)


def test_publish_audit_refuses_existing_output_without_modifying_it(tmp_path):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("retain", encoding="utf-8")

    with pytest.raises(FileExistsError, match="exists"):
        module.publish_audit(**arguments)

    assert sentinel.read_text(encoding="utf-8") == "retain"
    _assert_no_transaction_debris(output)


def test_publish_audit_refuses_lock_collision(tmp_path):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    lock = output.with_name(f".{output.name}.lock")
    lock.write_text('{"pid": 123}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="lock"):
        module.publish_audit(**arguments)

    assert not output.exists()
    assert lock.read_text(encoding="utf-8") == '{"pid": 123}'
    assert not list(output.parent.glob(f".{output.name}.*.staging"))
    lock.unlink()


def test_publish_audit_json_failure_removes_staging_and_owned_lock(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    original = module._atomic_write_json

    def fail_audit_json(path, payload):
        if Path(path).name == "reference_stability.json":
            raise RuntimeError("injected JSON failure")
        return original(path, payload)

    monkeypatch.setattr(module, "_atomic_write_json", fail_audit_json)

    with pytest.raises(RuntimeError, match="injected JSON failure"):
        module.publish_audit(**arguments)

    assert not output.exists()
    _assert_no_transaction_debris(output)


def test_publish_audit_npz_failure_removes_staging_and_owned_lock(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]

    def fail_npz(path, **arrays):
        raise RuntimeError("injected NPZ failure")

    monkeypatch.setattr(module, "_atomic_write_npz", fail_npz)

    with pytest.raises(RuntimeError, match="injected NPZ failure"):
        module.publish_audit(**arguments)

    assert not output.exists()
    _assert_no_transaction_debris(output)


def test_publish_audit_staged_postcondition_failure_is_not_publishable(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]

    def fail_validation(path, **kwargs):
        raise RuntimeError("injected staged postcondition failure")

    monkeypatch.setattr(module, "load_validated_audit", fail_validation)

    with pytest.raises(RuntimeError, match="postcondition"):
        module.publish_audit(**arguments)

    assert not output.exists()
    _assert_no_transaction_debris(output)


def test_publish_audit_post_input_hash_failure_is_not_publishable(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    original = module._verify_input_hashes
    calls = 0

    def fail_second_postcondition(paths, expected):
        nonlocal calls
        calls += 1
        original(paths, expected)
        if calls == 2:
            raise RuntimeError("injected post-input-hash failure")

    monkeypatch.setattr(module, "_verify_input_hashes", fail_second_postcondition)

    with pytest.raises(RuntimeError, match="post-input-hash"):
        module.publish_audit(**arguments)

    assert calls == 2
    assert not output.exists()
    _assert_no_transaction_debris(output)


def test_publish_audit_code_change_is_not_publishable(tmp_path, monkeypatch):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    code = tmp_path / "code/audit_code.py"
    code.parent.mkdir()
    code.write_text("VALUE = 1\n", encoding="utf-8")
    arguments["code_paths"] = {"audit_code.py": code}
    arguments["code_sha256"] = {
        "audit_code.py": module.sha256_file(code),
    }
    original = module._verify_code_hashes
    calls = 0

    def mutate_before_second_check(paths, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            code.write_text("VALUE = 2\n", encoding="utf-8")
        return original(paths, expected)

    monkeypatch.setattr(
        module,
        "_verify_code_hashes",
        mutate_before_second_check,
    )

    with pytest.raises(RuntimeError, match="audit code changed"):
        module.publish_audit(**arguments)

    assert calls == 2
    assert not output.exists()
    _assert_no_transaction_debris(output)


def test_loader_rejects_local_code_hash_tampering(tmp_path):
    module = _load_script()
    _, arguments = _publisher_inputs(tmp_path)
    output = arguments["output"]
    code = tmp_path / "code/audit_code.py"
    code.parent.mkdir()
    code.write_text("VALUE = 1\n", encoding="utf-8")
    code_paths = {"audit_code.py": code}
    arguments["code_paths"] = code_paths
    arguments["code_sha256"] = {
        "audit_code.py": module.sha256_file(code),
    }
    module.publish_audit(**arguments)
    module.load_validated_audit(output, code_paths=code_paths)

    code.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="code hash mismatch"):
        module.load_validated_audit(output, code_paths=code_paths)


def test_atomic_json_rejects_nan_and_cleans_up_temporary_file(tmp_path):
    module = _load_script()
    destination = tmp_path / "audit.json"

    with pytest.raises(ValueError):
        module._atomic_write_json(destination, {"invalid": np.nan})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_npz_rejects_nonfinite_arrays_without_artifacts(tmp_path):
    module = _load_script()
    destination = tmp_path / "audit.npz"

    with pytest.raises(ValueError, match="finite"):
        module._atomic_write_npz(
            destination,
            values=np.array([1.0, np.inf]),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_qwe_paths_transform_totals_and_frequency_difference_explicitly(
    monkeypatch,
):
    module = _load_script()
    times = np.geomspace(1.0e-4, 1.0e-3, 4)
    frequencies = np.array([1.0, 10.0, 100.0, 1000.0])
    noip = SimpleNamespace(
        variant="noip",
        times=times,
        signal=-1,
        coordinate_system="depth_down",
    )
    ip = SimpleNamespace(
        variant="ip",
        times=times,
        signal=-1,
        coordinate_system="depth_down",
    )
    noip_frequency = np.array([10.0, 20.0, 30.0, 40.0], dtype=complex)
    ip_frequency = np.array([11.0, 22.0, 33.0, 44.0], dtype=complex)
    grid = {
        "times": times,
        "frequencies": frequencies,
        "ft": "qwe",
        "ftarg": dict(module.QWE),
        "signal": 0,
        "scale": -1.0,
    }
    monkeypatch.setattr(module, "_checked_qwe_grid", lambda survey: grid)

    frequency_calls = []

    def fake_frequency_response(survey, requested_frequencies, *, srcpts):
        frequency_calls.append(
            (survey.variant, requested_frequencies.copy(), srcpts)
        )
        return noip_frequency if survey.variant == "noip" else ip_frequency

    monkeypatch.setattr(module, "_frequency_response", fake_frequency_response)
    transform_calls = []
    convergence = iter((False, True, True))

    def fake_tem_transform(values, requested_grid):
        transform_calls.append(np.asarray(values).copy())
        assert requested_grid is grid
        return np.asarray(values.real), next(convergence)

    monkeypatch.setattr(module, "_tem_qwe_transform", fake_tem_transform)

    result = module._compute_qwe_audit_signals(noip, ip, srcpts=17)

    assert [item[0] for item in frequency_calls] == ["noip", "ip"]
    assert all(item[2] == 17 for item in frequency_calls)
    np.testing.assert_array_equal(transform_calls[0], noip_frequency)
    np.testing.assert_array_equal(transform_calls[1], ip_frequency)
    np.testing.assert_array_equal(
        transform_calls[2],
        ip_frequency - noip_frequency,
    )
    np.testing.assert_array_equal(
        result["separate_total_qwe"],
        ip_frequency.real - noip_frequency.real,
    )
    np.testing.assert_array_equal(
        result["direct_frequency_qwe"],
        (ip_frequency - noip_frequency).real,
    )
    assert result["qwe_convergence"] == {
        "separate_noip_qwe_converged": False,
        "separate_ip_qwe_converged": True,
        "direct_difference_qwe_converged": True,
        "converged": False,
    }


def test_run_audit_loads_formal_inputs_and_aggregates_all_qwe_convergence(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    noip_table = np.genfromtxt(
        FORMAL_RUN / "reference/empymod_noip.csv",
        delimiter=",",
        names=True,
    )
    ip_table = np.genfromtxt(
        FORMAL_RUN / "reference/empymod_ip.csv",
        delimiter=",",
        names=True,
    )
    times = np.asarray(noip_table["time_s"], dtype=float)
    default_increment = (
        np.asarray(ip_table["dBzdt_T_per_s"], dtype=float)
        - np.asarray(noip_table["dBzdt_T_per_s"], dtype=float)
    )
    frequencies = np.arange(1, times.size + 1, dtype=float)
    noip_frequency = np.zeros(times.size, dtype=complex)
    ip_frequency = default_increment.astype(complex)
    survey_variants = []

    def fake_grid(survey):
        np.testing.assert_array_equal(survey.times, times)
        return {
            "times": times,
            "frequencies": frequencies,
            "ft": "qwe",
            "ftarg": dict(module.QWE),
            "signal": 0,
            "scale": -4.0 * np.pi * 1.0e-7,
        }

    def fake_frequency_response(survey, requested_frequencies, *, srcpts):
        np.testing.assert_array_equal(requested_frequencies, frequencies)
        assert srcpts == 17
        is_ip = isinstance(survey.resistivities, dict)
        survey_variants.append("ip" if is_ip else "noip")
        return ip_frequency if is_ip else noip_frequency

    transform_calls = []
    convergence = iter((False, True, True))

    def fake_tem_transform(values, grid):
        transform_calls.append(np.asarray(values).copy())
        return np.asarray(values.real), next(convergence)

    monkeypatch.setattr(module, "_checked_qwe_grid", fake_grid)
    monkeypatch.setattr(module, "_frequency_response", fake_frequency_response)
    monkeypatch.setattr(module, "_tem_qwe_transform", fake_tem_transform)
    output = tmp_path / "reference_audit"

    audit = module.run_audit(
        run=FORMAL_RUN,
        case=CASE,
        output=output,
        srcpts=17,
    )

    assert survey_variants == ["noip", "ip"]
    np.testing.assert_array_equal(transform_calls[0], noip_frequency)
    np.testing.assert_array_equal(transform_calls[1], ip_frequency)
    np.testing.assert_array_equal(
        transform_calls[2],
        ip_frequency - noip_frequency,
    )
    assert audit["status"] == "inconclusive"
    assert audit["qwe"] == {
        "separate_noip_qwe_converged": False,
        "separate_ip_qwe_converged": True,
        "direct_difference_qwe_converged": True,
        "converged": False,
    }
    validated = module.load_validated_audit(output)
    manifest = validated["manifest"]
    assert set(manifest["input_sha256"]) == {
        "case.yaml",
        "run_manifest.json",
        "reference_manifest.json",
        "empymod_metadata.json",
        "strict_comparison.json",
        "empymod_noip.csv",
        "empymod_ip.csv",
        "fenicsx_noip_predictions.csv",
        "fenicsx_ip_predictions.csv",
    }
    assert manifest["input_sha256"]["case.yaml"] == module.sha256_file(CASE)
    assert manifest["code_sha256"] == audit["code_sha256"]
    expected_code_paths = module._audit_code_paths()
    assert set(manifest["code_sha256"]) == set(expected_code_paths)
    assert {
        "scripts/audit_zhou2020_reference_stability.py",
        "src/atem3d/empymod_compare.py",
        "src/atem3d/zhou2020_reference.py",
        "src/atem3d/zhou2020_reference_stability.py",
    } <= set(manifest["code_sha256"])
    for relative_path, path in expected_code_paths.items():
        assert manifest["code_sha256"][relative_path] == module.sha256_file(path)
    assert manifest["methods"]["qwe_convergence"] == audit["qwe"]
    assert manifest["methods"]["separate_total_qwe"]["srcpts"] == 17
    assert manifest["methods"]["separate_total_qwe"]["ftarg"] == module.QWE
    assert manifest["methods"]["component_convention"]["empymod_signal"] == 0
    assert manifest["methods"]["component_convention"]["scale"] < 0.0
    assert set(manifest["runtime"]) >= {
        "python_version",
        "numpy_version",
        "empymod_version",
        "git_commit",
        "git_dirty",
    }


def test_run_audit_rejects_case_revision_mismatch_before_publication(tmp_path):
    module = _load_script()
    mismatched_case = tmp_path / "zhou_case.yaml"
    mismatched_case.write_text(
        CASE.read_text(encoding="utf-8") + "\n# mismatched revision\n",
        encoding="utf-8",
    )
    output = tmp_path / "reference_audit"

    with pytest.raises(ValueError, match="case.*hash|identity"):
        module.run_audit(
            run=FORMAL_RUN,
            case=mismatched_case,
            output=output,
            srcpts=17,
        )

    assert not output.exists()
    _assert_no_transaction_debris(output)


@pytest.mark.parametrize(
    "input_name",
    ["reference_manifest.json", "empymod_metadata.json"],
)
def test_run_identity_rejects_ancillary_manifest_hash_mismatch(input_name):
    module = _load_script()
    paths = module._audit_input_paths(FORMAL_RUN, CASE)
    hashes = module._hash_inputs(paths)
    hashes[input_name] = "0" * 64

    with pytest.raises(ValueError, match="identity hash mismatch"):
        module._validate_run_identity(FORMAL_RUN, hashes)


@pytest.mark.filterwarnings(
    "ignore:The signature of `empymod.utils.check_time_only` changed:"
    "DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The signature of `empymod.utils.check_time` changed:"
    "DeprecationWarning"
)
def test_real_empymod_qwe_private_path_smoke():
    module = _load_script()
    times = np.geomspace(1.0e-4, 1.0e-3, 3)
    noip = replace(
        module.build_zhou_empymod_survey(CASE, variant="noip"),
        times=times,
    )
    ip = replace(
        module.build_zhou_empymod_survey(CASE, variant="ip"),
        times=times,
    )

    result = module._compute_qwe_audit_signals(noip, ip, srcpts=1)

    assert result["separate_total_qwe"].shape == times.shape
    assert result["direct_frequency_qwe"].shape == times.shape
    assert np.isfinite(result["separate_total_qwe"]).all()
    assert np.isfinite(result["direct_frequency_qwe"]).all()
    assert set(result["qwe_convergence"]) == {
        "separate_noip_qwe_converged",
        "separate_ip_qwe_converged",
        "direct_difference_qwe_converged",
        "converged",
    }
    assert all(
        isinstance(value, bool)
        for value in result["qwe_convergence"].values()
    )
