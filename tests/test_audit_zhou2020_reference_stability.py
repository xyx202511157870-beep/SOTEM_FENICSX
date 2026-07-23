import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_zhou2020_reference_stability.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "zhou_reference_audit_under_test",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    run = tmp_path / "run"
    comparison = run / "comparisons/S1T1B1"
    comparison.mkdir(parents=True)
    strict = comparison / "strict_comparison.json"
    strict.write_text(
        '{"status":"failed_with_reproducible_evidence"}\n',
        encoding="utf-8",
    )
    before = strict.read_bytes()
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    direct = np.geomspace(1.0e-12, 1.0e-9, 8)

    module.publish_audit(
        run=run,
        output=tmp_path / "audit",
        times=times,
        default_dlf=direct * 1.02,
        separate_total_qwe=direct * 1.01,
        direct_frequency_qwe=direct,
        direct_qwe_converged=False,
        fenicsx_increment=direct * 1.05,
        consecutive=2,
    )

    assert strict.read_bytes() == before
    payload = json.loads(
        (tmp_path / "audit/reference_stability.json").read_text("utf-8")
    )
    with np.load(tmp_path / "audit/reference_stability.npz") as arrays:
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
    assert not list((tmp_path / "audit").glob(".*.tmp"))


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
