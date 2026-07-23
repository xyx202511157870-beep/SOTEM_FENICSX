"""Publish an immutable Zhou 2020 DLF/QWE reference-stability audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atem3d.empymod_compare import (
    _component_signal_and_scale,
    _receiver_mapping,
    _resistivity_model,
    _source_mapping,
)
from atem3d.zhou2020_reference import build_zhou_empymod_survey
from atem3d.zhou2020_reference_stability import build_reference_stability_audit


QWE = {
    "rtol": 1.0e-8,
    "atol": 1.0e-20,
    "nquad": 51,
    "maxint": 1000,
    "pts_per_dec": 60,
}
COMPONENT = "dBzdt"
MANIFEST_SCHEMA = "atem3d.zhou2020.reference-stability-manifest/v1"
ARTIFACT_NAMES = (
    "reference_stability.json",
    "reference_stability.npz",
)
NPZ_NAMES = (
    "time_s",
    "default_dlf",
    "separate_total_qwe",
    "direct_frequency_qwe",
    "fenicsx_increment",
)


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where POSIX supports directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write strict, finite JSON beside its final destination."""

    destination = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    """Atomically write finite arrays to a compressed NPZ artifact."""

    validated: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        try:
            finite = np.isfinite(array).all()
        except TypeError as exc:
            raise ValueError(f"{name} must contain finite numeric values") from exc
        if not finite:
            raise ValueError(f"{name} must contain only finite values")
        validated[name] = array

    destination = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **validated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _hash_inputs(paths: Mapping[str, Path]) -> dict[str, str]:
    if not paths:
        raise ValueError("at least one immutable input is required")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _verify_input_hashes(
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
) -> None:
    actual = _hash_inputs(paths)
    if actual != dict(expected):
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise RuntimeError(
            "immutable audit input changed during computation: "
            + ", ".join(changed)
        )


def _lock_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.lock")


def _acquire_lock(output: Path) -> Path:
    lock = _lock_path(output)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(output),
        },
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"audit publication lock exists: {lock}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(lock.parent)
    except BaseException:
        lock.unlink(missing_ok=True)
        _fsync_directory(lock.parent)
        raise
    return lock


def _is_verified_staging(staging: Path, output: Path) -> bool:
    expected_parent = output.parent.resolve()
    try:
        actual_parent = staging.parent.resolve(strict=True)
    except FileNotFoundError:
        return False
    return (
        actual_parent == expected_parent
        and staging.name.startswith(f".{output.name}.")
        and staging.name.endswith(".staging")
        and not staging.is_symlink()
    )


def _remove_staging(staging: Path, output: Path) -> None:
    if not _is_verified_staging(staging, output):
        raise RuntimeError(f"refusing to remove unverified staging path: {staging}")
    if staging.exists():
        shutil.rmtree(staging)
        _fsync_directory(staging.parent)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def load_validated_audit(output: str | Path) -> dict[str, Any]:
    """Load an audit directory only when its manifest binds every artifact."""

    directory = Path(output)
    if not directory.is_dir():
        raise FileNotFoundError(f"audit output directory does not exist: {directory}")
    required = {*ARTIFACT_NAMES, "manifest.json"}
    actual = {path.name for path in directory.iterdir()}
    if actual != required:
        raise ValueError("audit output must contain exactly the manifested artifacts")

    manifest = _load_json_object(directory / "manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported audit manifest schema")
    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, dict) or set(artifact_records) != set(
        ARTIFACT_NAMES
    ):
        raise ValueError("audit manifest artifact set is invalid")
    for name in ARTIFACT_NAMES:
        record = artifact_records[name]
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(directory / name)
        ):
            raise ValueError(f"audit artifact hash mismatch: {name}")

    audit = _load_json_object(directory / "reference_stability.json")
    if audit.get("status") != manifest.get("status"):
        raise ValueError("audit and manifest status differ")
    if audit.get("input_sha256") != manifest.get("input_sha256"):
        raise ValueError("audit and manifest input hashes differ")
    if audit.get("methods", {}) != manifest.get("methods", {}):
        raise ValueError("audit and manifest method metadata differ")
    if audit.get("qwe") != manifest.get("qwe"):
        raise ValueError("audit and manifest QWE convergence records differ")

    with np.load(
        directory / "reference_stability.npz",
        allow_pickle=False,
    ) as archive:
        if tuple(archive.files) != NPZ_NAMES:
            raise ValueError("audit NPZ member set or order is invalid")
        arrays = {name: np.asarray(archive[name]).copy() for name in NPZ_NAMES}
    sample_count = audit.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("audit sample count is invalid")
    for name, values in arrays.items():
        if values.shape != (sample_count,) or not np.isfinite(values).all():
            raise ValueError(f"audit NPZ member is invalid: {name}")
    return {"manifest": manifest, "audit": audit, "arrays": arrays}


def publish_audit(
    *,
    run: Path,
    output: Path,
    times,
    default_dlf,
    separate_total_qwe,
    direct_frequency_qwe,
    direct_qwe_converged: bool,
    fenicsx_increment,
    consecutive: int = 5,
    method_metadata: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, Path] | None = None,
    input_sha256: Mapping[str, str] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    qwe_convergence: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Publish one locked, manifest-bound audit directory transaction."""

    run = Path(run)
    output = Path(output)
    strict = run / "comparisons/S1T1B1/strict_comparison.json"
    immutable_inputs = dict(
        input_paths or {"strict_comparison.json": strict}
    )
    expected_hashes = dict(input_sha256 or _hash_inputs(immutable_inputs))
    if immutable_inputs.get("strict_comparison.json") != strict:
        raise ValueError("strict_comparison.json input path must match the formal run")
    if set(expected_hashes) != set(immutable_inputs):
        raise ValueError("input paths and SHA-256 records must have identical keys")

    audit = build_reference_stability_audit(
        times=times,
        default_dlf=default_dlf,
        separate_total_qwe=separate_total_qwe,
        direct_frequency_qwe=direct_frequency_qwe,
        direct_qwe_converged=direct_qwe_converged,
        fenicsx_increment=fenicsx_increment,
        consecutive=consecutive,
    )
    audit["input_sha256"] = expected_hashes
    if method_metadata is not None:
        audit["methods"] = dict(method_metadata)
    if qwe_convergence is not None:
        convergence = dict(qwe_convergence)
        if convergence.get("converged") is not audit["qwe"]["converged"]:
            raise ValueError("aggregate QWE convergence records must agree")
        audit["qwe"] = convergence

    output.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")

    lock: Path | None = None
    staging: Path | None = None
    renamed = False
    try:
        lock = _acquire_lock(output)
        if output.exists():
            raise FileExistsError(f"audit output already exists: {output}")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.",
                suffix=".staging",
                dir=output.parent,
            )
        )
        if not _is_verified_staging(staging, output):
            raise RuntimeError("created staging path failed safety verification")
        _fsync_directory(output.parent)

        _atomic_write_npz(
            staging / "reference_stability.npz",
            time_s=np.asarray(times),
            default_dlf=np.asarray(default_dlf),
            separate_total_qwe=np.asarray(separate_total_qwe),
            direct_frequency_qwe=np.asarray(direct_frequency_qwe),
            fenicsx_increment=np.asarray(fenicsx_increment),
        )
        _atomic_write_json(staging / "reference_stability.json", audit)
        _verify_input_hashes(immutable_inputs, expected_hashes)
        artifact_hashes = {
            name: {"sha256": sha256_file(staging / name)}
            for name in ARTIFACT_NAMES
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": audit["status"],
            "artifacts": artifact_hashes,
            "input_sha256": expected_hashes,
            "methods": dict(method_metadata or {}),
            "qwe": audit["qwe"],
            "runtime": dict(runtime_metadata or {}),
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        _verify_input_hashes(immutable_inputs, expected_hashes)
        load_validated_audit(staging)
        _fsync_directory(staging)
        if output.exists():
            raise FileExistsError(f"audit output already exists: {output}")
        os.replace(staging, output)
        renamed = True
        try:
            _fsync_directory(output.parent)
        except BaseException:
            os.replace(output, staging)
            renamed = False
            _fsync_directory(output.parent)
            raise
        staging = None
    finally:
        if staging is not None and staging.exists():
            _remove_staging(staging, output)
        if lock is not None:
            lock.unlink(missing_ok=True)
            _fsync_directory(lock.parent)
    if not renamed:
        raise RuntimeError("audit publication did not complete")
    return audit


def _load_signal(
    path: Path,
    *,
    time_column: str,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one named time/value signal from a CSV file."""

    table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    names = table.dtype.names or ()
    if time_column not in names or value_column not in names:
        raise ValueError(
            f"{path} must contain {time_column!r} and {value_column!r}"
        )
    times = np.atleast_1d(np.asarray(table[time_column], dtype=float))
    values = np.atleast_1d(np.asarray(table[value_column], dtype=float))
    if (
        times.shape != values.shape
        or times.ndim != 1
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{path} must contain matching finite 1D columns")
    return times, values


def _require_identical_times(
    left: np.ndarray,
    right: np.ndarray,
    *,
    description: str,
) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"{description} time axes must be identical")


def _frequency_response(
    survey,
    frequencies: np.ndarray,
    *,
    srcpts: int,
) -> np.ndarray:
    """Compute the complex canonical dBzdt frequency response."""

    import empymod  # noqa: PLC0415

    location = survey.receiver_locations[0]
    rec, mrec = _receiver_mapping(
        location,
        COMPONENT,
        survey.coordinate_system,
    )
    response = empymod.bipole(
        src=_source_mapping(
            survey.source_start,
            survey.source_end,
            survey.coordinate_system,
        ),
        rec=rec,
        depth=list(survey.depths),
        res=_resistivity_model(survey.resistivities),
        freqtime=frequencies,
        signal=None,
        strength=survey.strength,
        mrec=mrec,
        srcpts=srcpts,
        recpts=1,
        verb=0,
    )
    return np.asarray(response, dtype=complex).reshape(-1)


def _checked_qwe_grid(survey) -> dict[str, Any]:
    """Create the checked grid shared by all three QWE transforms."""

    from empymod.utils import check_time  # noqa: PLC0415

    signal, scale = _component_signal_and_scale(
        COMPONENT,
        survey.signal,
        survey.coordinate_system,
    )
    times, frequencies, ft, ftarg = check_time(
        survey.times,
        signal,
        "qwe",
        dict(QWE),
        0,
    )
    return {
        "times": np.asarray(times, dtype=float),
        "frequencies": np.asarray(frequencies, dtype=float),
        "ft": ft,
        "ftarg": ftarg,
        "signal": signal,
        "scale": float(scale),
    }


def _tem_qwe_transform(
    frequency_response: np.ndarray,
    grid: Mapping[str, Any],
) -> tuple[np.ndarray, bool]:
    """Transform one frequency response and retain QWE convergence."""

    from empymod.model import tem  # noqa: PLC0415

    response = np.asarray(frequency_response, dtype=complex).reshape(-1)
    frequencies = np.asarray(grid["frequencies"], dtype=float)
    if response.shape != frequencies.shape:
        raise ValueError("frequency response and checked QWE grid must match")
    transformed, converged = tem(
        response[:, None],
        np.ones(1),
        frequencies,
        np.asarray(grid["times"], dtype=float),
        grid["signal"],
        grid["ft"],
        grid["ftarg"],
    )
    convergence_values = np.asarray(converged)
    if (
        convergence_values.dtype.kind not in "biu"
        or convergence_values.size != 1
    ):
        raise ValueError("empymod QWE convergence must be boolean")
    convergence_value = convergence_values.reshape(-1)[0].item()
    if convergence_value not in (0, 1):
        raise ValueError("empymod QWE convergence must be boolean")
    values = float(grid["scale"]) * np.asarray(transformed[:, 0], dtype=float)
    return values, bool(convergence_value)


def _compute_qwe_audit_signals(
    noip_survey,
    ip_survey,
    *,
    srcpts: int,
) -> dict[str, Any]:
    """Run no-IP, IP, and direct-difference QWE transforms explicitly."""

    noip_times = np.asarray(noip_survey.times, dtype=float)
    ip_times = np.asarray(ip_survey.times, dtype=float)
    _require_identical_times(
        noip_times,
        ip_times,
        description="canonical no-IP/IP survey",
    )
    grid = _checked_qwe_grid(noip_survey)
    _require_identical_times(
        noip_times,
        np.asarray(grid["times"], dtype=float),
        description="canonical survey/checked QWE",
    )
    frequencies = np.asarray(grid["frequencies"], dtype=float)
    noip_frequency = _frequency_response(
        noip_survey,
        frequencies,
        srcpts=srcpts,
    )
    ip_frequency = _frequency_response(
        ip_survey,
        frequencies,
        srcpts=srcpts,
    )
    noip_total, noip_converged = _tem_qwe_transform(noip_frequency, grid)
    ip_total, ip_converged = _tem_qwe_transform(ip_frequency, grid)
    direct_difference, direct_converged = _tem_qwe_transform(
        ip_frequency - noip_frequency,
        grid,
    )
    convergence = {
        "separate_noip_qwe_converged": bool(noip_converged),
        "separate_ip_qwe_converged": bool(ip_converged),
        "direct_difference_qwe_converged": bool(direct_converged),
    }
    convergence["converged"] = all(convergence.values())
    return {
        "separate_total_qwe": ip_total - noip_total,
        "direct_frequency_qwe": direct_difference,
        "qwe_convergence": convergence,
        "grid": grid,
    }


def _audit_input_paths(run: Path, case: Path) -> dict[str, Path]:
    return {
        "case.yaml": case,
        "run_manifest.json": run / "run_manifest.json",
        "reference_manifest.json": run / "reference/reference_manifest.json",
        "empymod_metadata.json": run / "reference/empymod_metadata.json",
        "strict_comparison.json": (
            run / "comparisons/S1T1B1/strict_comparison.json"
        ),
        "empymod_noip.csv": run / "reference/empymod_noip.csv",
        "empymod_ip.csv": run / "reference/empymod_ip.csv",
        "fenicsx_noip_predictions.csv": (
            run / "fenicsx/noip/S1T1B1/predictions.csv"
        ),
        "fenicsx_ip_predictions.csv": (
            run / "fenicsx/ip/S1T1B1/predictions.csv"
        ),
    }


def _require_recorded_hash(
    record: Any,
    *,
    expected: str,
    description: str,
) -> None:
    if not isinstance(record, str) or record != expected:
        raise ValueError(f"{description} identity hash mismatch")


def _validate_run_identity(
    run: Path,
    input_hashes: Mapping[str, str],
) -> None:
    """Bind requested inputs to the formal Zhou case and run manifests."""

    reference_manifest = _load_json_object(
        run / "reference/reference_manifest.json"
    )
    if (
        reference_manifest.get("case_id") != "zhou2020_grounded_wire"
        or reference_manifest.get("status") != "reference_verified"
    ):
        raise ValueError("reference manifest identity or status is invalid")
    reference_hashes = reference_manifest.get("file_sha256")
    if not isinstance(reference_hashes, dict):
        raise ValueError("reference manifest file hashes are missing")
    for name in (
        "empymod_noip.csv",
        "empymod_ip.csv",
        "empymod_metadata.json",
    ):
        _require_recorded_hash(
            reference_hashes.get(name),
            expected=input_hashes[name],
            description=f"reference {name}",
        )

    reference_metadata = _load_json_object(
        run / "reference/empymod_metadata.json"
    )
    if reference_metadata.get("case_id") != "zhou2020_grounded_wire":
        raise ValueError("reference metadata case identity is invalid")
    _require_recorded_hash(
        reference_metadata.get("case_file_sha256"),
        expected=input_hashes["case.yaml"],
        description="case file",
    )

    run_manifest_path = run / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError("formal run manifest is required for audit identity")
    run_manifest = _load_json_object(run_manifest_path)
    snapshots = run_manifest.get("snapshots")
    if not isinstance(snapshots, dict) or not isinstance(
        snapshots.get("case"), dict
    ):
        raise ValueError("formal run case snapshot identity is missing")
    _require_recorded_hash(
        snapshots["case"].get("sha256"),
        expected=input_hashes["case.yaml"],
        description="formal run case",
    )
    comparisons = run_manifest.get("comparisons")
    if not isinstance(comparisons, dict) or not isinstance(
        comparisons.get("full/S1T1B1"), dict
    ):
        raise ValueError("formal strict comparison identity is missing")
    _require_recorded_hash(
        comparisons["full/S1T1B1"].get("sha256"),
        expected=input_hashes["strict_comparison.json"],
        description="strict comparison",
    )

    stages = run_manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("formal run stage identities are missing")
    expected_stage_files = {
        ("reference", "reference/empymod_noip.csv"): "empymod_noip.csv",
        ("reference", "reference/empymod_ip.csv"): "empymod_ip.csv",
        (
            "reference",
            "reference/empymod_metadata.json",
        ): "empymod_metadata.json",
        (
            "reference",
            "reference/reference_manifest.json",
        ): "reference_manifest.json",
        (
            "fenicsx/noip/S1T1B1",
            "fenicsx/noip/S1T1B1/predictions.csv",
        ): "fenicsx_noip_predictions.csv",
        (
            "fenicsx/ip/S1T1B1",
            "fenicsx/ip/S1T1B1/predictions.csv",
        ): "fenicsx_ip_predictions.csv",
    }
    for (stage_name, file_name), input_name in expected_stage_files.items():
        stage = stages.get(stage_name)
        files = stage.get("files") if isinstance(stage, dict) else None
        record = files.get(file_name) if isinstance(files, dict) else None
        _require_recorded_hash(
            record.get("sha256") if isinstance(record, dict) else None,
            expected=input_hashes[input_name],
            description=f"formal run {file_name}",
        )


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    commit_text = commit.stdout.strip() if commit.returncode == 0 else "unavailable"
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return commit_text, dirty


def _runtime_metadata() -> dict[str, Any]:
    git_commit, git_dirty = _git_state()
    try:
        empymod_version = importlib.metadata.version("empymod")
    except importlib.metadata.PackageNotFoundError:
        empymod_version = "unavailable"
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "empymod_version": empymod_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def run_audit(
    *,
    run: Path,
    case: Path,
    output: Path,
    srcpts: int = 17,
) -> dict[str, Any]:
    """Compute and publish the formal Zhou reference-transform audit."""

    if isinstance(srcpts, bool) or not isinstance(srcpts, int) or srcpts <= 0:
        raise ValueError("srcpts must be a positive integer")
    run = Path(run)
    case = Path(case)
    output = Path(output)
    input_paths = _audit_input_paths(run, case)
    input_hashes = _hash_inputs(input_paths)
    _validate_run_identity(run, input_hashes)

    reference = run / "reference"
    reference_noip_times, reference_noip = _load_signal(
        reference / "empymod_noip.csv",
        time_column="time_s",
        value_column="dBzdt_T_per_s",
    )
    reference_ip_times, reference_ip = _load_signal(
        reference / "empymod_ip.csv",
        time_column="time_s",
        value_column="dBzdt_T_per_s",
    )
    _require_identical_times(
        reference_noip_times,
        reference_ip_times,
        description="empymod no-IP/IP",
    )
    default_dlf = reference_ip - reference_noip

    fenicsx_noip_path = run / "fenicsx/noip/S1T1B1/predictions.csv"
    fenicsx_ip_path = run / "fenicsx/ip/S1T1B1/predictions.csv"
    fenicsx_noip_times, fenicsx_noip = _load_signal(
        fenicsx_noip_path,
        time_column="time_obs",
        value_column="dBzdt",
    )
    fenicsx_ip_times, fenicsx_ip = _load_signal(
        fenicsx_ip_path,
        time_column="time_obs",
        value_column="dBzdt",
    )
    _require_identical_times(
        fenicsx_noip_times,
        fenicsx_ip_times,
        description="FEniCSx no-IP/IP",
    )
    _require_identical_times(
        reference_noip_times,
        fenicsx_noip_times,
        description="empymod/FEniCSx",
    )
    fenicsx_increment = fenicsx_ip - fenicsx_noip

    noip_survey = build_zhou_empymod_survey(case, variant="noip")
    ip_survey = build_zhou_empymod_survey(case, variant="ip")
    _require_identical_times(
        reference_noip_times,
        np.asarray(noip_survey.times, dtype=float),
        description="formal reference/canonical survey",
    )
    _require_identical_times(
        np.asarray(noip_survey.times, dtype=float),
        np.asarray(ip_survey.times, dtype=float),
        description="canonical no-IP/IP survey",
    )
    qwe_result = _compute_qwe_audit_signals(
        noip_survey,
        ip_survey,
        srcpts=srcpts,
    )
    separate_total_qwe = qwe_result["separate_total_qwe"]
    direct_frequency_qwe = qwe_result["direct_frequency_qwe"]
    qwe_convergence = qwe_result["qwe_convergence"]
    grid = qwe_result["grid"]

    methods = {
        "default_dlf": {
            "operation": "empymod_ip.csv minus empymod_noip.csv",
            "transform": "formal_run_default_dlf",
        },
        "separate_total_qwe": {
            "operation": "IP total minus no-IP total after separate transforms",
            "component": COMPONENT,
            "ft": "qwe",
            "ftarg": dict(QWE),
            "srcpts": srcpts,
            "recpts": 1,
        },
        "direct_frequency_qwe": {
            "operation": "IP minus no-IP in frequency domain before empymod.model.tem",
            "component": COMPONENT,
            "ft": "qwe",
            "ftarg": dict(QWE),
            "srcpts": srcpts,
            "recpts": 1,
        },
        "qwe_convergence": dict(qwe_convergence),
        "component_convention": {
            "component": COMPONENT,
            "coordinate_system": noip_survey.coordinate_system,
            "empymod_signal": int(grid["signal"]),
            "scale": float(grid["scale"]),
            "frequency_operation": "IP minus no-IP before empymod.model.tem",
        },
        "fenicsx_increment": {
            "operation": "IP predictions minus no-IP predictions",
            "spatial_case": "S1T1B1",
        },
    }
    return publish_audit(
        run=run,
        output=output,
        times=reference_noip_times,
        default_dlf=default_dlf,
        separate_total_qwe=separate_total_qwe,
        direct_frequency_qwe=direct_frequency_qwe,
        direct_qwe_converged=qwe_convergence["converged"],
        fenicsx_increment=fenicsx_increment,
        method_metadata=methods,
        input_paths=input_paths,
        input_sha256=input_hashes,
        runtime_metadata=_runtime_metadata(),
        qwe_convergence=qwe_convergence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a non-promoting Zhou 2020 DLF/QWE stability audit."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--srcpts", type=int, default=17)
    args = parser.parse_args(argv)

    audit = run_audit(
        run=args.run,
        case=args.case,
        output=args.output,
        srcpts=args.srcpts,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
