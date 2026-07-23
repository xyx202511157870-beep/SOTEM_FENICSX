from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPORT_FORMATS = (".svg", ".png", ".tiff")
FIGURE_STEMS = (
    "fig01_model_contract",
    "fig02_total_fields",
    "fig03_reference_stability",
    "fig04_gate_summary",
    "fig05_debye_order_diagnostic",
)
DEBYE_JSON_NAME = "debye_order_diagnostic.json"
COMPLETION_MANIFEST_NAME = "completion_manifest.json"
PUBLISHED_BUNDLE_NAME = "publication_bundle"
BUNDLE_SCHEMA = "atem3d.zhou2020.figure-bundle/v1"
BUNDLE_ARTIFACT_NAMES = tuple(
    f"{stem}{suffix}"
    for stem in FIGURE_STEMS
    for suffix in EXPORT_FORMATS
) + (DEBYE_JSON_NAME,)
EXPORT_POLICY = {
    "pdf": "forbidden_by_user",
    "known_waiver": (
        "The static Nature-figure preflight requires PDF, but the user's "
        "explicit no-PDF constraint takes precedence."
    ),
}
AUDIT_BOUND_INPUTS = {
    "case.yaml": Path("snapshots/case.yaml"),
    "run_manifest.json": Path("run_manifest.json"),
    "reference_manifest.json": Path("reference/reference_manifest.json"),
    "empymod_metadata.json": Path("reference/empymod_metadata.json"),
    "strict_comparison.json": Path(
        "comparisons/S1T1B1/strict_comparison.json"
    ),
    "empymod_noip.csv": Path("reference/empymod_noip.csv"),
    "empymod_ip.csv": Path("reference/empymod_ip.csv"),
    "fenicsx_noip_predictions.csv": Path(
        "fenicsx/noip/S1T1B1/predictions.csv"
    ),
    "fenicsx_ip_predictions.csv": Path(
        "fenicsx/ip/S1T1B1/predictions.csv"
    ),
}
COMPONENTS = (
    ("Ex", "Ex (V/m)"),
    ("Hz", "Hz (A/m)"),
    ("dBzdt", "dBz/dt (T/s)"),
)
COLORS = {
    "fenicsx": "#0072B2",
    "empymod": "#D55E00",
    "noip": "#009E73",
    "ip": "#CC79A7",
    "qwe": "#6F4E7C",
    "separate_qwe": "#7A7A7A",
    "warning": "#A32A2A",
}


class _BundleLock:
    __slots__ = ("path", "owner_token", "payload")

    def __init__(self, path: Path, owner_token: str, payload: bytes) -> None:
        self.path = path
        self.owner_token = owner_token
        self.payload = payload


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def _time_column(data: dict[str, np.ndarray]) -> np.ndarray:
    for name in ("time_obs", "time_s"):
        if name in data:
            return data[name]
    raise KeyError("CSV does not contain time_obs or time_s")


def _component(data: dict[str, np.ndarray], component: str) -> np.ndarray:
    aliases = {
        "Ex": ("Ex", "Ex_V_per_m"),
        "Hz": ("Hz", "Hz_A_per_m"),
        "dBzdt": ("dBzdt", "dBzdt_T_per_s"),
    }
    for name in aliases[component]:
        if name in data:
            return data[name]
    raise KeyError(f"CSV does not contain {component}")


def _positive_magnitude(values: np.ndarray) -> np.ma.MaskedArray:
    """Return |values| for log plotting, masking exact zeros and nothing else."""

    magnitude = np.abs(np.asarray(values, dtype=float))
    return np.ma.masked_equal(magnitude, 0.0, copy=True)


def _require_positive_times(values: np.ndarray) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times <= 0.0):
        raise ValueError("logarithmic time coordinates must be finite and strictly positive")
    if times.size == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time coordinates must be non-empty and strictly increasing")
    return times


def _require_finite_1d(
    values: np.ndarray,
    *,
    label: str,
    sample_count: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or array.shape != (sample_count,)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{label} must be a finite 1-D array with {sample_count} samples"
        )
    return array


def _validated_total_field_time_grid(
    *datasets: tuple[str, dict[str, np.ndarray]],
) -> np.ndarray:
    grids: list[np.ndarray] = []
    for dataset_name, data in datasets:
        times = _require_positive_times(_time_column(data))
        for component, _ in COMPONENTS:
            _require_finite_1d(
                _component(data, component),
                label=f"{dataset_name}.{component}",
                sample_count=times.size,
            )
        grids.append(times)
    if any(not np.array_equal(grids[0], grid) for grid in grids[1:]):
        raise ValueError("all four total-field time grids must be exactly equal")
    return grids[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_file(path)


def _fsync_file(path: Path) -> None:
    mode = "rb+" if os.name == "nt" else "rb"
    with Path(path).open(mode) as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the platform exposes directory fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bundle_lock_path(target: Path) -> Path:
    target = Path(target)
    return target.with_name(f".{target.name}.lock")


def _acquire_bundle_lock(target: Path) -> _BundleLock:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    path = _bundle_lock_path(target)
    owner_token = secrets.token_hex(16)
    payload = (
        json.dumps(
            {"owner_token": owner_token, "pid": os.getpid()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"publication lock already exists: {path}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to write publication lock")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            if path.is_file() and path.read_bytes() == payload:
                path.unlink()
                _fsync_directory(path.parent)
        except BaseException:
            pass
        raise
    else:
        os.close(descriptor)
    try:
        _fsync_file(path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            if path.is_file() and path.read_bytes() == payload:
                path.unlink()
                _fsync_directory(path.parent)
        except BaseException:
            pass
        raise
    return _BundleLock(path=path, owner_token=owner_token, payload=payload)


def _release_bundle_lock(owner: _BundleLock) -> None:
    try:
        current = owner.path.read_bytes()
    except FileNotFoundError:
        return
    if current != owner.payload:
        return
    failure: tuple[type[BaseException], BaseException, Any] | None = None
    for _ in range(2):
        try:
            owner.path.unlink()
            failure = None
            break
        except FileNotFoundError:
            failure = None
            break
        except BaseException:
            failure = sys.exc_info()
    if failure is not None:
        _, error, traceback = failure
        raise error.with_traceback(traceback)
    _fsync_directory(owner.path.parent)


def _remove_tree_with_retry(path: Path) -> None:
    """Remove an owned transaction tree despite one asynchronous interruption."""

    path = Path(path)
    failure: tuple[type[BaseException], BaseException, Any] | None = None
    for _ in range(2):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except BaseException:
            failure = sys.exc_info()
    if failure is not None:
        _, error, traceback = failure
        raise error.with_traceback(traceback)


def _completion_manifest(
    staging: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual_entries = {
        path.name
        for path in Path(staging).iterdir()
    }
    if actual_entries != set(BUNDLE_ARTIFACT_NAMES):
        raise ValueError("figure bundle artifact set is incomplete or unexpected")
    artifacts = {}
    for name in BUNDLE_ARTIFACT_NAMES:
        path = Path(staging) / name
        artifacts[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": BUNDLE_SCHEMA,
        "status": "complete",
        "figure_count": len(FIGURE_STEMS),
        "artifact_count": len(BUNDLE_ARTIFACT_NAMES),
        "export_formats": list(EXPORT_FORMATS),
        "pdf_policy": EXPORT_POLICY,
        "metadata": metadata,
        "artifacts": artifacts,
    }


def _write_completion_manifest(
    staging: Path,
    metadata: dict[str, Any],
) -> None:
    manifest_path = Path(staging) / COMPLETION_MANIFEST_NAME
    if manifest_path.exists():
        raise ValueError("completion manifest must not exist before finalization")
    _write_json(manifest_path, _completion_manifest(staging, metadata))


def _validate_bundle_metadata(metadata: Any) -> None:
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("run_id"), str)
        or not metadata["run_id"]
        or metadata.get("spatial_case") != "S1T1B1"
        or not isinstance(metadata.get("reference_audit_status"), str)
        or not isinstance(metadata.get("qwe_converged"), bool)
    ):
        raise ValueError("figure bundle source metadata is invalid")
    input_hashes = metadata.get("input_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("figure bundle input hashes are missing")
    for name, value in input_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError("figure bundle input hash record is invalid")


def load_validated_bundle(bundle: Path) -> dict[str, Any]:
    """Load a complete generation only after exact-set and hash validation."""

    bundle = Path(bundle)
    manifest_path = bundle / COMPLETION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completion manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("figure_count") != len(FIGURE_STEMS)
        or manifest.get("artifact_count") != len(BUNDLE_ARTIFACT_NAMES)
        or manifest.get("export_formats") != list(EXPORT_FORMATS)
        or manifest.get("pdf_policy") != EXPORT_POLICY
    ):
        raise ValueError("figure bundle completion metadata is invalid")
    _validate_bundle_metadata(manifest.get("metadata"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        BUNDLE_ARTIFACT_NAMES
    ):
        raise ValueError("figure bundle manifest artifact set is invalid")
    actual_entries = {
        path.name
        for path in bundle.iterdir()
    }
    if actual_entries != set(BUNDLE_ARTIFACT_NAMES) | {
        COMPLETION_MANIFEST_NAME
    }:
        raise ValueError("published figure bundle file set is invalid")
    if any(
        path.suffix.lower().lstrip(".") == "pdf"
        for path in bundle.rglob("*")
        if path.is_file()
    ):
        raise ValueError("PDF artifacts are forbidden by the export policy")
    for name, item in artifacts.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("bytes"), int)
        ):
            raise ValueError(f"invalid artifact manifest record: {name}")
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact is missing: {name}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"bundle artifact size mismatch: {name}")
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"bundle artifact hash mismatch: {name}")
    diagnostic = json.loads(
        (bundle / DEBYE_JSON_NAME).read_text(encoding="utf-8")
    )
    if (
        not isinstance(diagnostic, dict)
        or diagnostic.get("schema")
        != "atem3d.zhou2020.debye-order-diagnostic/v2"
    ):
        raise ValueError("Debye diagnostic JSON schema is invalid")
    return {"manifest": manifest, "diagnostic": diagnostic}


def _replace_bundle_directory(staging: Path, target: Path) -> None:
    """Replace one complete directory generation and roll back on failure."""

    staging = Path(staging)
    target = Path(target)
    backup = staging.with_name(
        staging.name.replace(
            f".{target.name}.staging-",
            f".{target.name}.backup-",
            1,
        )
    )
    if backup.exists():
        raise FileExistsError(f"bundle backup already exists: {backup}")
    old_moved = False
    new_published = False
    try:
        if target.exists():
            if not target.is_dir():
                raise NotADirectoryError(f"bundle target is not a directory: {target}")
            os.replace(target, backup)
            old_moved = True
            _fsync_directory(target.parent)
        os.replace(staging, target)
        new_published = True
        _fsync_directory(target.parent)
        load_validated_bundle(target)
    except BaseException:
        rollback_failure: BaseException | None = None
        try:
            if (
                target.exists()
                and (new_published or not staging.exists())
            ):
                _remove_tree_with_retry(target)
                _fsync_directory(target.parent)
        except BaseException as exc:
            rollback_failure = exc
        try:
            if old_moved and backup.exists():
                os.replace(backup, target)
                _fsync_directory(target.parent)
        except BaseException as exc:
            if rollback_failure is None:
                rollback_failure = exc
        if rollback_failure is not None:
            raise RuntimeError("figure bundle rollback failed") from rollback_failure
        raise
    else:
        try:
            if backup.exists():
                _remove_tree_with_retry(backup)
                _fsync_directory(target.parent)
        except BaseException:
            if backup.exists() and target.exists():
                _remove_tree_with_retry(target)
                os.replace(backup, target)
                _fsync_directory(target.parent)
            raise


def _publish_bundle(
    target: Path,
    build_artifacts: Callable[[Path], None],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build, finalize, validate, and expose one indivisible figure generation."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    owner = _acquire_bundle_lock(target)
    staging: Path | None = None
    result: dict[str, Any] | None = None
    failure: tuple[type[BaseException], BaseException, Any] | None = None
    cleanup_failure: BaseException | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.staging-",
                dir=target.parent,
            )
        )
        _fsync_directory(target.parent)
        build_artifacts(staging)
        for name in BUNDLE_ARTIFACT_NAMES:
            artifact = staging / name
            if artifact.is_file():
                _fsync_file(artifact)
        _fsync_directory(staging)
        _write_completion_manifest(staging, metadata)
        _fsync_file(staging / COMPLETION_MANIFEST_NAME)
        _fsync_directory(staging)
        load_validated_bundle(staging)
        _replace_bundle_directory(staging, target)
        result = load_validated_bundle(target)
    except BaseException:
        failure = sys.exc_info()
    try:
        if staging is not None and staging.exists():
            _remove_tree_with_retry(staging)
            _fsync_directory(target.parent)
    except BaseException as exc:
        cleanup_failure = exc
    try:
        _release_bundle_lock(owner)
    except BaseException as exc:
        if cleanup_failure is None:
            cleanup_failure = exc
    if failure is not None:
        _, error, traceback = failure
        raise error.with_traceback(traceback)
    if cleanup_failure is not None:
        raise cleanup_failure
    assert result is not None
    return result


def _save_all(fig: plt.Figure, stem: Path) -> None:
    """Stage the complete no-PDF export set before atomically replacing files."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{stem.name}.", dir=stem.parent)
    )
    staged_stem = staging / stem.name
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        fig.savefig(staged_stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(
            staged_stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            staged_stem.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        for suffix in EXPORT_FORMATS:
            _fsync_file(staged_stem.with_suffix(suffix))
        _fsync_directory(staging)

        for suffix in EXPORT_FORMATS:
            target = stem.with_suffix(suffix)
            if target.exists():
                backup = staging / f"previous{suffix}"
                os.replace(target, backup)
                backups[target] = backup
        if backups:
            _fsync_directory(stem.parent)
        try:
            for suffix in EXPORT_FORMATS:
                target = stem.with_suffix(suffix)
                os.replace(staged_stem.with_suffix(suffix), target)
                published.append(target)
            _fsync_directory(stem.parent)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            for target, backup in backups.items():
                if backup.exists():
                    os.replace(backup, target)
            _fsync_directory(stem.parent)
            raise
    except BaseException:
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        _fsync_directory(stem.parent)
        raise
    finally:
        try:
            _remove_tree_with_retry(staging)
        finally:
            _fsync_directory(stem.parent)


def _finish_or_save(
    fig: plt.Figure,
    stem: Path | None,
    *,
    rect: tuple[float, float, float, float] | None = None,
    tight: bool = True,
) -> plt.Figure | None:
    if stem is None:
        if tight:
            fig.tight_layout(rect=rect)
        return fig
    try:
        if tight:
            fig.tight_layout(rect=rect)
        _save_all(fig, stem)
    finally:
        plt.close(fig)
    return None


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "lines.linewidth": 1.35,
            "legend.frameon": False,
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _load_audit_module():
    path = ROOT / "scripts/audit_zhou2020_reference_stability.py"
    spec = importlib.util.spec_from_file_location(
        "zhou_reference_audit_validator", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference-audit validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_audit(path: Path) -> dict[str, Any]:
    """Load only a manifest-bound Task-2 audit evidence directory."""

    return _load_audit_module().load_validated_audit(Path(path))


def _audit_bound_inputs(run: Path) -> dict[str, Path]:
    run = Path(run)
    return {
        name: run / relative_path
        for name, relative_path in AUDIT_BOUND_INPUTS.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cross_bind_run_to_audit(
    run: Path,
    validated_audit: dict[str, Any],
) -> None:
    """Require the plotted run to be byte-identical to the audited run inputs."""

    manifest_hashes = validated_audit.get("manifest", {}).get("input_sha256")
    audit = validated_audit.get("audit", {})
    audit_hashes = audit.get("input_sha256")
    paths = _audit_bound_inputs(run)
    if (
        not isinstance(manifest_hashes, dict)
        or manifest_hashes != audit_hashes
        or set(manifest_hashes) != set(paths)
    ):
        raise ValueError("audit input identity is incomplete or inconsistent")
    if (
        audit.get("methods", {})
        .get("fenicsx_increment", {})
        .get("spatial_case")
        != "S1T1B1"
    ):
        raise ValueError("audit spatial-case identity is not S1T1B1")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"audited run input is missing: {path}")
        if _sha256_file(path) != manifest_hashes[name]:
            raise ValueError(f"audited run input hash mismatch: {name}")


def plot_model(stem: Path | None) -> plt.Figure | None:
    fig, (ax_plan, ax_depth) = plt.subplots(
        1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": (1.25, 1.0)}
    )
    ax_plan.plot([-500, 500], [0, 0], color="#222222", lw=3.0)
    ax_plan.scatter([-500, 500], [0, 0], color="#222222", s=25, zorder=3)
    ax_plan.scatter([0], [1000], marker="v", color=COLORS["empymod"], s=45, zorder=4)
    ax_plan.plot([0, 0], [0, 1000], color="#777777", lw=0.9, ls=":")
    ax_plan.annotate("A", (-500, 55), ha="center")
    ax_plan.annotate("B", (500, 55), ha="center")
    ax_plan.text(
        180,
        1030,
        "Rx (0, 1000 m)",
        color=COLORS["empymod"],
        ha="left",
        va="center",
    )
    ax_plan.text(0, -125, "1000 m grounded wire, 10 A", ha="center")
    ax_plan.text(35, 500, "1000 m perpendicular offset", rotation=90, va="center")
    ax_plan.set_xlim(-650, 650)
    ax_plan.set_ylim(-250, 1120)
    ax_plan.set_aspect("equal", adjustable="box")
    ax_plan.set_xlabel("x (m)")
    ax_plan.set_ylabel("y (m)")
    ax_plan.set_title("(a) Plan view")
    ax_plan.grid(True, color="#E2E2E2", lw=0.45)

    ax_depth.axhspan(0, 70, color="#DCEAF7")
    ax_depth.axhspan(-500, 0, color="#E8D7B9")
    ax_depth.axhspan(-520, -500, color="#7FCDBB")
    ax_depth.axhspan(-650, -520, color="#C7B9A6")
    ax_depth.text(0.5, -250, "Layer 1\n100 ohm m\n500 m", ha="center", va="center")
    ax_depth.annotate(
        "IP: 10 ohm m, 20 m\nm=0.1, tau=1 s, c=0.3",
        xy=(0.30, -510),
        xytext=(0.35, -445),
        ha="center",
        va="center",
        fontsize=7.4,
        arrowprops={"arrowstyle": "->", "color": "#3D8075", "lw": 0.8},
    )
    ax_depth.text(0.5, -585, "Half-space\n200 ohm m", ha="center", va="center")
    ax_depth.set_xlim(0, 1)
    ax_depth.set_ylim(-650, 70)
    ax_depth.set_xticks([])
    ax_depth.set_ylabel("Elevation z (m)")
    ax_depth.set_title("(b) Layered earth")
    ax_depth.grid(False)
    fig.suptitle("Zhou et al. (2020) grounded-wire TEM-IP benchmark", y=0.995)
    return _finish_or_save(fig, stem, rect=(0.0, 0.0, 1.0, 0.97))


def plot_total_fields(
    noip_fem: dict[str, np.ndarray],
    ip_fem: dict[str, np.ndarray],
    noip_ref: dict[str, np.ndarray],
    ip_ref: dict[str, np.ndarray],
    stem: Path | None,
) -> plt.Figure | None:
    """Plot literature-style total-field magnitudes without altering source arrays."""

    times = _validated_total_field_time_grid(
        ("noip_fem", noip_fem),
        ("ip_fem", ip_fem),
        ("noip_ref", noip_ref),
        ("ip_ref", ip_ref),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True)
    variants = (
        ("No-IP", noip_fem, noip_ref),
        ("IP", ip_fem, ip_ref),
    )
    for row, (variant, fem, ref) in enumerate(variants):
        for col, (component, label) in enumerate(COMPONENTS):
            ax = axes[row, col]
            fem_values = _positive_magnitude(_component(fem, component))
            ref_values = _positive_magnitude(_component(ref, component))
            ax.plot(
                times,
                ref_values,
                color=COLORS["empymod"],
                ls="--",
                label="empymod",
            )
            ax.plot(times, fem_values, color=COLORS["fenicsx"], label="FEniCSx")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", color="#D8D8D8", lw=0.45)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{variant}\nAbsolute magnitude")
            if row == 1:
                ax.set_xlabel("Time after turn-off (s)")
            if row == 0 and col == 0:
                ax.legend(loc="best")
    fig.suptitle(
        "Absolute magnitude total fields: literature-style log-log; "
        "absolute magnitude does not contain sign information",
        y=0.995,
    )
    return _finish_or_save(fig, stem, rect=(0.0, 0.0, 1.0, 0.96))


def plot_reference_stability(
    arrays: dict[str, np.ndarray],
    audit: dict[str, Any],
    stem: Path | None,
) -> plt.Figure | None:
    """Show weak dBz/dt IP-increment stability without hiding any samples."""

    if audit.get("all_samples_retained") is not True:
        raise ValueError("audit evidence all_samples_retained must be true")
    times = _require_positive_times(arrays["time_s"])
    unstable_end = float(audit["stable_window"]["start_s"])
    series = (
        ("default_dlf", "default DLF", "--", COLORS["empymod"]),
        (
            "separate_total_qwe",
            "separate-total QWE",
            ":",
            COLORS["separate_qwe"],
        ),
        ("direct_frequency_qwe", "direct-frequency QWE", "-", COLORS["qwe"]),
        ("fenicsx_increment", "FEniCSx", "-", COLORS["fenicsx"]),
    )
    for key, *_ in series:
        values = np.asarray(arrays[key], dtype=float)
        if values.shape != times.shape:
            raise ValueError(f"audit array shape mismatch: {key}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax in axes:
        ax.axvspan(
            times[0],
            unstable_end,
            color="#D9D9D9",
            alpha=0.55,
            zorder=0,
        )
        ax.set_xscale("log")
        ax.grid(True, which="both", color="#D8D8D8", lw=0.45)

    for name, label, style, color in series:
        axes[0].plot(
            times,
            _positive_magnitude(arrays[name]),
            style,
            color=color,
            label=label,
        )
        axes[1].plot(times, arrays[name], style, color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("(a) Absolute magnitude")
    axes[0].set_ylabel("|IP increment dBz/dt| (T/s)")
    axes[0].legend(loc="upper right")

    axes[1].axhline(0.0, color="#222222", lw=0.8, zorder=1)
    axes[1].set_yscale("linear")
    early_count = min(20, times.size)
    early_end = float(times[early_count - 1])
    axes[1].set_xlim(float(times[0]), max(early_end, unstable_end))
    early_values = np.concatenate(
        [
            np.asarray(arrays[name], dtype=float)[:early_count]
            for name, *_ in series
        ]
    )
    early_low = min(0.0, float(np.min(early_values)))
    early_high = max(0.0, float(np.max(early_values)))
    early_span = early_high - early_low
    if early_span == 0.0:
        early_span = max(abs(early_low), np.finfo(float).tiny)
    early_padding = 0.08 * early_span
    axes[1].set_ylim(early_low - early_padding, early_high + early_padding)
    axes[1].set_title("(b) Signed diagnostic")
    axes[1].set_ylabel("IP increment dBz/dt (T/s)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes:
        ax.set_xlabel("Time after turn-off (s)")
    axes[0].text(
        0.02,
        0.95,
        "reference-transform unstable",
        transform=axes[0].transAxes,
        color="#4D4D4D",
        fontsize=6.8,
        weight="bold",
        va="top",
    )
    qwe_converged = bool(audit["qwe"]["converged"])
    sign_changes = int(audit["default_dlf"]["sign_changes_first20"])
    axes[1].text(
        0.50,
        -0.38,
        (
            f"default DLF first-20 sign changes={sign_changes}\n"
            f"QWE converged={qwe_converged}\n"
            f"status={audit['status']}; "
            f"all samples retained={audit['all_samples_retained']}"
        ),
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "#B0B0B0", "alpha": 0.9},
    )
    fig.suptitle(
        "dBz/dt IP-increment reference-transform stability audit",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.78,
        bottom=0.34,
        wspace=0.22,
    )
    return _finish_or_save(fig, stem, tight=False)


def _failed_hz_false_reversal_time(metrics: dict[str, Any]) -> float:
    try:
        record = metrics["zero_crossings"]["noip"]["Hz"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Hz false-reversal audit record is unavailable") from exc
    prediction = record.get("prediction")
    reference = record.get("reference")
    if (
        record.get("passed") is not False
        or record.get("count_match") is not False
        or not isinstance(prediction, list)
        or len(prediction) != 1
        or reference != []
    ):
        raise ValueError("Hz false-reversal audit record is not a failed event")
    value = prediction[0]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("Hz false-reversal time must be finite and positive")
    return float(value)


def plot_gate_summary(
    metrics: dict[str, Any],
    audit: dict[str, Any],
    stem: Path | None,
) -> plt.Figure | None:
    """Separate formal total-field gates from weak-increment sensitivity evidence."""

    labels: list[str] = []
    values: list[float] = []
    gates: list[float] = []
    colors: list[str] = []
    for variant in ("noip", "ip"):
        for component, _ in COMPONENTS:
            item = metrics["total_field"][variant][component]
            labels.append(f"{variant}\n{component}")
            values.append(float(item["relative_l2"]) * 100.0)
            gates.append(float(item["gate"]) * 100.0)
            colors.append(COLORS["noip"] if item["passed"] else COLORS["warning"])

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.72)
    for xi, gate in zip(x, gates):
        ax.hlines(gate, xi - 0.36, xi + 0.36, color="#202020", lw=1.4)
    peak = max(values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + peak * 0.025,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Formal total-field relative L2 error (%)")
    ax.set_ylim(0, max(max(gates) * 1.6, peak * 1.35))
    ax.grid(True, axis="y", color="#D8D8D8", lw=0.5)
    ax.set_title("Formal total-field L2 gates (black segment = 5% threshold)")

    increments = metrics["ip_increment"]
    sensitivity_text = (
        "IP increments: sensitivity only\n"
        f"Ex {increments['Ex']['relative_l2']*100:.2f}%, "
        f"Hz {increments['Hz']['relative_l2']*100:.2f}%, "
        f"dBz/dt {increments['dBzdt']['relative_l2']*100:.2f}% "
        "(default DLF)\n"
        "reference stability not established -> "
        f"{audit['status']}"
    )
    ax.text(
        0.01,
        0.98,
        sensitivity_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#4D4D4D",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ECECEC",
            "edgecolor": "#8A8A8A",
            "hatch": "///",
            "alpha": 0.95,
        },
    )
    hz_false_reversal = _failed_hz_false_reversal_time(metrics)
    ax.text(
        0.99,
        0.98,
        (
            f"WARNING: Hz false reversal near {hz_false_reversal:.3f} s\n"
            "strict zero-crossing audit remains failed"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["warning"],
        weight="bold",
        fontsize=7.3,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF4F0",
            "edgecolor": COLORS["warning"],
        },
    )
    return _finish_or_save(fig, stem)


def _relative_l2(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if (
        numerator.shape != denominator.shape
        or not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
    ):
        raise ValueError("relative-L2 arrays must have equal shape and be finite")
    common_scale = max(
        float(np.max(np.abs(numerator), initial=0.0)),
        float(np.max(np.abs(denominator), initial=0.0)),
    )
    if common_scale == 0.0:
        raise ValueError("Debye-16 increment norm must be finite and non-zero")
    scaled_denominator_norm = float(
        np.linalg.norm(denominator / common_scale)
    )
    if scaled_denominator_norm == 0.0:
        raise ValueError("Debye-16 increment norm must be finite and non-zero")
    return float(
        np.linalg.norm(numerator / common_scale)
        / scaled_denominator_norm
    )


def _load_verification(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"times", "components", "fem"} <= set(archive.files):
            raise ValueError("Debye NPZ member set is incomplete")
        times = _require_positive_times(
            np.asarray(archive["times"], dtype=float).copy()
        )
        raw_components = np.asarray(archive["components"])
        if raw_components.ndim != 1:
            raise ValueError("Debye components must be a 1-D array")
        if np.issubdtype(raw_components.dtype, np.number) and not np.isfinite(
            raw_components
        ).all():
            raise ValueError("Debye components must be finite")
        components = [str(value) for value in raw_components]
        fem = np.asarray(archive["fem"], dtype=float).copy()
    required_components = [component for component, _ in COMPONENTS]
    if len(components) != len(set(components)) or not set(
        required_components
    ) <= set(components):
        raise ValueError("Debye components must uniquely include Ex, Hz, dBzdt")
    if fem.shape != (times.size, len(components)):
        raise ValueError("Debye field array shape is incomplete")
    if not np.isfinite(fem).all():
        raise ValueError("Debye field array must be finite")
    return times, components, fem


def plot_debye_order_diagnostic(
    noip_path: Path,
    ip4_path: Path,
    ip16_path: Path,
    stem: Path | None,
) -> dict[str, Any] | tuple[dict[str, Any], plt.Figure]:
    """Compare four versus sixteen Debye terms as an internal sensitivity test."""

    times, names, noip = _load_verification(noip_path)
    times4, names4, ip4 = _load_verification(ip4_path)
    times16, names16, ip16 = _load_verification(ip16_path)
    if (
        names4 != names
        or names16 != names
        or not np.array_equal(times4, times)
        or not np.array_equal(times16, times)
        or noip.shape != ip4.shape
        or noip.shape != ip16.shape
    ):
        raise ValueError("Debye-order files must share time, component, and field grids")
    times = _require_positive_times(times)

    result: dict[str, Any] = {
        "schema": "atem3d.zhou2020.debye-order-diagnostic/v2",
        "level": "S0T0B0",
        "sample_count": int(times.size),
        "comparison": {},
        "interpretation": (
            "Internal pole-count sensitivity only; this is not an empymod "
            "cross-code validation gate."
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    for ax, (component, label) in zip(axes, COMPONENTS):
        index = names.index(component)
        delta4 = np.asarray(ip4[:, index] - noip[:, index], dtype=float)
        delta16 = np.asarray(ip16[:, index] - noip[:, index], dtype=float)
        change = _relative_l2(delta4 - delta16, delta16)
        result["comparison"][component] = {
            "debye_4_vs_16_relative_l2": change,
        }
        ax.plot(
            times,
            _positive_magnitude(delta4),
            color=COLORS["fenicsx"],
            label="4 Debye",
        )
        ax.plot(
            times,
            _positive_magnitude(delta16),
            color=COLORS["ip"],
            label="16 Debye",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#D8D8D8", lw=0.45)
        ax.set_title(label)
        ax.set_xlabel("Time after turn-off (s)")
        ax.text(
            0.04,
            0.05,
            f"4 vs 16: {change*100:.2f}%",
            transform=ax.transAxes,
            va="bottom",
        )
    axes[0].set_ylabel("|IP increment| (S0T0B0)")
    axes[0].legend(loc="best")
    fig.suptitle(
        "Debye pole-count sensitivity: 4 terms relative to 16 terms",
        y=0.995,
    )
    returned = _finish_or_save(fig, stem, rect=(0.0, 0.0, 1.0, 0.94))
    if stem is None:
        assert returned is fig
        return result, fig
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--compute-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-audit",
        type=Path,
        required=True,
        help="Manifest-bound reference-transform audit directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _set_style()
    run = args.run
    validated_audit = load_reference_audit(args.reference_audit)
    _cross_bind_run_to_audit(run, validated_audit)
    audit = validated_audit["audit"]
    audit_arrays = validated_audit["arrays"]

    formal = run / "fenicsx"
    noip_fem = _read_csv(formal / "noip" / "S1T1B1" / "predictions.csv")
    ip_fem = _read_csv(formal / "ip" / "S1T1B1" / "predictions.csv")
    noip_ref = _read_csv(run / "reference" / "empymod_noip.csv")
    ip_ref = _read_csv(run / "reference" / "empymod_ip.csv")
    metrics = json.loads(
        (run / "comparisons" / "S1T1B1" / "strict_comparison.json").read_text(
            encoding="utf-8"
        )
    )

    def build_artifacts(staging: Path) -> None:
        plot_model(staging / FIGURE_STEMS[0])
        plot_total_fields(
            noip_fem,
            ip_fem,
            noip_ref,
            ip_ref,
            staging / FIGURE_STEMS[1],
        )
        plot_reference_stability(
            audit_arrays,
            audit,
            staging / FIGURE_STEMS[2],
        )
        plot_gate_summary(metrics, audit, staging / FIGURE_STEMS[3])

        diagnostic = plot_debye_order_diagnostic(
            args.compute_root / "noip-S0T0B0" / "verification_data.npz",
            args.compute_root / "ip-S0T0B0-n4" / "verification_data.npz",
            args.compute_root / "ip-S0T0B0-n16" / "verification_data.npz",
            staging / FIGURE_STEMS[4],
        )
        assert isinstance(diagnostic, dict)
        _write_json(staging / DEBYE_JSON_NAME, diagnostic)

    metadata = {
        "run_id": run.name,
        "spatial_case": "S1T1B1",
        "input_sha256": validated_audit["manifest"]["input_sha256"],
        "reference_audit_status": audit["status"],
        "qwe_converged": bool(audit["qwe"]["converged"]),
    }
    _publish_bundle(
        args.output / PUBLISHED_BUNDLE_NAME,
        build_artifacts,
        metadata,
    )


if __name__ == "__main__":
    main()
