"""Publish an immutable Zhou 2020 DLF/QWE reference-stability audit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
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
    run_empymod_reference,
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


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
) -> dict[str, Any]:
    """Build and atomically publish audit evidence without changing the formal run."""

    run = Path(run)
    output = Path(output)
    strict = run / "comparisons/S1T1B1/strict_comparison.json"
    strict_before = sha256_file(strict)
    audit = build_reference_stability_audit(
        times=times,
        default_dlf=default_dlf,
        separate_total_qwe=separate_total_qwe,
        direct_frequency_qwe=direct_frequency_qwe,
        direct_qwe_converged=direct_qwe_converged,
        fenicsx_increment=fenicsx_increment,
        consecutive=consecutive,
    )
    audit["input_sha256"] = {"strict_comparison.json": strict_before}
    if method_metadata is not None:
        audit["methods"] = dict(method_metadata)

    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_npz(
        output / "reference_stability.npz",
        time_s=np.asarray(times),
        default_dlf=np.asarray(default_dlf),
        separate_total_qwe=np.asarray(separate_total_qwe),
        direct_frequency_qwe=np.asarray(direct_frequency_qwe),
        fenicsx_increment=np.asarray(fenicsx_increment),
    )
    _atomic_write_json(output / "reference_stability.json", audit)
    if sha256_file(strict) != strict_before:
        raise RuntimeError("strict_comparison.json changed during audit publication")
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


def _separate_total_qwe(noip_survey, ip_survey, *, srcpts: int) -> np.ndarray:
    """Transform totals separately and subtract IP minus no-IP."""

    kwargs = {
        "ft": "qwe",
        "ftarg": dict(QWE),
        "srcpts": srcpts,
        "recpts": 1,
    }
    noip = replace(noip_survey, components=(COMPONENT,))
    ip = replace(ip_survey, components=(COMPONENT,))
    noip_values = run_empymod_reference(noip, **kwargs)[:, 0]
    ip_values = run_empymod_reference(ip, **kwargs)[:, 0]
    return ip_values - noip_values


def _direct_frequency_qwe(
    noip_survey,
    ip_survey,
    *,
    srcpts: int,
) -> tuple[np.ndarray, bool]:
    """Subtract frequency responses before the canonical QWE time transform."""

    from empymod.model import tem  # noqa: PLC0415
    from empymod.utils import check_time  # noqa: PLC0415

    signal, scale = _component_signal_and_scale(
        COMPONENT,
        noip_survey.signal,
        noip_survey.coordinate_system,
    )
    times, frequencies, ft, ftarg = check_time(
        noip_survey.times,
        signal,
        "qwe",
        dict(QWE),
        0,
    )
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
    transformed, converged = tem(
        (ip_frequency - noip_frequency)[:, None],
        np.ones(1),
        frequencies,
        times,
        signal,
        ft,
        ftarg,
    )
    values = scale * np.asarray(transformed[:, 0], dtype=float)
    return values, converged


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
    separate_total_qwe = _separate_total_qwe(
        noip_survey,
        ip_survey,
        srcpts=srcpts,
    )
    direct_frequency_qwe, direct_qwe_converged = _direct_frequency_qwe(
        noip_survey,
        ip_survey,
        srcpts=srcpts,
    )

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
        direct_qwe_converged=direct_qwe_converged,
        fenicsx_increment=fenicsx_increment,
        method_metadata=methods,
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
