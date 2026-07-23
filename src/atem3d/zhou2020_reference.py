"""Independent empymod reference workflow for the Zhou 2020 benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .empymod_compare import (
    EmpymodSurvey,
    make_exact_pelton_resistivity_model,
    run_empymod_reference,
)
from .sotem_benchmark import (
    BenchmarkCase,
    load_benchmark_case,
    load_benchmark_provenance,
)


COMPONENTS = ("Ex", "Hz", "dBzdt")
COLUMNS = (
    "time_s",
    "Ex_V_per_m",
    "Hz_A_per_m",
    "dBzdt_T_per_s",
)
REFERENCE_SCHEMA = "atem3d.zhou2020.reference-manifest/v1"
SOURCE_CONVERGENCE_GATE = 0.005


def build_zhou_empymod_survey(
    case_path: str | Path,
    *,
    variant: str,
    surface_offset_m: float = 0.0,
) -> EmpymodSurvey:
    """Build the canonical finite-bipole empymod survey."""

    if variant not in {"noip", "ip"}:
        raise ValueError("variant must be 'noip' or 'ip'")
    offset = float(surface_offset_m)
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError("surface_offset_m must be finite and non-negative")

    case = load_benchmark_case(case_path)
    if case.case_id != "zhou2020_grounded_wire":
        raise ValueError("case must be zhou2020_grounded_wire")
    if case.validation_role != "strict_primary":
        raise ValueError("Zhou case must have strict_primary validation role")
    layers = tuple(case.earth.get("layers", ()))
    if len(layers) != 3:
        raise ValueError("Zhou reference requires exactly three earth layers")
    depths = [0.0]
    depths.extend(
        float(layer["bottom_m"])
        for layer in layers[:-1]
    )
    resistivities = [float(case.rho_air_ohm_m)]
    resistivities.extend(float(layer["rho_ohm_m"]) for layer in layers)

    if variant == "ip":
        polarization = case.polarization
        if polarization is None:
            raise ValueError("IP variant requires a polarization definition")
        middle = layers[1]
        if (
            float(middle["top_m"]) != float(polarization["top_m"])
            or float(middle["bottom_m"]) != float(polarization["bottom_m"])
            or float(middle["rho_ohm_m"]) != float(polarization["rho0_ohm_m"])
        ):
            raise ValueError("polarization interval must match the second layer")
        resistivity_model: Sequence[float] | dict[str, Any]
        resistivity_model = make_exact_pelton_resistivity_model(
            rho0=resistivities,
            chargeability=[
                0.0,
                0.0,
                float(polarization["m"]),
                0.0,
            ],
            tau=[
                1.0,
                1.0,
                float(polarization["tau_s"]),
                1.0,
            ],
            c=[
                1.0,
                1.0,
                float(polarization["c"]),
                1.0,
            ],
        )
    else:
        resistivity_model = resistivities

    start = tuple(case.source_start_down[:2]) + (
        float(case.source_start_down[2]) + offset,
    )
    end = tuple(case.source_end_down[:2]) + (
        float(case.source_end_down[2]) + offset,
    )
    receiver = tuple(case.receiver_down[:2]) + (
        float(case.receiver_down[2]) + offset,
    )
    return EmpymodSurvey(
        source_start=start,
        source_end=end,
        receiver_locations=[receiver],
        components=COMPONENTS,
        times=case.observation_times,
        depths=depths,
        resistivities=resistivity_model,
        strength=case.current_a,
        signal=-1,
        coordinate_system="depth_down",
    )


def run_reference_sweep(
    *,
    case_path: str | Path,
    provenance_path: str | Path,
    output_dir: str | Path,
    srcpts_values: Sequence[int] = (3, 5, 9, 17),
    surface_offsets_m: Sequence[float] = (0.0, 0.05, 0.1, 0.2),
    backend=None,
    empymod_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and publish the strict Zhou empymod reference sweep."""

    case_file = Path(case_path)
    provenance_file = Path(provenance_path)
    case = load_benchmark_case(case_file)
    provenance = load_benchmark_provenance(
        provenance_file,
        case_path=case_file,
    )
    srcpts = _strictly_increasing_positive_ints(srcpts_values, "srcpts_values")
    offsets = _strictly_increasing_nonnegative(surface_offsets_m)
    if 0.0 not in offsets:
        raise ValueError("surface_offsets_m must include the canonical zero offset")
    kwargs = dict(empymod_kwargs or {})
    kwargs.setdefault("recpts", 1)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("reference output directory must be empty")

    responses: dict[str, dict[int, np.ndarray]] = {"noip": {}, "ip": {}}
    for variant in ("noip", "ip"):
        survey = build_zhou_empymod_survey(
            case_file,
            variant=variant,
            surface_offset_m=0.0,
        )
        for value in srcpts:
            responses[variant][value] = run_empymod_reference(
                survey,
                backend=backend,
                srcpts=value,
                **kwargs,
            )

    convergence = _source_convergence(
        times=case.observation_times,
        responses=responses,
        srcpts=srcpts,
    )

    final_srcpts = srcpts[-1]
    _write_response_csv(
        output / "empymod_noip.csv",
        case.observation_times,
        responses["noip"][final_srcpts],
    )
    _write_response_csv(
        output / "empymod_ip.csv",
        case.observation_times,
        responses["ip"][final_srcpts],
    )
    _atomic_write_json(
        output / "empymod_srcpts_convergence.json",
        convergence,
    )

    offset_responses: dict[str, list[np.ndarray]] = {"noip": [], "ip": []}
    for variant in ("noip", "ip"):
        for offset in offsets:
            if offset == 0.0:
                values = responses[variant][final_srcpts]
            else:
                survey = build_zhou_empymod_survey(
                    case_file,
                    variant=variant,
                    surface_offset_m=offset,
                )
                values = run_empymod_reference(
                    survey,
                    backend=backend,
                    srcpts=final_srcpts,
                    **kwargs,
                )
            offset_responses[variant].append(values)
    offset_sensitivity = _surface_offset_sensitivity(
        times=case.observation_times,
        offsets=offsets,
        responses=offset_responses,
    )
    _atomic_write_json(
        output / "surface_offset_sensitivity.json",
        offset_sensitivity,
    )

    metadata = {
        "schema": "atem3d.zhou2020.empymod-metadata/v2",
        "case_id": case.case_id,
        "reference_modes": ["noip", "exact_pelton_ip"],
        "components": list(COMPONENTS),
        "units": {
            "Ex": "V/m",
            "Hz": "A/m",
            "dBzdt": "T/s",
        },
        "component_conventions": {
            "Ex": {
                "empymod_receiver": "electric",
                "empymod_signal": -1,
                "scale": "1",
                "source_waveform": "ideal_step_off",
            },
            "Hz": {
                "empymod_receiver": "H",
                "empymod_signal": -1,
                "scale": "1",
                "source_waveform": "ideal_step_off",
            },
            "dBzdt": {
                "empymod_receiver": "H",
                "empymod_signal": 0,
                "scale": "-mu0",
                "source_waveform": "ideal_step_off",
            },
        },
        "srcpts_values": list(srcpts),
        "selected_srcpts": final_srcpts,
        "recpts": int(kwargs["recpts"]),
        "surface_offsets_m": list(offsets),
        "empymod_version": _backend_version(backend),
        "transform_options": {
            "mode": "empymod_defaults"
            if set(kwargs) == {"recpts"}
            else _json_safe(kwargs),
        },
        "case_file_sha256": provenance["case_file_sha256"],
        "provenance_schema": provenance["schema"],
        "exact_ip_material": True,
        "debye_fit_used": False,
    }
    _atomic_write_json(output / "empymod_metadata.json", metadata)

    status = (
        "reference_verified"
        if convergence["passed"]
        else "failed_with_reproducible_evidence"
    )
    files = (
        "empymod_noip.csv",
        "empymod_ip.csv",
        "empymod_srcpts_convergence.json",
        "surface_offset_sensitivity.json",
        "empymod_metadata.json",
    )
    manifest = {
        "schema": REFERENCE_SCHEMA,
        "case_id": case.case_id,
        "status": status,
        "file_sha256": {
            name: _sha256_file(output / name)
            for name in files
        },
    }
    _atomic_write_json(output / "reference_manifest.json", manifest)
    return {
        "status": status,
        "source_convergence": convergence,
        "surface_offset_sensitivity": offset_sensitivity,
        "manifest": manifest,
    }


def _source_convergence(
    *,
    times: np.ndarray,
    responses: Mapping[str, Mapping[int, np.ndarray]],
    srcpts: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = True
    for variant, by_srcpts in responses.items():
        for left, right in zip(srcpts[:-1], srcpts[1:]):
            metrics = _component_change(
                times,
                by_srcpts[left],
                by_srcpts[right],
            )
            pair_passed = all(
                value <= SOURCE_CONVERGENCE_GATE
                for value in metrics.values()
            )
            if (left, right) == (srcpts[-2], srcpts[-1]):
                passed = passed and pair_passed
            rows.append(
                {
                    "variant": variant,
                    "srcpts_left": int(left),
                    "srcpts_right": int(right),
                    "max_robust_relative_change": metrics,
                    "passed": pair_passed,
                }
            )
    return {
        "schema": "atem3d.zhou2020.source-convergence/v1",
        "srcpts_values": [int(value) for value in srcpts],
        "gate": SOURCE_CONVERGENCE_GATE,
        "selected_pair": [int(srcpts[-2]), int(srcpts[-1])],
        "rows": rows,
        "passed": bool(passed),
    }


def _surface_offset_sensitivity(
    *,
    times: np.ndarray,
    offsets: Sequence[float],
    responses: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, values in responses.items():
        canonical = values[0]
        for offset, shifted in zip(offsets, values):
            rows.append(
                {
                    "variant": variant,
                    "offset_m": float(offset),
                    "change_from_zero": _component_change(
                        times,
                        canonical,
                        shifted,
                    ),
                }
            )
    return {
        "schema": "atem3d.zhou2020.surface-offset-sensitivity/v1",
        "offsets_m": [float(value) for value in offsets],
        "rows": rows,
    }


def _component_change(
    times: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    del times
    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("reference arrays must have matching shapes")
    if candidate_values.ndim != 2 or candidate_values.shape[1] != len(COMPONENTS):
        raise ValueError("reference arrays must be n_times by three components")
    metrics: dict[str, float] = {}
    for index, component in enumerate(COMPONENTS):
        ref = reference_values[:, index]
        peak = float(np.max(np.abs(ref)))
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError(f"reference peak for {component} must be positive")
        denominator = np.maximum(np.abs(ref), 0.01 * peak)
        metrics[component] = float(
            np.max(np.abs(candidate_values[:, index] - ref) / denominator)
        )
    return metrics


def _write_response_csv(
    path: Path,
    times: np.ndarray,
    values: np.ndarray,
) -> None:
    time_values = np.asarray(times, dtype=float)
    response = np.asarray(values, dtype=float)
    if response.shape != (time_values.size, len(COMPONENTS)):
        raise ValueError("response shape does not match canonical components")
    if not np.isfinite(time_values).all() or not np.isfinite(response).all():
        raise ValueError("reference response must contain only finite values")
    rows = [
        {
            COLUMNS[0]: f"{time:.17g}",
            COLUMNS[1]: f"{row[0]:.17g}",
            COLUMNS[2]: f"{row[1]:.17g}",
            COLUMNS[3]: f"{row[2]:.17g}",
        }
        for time, row in zip(time_values, response)
    ]
    _atomic_write_csv(path, COLUMNS, rows)


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_safe(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _strictly_increasing_positive_ints(
    values: Sequence[int],
    name: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) < 2 or any(type(value) is not int or value <= 0 for value in result):
        raise ValueError(f"{name} must contain at least two positive integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _strictly_increasing_nonnegative(
    values: Sequence[float],
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("surface_offsets_m must contain finite non-negative values")
    if tuple(sorted(set(result))) != result:
        raise ValueError("surface_offsets_m must be strictly increasing")
    return result


def _backend_version(backend) -> str:
    if backend is not None:
        return str(getattr(backend, "__version__", type(backend).__name__))
    return importlib.metadata.version("empymod")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON evidence cannot contain non-finite values")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported JSON evidence type: {type(value).__qualname__}")
