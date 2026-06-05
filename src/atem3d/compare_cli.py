"""Compare ATEM3D receiver data against empymod reference data."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_NUMBA_CACHE_DIR = Path.cwd() / ".numba_cache"
if not os.environ.get("NUMBA_CACHE_DIR"):
    _DEFAULT_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_DEFAULT_NUMBA_CACHE_DIR.resolve())

import h5py
import numpy as np
from scipy.constants import mu_0
import yaml

from .config import build_simulation
from .empymod_compare import (
    EmpymodSurvey,
    build_empymod_survey_from_result,
    make_debye_resistivity_model_from_config,
    run_empymod_reference,
)
from .metrics import fit_linear_response_components, summarize_errors
from .validation import ValidationCase, write_validation_report


@dataclass(frozen=True)
class ComponentPolarizationScaleFit:
    """Diagnostic component-wise polarization-current fit from saved fields."""

    weights: np.ndarray
    fitted_data: np.ndarray
    ohmic_data: np.ndarray
    polarization_components: list[np.ndarray]
    rank: int
    singular_values: np.ndarray


@dataclass(frozen=True)
class MemoryPolarizationScaleFit:
    """Diagnostic current-memory and initial-memory polarization-current fit."""

    weights: np.ndarray
    fitted_data: np.ndarray
    ohmic_data: np.ndarray
    current_memory_data: np.ndarray
    initial_memory_data: np.ndarray
    rank: int
    singular_values: np.ndarray


@dataclass(frozen=True)
class PerReceiverMemoryPolarizationScaleFit:
    """Independent memory-basis fit for each receiver/component column."""

    weights: np.ndarray
    fitted_data: np.ndarray
    ohmic_data: np.ndarray
    current_memory_data: np.ndarray
    initial_memory_data: np.ndarray
    ranks: np.ndarray
    singular_values: list[np.ndarray]


def main(argv: list[str] | None = None) -> int:
    _ensure_numba_cache_dir()
    parser = argparse.ArgumentParser(description="Compare ATEM3D HDF5 output with empymod.")
    parser.add_argument("result", type=Path, help="ATEM3D HDF5 output")
    parser.add_argument("--depths", nargs="+", type=float, required=True)
    parser.add_argument("--resistivities", nargs="+", type=float, required=True)
    parser.add_argument("--signal", type=int, default=-1)
    parser.add_argument("--srcpts", type=int, default=51, help="empymod source integration points")
    parser.add_argument("--recpts", type=int, default=1, help="empymod receiver integration points")
    parser.add_argument("--empymod-strength", type=float, default=None, help="Override source strength passed to empymod")
    parser.add_argument("--use-config-ip", action="store_true", help="Use layered Debye IP terms from the result config for empymod")
    parser.add_argument(
        "--recompute-current-biot",
        action="store_true",
        help="Recompute receiver data from saved e/b fields using magnetic_receiver_mode='current_biot'",
    )
    parser.add_argument(
        "--recompute-edge-current-biot",
        action="store_true",
        help=(
            "Recompute receiver data from saved e/b fields using "
            "magnetic_receiver_mode='edge_current_biot'"
        ),
    )
    parser.add_argument(
        "--recompute-edge-basis-biot",
        action="store_true",
        help=(
            "Recompute receiver data from saved e/b fields using "
            "magnetic_receiver_mode='edge_basis_biot'"
        ),
    )
    parser.add_argument(
        "--recompute-edge-basis-cell-biot",
        action="store_true",
        help=(
            "Recompute receiver data from saved e/b fields using "
            "magnetic_receiver_mode='edge_basis_cell_biot'"
        ),
    )
    parser.add_argument(
        "--magnetic-recovery-subdivisions",
        type=int,
        default=None,
        help="Override current_biot midpoint subdivisions when --recompute-current-biot is used",
    )
    parser.add_argument(
        "--magnetic-recovery-polarization-scale",
        type=_polarization_scale_arg,
        default=None,
        help=(
            "Diagnostic scale for Debye polarization current in recomputed "
            "current_biot data; use a nonnegative number, 'low_frequency_ratio', "
            "or comma-separated component scales sx,sy,sz"
        ),
    )
    parser.add_argument(
        "--magnetic-recovery-initial-polarization-scale",
        type=float,
        default=None,
        help=(
            "Diagnostic initial Debye memory scale for recomputed magnetic "
            "recovery; finite values are allowed"
        ),
    )
    parser.add_argument(
        "--magnetic-recovery-source-primary-delta6",
        action="store_true",
        help=(
            "Diagnostic only: add -6*mu0*delta_sigma*L^2*H_wire*exp(-t/(2*tau)) "
            "to recomputed magnetic H receiver data"
        ),
    )
    parser.add_argument(
        "--fit-magnetic-recovery-component-scale",
        action="store_true",
        help=(
            "Diagnostic only: fit sx,sy,sz polarization-current weights against "
            "the empymod reference after recomputing current_biot basis responses"
        ),
    )
    parser.add_argument(
        "--fit-magnetic-recovery-memory-scale",
        action="store_true",
        help=(
            "Diagnostic only: fit current-memory and initial-memory "
            "polarization-current weights against the empymod reference"
        ),
    )
    parser.add_argument(
        "--fit-magnetic-recovery-memory-scale-per-receiver",
        action="store_true",
        help=(
            "Diagnostic only: fit current/initial memory weights independently "
            "for each receiver column"
        ),
    )
    parser.add_argument("--include-t0", action="store_true", help="Include t=0 in the error comparison")
    parser.add_argument(
        "--skip-positive-times",
        type=int,
        default=0,
        help="Skip this many positive-time samples after applying the t=0 filter",
    )
    parser.add_argument(
        "--receiver-indices",
        nargs="+",
        type=int,
        default=None,
        help="Compare only these zero-based receiver/component columns from the result file",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG comparison plot path")
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include per-time numerical/reference samples in the JSON report",
    )
    args = parser.parse_args(argv)
    if args.skip_positive_times < 0:
        parser.error("--skip-positive-times must be non-negative")

    with h5py.File(args.result, "r") as h5:
        times = h5["times"][:]
        numerical = h5["data"][:]
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))
    receiver_indices = None
    if args.receiver_indices is not None:
        receiver_indices = [int(index) for index in args.receiver_indices]
        n_components = numerical.shape[1]
        if any(index < 0 or index >= n_components for index in receiver_indices):
            parser.error("--receiver-indices contains an out-of-range column")
    if args.magnetic_recovery_subdivisions is not None and args.magnetic_recovery_subdivisions <= 0:
        parser.error("--magnetic-recovery-subdivisions must be positive")
    if (
        args.magnetic_recovery_initial_polarization_scale is not None
        and not np.isfinite(args.magnetic_recovery_initial_polarization_scale)
    ):
        parser.error("--magnetic-recovery-initial-polarization-scale must be finite")
    if (
        int(bool(args.recompute_current_biot))
        + int(bool(args.recompute_edge_current_biot))
        + int(bool(args.recompute_edge_basis_biot))
        + int(bool(args.recompute_edge_basis_cell_biot))
    ) > 1:
        parser.error(
            "--recompute-current-biot, --recompute-edge-current-biot, "
            "--recompute-edge-basis-biot, and --recompute-edge-basis-cell-biot "
            "are mutually exclusive"
        )
    if args.fit_magnetic_recovery_component_scale and not args.recompute_current_biot:
        parser.error("--fit-magnetic-recovery-component-scale requires --recompute-current-biot")
    if args.fit_magnetic_recovery_memory_scale and not args.recompute_current_biot:
        parser.error("--fit-magnetic-recovery-memory-scale requires --recompute-current-biot")
    if (
        args.fit_magnetic_recovery_memory_scale_per_receiver
        and not args.recompute_current_biot
    ):
        parser.error(
            "--fit-magnetic-recovery-memory-scale-per-receiver requires "
            "--recompute-current-biot"
        )
    if sum(
        bool(flag)
        for flag in (
            args.fit_magnetic_recovery_component_scale,
            args.fit_magnetic_recovery_memory_scale,
            args.fit_magnetic_recovery_memory_scale_per_receiver,
        )
    ) > 1:
        parser.error(
            "magnetic recovery fit flags are mutually exclusive"
        )
    if (
        args.fit_magnetic_recovery_component_scale
        and (
            args.recompute_edge_current_biot
            or args.recompute_edge_basis_biot
            or args.recompute_edge_basis_cell_biot
        )
    ):
        parser.error(
            "--fit-magnetic-recovery-component-scale is only implemented for "
            "--recompute-current-biot"
        )
    if (
        args.fit_magnetic_recovery_memory_scale
        and (
            args.recompute_edge_current_biot
            or args.recompute_edge_basis_biot
            or args.recompute_edge_basis_cell_biot
        )
    ):
        parser.error(
            "--fit-magnetic-recovery-memory-scale is only implemented for "
            "--recompute-current-biot"
        )
    if (
        args.fit_magnetic_recovery_memory_scale_per_receiver
        and (
            args.recompute_edge_current_biot
            or args.recompute_edge_basis_biot
            or args.recompute_edge_basis_cell_biot
        )
    ):
        parser.error(
            "--fit-magnetic-recovery-memory-scale-per-receiver is only implemented for "
            "--recompute-current-biot"
        )
    if (
        (
            args.fit_magnetic_recovery_component_scale
            or args.fit_magnetic_recovery_memory_scale
            or args.fit_magnetic_recovery_memory_scale_per_receiver
        )
        and args.magnetic_recovery_polarization_scale is not None
    ):
        parser.error(
            "magnetic recovery fit flags cannot be combined with "
            "--magnetic-recovery-polarization-scale"
        )
    if (
        (
            args.fit_magnetic_recovery_component_scale
            or args.fit_magnetic_recovery_memory_scale
            or args.fit_magnetic_recovery_memory_scale_per_receiver
        )
        and args.magnetic_recovery_initial_polarization_scale is not None
    ):
        parser.error(
            "magnetic recovery fit flags cannot be combined with "
            "--magnetic-recovery-initial-polarization-scale"
        )
    recompute_biot = (
        args.recompute_current_biot
        or args.recompute_edge_current_biot
        or args.recompute_edge_basis_biot
        or args.recompute_edge_basis_cell_biot
    )
    fit_requested = (
        args.fit_magnetic_recovery_component_scale
        or args.fit_magnetic_recovery_memory_scale
        or args.fit_magnetic_recovery_memory_scale_per_receiver
    )
    if args.magnetic_recovery_source_primary_delta6 and not recompute_biot:
        parser.error(
            "--magnetic-recovery-source-primary-delta6 requires a recomputed "
            "magnetic receiver mode"
        )
    if args.magnetic_recovery_source_primary_delta6 and fit_requested:
        parser.error(
            "--magnetic-recovery-source-primary-delta6 cannot be combined with "
            "magnetic recovery fit flags"
        )
    if recompute_biot and not fit_requested:
        recompute_kwargs = {
            "subdivisions": args.magnetic_recovery_subdivisions,
            "receiver_indices": receiver_indices,
        }
        if args.recompute_edge_current_biot:
            recompute_kwargs["magnetic_receiver_mode"] = "edge_current_biot"
        elif args.recompute_edge_basis_biot:
            recompute_kwargs["magnetic_receiver_mode"] = "edge_basis_biot"
        elif args.recompute_edge_basis_cell_biot:
            recompute_kwargs["magnetic_receiver_mode"] = "edge_basis_cell_biot"
        if args.magnetic_recovery_polarization_scale is not None:
            recompute_kwargs["polarization_scale"] = args.magnetic_recovery_polarization_scale
        if args.magnetic_recovery_initial_polarization_scale is not None:
            recompute_kwargs["initial_polarization_scale"] = (
                args.magnetic_recovery_initial_polarization_scale
            )
        numerical = recompute_current_biot_receiver_data(args.result, **recompute_kwargs)

    mask = np.ones(times.shape, dtype=bool)
    if not args.include_t0:
        mask = times > 0.0
    if args.skip_positive_times:
        selected = np.flatnonzero(mask)
        mask[selected[: args.skip_positive_times]] = False
    survey, names = build_empymod_survey_from_result(
        args.result,
        depths=args.depths,
        resistivities=args.resistivities,
        signal=args.signal,
    )
    if args.receiver_indices is not None:
        names = [names[index] for index in receiver_indices]
        if not recompute_biot:
            numerical = numerical[:, receiver_indices]
        receiver_components = list(survey.receiver_components or [])
        receiver_components = [receiver_components[index] for index in receiver_indices]
    else:
        receiver_components = survey.receiver_components
    if args.magnetic_recovery_source_primary_delta6:
        numerical = _apply_source_primary_delta6_correction(
            numerical,
            times=times,
            config=config,
            receiver_components=list(receiver_components or []),
        )
    report_numerical = numerical if recompute_biot else None
    empymod_strength = survey.strength if args.empymod_strength is None else float(args.empymod_strength)
    survey = EmpymodSurvey(
        source_start=survey.source_start,
        source_end=survey.source_end,
        receiver_locations=survey.receiver_locations,
        components=survey.components,
        times=survey.times[mask],
        depths=survey.depths,
        resistivities=(
            make_debye_resistivity_model_from_config(config, args.depths)
            if args.use_config_ip
            else survey.resistivities
        ),
        strength=empymod_strength,
        signal=survey.signal,
        receiver_components=receiver_components,
        coordinate_system=survey.coordinate_system,
    )
    numerical = numerical[mask]
    empymod_kwargs = {"srcpts": args.srcpts, "recpts": args.recpts}
    _ensure_numba_cache_dir()
    reference = run_empymod_reference(survey, **empymod_kwargs)
    fit_result = None
    memory_fit_result = None
    per_receiver_memory_fit_result = None
    if args.fit_magnetic_recovery_component_scale:
        fit_result = fit_recomputed_component_polarization_scale(
            args.result,
            reference,
            fit_mask=mask,
            subdivisions=args.magnetic_recovery_subdivisions,
            receiver_indices=receiver_indices,
        )
        report_numerical = fit_result.fitted_data
        numerical = fit_result.fitted_data[mask]
    if args.fit_magnetic_recovery_memory_scale:
        memory_fit_result = fit_recomputed_memory_polarization_scale(
            args.result,
            reference,
            fit_mask=mask,
            subdivisions=args.magnetic_recovery_subdivisions,
            receiver_indices=receiver_indices,
        )
        report_numerical = memory_fit_result.fitted_data
        numerical = memory_fit_result.fitted_data[mask]
    if args.fit_magnetic_recovery_memory_scale_per_receiver:
        per_receiver_memory_fit_result = (
            fit_recomputed_memory_polarization_scale_per_receiver(
                args.result,
                reference,
                fit_mask=mask,
                subdivisions=args.magnetic_recovery_subdivisions,
                receiver_indices=receiver_indices,
            )
        )
        report_numerical = per_receiver_memory_fit_result.fitted_data
        numerical = per_receiver_memory_fit_result.fitted_data[mask]
    summary = summarize_errors(numerical, reference, names)
    if args.plot is not None:
        _write_comparison_plot(args.plot, survey.times, numerical, reference, names)
    if args.output is not None:
        metadata = {
            "empymod": {
                "depths": list(args.depths),
                "resistivities": list(args.resistivities),
                "signal": args.signal,
                "srcpts": args.srcpts,
                "recpts": args.recpts,
                "strength": survey.strength,
                "use_config_ip": args.use_config_ip,
                "coordinate_system": survey.coordinate_system,
                "skip_positive_times": args.skip_positive_times,
                "receiver_indices": receiver_indices,
            }
        }
        if recompute_biot:
            if args.recompute_edge_current_biot:
                magnetic_receiver_mode = "edge_current_biot"
            elif args.recompute_edge_basis_biot:
                magnetic_receiver_mode = "edge_basis_biot"
            elif args.recompute_edge_basis_cell_biot:
                magnetic_receiver_mode = "edge_basis_cell_biot"
            else:
                magnetic_receiver_mode = "current_biot"
            metadata["atem3d"] = {
                "recomputed_current_biot": bool(args.recompute_current_biot),
                "recomputed_edge_current_biot": bool(args.recompute_edge_current_biot),
                "recomputed_edge_basis_biot": bool(args.recompute_edge_basis_biot),
                "recomputed_edge_basis_cell_biot": bool(
                    args.recompute_edge_basis_cell_biot
                ),
                "magnetic_receiver_mode": magnetic_receiver_mode,
                "magnetic_recovery_subdivisions": (
                    int(args.magnetic_recovery_subdivisions)
                    if args.magnetic_recovery_subdivisions is not None
                    else int(config.get("magnetic_recovery_subdivisions", 1))
                ),
                "magnetic_recovery_polarization_scale": _metadata_polarization_scale(
                    config,
                    args.magnetic_recovery_polarization_scale,
                    fit_result,
                ),
                "magnetic_recovery_initial_polarization_scale": (
                    float(args.magnetic_recovery_initial_polarization_scale)
                    if args.magnetic_recovery_initial_polarization_scale is not None
                    else float(
                        config.get(
                            "magnetic_recovery_initial_polarization_scale",
                            0.0,
                        )
                    )
                ),
            }
            if args.magnetic_recovery_source_primary_delta6:
                metadata["atem3d"]["magnetic_recovery_source_primary_correction"] = {
                    "diagnostic_only": True,
                    "kind": "delta6",
                    "formula": "-6 * mu0 * delta_sigma * L^2 * H_wire * exp(-t / (2 tau))",
                }
            if fit_result is not None:
                metadata["atem3d"]["magnetic_recovery_polarization_scale"] = (
                    fit_result.weights.tolist()
                )
                metadata["atem3d"]["magnetic_recovery_component_fit"] = {
                    "diagnostic_only": True,
                    "weights": fit_result.weights.tolist(),
                    "rank": int(fit_result.rank),
                    "singular_values": fit_result.singular_values.tolist(),
                }
            if memory_fit_result is not None:
                metadata["atem3d"]["magnetic_recovery_polarization_scale"] = (
                    float(memory_fit_result.weights[0])
                )
                metadata["atem3d"]["magnetic_recovery_initial_polarization_scale"] = (
                    float(memory_fit_result.weights[1])
                )
                metadata["atem3d"]["magnetic_recovery_memory_fit"] = {
                    "diagnostic_only": True,
                    "formula": "H = H_ohmic - lambda H_delta_y(t) + gamma H_delta_y(0-)",
                    "weights": memory_fit_result.weights.tolist(),
                    "rank": int(memory_fit_result.rank),
                    "singular_values": memory_fit_result.singular_values.tolist(),
                }
            if per_receiver_memory_fit_result is not None:
                metadata["atem3d"]["magnetic_recovery_polarization_scale"] = (
                    "per_receiver"
                )
                metadata["atem3d"]["magnetic_recovery_initial_polarization_scale"] = (
                    "per_receiver"
                )
                metadata["atem3d"]["magnetic_recovery_memory_fit_per_receiver"] = {
                    name: {
                        "diagnostic_only": True,
                        "formula": (
                            "H = H_ohmic - lambda H_delta_y(t) "
                            "+ gamma H_delta_y(0-)"
                        ),
                        "weights": per_receiver_memory_fit_result.weights[
                            index
                        ].tolist(),
                        "rank": int(per_receiver_memory_fit_result.ranks[index]),
                        "singular_values": per_receiver_memory_fit_result.singular_values[
                            index
                        ].tolist(),
                    }
                    for index, name in enumerate(names)
                }
        write_validation_report(
            ValidationCase(
                args.result,
                reference=reference,
                component_names=names,
                positive_times_only=not args.include_t0,
                skip_positive_times=args.skip_positive_times,
                metadata=metadata,
                component_indices=None if report_numerical is not None else receiver_indices,
                include_samples=args.include_samples,
                numerical=report_numerical,
            ),
            args.output,
        )
    for name, values in summary.items():
        print(
            f"{name}: relative_l2={values['relative_l2']:.6e}, "
            f"relative_linf={values['relative_linf']:.6e}"
        )
    return 0


def _apply_source_primary_delta6_correction(
    numerical: np.ndarray,
    *,
    times: np.ndarray,
    config: dict,
    receiver_components,
) -> np.ndarray:
    """Add the diagnostic ``-6 mu0 delta_sigma L^2`` source-primary correction."""

    terms = _source_primary_delta6_terms(config)
    if not terms:
        return np.asarray(numerical, dtype=float).copy()

    try:
        from geoana.em.static import LineCurrentWholeSpace  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError(
            "--magnetic-recovery-source-primary-delta6 requires geoana"
        ) from err

    source_cfg = config["source"]
    start = np.asarray(source_cfg["start"], dtype=float)
    end = np.asarray(source_cfg["end"], dtype=float)
    length = float(np.linalg.norm(end - start))
    if length == 0.0:
        raise ValueError("source start and end must be distinct")
    current = float(source_cfg.get("current", 1.0))
    line_current = LineCurrentWholeSpace(
        np.vstack([start, end]),
        current=current,
        mu=mu_0,
    )

    locations = np.array([location for location, _ in receiver_components], dtype=float)
    if locations.size == 0:
        return np.asarray(numerical, dtype=float).copy()
    h_wire = np.asarray(line_current.magnetic_field(locations), dtype=float)
    times = np.asarray(times, dtype=float)
    kernel = np.zeros(times.shape, dtype=float)
    for delta_sigma, tau in terms:
        kernel += (
            -6.0
            * mu_0
            * float(delta_sigma)
            * length**2
            * np.exp(-times / (2.0 * float(tau)))
        )

    corrected = np.asarray(numerical, dtype=float).copy()
    component_index = {"x": 0, "y": 1, "z": 2}
    for column, (_, component) in enumerate(receiver_components):
        normalized = str(component).strip()
        if len(normalized) < 2:
            continue
        family = normalized[0].upper()
        axis = normalized[-1].lower()
        if family not in {"H", "B"} or axis not in component_index:
            continue
        values = h_wire[:, component_index[axis]]
        if family == "B":
            values = mu_0 * values
        corrected[:, column] += kernel * values[column]
    return corrected


def _source_primary_delta6_terms(config: dict) -> list[tuple[float, float]]:
    model_cfg = dict(config.get("model", {}))
    layers = model_cfg.get("layers")
    if layers:
        source_cfg = config["source"]
        z = 0.5 * (float(source_cfg["start"][2]) + float(source_cfg["end"][2]))
        for layer in layers:
            terms = layer.get("debye_terms", [])
            if not terms:
                continue
            top = float(layer.get("top", np.inf))
            bottom = float(layer.get("bottom", -np.inf))
            if min(top, bottom) <= z <= max(top, bottom):
                return _delta6_terms_from_specs(terms)
        for layer in layers:
            terms = layer.get("debye_terms", [])
            if terms:
                return _delta6_terms_from_specs(terms)
        return []
    return _delta6_terms_from_specs(model_cfg.get("debye_terms", []))


def _delta6_terms_from_specs(term_specs) -> list[tuple[float, float]]:
    terms: list[tuple[float, float]] = []
    for spec in term_specs or []:
        delta = np.asarray(spec["delta_sigma"], dtype=float)
        delta_sigma = float(delta.reshape(-1)[0])
        tau = float(spec["tau"])
        if delta_sigma > 0.0 and tau > 0.0:
            terms.append((delta_sigma, tau))
    return terms


def recompute_current_biot_receiver_data(
    result_path: Path,
    *,
    subdivisions: int | None = None,
    polarization_scale: float | str | list[float] | None = None,
    initial_polarization_scale: float | None = None,
    receiver_indices: list[int] | None = None,
    magnetic_receiver_mode: str = "current_biot",
) -> np.ndarray:
    """Recompute receiver data from saved fields using a Biot recovery mode."""

    if magnetic_receiver_mode not in {
        "current_biot",
        "edge_basis_biot",
        "edge_basis_cell_biot",
        "edge_current_biot",
    }:
        raise ValueError(
            "magnetic_receiver_mode must be 'current_biot', 'edge_basis_biot', "
            "'edge_basis_cell_biot', or 'edge_current_biot'"
        )

    with h5py.File(result_path, "r") as h5:
        if "e" not in h5 or "b" not in h5:
            raise ValueError("result file must contain e and b datasets")
        times = h5["times"][:]
        e = h5["e"][:]
        b = h5["b"][:]
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))

    if subdivisions is not None:
        config["magnetic_recovery_subdivisions"] = int(subdivisions)
    if polarization_scale is not None:
        config["magnetic_recovery_polarization_scale"] = _polarization_scale_config_value(
            polarization_scale
        )
    if initial_polarization_scale is not None:
        config["magnetic_recovery_initial_polarization_scale"] = float(
            initial_polarization_scale
        )
    config["magnetic_receiver_mode"] = magnetic_receiver_mode
    sim = build_simulation(config)
    if receiver_indices is not None:
        if any(index < 0 or index >= len(sim.receivers) for index in receiver_indices):
            raise ValueError("receiver_indices contains an out-of-range receiver column")
        sim.receivers = [sim.receivers[index] for index in receiver_indices]
    if e.shape != (times.size, sim.mesh.n_edges):
        raise ValueError("saved e dataset shape does not match the configured mesh and times")
    if b.shape != (times.size, sim.mesh.n_faces):
        raise ValueError("saved b dataset shape does not match the configured mesh and times")
    if sim.time_steps.size + 1 != times.size or not np.allclose(sim.times, times):
        raise ValueError("saved times do not match the configured time_steps")

    data = np.zeros((times.size, len(sim.receivers)), dtype=float)
    initial_memories = sim.ip_model.initial_memory(sim.mesh.n_edges, e[0])
    memories = [memory.copy() for memory in initial_memories]
    data[0] = sim._sample_receivers(e[0], b[0])
    for step_index, dt in enumerate(sim.time_steps):
        new_memories = sim.ip_model.update_memory(memories, e[step_index + 1], float(dt))
        data[step_index + 1] = sim._sample_receivers_with_current_biot(
            e[step_index + 1],
            b[step_index + 1],
            b[step_index],
            float(dt),
            new_memories,
            float(times[step_index + 1]),
            initial_memories,
        )
        memories = new_memories
    return data


def fit_recomputed_component_polarization_scale(
    result_path: Path,
    reference: np.ndarray,
    *,
    fit_mask: np.ndarray,
    subdivisions: int | None = None,
    receiver_indices: list[int] | None = None,
) -> ComponentPolarizationScaleFit:
    """Fit component-wise Debye polarization-current weights from saved fields.

    This is intentionally a diagnostic fit against an external reference.  It
    should be used to localize magnetic-recovery error, not as a production
    constitutive model.
    """

    fit_mask = np.asarray(fit_mask, dtype=bool)
    ohmic = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=[0.0, 0.0, 0.0],
        receiver_indices=receiver_indices,
    )
    if fit_mask.shape != (ohmic.shape[0],):
        raise ValueError("fit_mask length must match the saved time samples")
    if not np.any(fit_mask):
        raise ValueError("fit_mask selects no samples")

    unit_responses = [
        recompute_current_biot_receiver_data(
            result_path,
            subdivisions=subdivisions,
            polarization_scale=scale,
            receiver_indices=receiver_indices,
        )
        for scale in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    ]
    polarization_components = [ohmic - unit for unit in unit_responses]

    reference = np.asarray(reference, dtype=float)
    if reference.shape != ohmic[fit_mask].shape:
        raise ValueError("reference must match the selected fit window and receiver columns")

    fit = fit_linear_response_components(
        ohmic[fit_mask],
        [component[fit_mask] for component in polarization_components],
        reference,
        signs=[-1.0, -1.0, -1.0],
    )
    fitted_data = ohmic.copy()
    for weight, component in zip(fit.weights, polarization_components):
        fitted_data -= float(weight) * component
    return ComponentPolarizationScaleFit(
        weights=fit.weights,
        fitted_data=fitted_data,
        ohmic_data=ohmic,
        polarization_components=polarization_components,
        rank=fit.rank,
        singular_values=fit.singular_values,
    )


def fit_recomputed_memory_polarization_scale(
    result_path: Path,
    reference: np.ndarray,
    *,
    fit_mask: np.ndarray,
    subdivisions: int | None = None,
    receiver_indices: list[int] | None = None,
) -> MemoryPolarizationScaleFit:
    """Fit current-memory and initial-memory polarization weights.

    The fitted diagnostic model is
    ``H = H_ohmic - lambda * H_delta_y(t) + gamma * H_delta_y(0-)``.
    It is used to localize step-off primary-secondary/IP memory convention
    errors; it is not a production constitutive law.
    """

    fit_mask = np.asarray(fit_mask, dtype=bool)
    ohmic = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=0.0,
        initial_polarization_scale=0.0,
        receiver_indices=receiver_indices,
    )
    if fit_mask.shape != (ohmic.shape[0],):
        raise ValueError("fit_mask length must match the saved time samples")
    if not np.any(fit_mask):
        raise ValueError("fit_mask selects no samples")

    current_unit = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=1.0,
        initial_polarization_scale=0.0,
        receiver_indices=receiver_indices,
    )
    initial_unit = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=0.0,
        initial_polarization_scale=1.0,
        receiver_indices=receiver_indices,
    )
    current_memory = ohmic - current_unit
    initial_memory = initial_unit - ohmic

    reference = np.asarray(reference, dtype=float)
    if reference.shape != ohmic[fit_mask].shape:
        raise ValueError("reference must match the selected fit window and receiver columns")

    fit = fit_linear_response_components(
        ohmic[fit_mask],
        [current_memory[fit_mask], initial_memory[fit_mask]],
        reference,
        signs=[-1.0, 1.0],
    )
    fitted_data = ohmic - float(fit.weights[0]) * current_memory
    fitted_data += float(fit.weights[1]) * initial_memory
    return MemoryPolarizationScaleFit(
        weights=fit.weights,
        fitted_data=fitted_data,
        ohmic_data=ohmic,
        current_memory_data=current_memory,
        initial_memory_data=initial_memory,
        rank=fit.rank,
        singular_values=fit.singular_values,
    )


def fit_recomputed_memory_polarization_scale_per_receiver(
    result_path: Path,
    reference: np.ndarray,
    *,
    fit_mask: np.ndarray,
    subdivisions: int | None = None,
    receiver_indices: list[int] | None = None,
) -> PerReceiverMemoryPolarizationScaleFit:
    """Fit current/initial memory weights independently for each column."""

    fit_mask = np.asarray(fit_mask, dtype=bool)
    ohmic = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=0.0,
        initial_polarization_scale=0.0,
        receiver_indices=receiver_indices,
    )
    if fit_mask.shape != (ohmic.shape[0],):
        raise ValueError("fit_mask length must match the saved time samples")
    if not np.any(fit_mask):
        raise ValueError("fit_mask selects no samples")

    current_unit = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=1.0,
        initial_polarization_scale=0.0,
        receiver_indices=receiver_indices,
    )
    initial_unit = recompute_current_biot_receiver_data(
        result_path,
        subdivisions=subdivisions,
        polarization_scale=0.0,
        initial_polarization_scale=1.0,
        receiver_indices=receiver_indices,
    )
    current_memory = ohmic - current_unit
    initial_memory = initial_unit - ohmic

    reference = np.asarray(reference, dtype=float)
    if reference.shape != ohmic[fit_mask].shape:
        raise ValueError("reference must match the selected fit window and receiver columns")

    n_columns = ohmic.shape[1]
    weights = np.zeros((n_columns, 2), dtype=float)
    ranks = np.zeros(n_columns, dtype=int)
    singular_values: list[np.ndarray] = []
    fitted_data = ohmic.copy()
    for column in range(n_columns):
        column_slice = slice(column, column + 1)
        fit = fit_linear_response_components(
            ohmic[fit_mask, column_slice],
            [
                current_memory[fit_mask, column_slice],
                initial_memory[fit_mask, column_slice],
            ],
            reference[:, column_slice],
            signs=[-1.0, 1.0],
        )
        weights[column] = fit.weights
        ranks[column] = int(fit.rank)
        singular_values.append(fit.singular_values)
        fitted_data[:, column] -= float(fit.weights[0]) * current_memory[:, column]
        fitted_data[:, column] += float(fit.weights[1]) * initial_memory[:, column]

    return PerReceiverMemoryPolarizationScaleFit(
        weights=weights,
        fitted_data=fitted_data,
        ohmic_data=ohmic,
        current_memory_data=current_memory,
        initial_memory_data=initial_memory,
        ranks=ranks,
        singular_values=singular_values,
    )


def _write_comparison_plot(
    path: Path,
    times: np.ndarray,
    numerical: np.ndarray,
    reference: np.ndarray,
    names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_components = numerical.shape[1]
    fig, axes = plt.subplots(
        2,
        n_components,
        figsize=(max(4.0, 3.2 * n_components), 5.8),
        squeeze=False,
    )
    for i, name in enumerate(names):
        ax = axes[0, i]
        ax.loglog(times, np.abs(numerical[:, i]), "o-", label="ATEM3D")
        ax.loglog(times, np.abs(reference[:, i]), "s--", label="empymod")
        ax.set_title(name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|response|")
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.legend()

        err = np.abs(numerical[:, i] - reference[:, i]) / np.maximum(np.abs(reference[:, i]), 1.0e-300)
        ax_err = axes[1, i]
        ax_err.loglog(times, err, "o-")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("relative error")
        ax_err.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ensure_numba_cache_dir() -> None:
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    _DEFAULT_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_DEFAULT_NUMBA_CACHE_DIR.resolve())


def _metadata_polarization_scale(
    config: dict,
    cli_scale: float | str | list[float] | None,
    fit_result: ComponentPolarizationScaleFit | None,
) -> float | str | list[float]:
    if fit_result is not None:
        return fit_result.weights.tolist()
    if cli_scale is not None:
        return cli_scale
    return config.get("magnetic_recovery_polarization_scale", 1.0)


def _polarization_scale_arg(value: str) -> float | str:
    normalized = value.strip().lower()
    if normalized == "low_frequency_ratio":
        return normalized
    if "," in value:
        try:
            scale = [float(part.strip()) for part in value.split(",")]
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                "must be a nonnegative number, 'low_frequency_ratio', or sx,sy,sz"
            ) from err
        if len(scale) != 3 or any(part < 0.0 for part in scale):
            raise argparse.ArgumentTypeError(
                "component scales must be three nonnegative numbers sx,sy,sz"
            )
        return scale
    try:
        scale = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "must be a nonnegative number, 'low_frequency_ratio', or sx,sy,sz"
        ) from err
    if scale < 0.0:
        raise argparse.ArgumentTypeError(
            "must be a nonnegative number, 'low_frequency_ratio', or sx,sy,sz"
        )
    return scale


def _polarization_scale_config_value(value: float | str | list[float]) -> float | str | list[float]:
    if isinstance(value, str) and value.strip().lower() == "low_frequency_ratio":
        return "low_frequency_ratio"
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(
                "component magnetic_recovery_polarization_scale must have length 3"
            )
        if np.any(array < 0.0):
            raise ValueError("magnetic_recovery_polarization_scale must be nonnegative")
        return array.tolist()
    scale = float(array)
    if scale < 0.0:
        raise ValueError(
            "magnetic_recovery_polarization_scale must be nonnegative "
            "or 'low_frequency_ratio'"
        )
    return scale


if __name__ == "__main__":
    raise SystemExit(main())
