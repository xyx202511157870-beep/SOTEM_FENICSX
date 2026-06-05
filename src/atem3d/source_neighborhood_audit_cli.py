"""Audit H/J near-source face-current contributions to magnetic receivers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .config import build_simulation
from .hj import hj_dc_initial_current_density
from .magnetic_recovery import face_current_biot_matrix
from .recovery_spectrum import magnetic_diffusion_mmr_initial_state
from .source_history_runtime import charge_conserving_face_current


_CANDIDATE_CHOICES = (
    "active_source",
    "charge_conserving_source",
    "mmr_initial_curl_active_source",
    "mmr_initial_curl_charge_conserving_source",
    "dc_initial_current",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose saved H/J current_biot magnetic receiver traces by "
            "radial face-current neighborhoods around the grounded wire."
        )
    )
    parser.add_argument("full_hdf5", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--components", nargs="+")
    parser.add_argument(
        "--radial-edges",
        nargs="+",
        type=float,
        default=[0.0, 2.5, 5.0, 10.0, 20.0],
        help="Radial shell edges in metres around the finite source segment.",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=None,
        help="Biot midpoint subdivisions; defaults to config value.",
    )
    parser.add_argument("--time-tolerance", type=float, default=1.0e-9)
    parser.add_argument(
        "--exclude-outside",
        action="store_true",
        help="Do not add an outside group for faces beyond the final radial edge.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=_CANDIDATE_CHOICES,
        default=[],
        help=(
            "Non-fitted static face-current candidates to project against the "
            "reference-minus-numerical residual."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.time_tolerance < 0.0:
        parser.error("--time-tolerance must be non-negative")
    if args.subdivisions is not None and args.subdivisions <= 0:
        parser.error("--subdivisions must be positive")
    radial_edges = _validate_radial_edges(args.radial_edges)

    config, times, h_history = _load_hj_hdf5(args.full_hdf5)
    sim = build_simulation(config)
    subdivisions = (
        int(args.subdivisions)
        if args.subdivisions is not None
        else int(config.get("magnetic_recovery_subdivisions", 1))
    )

    samples = _load_samples(args.report)
    component_names = args.components or [
        name for name in sorted(samples) if name[:1] in {"H", "B"}
    ]
    if not component_names:
        raise ValueError("no magnetic components selected")

    sample_times, numerical, reference = _component_sample_arrays(samples, component_names)
    time_indices = _match_time_indices(times, sample_times, args.time_tolerance)
    receiver_matrix = _component_receiver_matrix(sim, component_names, subdivisions)
    currents = _hj_face_current_history(sim, h_history[time_indices], sample_times)
    recomputed = currents @ receiver_matrix.T

    groups = _source_radial_face_groups(
        sim.mesh,
        sim.sources,
        radial_edges,
        include_outside=not args.exclude_outside,
    )
    group_contribs = [
        currents[:, group["face_indices"]] @ receiver_matrix[:, group["face_indices"]].T
        for group in groups
    ]
    candidate_vectors = _candidate_face_vectors(sim, args.candidates)

    report = _summarize_audit(
        component_names=component_names,
        sample_times=sample_times,
        numerical=numerical,
        reference=reference,
        recomputed=recomputed,
        groups=groups,
        group_contribs=group_contribs,
        full_hdf5=args.full_hdf5,
        validation_report=args.report,
        radial_edges=radial_edges,
        subdivisions=subdivisions,
        candidate_vectors=candidate_vectors,
        receiver_matrix=receiver_matrix,
        source_diffusion_time=_source_diffusion_time(sim),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"base_l2={report['base_relative_l2']:.6g} "
        f"recomputed_l2={report['recomputed_relative_l2']:.6g} "
        f"neighborhood_fit_l2={report['neighborhood_fit']['relative_l2']:.6g}"
    )
    print(f"wrote {args.output}")
    return 0


def _validate_radial_edges(edges: list[float]) -> np.ndarray:
    array = np.asarray(edges, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("--radial-edges requires at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError("--radial-edges must be finite")
    if np.any(array < 0.0):
        raise ValueError("--radial-edges must be non-negative")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError("--radial-edges must be strictly increasing")
    return array


def _load_hj_hdf5(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        formulation = str(h5.attrs.get("formulation", "")).lower()
        if formulation != "hj":
            raise ValueError("source-neighborhood audit currently requires an H/J HDF5 result")
        config = yaml.safe_load(h5.attrs["config_yaml"])
        times = np.asarray(h5["times"][:], dtype=float)
        h_history = np.asarray(h5["h"][:], dtype=float)
    if not isinstance(config, dict):
        raise ValueError("HDF5 config_yaml must decode to a mapping")
    if times.ndim != 1:
        raise ValueError("HDF5 times must be 1D")
    if h_history.ndim != 2 or h_history.shape[0] != times.size:
        raise ValueError("HDF5 h history must have shape (n_times, n_edges)")
    return config, times, h_history


def _load_samples(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("validation report must contain a non-empty samples object")
    return samples


def _component_sample_arrays(
    samples: dict[str, list[dict[str, Any]]],
    component_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: np.ndarray | None = None
    numerical_columns: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    for name in component_names:
        if name not in samples:
            raise ValueError(f"component {name!r} is not present in report samples")
        rows = samples[name]
        if not rows:
            raise ValueError(f"component {name!r} has no samples")
        component_times = np.asarray([row["time"] for row in rows], dtype=float)
        if times is None:
            times = component_times
        elif not np.allclose(times, component_times, rtol=0.0, atol=1.0e-15):
            raise ValueError("all selected components must share identical sample times")
        numerical_columns.append(np.asarray([row["numerical"] for row in rows], dtype=float))
        reference_columns.append(np.asarray([row["reference"] for row in rows], dtype=float))
    assert times is not None
    return times, np.column_stack(numerical_columns), np.column_stack(reference_columns)


def _match_time_indices(
    available_times: np.ndarray,
    sample_times: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    indices = []
    for time in sample_times:
        differences = np.abs(available_times - float(time))
        index = int(np.argmin(differences))
        if float(differences[index]) > tolerance:
            raise ValueError(
                f"sample time {time:g} is not present in HDF5 times within {tolerance:g}"
            )
        indices.append(index)
    return np.asarray(indices, dtype=int)


def _component_receiver_matrix(sim, component_names: list[str], subdivisions: int) -> np.ndarray:
    receivers = [_receiver_for_component(sim.receivers, name) for name in component_names]
    locations = np.asarray([receiver.location for receiver in receivers], dtype=float)
    biot = face_current_biot_matrix(sim.mesh, locations, subdivisions=subdivisions)
    matrix = np.zeros((len(receivers), sim.mesh.n_faces), dtype=float)
    for index, receiver in enumerate(receivers):
        row = biot[index, receiver.vector_component_index, :]
        if receiver.component.startswith("B"):
            row = sim.mu * row
        matrix[index] = row
    return matrix


def _receiver_for_component(receivers, name: str):
    by_name: dict[str, object] = {}
    for receiver in receivers:
        x_name = f"{receiver.component}@x={float(receiver.location[0]):g}"
        by_name[x_name] = receiver
        if receiver.component not in by_name:
            by_name[receiver.component] = receiver
    if name in by_name:
        return by_name[name]
    if "@x=" in name:
        component, x_text = name.split("@x=", 1)
        target_x = float(x_text)
        matches = [
            receiver
            for receiver in receivers
            if receiver.component == component
            and np.isclose(float(receiver.location[0]), target_x, rtol=0.0, atol=1.0e-9)
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"could not map component name {name!r} to a configured receiver")


def _hj_face_current_history(sim, h_history: np.ndarray, times: np.ndarray) -> np.ndarray:
    currents = (sim.mesh.edge_curl @ h_history.T).T
    for index, time in enumerate(times):
        currents[index] = currents[index] - sim.electric_source_term(float(time))
    return np.asarray(currents, dtype=float)


def _source_radial_face_groups(
    mesh,
    sources,
    radial_edges: np.ndarray,
    *,
    include_outside: bool,
) -> list[dict[str, Any]]:
    centers = _face_centers(mesh)
    distances = _distance_to_sources(centers, sources)
    source_vector = np.zeros(mesh.n_faces, dtype=float)
    for source in sources:
        source_vector += np.asarray(source.initial_face_vector(mesh), dtype=float)

    groups: list[dict[str, Any]] = []
    covered = np.zeros(mesh.n_faces, dtype=bool)
    for start, stop in zip(radial_edges[:-1], radial_edges[1:]):
        mask = (distances >= float(start)) & (distances < float(stop))
        covered |= mask
        if np.any(mask):
            groups.append(_group_payload(f"{start:g}-{stop:g}m", mask, source_vector))
    if include_outside:
        mask = ~covered
        if np.any(mask):
            groups.append(_group_payload(f">={radial_edges[-1]:g}m", mask, source_vector))
    if not groups:
        raise ValueError("radial edges selected no face groups")
    return groups


def _face_centers(mesh) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(mesh.faces_x, dtype=float),
            np.asarray(mesh.faces_y, dtype=float),
            np.asarray(mesh.faces_z, dtype=float),
        ]
    )


def _distance_to_sources(points: np.ndarray, sources) -> np.ndarray:
    distances = np.full(points.shape[0], np.inf, dtype=float)
    for source in sources:
        locations = np.asarray(source.locations, dtype=float)
        for start, stop in zip(locations[:-1], locations[1:]):
            segment = stop - start
            length2 = float(np.dot(segment, segment))
            if length2 == 0.0:
                candidate = np.linalg.norm(points - start[None, :], axis=1)
            else:
                t = np.clip(((points - start[None, :]) @ segment) / length2, 0.0, 1.0)
                closest = start[None, :] + t[:, None] * segment[None, :]
                candidate = np.linalg.norm(points - closest, axis=1)
            distances = np.minimum(distances, candidate)
    if not np.all(np.isfinite(distances)):
        raise ValueError("at least one source segment is required")
    return distances


def _group_payload(name: str, mask: np.ndarray, source_vector: np.ndarray) -> dict[str, Any]:
    face_indices = np.flatnonzero(mask)
    active = np.count_nonzero(np.abs(source_vector[face_indices]) > 0.0)
    return {
        "name": name,
        "face_indices": face_indices,
        "face_count": int(face_indices.size),
        "active_source_face_count": int(active),
    }


def _summarize_audit(
    *,
    component_names: list[str],
    sample_times: np.ndarray,
    numerical: np.ndarray,
    reference: np.ndarray,
    recomputed: np.ndarray,
    groups: list[dict[str, Any]],
    group_contribs: list[np.ndarray],
    full_hdf5: Path,
    validation_report: Path,
    radial_edges: np.ndarray,
    subdivisions: int,
    candidate_vectors: dict[str, np.ndarray],
    receiver_matrix: np.ndarray,
    source_diffusion_time: float,
) -> dict[str, Any]:
    residual = reference - recomputed
    group_reports = []
    for group, contribution in zip(groups, group_contribs):
        item = {
            key: value
            for key, value in group.items()
            if key != "face_indices"
        }
        item.update(_group_metrics(contribution, recomputed, reference, residual))
        group_reports.append(item)

    fit = _fit_group_weights(reference, recomputed, groups, group_contribs)
    return {
        "diagnostic_only": True,
        "full_hdf5": str(full_hdf5),
        "validation_report": str(validation_report),
        "components": list(component_names),
        "time_count": int(sample_times.size),
        "time_min": float(sample_times[0]),
        "time_max": float(sample_times[-1]),
        "radial_edges_m": [float(value) for value in radial_edges],
        "subdivisions": int(subdivisions),
        "source_diffusion_time_s": float(source_diffusion_time),
        "base_relative_l2": _relative_l2(numerical, reference),
        "recomputed_relative_l2": _relative_l2(recomputed, reference),
        "max_abs_difference_from_report_numerical": float(
            np.max(np.abs(recomputed - numerical))
        ),
        "groups": group_reports,
        "neighborhood_fit": fit,
        "candidate_static_fits": {
            name: _candidate_static_fit_report(
                vector,
                receiver_matrix,
                recomputed,
                reference,
                sample_times,
                source_diffusion_time,
            )
            for name, vector in candidate_vectors.items()
        },
    }


def _group_metrics(
    contribution: np.ndarray,
    recomputed: np.ndarray,
    reference: np.ndarray,
    residual: np.ndarray,
) -> dict[str, float]:
    contribution_norm = float(np.linalg.norm(contribution))
    residual_norm = float(np.linalg.norm(residual))
    if contribution_norm > 0.0:
        correlation = float(np.vdot(contribution.ravel(), residual.ravel()).real)
        correlation /= contribution_norm * residual_norm if residual_norm > 0.0 else 1.0
    else:
        correlation = 0.0
    outside = recomputed - contribution
    target = reference - outside
    denom = float(np.vdot(contribution.ravel(), contribution.ravel()).real)
    if denom > 0.0:
        weight = float(np.vdot(contribution.ravel(), target.ravel()).real / denom)
    else:
        weight = 0.0
    replaced = outside + weight * contribution
    return {
        "relative_norm_vs_reference": _relative_norm(contribution, reference),
        "correlation_with_residual": correlation,
        "relative_l2_if_removed": _relative_l2(outside, reference),
        "best_weight_with_other_groups_fixed": weight,
        "relative_l2_after_best_weight": _relative_l2(replaced, reference),
    }


def _fit_group_weights(
    reference: np.ndarray,
    recomputed: np.ndarray,
    groups: list[dict[str, Any]],
    group_contribs: list[np.ndarray],
) -> dict[str, Any]:
    fit_indices = [index for index, group in enumerate(groups) if not group["name"].startswith(">=")]
    if not fit_indices:
        return {
            "group_names": [],
            "weights": [],
            "relative_l2": _relative_l2(recomputed, reference),
            "rank": 0,
            "singular_values": [],
        }

    fit_sum = np.zeros_like(recomputed)
    for index in fit_indices:
        fit_sum += group_contribs[index]
    outside = recomputed - fit_sum
    design = np.column_stack([group_contribs[index].ravel() for index in fit_indices])
    target = (reference - outside).ravel()
    weights, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    fitted = outside + sum(
        float(weight) * group_contribs[index]
        for weight, index in zip(weights, fit_indices)
    )
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "group_names": [groups[index]["name"] for index in fit_indices],
        "weights": [float(value) for value in weights],
        "relative_l2": _relative_l2(fitted, reference),
        "rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition_number,
    }


def _candidate_face_vectors(sim, names: list[str]) -> dict[str, np.ndarray]:
    if not names:
        return {}
    if len(sim.sources) != 1:
        raise ValueError("candidate source audit requires exactly one source")
    source = sim.sources[0]
    mesh = sim.mesh
    source_vector = -np.asarray(source.initial_face_vector(mesh), dtype=float)
    conductivity = np.asarray(sim.initial_ip_model.low_frequency_sigma(), dtype=float)
    candidates: dict[str, np.ndarray] = {}
    for name in names:
        if name == "active_source":
            vector = source_vector.copy()
        elif name == "charge_conserving_source":
            vector = charge_conserving_face_current(mesh, conductivity, source_vector)
        elif name == "mmr_initial_curl_active_source":
            state = magnetic_diffusion_mmr_initial_state(
                mesh,
                source_vector,
                conductivity=conductivity,
                source_projection="charge_conserving",
                mu=sim.mu,
            )
            vector = np.asarray(mesh.edge_curl @ state[:, 0], dtype=float)
        elif name == "mmr_initial_curl_charge_conserving_source":
            projected = charge_conserving_face_current(mesh, conductivity, source_vector)
            state = magnetic_diffusion_mmr_initial_state(
                mesh,
                projected,
                conductivity=conductivity,
                source_projection="raw",
                mu=sim.mu,
            )
            vector = np.asarray(mesh.edge_curl @ state[:, 0], dtype=float)
        elif name == "dc_initial_current":
            vector = hj_dc_initial_current_density(mesh, sim.initial_ip_model, sim.sources)
        else:  # pragma: no cover - argparse choices should prevent this
            raise ValueError(f"unsupported candidate: {name}")
        if vector.shape != (mesh.n_faces,):
            raise ValueError(f"candidate {name!r} did not produce a face-current vector")
        candidates[name] = np.asarray(vector, dtype=float)
    return candidates


def _candidate_static_fit_report(
    candidate_vector: np.ndarray,
    receiver_matrix: np.ndarray,
    recomputed: np.ndarray,
    reference: np.ndarray,
    sample_times: np.ndarray,
    source_diffusion_time: float,
) -> dict[str, Any]:
    response = np.asarray(receiver_matrix @ candidate_vector, dtype=float)
    target = reference - recomputed
    denom = float(np.vdot(response, response).real)
    if denom > 0.0:
        all_window_coefficient = float(
            np.vdot(
                np.broadcast_to(response, target.shape).ravel(),
                target.ravel(),
            ).real
            / (target.shape[0] * denom)
        )
        per_time_coefficients = np.asarray(
            [np.vdot(response, row).real / denom for row in target],
            dtype=float,
        )
    else:
        all_window_coefficient = 0.0
        per_time_coefficients = np.zeros(target.shape[0], dtype=float)
    all_window_residual = all_window_coefficient * response[None, :]
    per_time_residual = per_time_coefficients[:, None] * response[None, :]
    all_window_fitted = recomputed + all_window_coefficient * response[None, :]
    per_time_fitted = recomputed + per_time_coefficients[:, None] * response[None, :]
    kernel_fits = _coefficient_kernel_fits(
        per_time_coefficients,
        sample_times,
        source_diffusion_time=source_diffusion_time,
    )
    for fit in kernel_fits:
        kernel = _exponential_kernel(sample_times, fit["tau"])
        corrected = recomputed + float(fit["amplitude"]) * kernel[:, None] * response[None, :]
        fit["corrected_relative_l2"] = _relative_l2(corrected, reference)
    return {
        "candidate_response": [float(value) for value in response],
        "candidate_response_relative_norm_vs_reference": _relative_norm(
            np.broadcast_to(response, reference.shape),
            reference,
        ),
        "all_window_coefficient": all_window_coefficient,
        "all_window_relative_l2": _relative_l2(all_window_fitted, reference),
        "all_window_residual_relative_l2": _relative_l2(
            all_window_residual,
            target,
        ),
        "all_window_residual_projection_fraction": _projection_fraction(
            all_window_residual,
            target,
        ),
        "per_time_relative_l2": _relative_l2(per_time_fitted, reference),
        "per_time_residual_relative_l2": _relative_l2(
            per_time_residual,
            target,
        ),
        "per_time_residual_projection_fraction": _projection_fraction(
            per_time_residual,
            target,
        ),
        "per_time_response_target_cosines": _row_cosines(response, target),
        "per_time_coefficients": [float(value) for value in per_time_coefficients],
        "per_time_coefficient_first": float(per_time_coefficients[0]),
        "per_time_coefficient_last": float(per_time_coefficients[-1]),
        "per_time_coefficient_min": float(np.min(per_time_coefficients)),
        "per_time_coefficient_max": float(np.max(per_time_coefficients)),
        "coefficient_kernel_fits": kernel_fits,
    }


def _projection_fraction(predicted: np.ndarray, target: np.ndarray) -> float:
    target_norm2 = float(np.vdot(target.ravel(), target.ravel()).real)
    if target_norm2 == 0.0:
        return 0.0
    residual = np.asarray(predicted, dtype=float) - np.asarray(target, dtype=float)
    residual_norm2 = float(np.vdot(residual.ravel(), residual.ravel()).real)
    return float(1.0 - residual_norm2 / target_norm2)


def _row_cosines(response: np.ndarray, target: np.ndarray) -> list[float]:
    response = np.asarray(response, dtype=float)
    target = np.asarray(target, dtype=float)
    response_norm = float(np.linalg.norm(response))
    if response_norm == 0.0:
        return [0.0 for _ in range(target.shape[0])]
    cosines = []
    for row in target:
        row_norm = float(np.linalg.norm(row))
        if row_norm == 0.0:
            cosines.append(0.0)
        else:
            cosines.append(float(np.vdot(response, row).real / (response_norm * row_norm)))
    return cosines


def _coefficient_kernel_fits(
    coefficients: np.ndarray,
    sample_times: np.ndarray,
    *,
    source_diffusion_time: float,
    multipliers: list[float] | None = None,
) -> list[dict[str, float]]:
    coefficients = np.asarray(coefficients, dtype=float)
    sample_times = np.asarray(sample_times, dtype=float)
    if coefficients.ndim != 1 or sample_times.ndim != 1:
        raise ValueError("coefficients and sample_times must be 1D")
    if coefficients.shape != sample_times.shape:
        raise ValueError("coefficients and sample_times must have the same shape")
    source_diffusion_time = float(source_diffusion_time)
    if source_diffusion_time <= 0.0:
        raise ValueError("source_diffusion_time must be positive")
    if multipliers is None:
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    denominator = float(np.linalg.norm(coefficients))
    fits: list[dict[str, float]] = []
    for multiplier in multipliers:
        multiplier = float(multiplier)
        if multiplier <= 0.0:
            raise ValueError("kernel multipliers must be positive")
        tau = source_diffusion_time * multiplier
        kernel = _exponential_kernel(sample_times, tau)
        kernel_norm2 = float(np.vdot(kernel, kernel).real)
        amplitude = (
            float(np.vdot(kernel, coefficients).real / kernel_norm2)
            if kernel_norm2 > 0.0
            else 0.0
        )
        fitted = amplitude * kernel
        first_amplitude = float(coefficients[0])
        first_fitted = first_amplitude * kernel
        fits.append(
            {
                "multiplier": multiplier,
                "tau": float(tau),
                "amplitude": amplitude,
                "coefficient_relative_l2": _relative_vector_l2(fitted, coefficients),
                "first_coefficient_relative_l2": _relative_vector_l2(
                    first_fitted,
                    coefficients,
                )
                if denominator > 0.0
                else float(np.linalg.norm(first_fitted - coefficients)),
            }
        )
    return fits


def _exponential_kernel(sample_times: np.ndarray, tau: float) -> np.ndarray:
    sample_times = np.asarray(sample_times, dtype=float)
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return np.exp(-(sample_times - float(sample_times[0])) / tau)


def _source_diffusion_time(sim) -> float:
    if len(sim.sources) != 1:
        raise ValueError("source diffusion time requires exactly one source")
    source = sim.sources[0]
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    if length <= 0.0:
        raise ValueError("source length must be positive")
    conductivity = np.asarray(sim.initial_ip_model.low_frequency_sigma(), dtype=float)
    midpoint = np.mean(source.locations, axis=0)
    centers = np.asarray(sim.mesh.cell_centers, dtype=float)
    index = int(np.argmin(np.sum((centers - midpoint[None, :]) ** 2, axis=1)))
    sigma = float(conductivity[index])
    if sigma <= 0.0:
        raise ValueError("source midpoint conductivity must be positive")
    return float(sim.mu * sigma * length**2)


def _relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(numerical - reference))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


def _relative_vector_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(numerical - reference))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


def _relative_norm(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(value))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
