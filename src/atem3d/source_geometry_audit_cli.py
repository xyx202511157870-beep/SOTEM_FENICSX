"""Audit grounded-source geometry metrics used by source-history diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
import yaml
from scipy.constants import mu_0

from .config import build_simulation, load_config
from .magnetic_recovery import face_current_biot_matrix
from .source_history_runtime import (
    SourceDiffusionKernelSourceHistoryCorrection,
    source_history_correction_terms,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit grounded-wire source-vector geometry and the "
            "mu*sigma*L^2 source-diffusion normalization for one YAML config."
        )
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mu", type=float, default=mu_0)
    parser.add_argument(
        "--sweep-cases",
        type=Path,
        default=None,
        help="Optional YAML file containing named config overrides to audit",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.sweep_cases is None:
        report = audit_source_geometry(
            config,
            config_path=str(args.config),
            mu=float(args.mu),
        )
    else:
        report = audit_source_geometry_sweep(
            config,
            _load_sweep_cases(args.sweep_cases),
            base_config_path=str(args.config),
            sweep_cases_path=str(args.sweep_cases),
            mu=float(args.mu),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.sweep_cases is None:
        print(
            "source geometry: "
            f"L={report['source']['length_m']:.6g} m, "
            f"active={report['source_vector']['active_count']}, "
            f"tau0={report['diffusion_scale']['tau0_s']:.6g} s"
        )
    else:
        print(f"source geometry sweep cases: {report['case_count']}")
    print(f"wrote {args.output}")
    return 0


def audit_source_geometry(
    config: dict[str, Any],
    *,
    config_path: str | None = None,
    mu: float = mu_0,
) -> dict[str, Any]:
    """Return source-vector and source-diffusion normalization metrics."""

    simulation = build_simulation(config)
    if len(simulation.sources) != 1:
        raise ValueError("source geometry audit requires exactly one source")
    source = simulation.sources[0]
    mesh = simulation.mesh
    formulation = str(config.get("formulation", "eb")).strip().lower()
    if formulation not in {"eb", "hj"}:
        raise ValueError("formulation must be 'eb' or 'hj'")

    field_location = "face" if formulation == "hj" else "edge"
    source_vector = _runtime_source_vector(mesh, source, field_location)
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    if length <= 0.0:
        raise ValueError("source length must be positive")
    ip_model = getattr(simulation, "initial_ip_model", None) or simulation.ip_model
    sigma_midpoint = _source_midpoint_conductivity(mesh, ip_model, source)
    tau0 = float(mu) * sigma_midpoint * length**2

    source_vector_metrics = _source_vector_metrics(
        mesh,
        source_vector,
        field_location,
        ip_model,
    )
    source_vector_metrics["receiver_static_response"] = _receiver_static_response(
        simulation,
        source_vector,
        field_location,
        mu=float(mu),
    )

    return {
        "config_path": config_path,
        "formulation": formulation,
        "mesh": _mesh_metrics(mesh),
        "source": {
            "start": _float_list(source.locations[0]),
            "end": _float_list(source.locations[-1]),
            "midpoint": _float_list(np.mean(source.locations, axis=0)),
            "length_m": length,
            "current_a": float(source.current),
            "face_projection": str(getattr(source, "face_projection", "")),
            "midpoint_cell": _midpoint_cell_metrics(mesh, source),
        },
        "source_vector": source_vector_metrics,
        "diffusion_scale": {
            "mu_h_per_m": float(mu),
            "sigma_midpoint_s_per_m": sigma_midpoint,
            "tau0_s": tau0,
            "normalization": "mu * sigma_midpoint * source_length**2",
        },
        "configured_source_diffusion": _configured_source_diffusion(
            getattr(simulation, "magnetic_recovery_source_history", None),
            tau0,
        ),
        "diagnostic_note": (
            "These metrics audit the geometry entering source-diffusion "
            "diagnostics; they do not derive the production H/J MMR law."
        ),
    }


def audit_source_geometry_sweep(
    config: dict[str, Any],
    cases: Any,
    *,
    base_config_path: str | None = None,
    sweep_cases_path: str | None = None,
    mu: float = mu_0,
) -> dict[str, Any]:
    """Return source-geometry audits for named config override cases."""

    normalized_cases = _normalize_sweep_cases(cases)
    audits: dict[str, Any] = {}
    for name, overrides in normalized_cases:
        case_config = _deep_merge(config, overrides)
        audit = audit_source_geometry(
            case_config,
            config_path=(
                None
                if base_config_path is None
                else f"{base_config_path}::{name}"
            ),
            mu=mu,
        )
        audit["overrides"] = deepcopy(overrides)
        audits[name] = audit
    return {
        "base_config_path": base_config_path,
        "sweep_cases_path": sweep_cases_path,
        "case_count": len(audits),
        "cases": audits,
    }


def _runtime_source_vector(mesh, source, field_location: str) -> np.ndarray:
    if field_location == "edge":
        return np.asarray(source.initial_edge_vector(mesh), dtype=float)
    return -np.asarray(source.initial_face_vector(mesh), dtype=float)


def _mesh_metrics(mesh) -> dict[str, Any]:
    widths = [np.asarray(axis_widths, dtype=float) for axis_widths in mesh.h]
    return {
        "n_cells": int(mesh.n_cells),
        "n_edges": int(mesh.n_edges),
        "n_faces": int(mesh.n_faces),
        "shape_cells": [int(value) for value in mesh.shape_cells],
        "origin": _float_list(mesh.origin),
        "min_cell_width_m": [float(np.min(values)) for values in widths],
        "max_cell_width_m": [float(np.max(values)) for values in widths],
    }


def _source_vector_metrics(
    mesh,
    source_vector: np.ndarray,
    field_location: str,
    ip_model,
) -> dict[str, Any]:
    vector = np.asarray(source_vector, dtype=float)
    active = np.abs(vector) > 0.0
    active_values = vector[active]
    metrics: dict[str, Any] = {
        "location": field_location,
        "size": int(vector.size),
        "active_count": int(np.count_nonzero(active)),
        "signed_sum": float(np.sum(vector)),
        "l1_abs": float(np.sum(np.abs(vector))),
        "l2": float(np.linalg.norm(vector)),
        "linf_abs": float(np.max(np.abs(vector))) if vector.size else 0.0,
        "active_min": float(np.min(active_values)) if active_values.size else 0.0,
        "active_max": float(np.max(active_values)) if active_values.size else 0.0,
        "active_mean": float(np.mean(active_values)) if active_values.size else 0.0,
        "active_abs_mean": (
            float(np.mean(np.abs(active_values))) if active_values.size else 0.0
        ),
        "active_bounds": _active_bounds(mesh, active, field_location),
    }
    if field_location == "face":
        metrics.update(_face_source_vector_metrics(mesh, vector, active, ip_model))
    else:
        metrics.update(_edge_source_vector_metrics(mesh, active))
    return metrics


def _face_source_vector_metrics(
    mesh,
    vector: np.ndarray,
    active: np.ndarray,
    ip_model,
) -> dict[str, Any]:
    areas = np.asarray(mesh.face_areas, dtype=float)
    active_areas = areas[active]
    return {
        "active_count_by_orientation": _orientation_counts(
            active,
            [("x", mesh.nFx), ("y", mesh.nFy), ("z", mesh.nFz)],
        ),
        "active_area_sum_m2": float(np.sum(active_areas)),
        "active_area_min_m2": float(np.min(active_areas)) if active_areas.size else 0.0,
        "active_area_max_m2": float(np.max(active_areas)) if active_areas.size else 0.0,
        "area_weighted_signed_sum": float(np.sum(vector * areas)),
        "area_weighted_l1_abs": float(np.sum(np.abs(vector) * areas)),
        "area_weighted_l2": float(np.sqrt(np.sum((vector**2) * areas))),
        "face_inner_products": _face_inner_product_metrics(mesh, vector, active, ip_model),
    }


def _face_inner_product_metrics(
    mesh,
    vector: np.ndarray,
    active: np.ndarray,
    ip_model,
) -> dict[str, Any]:
    unit_face = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    rho_face = _face_resistivity_inner_product(mesh, ip_model, unit_face)
    unit_quadratic = _quadratic_form(unit_face, vector)
    rho_quadratic = _quadratic_form(rho_face, vector)
    curl = mesh.edge_curl.tocsr()
    unit_source_rhs = curl.T @ (unit_face @ vector)
    rho_source_rhs = curl.T @ (rho_face @ vector)
    return {
        "unit_quadratic": unit_quadratic,
        "unit_l2": float(np.sqrt(max(unit_quadratic, 0.0))),
        "rho_quadratic": rho_quadratic,
        "rho_l2": float(np.sqrt(max(rho_quadratic, 0.0))),
        "unit_diagonal_active_sum": float(np.sum(unit_face.diagonal()[active])),
        "rho_diagonal_active_sum": float(np.sum(rho_face.diagonal()[active])),
        "curl_transpose_unit_source_l2": float(np.linalg.norm(unit_source_rhs)),
        "curl_transpose_rho_source_l2": float(np.linalg.norm(rho_source_rhs)),
    }


def _face_resistivity_inner_product(mesh, ip_model, unit_face) -> sp.csr_matrix:
    conductivity = _low_frequency_conductivity(ip_model)
    if conductivity.size == 1:
        rho = np.full(mesh.n_cells, 1.0 / float(conductivity.reshape(-1)[0]))
        return mesh.get_face_inner_product(rho).tocsr()
    if conductivity.shape == (mesh.n_cells,):
        return mesh.get_face_inner_product(1.0 / conductivity).tocsr()
    if conductivity.shape == (mesh.n_faces,):
        return (unit_face @ sp.diags(1.0 / conductivity, format="csr")).tocsr()
    raise ValueError(
        "source geometry audit requires scalar, cell-centered, or face-centered conductivity"
    )


def _low_frequency_conductivity(ip_model) -> np.ndarray:
    if hasattr(ip_model, "low_frequency_sigma"):
        return np.asarray(ip_model.low_frequency_sigma(), dtype=float)
    return np.asarray(ip_model.sigma_infinity, dtype=float)


def _quadratic_form(matrix: sp.spmatrix, vector: np.ndarray) -> float:
    return float(np.asarray(vector @ (matrix @ vector)).reshape(-1)[0])


def _edge_source_vector_metrics(mesh, active: np.ndarray) -> dict[str, Any]:
    return {
        "active_count_by_orientation": _orientation_counts(
            active,
            [("x", mesh.nEx), ("y", mesh.nEy), ("z", mesh.nEz)],
        )
    }


def _receiver_static_response(
    simulation,
    source_vector: np.ndarray,
    field_location: str,
    *,
    mu: float,
) -> dict[str, Any]:
    magnetic_receivers = [
        receiver
        for receiver in simulation.receivers
        if receiver.uses_magnetic_field_vector
    ]
    if not magnetic_receivers:
        return {"present": False, "reason": "no magnetic receivers"}
    mode = str(getattr(simulation, "magnetic_receiver_mode", "")).strip().lower()
    if field_location != "face" or mode != "current_biot":
        return {
            "present": False,
            "reason": (
                "receiver static response is currently implemented for "
                "H/J face current_biot only"
            ),
            "magnetic_receiver_mode": mode,
        }

    locations, receiver_location_indices = _unique_receiver_locations(magnetic_receivers)
    matrix = face_current_biot_matrix(
        simulation.mesh,
        locations,
        subdivisions=int(getattr(simulation, "magnetic_recovery_subdivisions", 1)),
    )
    h_vectors = np.einsum("kcf,f->kc", matrix, source_vector)
    component_values = []
    for receiver, location_index in zip(magnetic_receivers, receiver_location_indices):
        component_values.append(
            {
                "location": _float_list(receiver.location),
                "component": receiver.component,
                "value": receiver.sample_magnetic_field_vector(
                    h_vectors[location_index],
                    mu=mu,
                ),
            }
        )
    return {
        "present": True,
        "receiver_matrix": "face_current_biot",
        "magnetic_receiver_mode": mode,
        "subdivisions": int(getattr(simulation, "magnetic_recovery_subdivisions", 1)),
        "locations": [_float_list(location) for location in locations],
        "h_vectors": [_float_list(vector) for vector in h_vectors],
        "h_l2": float(np.linalg.norm(h_vectors)),
        "component_values": component_values,
    }


def _unique_receiver_locations(receivers) -> tuple[np.ndarray, list[int]]:
    locations: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for receiver in receivers:
        location = tuple(float(value) for value in receiver.location)
        try:
            index = locations.index(location)
        except ValueError:
            index = len(locations)
            locations.append(location)
        indices.append(index)
    return np.asarray(locations, dtype=float), indices


def _orientation_counts(
    active: np.ndarray,
    blocks: list[tuple[str, int]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    start = 0
    for label, size in blocks:
        stop = start + int(size)
        counts[label] = int(np.count_nonzero(active[start:stop]))
        start = stop
    return counts


def _active_bounds(mesh, active: np.ndarray, field_location: str) -> dict[str, Any]:
    grids = _location_grids(mesh, field_location)
    locations = grids[active]
    if locations.size == 0:
        return {"min": None, "max": None}
    return {
        "min": _float_list(np.min(locations, axis=0)),
        "max": _float_list(np.max(locations, axis=0)),
    }


def _location_grids(mesh, field_location: str) -> np.ndarray:
    if field_location == "face":
        return np.vstack([mesh.gridFx, mesh.gridFy, mesh.gridFz])
    return np.vstack([mesh.gridEx, mesh.gridEy, mesh.gridEz])


def _source_midpoint_conductivity(mesh, ip_model, source) -> float:
    conductivity = _low_frequency_conductivity(ip_model)
    if conductivity.size == 1:
        sigma = float(conductivity.reshape(-1)[0])
    elif conductivity.shape == (mesh.n_cells,):
        sigma = float(conductivity[_midpoint_cell_index(mesh, source)])
    else:
        raise ValueError(
            "source geometry audit requires scalar or cell-centered conductivity"
        )
    if sigma <= 0.0:
        raise ValueError("source midpoint conductivity must be positive")
    return sigma


def _midpoint_cell_index(mesh, source) -> int:
    midpoint = np.mean(source.locations, axis=0).reshape(1, 3)
    if hasattr(mesh, "closest_points_index"):
        index = mesh.closest_points_index(midpoint, "CC")
        return int(np.asarray(index).reshape(-1)[0])
    centers = np.asarray(mesh.cell_centers, dtype=float)
    distances = np.sum((centers - midpoint[0]) ** 2, axis=1)
    return int(np.argmin(distances))


def _midpoint_cell_metrics(mesh, source) -> dict[str, Any]:
    midpoint = np.mean(source.locations, axis=0)
    indices = [
        _axis_cell_index(np.asarray(nodes, dtype=float), float(value))
        for nodes, value in zip((mesh.nodes_x, mesh.nodes_y, mesh.nodes_z), midpoint)
    ]
    widths = [
        float(np.asarray(axis_widths, dtype=float)[index])
        for axis_widths, index in zip(mesh.h, indices)
    ]
    center = [
        float((np.asarray(nodes, dtype=float)[index] + np.asarray(nodes, dtype=float)[index + 1]) / 2.0)
        for nodes, index in zip((mesh.nodes_x, mesh.nodes_y, mesh.nodes_z), indices)
    ]
    return {
        "axis_indices": [int(value) for value in indices],
        "flat_index": _midpoint_cell_index(mesh, source),
        "center": center,
        "widths_m": widths,
        "volume_m3": float(np.prod(widths)),
    }


def _axis_cell_index(nodes: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(nodes, value, side="right") - 1)
    return int(np.clip(index, 0, nodes.size - 2))


def _configured_source_diffusion(correction_config, tau0: float) -> dict[str, Any]:
    if correction_config is None:
        return {"present": False}
    corrections = source_history_correction_terms(correction_config)
    source_diffusion_terms = [
        term
        for term in corrections
        if isinstance(term, SourceDiffusionKernelSourceHistoryCorrection)
    ]
    if not source_diffusion_terms:
        return {"present": False}
    if len(source_diffusion_terms) > 1:
        raise ValueError("source geometry audit supports at most one source-diffusion term")
    term = source_diffusion_terms[0]
    if term.normalized_amplitude is None:
        coefficients = np.asarray(term.coefficients, dtype=float)
        amplitude = (
            float(coefficients[0])
            if len(term.source_moment_degrees) == 1
            else float(coefficients[term.source_moment_degrees.index(0)])
        )
        normalized = amplitude / tau0
    else:
        normalized = float(term.normalized_amplitude)
        amplitude = normalized * tau0
    return {
        "present": True,
        "kind": term.kind,
        "source_moment_degrees": [int(value) for value in term.source_moment_degrees],
        "tau_multiplier": float(term.tau_multiplier),
        "amplitude_time_s": float(term.amplitude_time),
        "basis_kind": term.basis_kind,
        "amplitude": float(amplitude),
        "normalized_amplitude": float(normalized),
    }


def _load_sweep_cases(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if isinstance(payload, dict) and "cases" in payload:
        return payload["cases"]
    return payload


def _normalize_sweep_cases(cases: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(cases, Mapping):
        return [(str(name), dict(overrides)) for name, overrides in cases.items()]
    if not isinstance(cases, list):
        raise ValueError("sweep cases must be a mapping or a list")
    normalized: list[tuple[str, dict[str, Any]]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("each sweep case must be a mapping")
        name = str(case.get("name", f"case_{index}"))
        overrides = case.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("sweep case overrides must be a mapping")
        normalized.append((name, dict(overrides)))
    return normalized


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    _merge_into(merged, overrides)
    return merged


def _merge_into(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and isinstance(target.get(key), dict)
        ):
            _merge_into(target[key], value)
        else:
            target[key] = deepcopy(value)


def _float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


if __name__ == "__main__":
    raise SystemExit(main())
