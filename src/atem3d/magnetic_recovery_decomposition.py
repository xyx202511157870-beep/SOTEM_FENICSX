"""Diagnostic decomposition of magnetic receiver recovery terms."""

from __future__ import annotations

from typing import Any

import numpy as np

from .magnetic_recovery import (
    biot_savart_h_from_cell_currents,
    biot_savart_h_from_edge_basis_cell_ip_currents,
    biot_savart_h_from_edge_basis_currents,
    biot_savart_h_from_edge_current_moments,
    biot_savart_h_from_face_basis_cell_ip_currents,
    biot_savart_h_from_face_basis_currents,
)


def magnetic_recovery_decomposition_at_time(
    simulation,
    result,
    *,
    time_index: int,
    include_fields: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready magnetic receiver recovery decomposition.

    The report is diagnostic-only: each term is evaluated with the same recovery
    operators used by the runtime receiver sampler, and the reported total is
    checked against the saved receiver data at ``time_index``.
    """

    times = np.asarray(result.times, dtype=float)
    index = int(time_index)
    if index < 0 or index >= times.size:
        raise ValueError("time_index is out of range")
    magnetic_indices = _magnetic_receiver_indices(simulation)
    if not magnetic_indices:
        raise ValueError("simulation has no magnetic H/B receivers to decompose")

    locations = np.asarray(
        [simulation.receivers[item].location for item in magnetic_indices],
        dtype=float,
    )
    time = float(times[index])
    formulation = _formulation(result)
    if formulation == "eb" and index == 0:
        raise ValueError(
            "EB current-Biot decomposition is defined for positive time nodes; "
            "t=0 receiver data are sampled from the stored initial B field"
        )

    if formulation == "eb":
        terms, details = _eb_terms(simulation, result, index, locations, time)
    else:
        terms, details = _hj_terms(simulation, result, index, locations, time)

    total_field = np.zeros((len(magnetic_indices), 3), dtype=float)
    for term in terms.values():
        total_field += term
    receiver_data = np.asarray(result.data[index, magnetic_indices], dtype=float)
    term_data = {
        name: _receiver_data_from_h(simulation, magnetic_indices, field)
        for name, field in terms.items()
    }
    detail_data = {
        name: _term_payload(
            simulation,
            magnetic_indices,
            field,
            include_fields=include_fields,
        )
        for name, field in details.items()
    }
    total_data = _receiver_data_from_h(simulation, magnetic_indices, total_field)
    residual = total_data - receiver_data

    return {
        "diagnostic_only": True,
        "formulation": formulation,
        "time_index": index,
        "time": time,
        "receiver_indices": [int(item) for item in magnetic_indices],
        "components": [simulation.receivers[item].component for item in magnetic_indices],
        "locations": locations.tolist(),
        "terms": {
            name: _term_payload(
                simulation,
                magnetic_indices,
                field,
                include_fields=include_fields,
            )
            for name, field in terms.items()
        },
        "term_details": detail_data,
        "total_data": total_data.tolist(),
        "numerical_data": receiver_data.tolist(),
        "data_residual": residual.tolist(),
        "relative_l2_to_numerical": _relative_l2(total_data, receiver_data),
        "term_l2_norms": {
            name: float(np.linalg.norm(data)) for name, data in term_data.items()
        },
    }


def summarize_decompositions(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one or more decomposition samples."""

    if not samples:
        return {
            "sample_count": 0,
            "total_recomputed_relative_l2_to_numerical": None,
            "term_l2_norms": {},
        }
    total = np.concatenate([np.asarray(sample["total_data"], dtype=float) for sample in samples])
    numerical = np.concatenate(
        [np.asarray(sample["numerical_data"], dtype=float) for sample in samples]
    )
    term_names = sorted(
        {
            name
            for sample in samples
            for name in sample.get("terms", {})
        }
    )
    term_norms = {}
    for name in term_names:
        values = []
        for sample in samples:
            data = sample.get("terms", {}).get(name, {}).get("data")
            if data is not None:
                values.append(np.asarray(data, dtype=float))
        if values:
            term_norms[name] = float(np.linalg.norm(np.concatenate(values)))
    total_norm = float(np.linalg.norm(total))
    return {
        "sample_count": int(len(samples)),
        "total_recomputed_relative_l2_to_numerical": _relative_l2(total, numerical),
        "total_l2_norm": total_norm,
        "numerical_l2_norm": float(np.linalg.norm(numerical)),
        "term_l2_norms": term_norms,
        "term_fraction_of_total_l2": {
            name: (None if total_norm == 0.0 else float(norm / total_norm))
            for name, norm in term_norms.items()
        },
    }


def align_decomposition_samples_to_validation(
    samples: list[dict[str, Any]],
    validation_report: dict[str, Any],
    *,
    time_atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Project validation residuals onto decomposition terms.

    The target is ``reference - numerical``: the response that would have to be
    added to the current ATEM3D magnetic data to match the validation reference.
    """

    if not samples:
        raise ValueError("at least one decomposition sample is required")
    validation_samples = validation_report.get("samples")
    if not isinstance(validation_samples, dict) or not validation_samples:
        raise ValueError("validation report must contain a non-empty samples object")
    time_atol = float(time_atol)
    if time_atol < 0.0:
        raise ValueError("time_atol must be nonnegative")

    component_names = _validation_component_names(
        validation_report,
        samples[0]["components"],
    )
    arrays = _alignment_arrays(
        samples,
        validation_samples,
        component_names,
        time_atol=time_atol,
    )
    target = arrays["reference"] - arrays["validation_numerical"]
    term_names = sorted(arrays["terms"])
    term_alignment = {
        name: _single_term_alignment(arrays["terms"][name], target)
        for name in term_names
    }
    best_name = min(
        term_alignment,
        key=lambda name: term_alignment[name]["relative_l2_after_best_scalar"],
    )
    design = np.column_stack([arrays["terms"][name] for name in term_names])
    coefficients, residual_relative_l2, rank, singular_values = _least_squares_alignment(
        design,
        target,
    )
    return {
        "diagnostic_only": True,
        "target": "reference_minus_numerical",
        "time_count": int(len(samples)),
        "component_count": int(len(component_names)),
        "sample_count": int(target.size),
        "validation_component_names": component_names,
        "target_l2_norm": float(np.linalg.norm(target)),
        "decomposition_to_validation_numerical_relative_l2": _relative_l2(
            arrays["decomposition_numerical"],
            arrays["validation_numerical"],
        ),
        "term_alignment": term_alignment,
        "best_single_term": {
            "name": best_name,
            "best_scalar": term_alignment[best_name]["best_scalar"],
            "relative_l2_after_best_scalar": term_alignment[best_name][
                "relative_l2_after_best_scalar"
            ],
        },
        "all_terms_fit": {
            "coefficients": {
                name: float(value) for name, value in zip(term_names, coefficients)
            },
            "relative_l2_after_fit": float(residual_relative_l2),
            "rank": int(rank),
            "singular_values": [float(value) for value in singular_values],
        },
        "per_time_alignment": [
            _per_time_alignment(
                sample,
                validation_samples,
                component_names,
                time_atol=time_atol,
            )
            for sample in samples
        ],
    }


def _eb_terms(sim, result, index: int, locations: np.ndarray, time: float):
    if sim.magnetic_receiver_mode not in {
        "current_biot",
        "edge_current_biot",
        "edge_basis_biot",
        "edge_basis_cell_biot",
    }:
        raise ValueError("EB decomposition requires a Biot magnetic_receiver_mode")
    memories = _eb_memories_at(result, index)
    initial_memories = _eb_memories_at(result, 0)
    if sim.magnetic_receiver_mode == "edge_basis_cell_biot":
        terms, details = _eb_edge_basis_cell_terms(
            sim,
            result.e[index],
            memories,
            initial_memories,
            locations,
        )
    else:
        terms, details = _eb_edge_moment_terms(
            sim,
            result.e[index],
            memories,
            initial_memories,
            locations,
        )

    source_current = sim._source_magnetic_field(locations, time)
    source_primary = (
        sim._source_primary_delta6_magnetic_field(locations, time)
        if sim.magnetic_recovery_source_primary_delta6
        else np.zeros_like(source_current)
    )
    source_history = (
        sim._source_history_magnetic_field(locations, time, initial_memories)
        if sim.magnetic_recovery_source_history is not None
        else np.zeros_like(source_current)
    )
    terms.update(
        {
            "source_current": source_current,
            "source_primary_delta6": source_primary,
            "source_history": source_history,
        }
    )
    return terms, details


def _hj_terms(sim, result, index: int, locations: np.ndarray, time: float):
    if sim.magnetic_receiver_mode == "stored_h":
        raise ValueError("H/J decomposition requires a Biot magnetic_receiver_mode")
    memories = [np.asarray(memory[index], dtype=float) for memory in result.memories]
    initial_memories = [np.asarray(memory[0], dtype=float) for memory in result.memories]
    zero = np.zeros((locations.shape[0], 3), dtype=float)

    if sim.magnetic_receiver_mode == "face_basis_cell_biot":
        terms, details = _hj_face_basis_cell_terms(
            sim,
            result.e[index],
            memories,
            initial_memories,
            locations,
        )
    elif sim.magnetic_receiver_mode == "face_basis_biot":
        field = sim._face_basis_biot_h(result.h[index], locations, time)
        terms = {
            "conductive_current": field,
            "ohmic_current": zero.copy(),
            "polarization_memory": zero.copy(),
            "initial_polarization": zero.copy(),
        }
        details = {}
    else:
        field = sim._current_biot_h(result.h[index], locations, time)
        terms = {
            "conductive_current": field,
            "ohmic_current": zero.copy(),
            "polarization_memory": zero.copy(),
            "initial_polarization": zero.copy(),
        }
        details = {}

    source_primary = (
        sim._source_primary_delta6_magnetic_field(locations, time)
        if sim.magnetic_recovery_source_primary_delta6
        else zero.copy()
    )
    source_history = (
        sim._source_history_magnetic_field(locations, time, initial_memories)
        if sim.magnetic_recovery_source_history is not None
        else zero.copy()
    )
    terms.update(
        {
            "source_current": zero.copy(),
            "source_primary_delta6": source_primary,
            "source_history": source_history,
        }
    )
    return terms, details


def _eb_edge_moment_terms(sim, e, memories, initial_memories, locations):
    ohmic_moments = sim.mesh.get_edge_inner_product(sim.ip_model.sigma_infinity).tocsr() @ e
    ohmic = _recover_eb_edge_moments(sim, ohmic_moments, locations)
    details = {}
    polarization = np.zeros_like(ohmic)
    for term_index, (term, memory) in enumerate(zip(sim.ip_model.terms, memories)):
        field = _recover_eb_edge_moments(
            sim,
            -(sim._polarization_current_matrix(term) @ memory),
            locations,
        )
        details[f"polarization_memory_{term_index}"] = field
        polarization += field

    initial = np.zeros_like(ohmic)
    if sim.magnetic_recovery_initial_polarization_scale != 0.0:
        for term_index, (term, memory) in enumerate(
            zip(sim.ip_model.terms, initial_memories)
        ):
            field = _recover_eb_edge_moments(
                sim,
                sim.magnetic_recovery_initial_polarization_scale
                * (sim.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ memory),
                locations,
            )
            details[f"initial_polarization_{term_index}"] = field
            initial += field
    return (
        {
            "ohmic_current": ohmic,
            "polarization_memory": polarization,
            "initial_polarization": initial,
        },
        details,
    )


def _eb_edge_basis_cell_terms(sim, e, memories, initial_memories, locations):
    zero_edge = np.zeros(sim.mesh.n_edges, dtype=float)
    ohmic = biot_savart_h_from_edge_basis_cell_ip_currents(
        sim.mesh,
        e,
        sim.ip_model.sigma_infinity,
        [],
        [],
        locations,
        subdivisions=sim.magnetic_recovery_subdivisions,
    )
    details = {}
    polarization = np.zeros_like(ohmic)
    for term_index, (term, memory) in enumerate(zip(sim.ip_model.terms, memories)):
        field = biot_savart_h_from_edge_basis_cell_ip_currents(
            sim.mesh,
            zero_edge,
            sim.ip_model.sigma_infinity,
            [term],
            [memory],
            locations,
            subdivisions=sim.magnetic_recovery_subdivisions,
            polarization_scale=sim.magnetic_recovery_polarization_scale,
        )
        details[f"polarization_memory_{term_index}"] = field
        polarization += field
    initial = np.zeros_like(ohmic)
    if sim.magnetic_recovery_initial_polarization_scale != 0.0:
        for term_index, (term, memory) in enumerate(
            zip(sim.ip_model.terms, initial_memories)
        ):
            field = biot_savart_h_from_edge_basis_cell_ip_currents(
                sim.mesh,
                zero_edge,
                sim.ip_model.sigma_infinity,
                [term],
                [zero_edge],
                locations,
                subdivisions=sim.magnetic_recovery_subdivisions,
                initial_polarization_scale=(
                    sim.magnetic_recovery_initial_polarization_scale
                ),
                initial_memories=[memory],
            )
            details[f"initial_polarization_{term_index}"] = field
            initial += field
    return (
        {
            "ohmic_current": ohmic,
            "polarization_memory": polarization,
            "initial_polarization": initial,
        },
        details,
    )


def _hj_face_basis_cell_terms(sim, e, memories, initial_memories, locations):
    zero_face = np.zeros(sim.mesh.n_faces, dtype=float)
    ohmic = biot_savart_h_from_face_basis_cell_ip_currents(
        sim.mesh,
        e,
        sim.ip_model.sigma_infinity,
        [],
        [],
        locations,
        subdivisions=sim.magnetic_recovery_subdivisions,
    )
    details = {}
    polarization = np.zeros_like(ohmic)
    for term_index, (term, memory) in enumerate(zip(sim.ip_model.terms, memories)):
        field = biot_savart_h_from_face_basis_cell_ip_currents(
            sim.mesh,
            zero_face,
            sim.ip_model.sigma_infinity,
            [term],
            [memory],
            locations,
            subdivisions=sim.magnetic_recovery_subdivisions,
            polarization_scale=sim.magnetic_recovery_polarization_scale,
        )
        details[f"polarization_memory_{term_index}"] = field
        polarization += field
    initial = np.zeros_like(ohmic)
    if sim.magnetic_recovery_initial_polarization_scale != 0.0:
        for term_index, (term, memory) in enumerate(
            zip(sim.ip_model.terms, initial_memories)
        ):
            field = biot_savart_h_from_face_basis_cell_ip_currents(
                sim.mesh,
                zero_face,
                sim.ip_model.sigma_infinity,
                [term],
                [zero_face],
                locations,
                subdivisions=sim.magnetic_recovery_subdivisions,
                initial_polarization_scale=(
                    sim.magnetic_recovery_initial_polarization_scale
                ),
                initial_memories=[memory],
            )
            details[f"initial_polarization_{term_index}"] = field
            initial += field
    return (
        {
            "ohmic_current": ohmic,
            "polarization_memory": polarization,
            "initial_polarization": initial,
        },
        details,
    )


def _recover_eb_edge_moments(sim, edge_moments, locations):
    edge_moments = np.asarray(edge_moments, dtype=float)
    if sim.magnetic_receiver_mode == "edge_current_biot":
        return biot_savart_h_from_edge_current_moments(sim.mesh, edge_moments, locations)
    if sim.magnetic_receiver_mode == "edge_basis_biot":
        return biot_savart_h_from_edge_basis_currents(
            sim.mesh,
            edge_moments / sim.unit_edge_mass_diagonal,
            locations,
            subdivisions=sim.magnetic_recovery_subdivisions,
        )
    edge_current_field = edge_moments / sim.unit_edge_mass_diagonal
    current = sim.mesh.average_edge_to_cell_vector @ edge_current_field
    current = np.asarray(current).reshape((sim.mesh.n_cells, 3), order="F")
    return biot_savart_h_from_cell_currents(
        sim.mesh,
        current,
        locations,
        subdivisions=sim.magnetic_recovery_subdivisions,
    )


def _eb_memories_at(result, index: int) -> list[np.ndarray]:
    memory_history = getattr(result, "memory_history", None)
    if memory_history is None:
        raise ValueError("EB decomposition requires SimulationResult.memory_history")
    return [np.asarray(memory[index], dtype=float) for memory in memory_history]


def _magnetic_receiver_indices(simulation) -> list[int]:
    return [
        index
        for index, receiver in enumerate(simulation.receivers)
        if receiver.uses_magnetic_field_vector
    ]


def _receiver_data_from_h(simulation, receiver_indices: list[int], field: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            simulation.receivers[receiver_index].sample_magnetic_field_vector(
                field[local_index],
                simulation.mu,
            )
            for local_index, receiver_index in enumerate(receiver_indices)
        ],
        dtype=float,
    )


def _term_payload(
    simulation,
    receiver_indices: list[int],
    field: np.ndarray,
    *,
    include_fields: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data": _receiver_data_from_h(simulation, receiver_indices, field).tolist(),
        "data_l2_norm": float(
            np.linalg.norm(_receiver_data_from_h(simulation, receiver_indices, field))
        ),
    }
    if include_fields:
        payload["h_field"] = np.asarray(field, dtype=float).tolist()
    return payload


def _relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(numerical) - np.asarray(reference)))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _validation_component_names(
    validation_report: dict[str, Any],
    components: list[str],
) -> list[str]:
    groups = validation_report.get("component_groups")
    if isinstance(groups, dict):
        magnetic = groups.get("magnetic")
        if isinstance(magnetic, dict):
            names = magnetic.get("components")
            if isinstance(names, list):
                selected = [str(name) for name in names]
                if _component_names_match(selected, components):
                    return selected
    samples = validation_report.get("samples", {})
    selected = [
        str(name)
        for name in samples
        if _component_prefix(str(name)) in set(str(component) for component in components)
    ]
    if _component_names_match(selected, components):
        return selected
    raise ValueError(
        "validation magnetic component names do not match decomposition receivers"
    )


def _component_names_match(names: list[str], components: list[str]) -> bool:
    if len(names) != len(components):
        return False
    return all(
        _component_prefix(name) == str(component)
        for name, component in zip(names, components)
    )


def _component_prefix(name: str) -> str:
    return str(name).split("@", 1)[0]


def _alignment_arrays(
    samples: list[dict[str, Any]],
    validation_samples: dict[str, list[dict[str, Any]]],
    component_names: list[str],
    *,
    time_atol: float,
) -> dict[str, Any]:
    term_names = sorted(
        {
            name
            for sample in samples
            for name in sample.get("terms", {})
        }
    )
    decomposition_numerical = []
    validation_numerical = []
    reference = []
    terms = {name: [] for name in term_names}

    for sample in samples:
        time = float(sample["time"])
        numerical = np.asarray(sample["numerical_data"], dtype=float)
        for local_index, component_name in enumerate(component_names):
            row = _validation_row_at_time(
                validation_samples,
                component_name,
                time,
                time_atol=time_atol,
            )
            decomposition_numerical.append(float(numerical[local_index]))
            validation_numerical.append(float(row["numerical"]))
            reference.append(float(row["reference"]))
            for term_name in term_names:
                term_data = sample.get("terms", {}).get(term_name, {}).get("data")
                if term_data is None:
                    value = 0.0
                else:
                    value = float(np.asarray(term_data, dtype=float)[local_index])
                terms[term_name].append(value)
    return {
        "decomposition_numerical": np.asarray(decomposition_numerical, dtype=float),
        "validation_numerical": np.asarray(validation_numerical, dtype=float),
        "reference": np.asarray(reference, dtype=float),
        "terms": {
            name: np.asarray(values, dtype=float) for name, values in terms.items()
        },
    }


def _per_time_alignment(
    sample: dict[str, Any],
    validation_samples: dict[str, list[dict[str, Any]]],
    component_names: list[str],
    *,
    time_atol: float,
) -> dict[str, Any]:
    arrays = _alignment_arrays(
        [sample],
        validation_samples,
        component_names,
        time_atol=time_atol,
    )
    target = arrays["reference"] - arrays["validation_numerical"]
    term_names = sorted(arrays["terms"])
    term_alignment = {
        name: _single_term_alignment(arrays["terms"][name], target)
        for name in term_names
    }
    best_name = min(
        term_alignment,
        key=lambda name: term_alignment[name]["relative_l2_after_best_scalar"],
    )
    design = np.column_stack([arrays["terms"][name] for name in term_names])
    coefficients, residual_relative_l2, rank, singular_values = _least_squares_alignment(
        design,
        target,
    )
    return {
        "time_index": int(sample["time_index"]),
        "time": float(sample["time"]),
        "target_l2_norm": float(np.linalg.norm(target)),
        "decomposition_to_validation_numerical_relative_l2": _relative_l2(
            arrays["decomposition_numerical"],
            arrays["validation_numerical"],
        ),
        "term_alignment": term_alignment,
        "best_single_term": {
            "name": best_name,
            "best_scalar": term_alignment[best_name]["best_scalar"],
            "relative_l2_after_best_scalar": term_alignment[best_name][
                "relative_l2_after_best_scalar"
            ],
        },
        "all_terms_fit": {
            "coefficients": {
                name: float(value) for name, value in zip(term_names, coefficients)
            },
            "relative_l2_after_fit": float(residual_relative_l2),
            "rank": int(rank),
            "singular_values": [float(value) for value in singular_values],
        },
    }


def _validation_row_at_time(
    validation_samples: dict[str, list[dict[str, Any]]],
    component_name: str,
    time: float,
    *,
    time_atol: float,
) -> dict[str, Any]:
    if component_name not in validation_samples:
        raise ValueError(f"validation report has no samples for {component_name!r}")
    rows = validation_samples[component_name]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"validation samples for {component_name!r} are empty")
    differences = np.asarray([abs(float(row["time"]) - time) for row in rows])
    index = int(np.argmin(differences))
    if float(differences[index]) > time_atol:
        raise ValueError(
            f"no validation sample for {component_name!r} at time {time:g} "
            f"within {time_atol:g}"
        )
    return rows[index]


def _single_term_alignment(term: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    term = np.asarray(term, dtype=float)
    target = np.asarray(target, dtype=float)
    target_norm = float(np.linalg.norm(target))
    term_norm = float(np.linalg.norm(term))
    if term_norm == 0.0:
        relative = 0.0 if target_norm == 0.0 else 1.0
        return {
            "best_scalar": 0.0,
            "corrected_existing_term_multiplier": 1.0,
            "term_l2_norm": 0.0,
            "relative_l2_after_best_scalar": relative,
            "cosine": None,
            "explained_energy_fraction": 0.0,
        }
    scalar = float(np.dot(term, target) / np.dot(term, term))
    residual = target - scalar * term
    relative = _relative_norm(residual, target)
    cosine = None if target_norm == 0.0 else float(np.dot(term, target) / (term_norm * target_norm))
    return {
        "best_scalar": scalar,
        "corrected_existing_term_multiplier": float(1.0 + scalar),
        "term_l2_norm": term_norm,
        "relative_l2_after_best_scalar": relative,
        "cosine": cosine,
        "explained_energy_fraction": float(max(0.0, 1.0 - relative**2)),
    }


def _least_squares_alignment(
    design: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    if design.size == 0:
        return np.zeros(0, dtype=float), _relative_norm(target, target), 0, np.zeros(0)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return coefficients, _relative_norm(residual, target), int(rank), singular_values


def _relative_norm(vector: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(vector, dtype=float)))
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=float)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _formulation(result) -> str:
    if hasattr(result, "b"):
        return "eb"
    if hasattr(result, "h"):
        return "hj"
    raise ValueError("result must contain EB b or H/J h field histories")
