"""Diagnostics for step-off initial magnetic fields."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from .config import build_simulation
from .empymod_compare import EmpymodSurvey, build_empymod_survey_from_result, run_empymod_reference


def run_initial_field_diagnostics(
    result_path: str | Path,
    modes: list[str],
    empymod_depths: list[float] | None = None,
    empymod_resistivities: list[float] | None = None,
    empymod_frequency: float = 1.0e-6,
    srcpts: int = 51,
    recpts: int = 1,
) -> dict:
    """Return initial-field diagnostics for each requested magnetic mode."""

    result_path = Path(result_path)
    with h5py.File(result_path, "r") as h5:
        config = yaml.safe_load(h5.attrs.get("config_yaml", "{}"))

    entries = []
    for mode in modes:
        mode_config = dict(config)
        mode_config["initial_magnetic_field"] = str(mode)
        simulation = build_simulation(mode_config)
        e0 = simulation.initial_electric_field()
        b0 = simulation.initial_magnetic_flux_density(e0)
        entries.append(_diagnose_mode(simulation, e0, b0, str(mode)))

    report = {
        "result_path": str(result_path),
        "modes": entries,
    }
    if empymod_depths is not None or empymod_resistivities is not None:
        if empymod_depths is None or empymod_resistivities is None:
            raise ValueError("empymod_depths and empymod_resistivities must be provided together")
        report["empymod_frequency_reference"] = _empymod_frequency_reference(
            result_path,
            empymod_depths,
            empymod_resistivities,
            empymod_frequency,
            srcpts,
            recpts,
        )
    return report


def _empymod_frequency_reference(
    result_path: Path,
    depths: list[float],
    resistivities: list[float],
    frequency: float,
    srcpts: int,
    recpts: int,
) -> dict:
    survey, names = build_empymod_survey_from_result(
        result_path,
        depths=depths,
        resistivities=resistivities,
        signal=None,
    )
    frequency_survey = EmpymodSurvey(
        source_start=survey.source_start,
        source_end=survey.source_end,
        receiver_locations=survey.receiver_locations,
        components=survey.components,
        times=np.array([float(frequency)]),
        depths=survey.depths,
        resistivities=survey.resistivities,
        strength=survey.strength,
        signal=None,
        receiver_components=survey.receiver_components,
        coordinate_system=survey.coordinate_system,
    )
    reference = run_empymod_reference(frequency_survey, srcpts=srcpts, recpts=recpts)
    return {
        "frequency": float(frequency),
        "depths": [float(value) for value in depths],
        "resistivities": [float(value) for value in resistivities],
        "srcpts": int(srcpts),
        "recpts": int(recpts),
        "receivers": {
            name: float(value)
            for name, value in zip(names, reference[0])
        },
    }


def _diagnose_mode(simulation, e0: np.ndarray, b0: np.ndarray, mode: str) -> dict:
    mesh = simulation.mesh
    source_vec = np.zeros(mesh.n_edges, dtype=float)
    for source in simulation.sources:
        source_vec += source.initial_edge_vector(mesh)
    current = (
        mesh.get_edge_inner_product(simulation.ip_model.low_frequency_sigma()).tocsr() @ e0
        + source_vec
    )
    ampere_residual = mesh.edge_curl.T @ simulation.face_mu_inverse_matrix @ b0 - current
    current_norm = float(np.linalg.norm(current))
    residual_norm = float(np.linalg.norm(ampere_residual))
    divergence = mesh.face_divergence @ b0

    return {
        "mode": mode,
        "b_norm": float(np.linalg.norm(b0)),
        "divergence_norm": float(np.linalg.norm(divergence)),
        "ampere_residual_norm": residual_norm,
        "ampere_relative_residual": (
            residual_norm / current_norm if current_norm > 0.0 else residual_norm
        ),
        "receivers": {
            _receiver_name(receiver): receiver.sample(mesh, e0, b0, simulation.mu)
            for receiver in simulation.receivers
        },
    }


def _receiver_name(receiver) -> str:
    return f"{receiver.component}@x={receiver.location[0]:g}"
