"""Direct time-domain grounded-wire TDEM solver with Debye IP coupling."""

from __future__ import annotations

import os
from pathlib import Path


if not os.environ.get("NUMBA_CACHE_DIR"):
    _numba_cache_dir = Path.cwd() / ".numba_cache"
    _numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_numba_cache_dir.resolve())

from .metrics import (
    LinearResponseFit,
    absolute_linf,
    fit_linear_response_components,
    relative_l2,
    relative_linf,
    robust_component_errors,
    robust_relative_error,
    summarize_errors,
)
from .corrected_model import (
    CorrectedModelValidationConfig,
    build_corrected_model_case_specs,
)
from .receivers import AverageReceiver, PointReceiver, build_receiver
from .waveforms import (
    LinearRampOffWaveform,
    StepOffWaveform,
    TabulatedWaveform,
    Waveform,
    build_internal_time_grid,
    build_internal_time_grid_from_turnoff,
    summarize_internal_time_grid,
)

__all__ = [
    "LinearResponseFit",
    "LinearRampOffWaveform",
    "AverageReceiver",
    "CorrectedModelValidationConfig",
    "PointReceiver",
    "StepOffWaveform",
    "TabulatedWaveform",
    "Waveform",
    "absolute_linf",
    "build_corrected_model_case_specs",
    "build_internal_time_grid",
    "build_internal_time_grid_from_turnoff",
    "build_receiver",
    "fit_linear_response_components",
    "relative_l2",
    "relative_linf",
    "robust_component_errors",
    "robust_relative_error",
    "summarize_internal_time_grid",
    "summarize_errors",
]


def _try_load_heavy_api() -> None:
    """Expose the simulation API when optional modelling dependencies exist."""

    try:
        from .ip import DebyeIPModel, DebyeTerm
        from .receivers import PointReceiver
        from .sources import GroundedWireSource
        from .simulation import ReceiverDataResult, SimulationResult, TDEMIPSimulation
        from .config import build_simulation, load_config
        from .validation import ValidationCase, write_validation_report
        from .recovery_spectrum import (
            LocalMagneticDiffusionSpectrum,
            LocalTensorMeshSupport,
            MagneticDiffusionModalCoupling,
            MagneticDiffusionSpectrum,
            ModalSourceMomentProjection,
            local_magnetic_diffusion_positive_spectrum,
            magnetic_diffusion_driven_response,
            magnetic_diffusion_matrices,
            magnetic_diffusion_mmr_initial_state,
            magnetic_diffusion_modal_coupling,
            magnetic_diffusion_positive_spectrum,
            magnetic_diffusion_time_constants,
            project_modal_response_to_source_moments,
            tensor_mesh_cell_submesh,
        )
        from .fit import (
            DebyeFitResult,
            cole_cole_conductivity,
            fit_cole_cole_conductivity_debye,
            fit_pelton_resistivity_debye,
            pelton_resistivity_to_conductivity,
        )
    except ModuleNotFoundError as exc:
        optional = {"discretize", "simpeg", "pymatsolver", "empymod"}
        if exc.name in optional:
            return
        raise

    globals().update(
        {
            "DebyeFitResult": DebyeFitResult,
            "DebyeIPModel": DebyeIPModel,
            "DebyeTerm": DebyeTerm,
            "GroundedWireSource": GroundedWireSource,
            "LocalMagneticDiffusionSpectrum": LocalMagneticDiffusionSpectrum,
            "LocalTensorMeshSupport": LocalTensorMeshSupport,
            "MagneticDiffusionModalCoupling": MagneticDiffusionModalCoupling,
            "MagneticDiffusionSpectrum": MagneticDiffusionSpectrum,
            "ModalSourceMomentProjection": ModalSourceMomentProjection,
            "PointReceiver": PointReceiver,
            "ReceiverDataResult": ReceiverDataResult,
            "SimulationResult": SimulationResult,
            "TDEMIPSimulation": TDEMIPSimulation,
            "ValidationCase": ValidationCase,
            "build_simulation": build_simulation,
            "cole_cole_conductivity": cole_cole_conductivity,
            "fit_cole_cole_conductivity_debye": fit_cole_cole_conductivity_debye,
            "fit_pelton_resistivity_debye": fit_pelton_resistivity_debye,
            "local_magnetic_diffusion_positive_spectrum": local_magnetic_diffusion_positive_spectrum,
            "load_config": load_config,
            "magnetic_diffusion_driven_response": magnetic_diffusion_driven_response,
            "magnetic_diffusion_matrices": magnetic_diffusion_matrices,
            "magnetic_diffusion_mmr_initial_state": magnetic_diffusion_mmr_initial_state,
            "magnetic_diffusion_modal_coupling": magnetic_diffusion_modal_coupling,
            "magnetic_diffusion_positive_spectrum": magnetic_diffusion_positive_spectrum,
            "magnetic_diffusion_time_constants": magnetic_diffusion_time_constants,
            "pelton_resistivity_to_conductivity": pelton_resistivity_to_conductivity,
            "project_modal_response_to_source_moments": project_modal_response_to_source_moments,
            "tensor_mesh_cell_submesh": tensor_mesh_cell_submesh,
            "write_validation_report": write_validation_report,
        }
    )
    __all__.extend(
        [
            "DebyeFitResult",
            "DebyeIPModel",
            "DebyeTerm",
            "GroundedWireSource",
            "LocalMagneticDiffusionSpectrum",
            "LocalTensorMeshSupport",
            "MagneticDiffusionModalCoupling",
            "MagneticDiffusionSpectrum",
            "ModalSourceMomentProjection",
            "PointReceiver",
            "ReceiverDataResult",
            "SimulationResult",
            "TDEMIPSimulation",
            "ValidationCase",
            "build_simulation",
            "cole_cole_conductivity",
            "fit_cole_cole_conductivity_debye",
            "fit_pelton_resistivity_debye",
            "local_magnetic_diffusion_positive_spectrum",
            "load_config",
            "magnetic_diffusion_driven_response",
            "magnetic_diffusion_matrices",
            "magnetic_diffusion_mmr_initial_state",
            "magnetic_diffusion_modal_coupling",
            "magnetic_diffusion_positive_spectrum",
            "magnetic_diffusion_time_constants",
            "pelton_resistivity_to_conductivity",
            "project_modal_response_to_source_moments",
            "tensor_mesh_cell_submesh",
            "write_validation_report",
        ]
    )


_try_load_heavy_api()
