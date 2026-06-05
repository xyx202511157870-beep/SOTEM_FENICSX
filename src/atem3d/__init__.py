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
from .waveforms import (
    LinearRampOffWaveform,
    StepOffWaveform,
    TabulatedWaveform,
    Waveform,
    build_internal_time_grid,
)

__all__ = [
    "LinearResponseFit",
    "LinearRampOffWaveform",
    "StepOffWaveform",
    "TabulatedWaveform",
    "Waveform",
    "absolute_linf",
    "build_internal_time_grid",
    "fit_linear_response_components",
    "relative_l2",
    "relative_linf",
    "robust_component_errors",
    "robust_relative_error",
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
            "PointReceiver": PointReceiver,
            "ReceiverDataResult": ReceiverDataResult,
            "SimulationResult": SimulationResult,
            "TDEMIPSimulation": TDEMIPSimulation,
            "ValidationCase": ValidationCase,
            "build_simulation": build_simulation,
            "cole_cole_conductivity": cole_cole_conductivity,
            "fit_cole_cole_conductivity_debye": fit_cole_cole_conductivity_debye,
            "fit_pelton_resistivity_debye": fit_pelton_resistivity_debye,
            "load_config": load_config,
            "pelton_resistivity_to_conductivity": pelton_resistivity_to_conductivity,
            "write_validation_report": write_validation_report,
        }
    )
    __all__.extend(
        [
            "DebyeFitResult",
            "DebyeIPModel",
            "DebyeTerm",
            "GroundedWireSource",
            "PointReceiver",
            "ReceiverDataResult",
            "SimulationResult",
            "TDEMIPSimulation",
            "ValidationCase",
            "build_simulation",
            "cole_cole_conductivity",
            "fit_cole_cole_conductivity_debye",
            "fit_pelton_resistivity_debye",
            "load_config",
            "pelton_resistivity_to_conductivity",
            "write_validation_report",
        ]
    )


_try_load_heavy_api()
