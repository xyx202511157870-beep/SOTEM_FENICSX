"""Canonical corrected-model validation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

CORRECTED_SOURCE_START = (-500.0, 200.0, -0.1)
CORRECTED_SOURCE_END = (500.0, 200.0, -0.1)
CORRECTED_RECEIVER = (0.0, -300.0, -0.1)
CORRECTED_SOURCE_LENGTH = 1000.0
CORRECTED_PARALLEL_OFFSET = 500.0


@dataclass(frozen=True)
class CorrectedModelValidationConfig:
    """Single source of truth for the latest corrected validation model."""

    source_start: tuple[float, float, float] = CORRECTED_SOURCE_START
    source_end: tuple[float, float, float] = CORRECTED_SOURCE_END
    receiver: tuple[float, float, float] = CORRECTED_RECEIVER
    source_current: float = 10.0
    ramp_off_time: float = 1.0e-5
    observation_time_min: float = 1.0e-5
    observation_time_max: float = 1.0
    n_observation_times: int = 80
    components: tuple[str, ...] = ("Ex", "Ey", "dBzdt")
    magnetic_quantity: str = "dBzdt"
    depths: tuple[float, ...] = (350.0, 650.0)
    resistivities: tuple[float, ...] = (100.0, 100.0, 100.0)
    ip_sigma_inf: float = 0.012
    ip_delta_sigma: tuple[float, ...] = (0.002,)
    ip_tau: tuple[float, ...] = (0.1,)
    validation_scope: str = "corrected_model_full"
    reference_type: str = "empymod"
    coordinate_system: str = "depth_down"

    @property
    def source_length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.source_end) - np.asarray(self.source_start)))

    @property
    def parallel_offset(self) -> float:
        source_y = 0.5 * (float(self.source_start[1]) + float(self.source_end[1]))
        return abs(float(self.receiver[1]) - source_y)

    @property
    def sigma_background(self) -> float:
        return 1.0 / float(self.resistivities[0])

    def validate_geometry(self) -> None:
        if not math.isclose(self.source_length, CORRECTED_SOURCE_LENGTH, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"source length must be {CORRECTED_SOURCE_LENGTH:g} m")
        if not math.isclose(self.parallel_offset, CORRECTED_PARALLEL_OFFSET, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"parallel offset must be {CORRECTED_PARALLEL_OFFSET:g} m")
        if tuple(float(value) for value in self.source_start) != CORRECTED_SOURCE_START:
            raise ValueError("source_start does not match corrected model")
        if tuple(float(value) for value in self.source_end) != CORRECTED_SOURCE_END:
            raise ValueError("source_end does not match corrected model")
        if tuple(float(value) for value in self.receiver) != CORRECTED_RECEIVER:
            raise ValueError("receiver does not match corrected model")

    def observation_times(self) -> np.ndarray:
        if self.n_observation_times < 2:
            raise ValueError("n_observation_times must be at least 2")
        if self.observation_time_min <= 0.0 or self.observation_time_max <= self.observation_time_min:
            raise ValueError("observation time bounds must be positive and increasing")
        return np.geomspace(
            float(self.observation_time_min),
            float(self.observation_time_max),
            int(self.n_observation_times),
        )

    def empymod_primary_config(self) -> dict:
        self.validate_geometry()
        return {
            "source_start": tuple(float(value) for value in self.source_start),
            "source_end": tuple(float(value) for value in self.source_end),
            "current": float(self.source_current),
            "strength": float(self.source_current),
            "depths": tuple(float(value) for value in self.depths),
            "resistivities": tuple(float(value) for value in self.resistivities),
            "signal": -1,
            "coordinate_system": str(self.coordinate_system),
        }


def build_corrected_model_case_specs(
    output_root: str | Path,
    config: CorrectedModelValidationConfig | None = None,
) -> dict[str, dict]:
    """Return no-IP/IP validation case metadata for the corrected model."""

    cfg = config or CorrectedModelValidationConfig()
    cfg.validate_geometry()
    output = Path(output_root)
    runner = {
        "backend": "dolfinx_primary_secondary",
        "reference": cfg.reference_type,
        "components": list(cfg.components),
        "output_root": str(output),
    }
    sigma0 = cfg.sigma_background
    sum_delta = sum(float(value) for value in cfg.ip_delta_sigma)
    ip_material = {
        "kind": "ip_prony",
        "sigma0": float(sigma0),
        "sigma_inf": float(cfg.ip_sigma_inf),
        "delta_sigma_list": [float(value) for value in cfg.ip_delta_sigma],
        "tau_list": [float(value) for value in cfg.ip_tau],
        "sum_delta_sigma": float(sum_delta),
        "prony_dc_constraint_error": float(cfg.ip_sigma_inf - sum_delta - sigma0),
    }
    common = {
        "reference_type": cfg.reference_type,
        "validation_scope": cfg.validation_scope,
        "components": list(cfg.components),
        "magnetic_quantity": cfg.magnetic_quantity,
        "source_start": list(cfg.source_start),
        "source_end": list(cfg.source_end),
        "receiver": list(cfg.receiver),
        "source_current": float(cfg.source_current),
        "observation_times": [float(value) for value in cfg.observation_times()],
        "empymod_primary": cfg.empymod_primary_config(),
        "runner": runner,
    }
    return {
        "noip": {
            **common,
            "case_type": "noip",
            "output_dir": str(output / "noip_3comp"),
            "material": {
                "kind": "noip",
                "sigma": float(sigma0),
                "resistivity": float(1.0 / sigma0),
            },
        },
        "ip": {
            **common,
            "case_type": "ip",
            "output_dir": str(output / "ip_3comp"),
            "material": ip_material,
        },
    }


def build_corrected_leakage_channel_case_specs(
    output_root: str | Path,
    config: CorrectedModelValidationConfig | None = None,
) -> dict[str, dict]:
    """Return memory-safe corrected-scale no-IP/IP leakage-channel specs."""

    cfg = config or CorrectedModelValidationConfig()
    specs = build_corrected_model_case_specs(output_root, config=cfg)
    common_forward = {
        "domain_min": [-2000.0, -2000.0, -1000.0],
        "domain_max": [2000.0, 2000.0, 100.0],
        "cells": [2, 2, 1],
        "receiver_evaluation_mode": "first_cell",
        "outer_boundary_mode": "natural",
        "ksp_type": "cg",
        "rtol": 1.0e-8,
        "atol": 1.0e-10,
        "max_it": 400,
        "terrain_model": {
            "kind": "diagnostic_box_with_surface_metadata",
            "surface_reference_z": 0.0,
            "note": "The default corrected runner currently creates a box mesh; terrain metadata is recorded for the corrected-scale P8 diagnostic contract.",
        },
    }
    leakage_points = [
        [-700.0, -500.0, -120.0],
        [-250.0, -320.0, -90.0],
        [150.0, -120.0, -140.0],
        [650.0, 80.0, -110.0],
    ]
    specs["noip"]["validation_scope"] = "corrected_model_terrain_leakage_diagnostic"
    specs["noip"]["empymod_kwargs"] = {"srcpts": 3}
    specs["noip"]["dolfinx_forward"] = {
        **common_forward,
        "leakage_channel": {
            "points": leakage_points,
            "radius": 900.0,
            "sigma": 0.04,
        },
    }
    specs["ip"]["validation_scope"] = "corrected_model_terrain_leakage_diagnostic"
    specs["ip"]["empymod_kwargs"] = {"srcpts": 3}
    specs["ip"]["dolfinx_forward"] = {
        **common_forward,
        "leakage_channel": {
            "points": leakage_points,
            "radius": 900.0,
            "sigma_inf": 0.05,
            "delta_sigma_list": [0.015],
            "tau_list": [0.1],
        },
    }
    return specs


def build_published_paper_model_target_spec(output_root: str | Path) -> dict:
    """Return the published SOTEM paper target metadata for later reproduction.

    The public abstract/search metadata identifies the paper and its broad SOTEM
    setup. Full numerical reproduction still needs the paper's tabulated model,
    receiver, and IP-anomaly parameters before this can be used as an accuracy
    acceptance reference.
    """

    cfg = CorrectedModelValidationConfig()
    cfg.validate_geometry()
    return {
        "published_reference": {
            "title": "Analysis of 3D induced polarization effects of SOTEM",
            "journal": "Journal of Applied Geophysics",
            "volume": "233",
            "publication_date": "February 2025",
            "article_number": "105613",
            "article_id": "S092698512400329X",
            "doi": "10.1016/j.jappgeo.2024.105613",
            "url": "https://www.sciencedirect.com/science/article/pii/S092698512400329X",
            "reproduction_status": "target_defined_full_text_parameters_pending",
        },
        "model": {
            "source_start": [float(value) for value in cfg.source_start],
            "source_end": [float(value) for value in cfg.source_end],
            "receiver": [float(value) for value in cfg.receiver],
            "source_current": float(cfg.source_current),
            "source_length_m": float(cfg.source_length),
            "parallel_offset_m": float(cfg.parallel_offset),
            "ramp_off_time_s": float(cfg.ramp_off_time),
            "observation_time_min_s": float(cfg.observation_time_min),
            "observation_time_max_s": float(cfg.observation_time_max),
            "components": list(cfg.components),
            "magnetic_quantity": str(cfg.magnetic_quantity),
            "air_conductivity_s_per_m": 1.0e-6,
            "background_conductivity_s_per_m": float(cfg.sigma_background),
            "calculation_domain_m": [4000.0, 4000.0, 1000.0],
        },
        "run_contract": {
            "output_root": str(Path(output_root)),
            "validation_scope": "published_paper_reproduction_target",
            "reference_type": "published_response_curve",
            "algorithm_under_test": "atem3d_primary_secondary",
            "comparison_outputs": [
                "paper_response_overlay.png",
                "paper_relative_error_curves.png",
                "runtime_diagnostics.json",
            ],
        },
        "full_text_parameters_required": [
            "terrain_surface_or_layer_geometry",
            "ip_anomaly_geometry",
            "ip_anomaly_prony_or_cole_cole_parameters",
            "all_receiver_locations_and_components",
            "paper_plot_time_channels",
            "digitized_or_tabulated_published_response_values",
        ],
    }
