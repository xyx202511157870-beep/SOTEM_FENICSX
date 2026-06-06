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
        "ramp_off_time": float(cfg.ramp_off_time),
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
    convergence_reference = {
        "dolfinx_forward": {
            "cells": [4, 4, 2],
            "rtol": 1.0e-9,
            "atol": 1.0e-11,
            "max_it": 800,
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
            "min_marked_cells": 1,
            "sigma": 0.04,
        },
    }
    specs["noip"]["convergence_reference"] = convergence_reference
    specs["ip"]["validation_scope"] = "corrected_model_terrain_leakage_diagnostic"
    specs["ip"]["empymod_kwargs"] = {"srcpts": 3}
    specs["ip"]["dolfinx_forward"] = {
        **common_forward,
        "leakage_channel": {
            "points": leakage_points,
            "radius": 900.0,
            "min_marked_cells": 1,
            "sigma_inf": 0.05,
            "delta_sigma_list": [0.015],
            "tau_list": [0.1],
        },
    }
    specs["ip"]["convergence_reference"] = convergence_reference
    return specs


def build_published_paper_model_target_spec(output_root: str | Path) -> dict:
    """Return the published SOTEM paper target metadata for later reproduction.

    The local full-text extraction identifies the paper's benchmark, layered,
    and 3D polarized-body model parameters. Numerical reproduction still needs
    digitized or tabulated response curves before this can be used as an
    accuracy acceptance reference.
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
            "reproduction_status": "full_text_model_parameters_extracted_response_digitization_pending",
            "parameter_source": "local_full_text_extraction_tmp_pdfs_song2025_full_text_txt",
            "public_method_summary": {
                "frequency_domain_solver": "COMSOL",
                "time_domain_transform": "frequency-time transformation",
                "reported_components": ["Ex", "Hz"],
            },
            "public_model_classes": [
                "polarized_layer",
                "high_resistivity_polarized_body",
                "low_resistivity_polarized_body",
            ],
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
        "paper_model_parameters": {
            "accuracy_benchmark_layer": {
                "purpose": "3D finite-element accuracy check against 1D analytical solutions",
                "homogeneous_earth_sigma_s_per_m": 0.01,
                "polarized_layer_thickness_m": 200.0,
                "source_current_a": 10.0,
                "source_length_m": 1000.0,
                "offset_m": 500.0,
                "components": ["Ex", "Hz"],
                "paper_figures": ["Fig. 2", "Fig. 3"],
                "paper_reported_error": "relative errors within 5%",
                "cole_cole": {
                    "M": 0.3,
                    "c": 0.5,
                    "tau_s": 1.0,
                    "sigma0_s_per_m": 0.01,
                },
            },
            "layered_polarization_model": {
                "purpose": "half-space with middle polarizable layer, with and without IP",
                "air_sigma_s_per_m": 1.0e-6,
                "halfspace_sigma_s_per_m": 0.01,
                "calculation_domain_m": [4000.0, 4000.0, 1000.0],
                "infinite_element_layer_thickness_m": 100.0,
                "minimum_cell_size_m": 10.0,
                "element_count": 265670,
                "frequency_range_hz": [0.001, 10000.0],
                "frequencies_per_decade": 10,
                "frequency_count": 81,
                "reported_runtime_min": 248.0,
                "memory_gbytes": 15.7,
                "source_current_a": 10.0,
                "source_length_m": 1000.0,
                "offset_m": 500.0,
                "observation_point_printed_m": [-300.0, 0.0, 0.0],
                "printed_source_position_note": (
                    "paper prints source along x=200 m; corrected working geometry uses y=200 m"
                ),
                "polarized_layer_thickness_m": 300.0,
                "polarized_layer_resistivity_ohm_m": 100.0,
                "cole_cole": {
                    "M": 0.3,
                    "c": 0.3,
                    "tau_s": 1.0,
                },
                "time_domain_figures": ["Fig. 7", "Fig. 8", "Fig. 9", "Fig. 10"],
                "frequency_domain_figures": ["Fig. 5", "Fig. 6"],
                "reported_time_samples_s": [0.0001, 0.1],
            },
            "three_dimensional_polarized_body": {
                "purpose": "high- and low-resistivity 3D polarized-body IP response",
                "body_size_m": [400.0, 400.0, 400.0],
                "body_center_m": [0.0, -300.0, 400.0],
                "source_center_m": [0.0, 0.0, 0.0],
                "source_orientation": "x_axis",
                "observation_point_m": [0.0, -400.0, 0.0],
                "background_resistivity_ohm_m": 100.0,
                "high_resistivity_ohm_m": 1000.0,
                "low_resistivity_ohm_m": 10.0,
                "cole_cole": {
                    "M": 0.6,
                    "c": 0.6,
                    "tau_s": 1.0,
                },
                "time_window_after_turnoff_s": [0.0001, 1.0],
                "time_count": 41,
                "responses_use_absolute_value_when_sign_changes": True,
                "low_resistivity_ex_sign_reversal_approx_s": 0.003,
                "low_resistivity_hz_sign_reversals_approx_s": [0.03, 0.58],
                "response_figures": {
                    "low_resistivity_ex": "Fig. 12",
                    "low_resistivity_hz": "Fig. 15",
                    "low_resistivity_ex_sections": ["Fig. 13", "Fig. 14"],
                    "low_resistivity_hz_sections": ["Fig. 16", "Fig. 17"],
                    "high_resistivity_ex": "Fig. 18",
                },
            },
        },
        "paper_response_targets": {
            "digitized_response_required": True,
            "candidate_overlay_figures": ["Fig. 2", "Fig. 3", "Fig. 7", "Fig. 8", "Fig. 12", "Fig. 15"],
            "components": ["Ex", "Hz"],
            "comparison_outputs": [
                "paper_response_overlay.png",
                "paper_relative_error_curves.png",
                "runtime_diagnostics.json",
            ],
            "acceptance_blocker": "paper curves are currently image figures, not tabulated numeric references",
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
        "remaining_reproduction_requirements": [
            "paper_plot_time_channels",
            "digitized_or_tabulated_published_response_values",
        ],
        "full_text_parameters_required": [],
    }
