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
