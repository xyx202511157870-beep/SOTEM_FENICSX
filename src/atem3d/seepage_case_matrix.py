"""Deterministic physical-limit and convergence cases for the thin channel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .seepage_channel_model import ChannelBox, model_for_variant
from .seepage_verification import canonical_model_contract, model_fingerprint


@dataclass(frozen=True)
class VerificationCase:
    case_id: str
    solver: str
    study: str
    role: str
    conductivity_s_per_m: float
    cross_section_m: float
    local_mesh_size_m: float
    time_step_factor: float
    model_fingerprint: str
    case_fingerprint: str
    execution_fingerprint: str
    receiver_count: int = 5
    output_time_count: int = 31

    @property
    def expected_output(self) -> str:
        return f"verification_runs/{self.solver}/{self.case_id}/normalized.npz"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["expected_output"] = self.expected_output
        return payload


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _case_fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_case(
    *,
    solver: str,
    study: str,
    role: str,
    suffix: str,
    conductivity: float = 1.0,
    cross_section: float = 1.0,
    mesh_size: float = 0.25,
    time_factor: float = 1.0,
) -> VerificationCase:
    model = model_for_variant("thin_60x1x1")
    base_fingerprint = model_fingerprint(model)
    controlled = {
        "base_model_fingerprint": base_fingerprint,
        "solver": solver,
        "study": study,
        "role": role,
        "conductivity_s_per_m": float(conductivity),
        "cross_section_m": float(cross_section),
        "local_mesh_size_m": float(mesh_size),
        "time_step_factor": float(time_factor),
    }
    execution = {
        key: value for key, value in controlled.items() if key != "study"
    }
    return VerificationCase(
        case_id=f"{solver}-{study}-{role}-{suffix}",
        solver=solver,
        study=study,
        role=role,
        conductivity_s_per_m=float(conductivity),
        cross_section_m=float(cross_section),
        local_mesh_size_m=float(mesh_size),
        time_step_factor=float(time_factor),
        model_fingerprint=base_fingerprint,
        case_fingerprint=_case_fingerprint(controlled),
        execution_fingerprint=_case_fingerprint(execution),
    )


def build_case_matrix() -> tuple[VerificationCase, ...]:
    """Return all approved cases, including matched backgrounds per study."""

    cases: list[VerificationCase] = []
    for solver in ("simpeg", "fenicsx"):
        cases.append(
            _make_case(
                solver=solver,
                study="conductivity",
                role="background",
                suffix="reference",
                conductivity=0.01,
            )
        )
        for conductivity in (0.01, 0.02, 0.1, 1.0):
            cases.append(
                _make_case(
                    solver=solver,
                    study="conductivity",
                    role="channel",
                    suffix=f"sigma-{_slug(conductivity)}",
                    conductivity=conductivity,
                )
            )

        for cross_section in (1.0, 2.0, 10.0):
            for role in ("background", "channel"):
                cases.append(
                    _make_case(
                        solver=solver,
                        study="volume",
                        role=role,
                        suffix=f"cross-{_slug(cross_section)}",
                        conductivity=0.01 if role == "background" else 1.0,
                        cross_section=cross_section,
                        mesh_size=(
                            0.5
                            if solver == "fenicsx" and cross_section == 10.0
                            else 0.25
                        ),
                    )
                )

        for mesh_size in (0.5, 0.25, 0.125):
            for role in ("background", "channel"):
                cases.append(
                    _make_case(
                        solver=solver,
                        study="spatial",
                        role=role,
                        suffix=f"h-{_slug(mesh_size)}",
                        conductivity=0.01 if role == "background" else 1.0,
                        mesh_size=mesh_size,
                    )
                )

        for time_factor in (1.0, 0.5, 0.25):
            for role in ("background", "channel"):
                cases.append(
                    _make_case(
                        solver=solver,
                        study="temporal",
                        role=role,
                        suffix=f"dt-{_slug(time_factor)}",
                        conductivity=0.01 if role == "background" else 1.0,
                        time_factor=time_factor,
                    )
                )
    return tuple(cases)


def case_model(case: VerificationCase):
    """Return the physical model represented by one controlled matrix case."""

    base = model_for_variant("thin_60x1x1")
    return replace(
        base,
        channel=ChannelBox(
            center=base.channel.center,
            size=(60.0, case.cross_section_m, case.cross_section_m),
            conductivity=case.conductivity_s_per_m,
        ),
    )


def write_case_manifest(
    output_root: str | Path, cases: tuple[VerificationCase, ...]
) -> Path:
    """Write a deterministic dry-run manifest for resumable execution."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model = model_for_variant("thin_60x1x1")
    manifest = {
        "variant": "thin_60x1x1",
        "model_fingerprint": model_fingerprint(model),
        "model_contract": canonical_model_contract(model),
        "cases": [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)],
    }
    path = output / "verification_case_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "VerificationCase",
    "build_case_matrix",
    "case_model",
    "write_case_manifest",
]
