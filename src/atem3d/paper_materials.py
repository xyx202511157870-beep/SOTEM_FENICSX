"""Prony material specs for published-paper SOTEM models."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from atem3d.fit import fit_pelton_resistivity_debye, pelton_resistivity_to_conductivity
from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.metrics import relative_l2


def write_published_paper_prony_materials(
    paper_spec: dict,
    output: str | Path,
    *,
    n_terms: int = 10,
) -> dict:
    """Fit paper Cole-Cole/Pelton materials and write a Prony material JSON."""

    payload = build_published_paper_prony_materials(paper_spec, n_terms=n_terms)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_published_paper_prony_materials(paper_spec: dict, *, n_terms: int = 10) -> dict:
    """Return Prony fit parameters for paper benchmark/layer/body materials."""

    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    models = dict(paper_spec["paper_model_parameters"])
    layered = dict(models["layered_polarization_model"])
    freq_min, freq_max = [float(value) for value in layered["frequency_range_hz"]]
    frequency_count = int(layered["frequency_count"])
    frequencies = np.geomspace(freq_min, freq_max, frequency_count)
    materials = {
        "accuracy_benchmark_layer": _fit_from_rho_m_tau_c(
            rho0=1.0 / float(models["accuracy_benchmark_layer"]["cole_cole"]["sigma0_s_per_m"]),
            chargeability=float(models["accuracy_benchmark_layer"]["cole_cole"]["M"]),
            tau=float(models["accuracy_benchmark_layer"]["cole_cole"]["tau_s"]),
            c=float(models["accuracy_benchmark_layer"]["cole_cole"]["c"]),
            frequencies=frequencies,
            n_terms=n_terms,
        ),
        "layered_polarization_model": _fit_from_rho_m_tau_c(
            rho0=float(layered["polarized_layer_resistivity_ohm_m"]),
            chargeability=float(layered["cole_cole"]["M"]),
            tau=float(layered["cole_cole"]["tau_s"]),
            c=float(layered["cole_cole"]["c"]),
            frequencies=frequencies,
            n_terms=n_terms,
        ),
    }
    body = dict(models["three_dimensional_polarized_body"])
    body_cole = dict(body["cole_cole"])
    materials["three_dimensional_high_resistivity_body"] = _fit_from_rho_m_tau_c(
        rho0=float(body["high_resistivity_ohm_m"]),
        chargeability=float(body_cole["M"]),
        tau=float(body_cole["tau_s"]),
        c=float(body_cole["c"]),
        frequencies=frequencies,
        n_terms=n_terms,
    )
    materials["three_dimensional_low_resistivity_body"] = _fit_from_rho_m_tau_c(
        rho0=float(body["low_resistivity_ohm_m"]),
        chargeability=float(body_cole["M"]),
        tau=float(body_cole["tau_s"]),
        c=float(body_cole["c"]),
        frequencies=frequencies,
        n_terms=n_terms,
    )
    return {
        "source_article_id": str(dict(paper_spec["published_reference"]).get("article_id", "")),
        "frequency_range_hz": [freq_min, freq_max],
        "frequency_count": frequency_count,
        "n_terms": int(n_terms),
        "materials": materials,
    }


def _fit_from_rho_m_tau_c(
    *,
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
    frequencies: np.ndarray,
    n_terms: int,
) -> dict:
    fit = fit_pelton_resistivity_debye(
        rho0=rho0,
        chargeability=chargeability,
        tau=tau,
        c=c,
        frequencies=frequencies,
        n_terms=n_terms,
    )
    material = _enforce_dc_conductivity(fit.to_prony_conductivity(), sigma0=1.0 / float(rho0))
    relative_error = _relative_l2_for_material(
        material,
        rho0=rho0,
        chargeability=chargeability,
        tau=tau,
        c=c,
        frequencies=frequencies,
    )
    return {
        "model": "pelton_cole_cole_prony_fit",
        "rho0_ohm_m": float(rho0),
        "chargeability": float(chargeability),
        "tau_s": float(tau),
        "c": float(c),
        "target_sigma0": float(1.0 / rho0),
        "sigma0": float(material.sigma0),
        "sigma_inf": float(material.sigma_inf),
        "relative_l2": float(relative_error),
        "terms": [
            {
                "delta_sigma": float(term.delta_sigma),
                "tau": float(term.tau),
            }
            for term in material.terms
        ],
    }


def _enforce_dc_conductivity(material: PronyConductivity, *, sigma0: float) -> PronyConductivity:
    target_delta_sum = float(material.sigma_inf) - float(sigma0)
    if target_delta_sum < 0.0:
        raise ValueError("target sigma0 cannot exceed sigma_inf")
    current_delta_sum = sum(term.delta_sigma for term in material.terms)
    if current_delta_sum <= 0.0:
        if target_delta_sum == 0.0:
            return PronyConductivity.no_ip(material.sigma_inf)
        raise ValueError("cannot enforce nonzero DC constraint without Debye terms")
    scale = target_delta_sum / current_delta_sum
    return PronyConductivity(
        sigma_inf=material.sigma_inf,
        terms=[
            DebyeTerm(delta_sigma=term.delta_sigma * scale, tau=term.tau)
            for term in material.terms
        ],
    )


def _relative_l2_for_material(
    material: PronyConductivity,
    *,
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
    frequencies: np.ndarray,
) -> float:
    target = pelton_resistivity_to_conductivity(
        frequencies,
        rho0=rho0,
        chargeability=chargeability,
        tau=tau,
        c=c,
    )
    basis = np.column_stack(
        [
            1.0 / (1.0 + 1j * 2.0 * np.pi * frequencies * term.tau)
            for term in material.terms
        ]
    )
    delta = np.asarray([term.delta_sigma for term in material.terms], dtype=float)
    fitted = material.sigma_inf - basis @ delta
    return relative_l2(np.r_[fitted.real, fitted.imag], np.r_[target.real, target.imag])
