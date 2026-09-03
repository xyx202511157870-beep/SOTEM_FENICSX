from dataclasses import replace

import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.passive_fit import (
    fit_debye_passive_hard_dc,
    fit_pelton_passive_hard_dc,
    hard_gate_failures,
    validate_pelton_parameters,
)
from atem3d.metrics import relative_l2


FREQS = np.logspace(-2, 4, 61)
PELTON = dict(rho0=100.0, chargeability=0.3, tau=0.1, c=0.6)
GRID8 = np.logspace(-4, 2, 8)


def test_c_equals_one_single_pole_recovers_exact_debye():
    fit = fit_pelton_passive_hard_dc(100.0, 0.3, 0.1, 1.0, FREQS, [0.1])
    expected = 1.0 / (100.0 * 0.7) - 1.0 / 100.0
    np.testing.assert_allclose(fit.delta_sigma, [expected], rtol=1.0e-10)
    assert fit.spectral_error < 1.0e-10
    assert fit.relative_dc_error <= 1.0e-10
    assert fit.passive
    assert fit.passes_hard_gates()
    assert fit.optimizer_status.success


def test_m_zero_gives_no_ip():
    fit = fit_pelton_passive_hard_dc(100.0, 0.0, 0.1, 0.6, FREQS, GRID8)
    np.testing.assert_allclose(fit.delta_sigma, 0.0, atol=0.0)
    assert fit.sigma0 == pytest.approx(fit.sigma_infinity)
    assert fit.optimizer_status.method == "zero_ip_budget"
    assert fit.passes_hard_gates()


def test_hard_dc_equality_for_fractional_c():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8)
    assert fit.relative_dc_error <= 1.0e-10
    np.testing.assert_allclose(fit.delta_sigma.sum(), fit.sigma_infinity - 0.01, rtol=1.0e-12)
    assert fit.passive
    assert fit.passes_hard_gates()
    assert fit.spectral_error < 0.05


def test_deltas_nonnegative():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8)
    assert float(fit.delta_sigma.min()) >= -1.0e-12
    np.testing.assert_allclose(fit.to_prony_conductivity().sigma0, 0.01, rtol=1.0e-10)


def test_weights_must_be_nonnegative_and_shaped():
    target = validate_pelton_parameters(**PELTON).complex_conductivity(FREQS)
    sigma_inf = 1.0 / (100.0 * 0.7)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, sigma_inf, GRID8, 0.01, weights=-np.ones(FREQS.size))
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, sigma_inf, GRID8, 0.01, weights=np.ones(FREQS.size - 1))
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, sigma_inf, GRID8, 0.01, weights=np.zeros(FREQS.size))
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, sigma_inf, GRID8, 0.01, weights=np.full(FREQS.size, np.nan))


def test_weight_scaling_is_invariant():
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.5, 2.0, FREQS.size)
    left = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8, weights=weights)
    right = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8, weights=1.0e3 * weights)
    np.testing.assert_allclose(left.delta_sigma, right.delta_sigma, rtol=1.0e-9, atol=1.0e-16)
    np.testing.assert_allclose(left.spectral_error, right.spectral_error, rtol=1.0e-12)
    np.testing.assert_allclose(left.condition_number, right.condition_number, rtol=1.0e-12)
    np.testing.assert_allclose(left.weights.mean(), 1.0)


def test_zero_weights_equal_dropping_frequencies():
    weights = np.concatenate([np.ones(40), np.zeros(21)])
    full = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8, weights=weights)
    dropped = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS[:40], tau_grid=GRID8)
    np.testing.assert_allclose(full.delta_sigma, dropped.delta_sigma, rtol=1.0e-8)


def test_unweighted_spectral_error_matches_relative_l2():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8)
    stacked_fit = np.r_[fit.fitted_sigma.real, fit.fitted_sigma.imag]
    stacked_target = np.r_[fit.target_sigma.real, fit.target_sigma.imag]
    np.testing.assert_allclose(fit.spectral_error, relative_l2(stacked_fit, stacked_target), rtol=1.0e-12)


def test_duplicate_or_unsorted_poles_rejected():
    with pytest.raises(ValueError):
        fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=[1.0e-3, 1.0e-3, 1.0e-2])
    with pytest.raises(ValueError):
        fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=[1.0e-2, 1.0e-3])
    with pytest.raises(ValueError):
        fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=[1.0e-3, 1.0e-3 * (1.0 + 1.0e-12)])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rho0=0.0, chargeability=0.3, tau=0.1, c=0.6),
        dict(rho0=-1.0, chargeability=0.3, tau=0.1, c=0.6),
        dict(rho0=100.0, chargeability=-0.1, tau=0.1, c=0.6),
        dict(rho0=100.0, chargeability=1.0, tau=0.1, c=0.6),
        dict(rho0=100.0, chargeability=0.3, tau=0.0, c=0.6),
        dict(rho0=100.0, chargeability=0.3, tau=0.1, c=0.0),
        dict(rho0=100.0, chargeability=0.3, tau=0.1, c=1.5),
    ],
)
def test_illegal_pelton_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_pelton_passive_hard_dc(**kwargs, frequencies=FREQS, tau_grid=GRID8)


def test_illegal_fit_inputs_rejected():
    material = validate_pelton_parameters(**PELTON)
    target = material.complex_conductivity(FREQS)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, material.sigma_inf, GRID8, 0.0)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, material.sigma_inf, GRID8, material.sigma_inf + 1.0)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target, 0.0, GRID8, 0.01)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(np.array([0.0, 1.0]), target[:2], material.sigma_inf, GRID8, 0.01)
    with pytest.raises(ValueError):
        fit_debye_passive_hard_dc(FREQS, target[:-1], material.sigma_inf, GRID8, 0.01)


def test_condition_number_and_status_emitted():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8)
    assert np.isfinite(fit.condition_number)
    assert fit.condition_number >= 1.0
    crowded = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=np.logspace(-3.0, -2.99, 8))
    assert crowded.condition_number > fit.condition_number
    assert isinstance(fit.optimizer_status.status_code, int)
    assert isinstance(fit.optimizer_status.method, str)
    assert fit.optimizer_status.active_set_size <= GRID8.size


def test_hard_gate_predicate_flags_violations():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=GRID8)
    broken_delta = fit.delta_sigma.copy()
    broken_delta[0] = -1.0e-6
    negative = replace(fit, delta_sigma=broken_delta)
    assert hard_gate_failures(negative) == ("delta_sigma_negative",)
    failed = replace(fit, optimizer_status=replace(fit.optimizer_status, success=False))
    assert "optimizer_failed" in hard_gate_failures(failed)
    with pytest.raises(ValueError):
        failed.to_prony_conductivity()


def test_k_equals_one_fractional_c_still_hard_dc():
    fit = fit_pelton_passive_hard_dc(**PELTON, frequencies=FREQS, tau_grid=[0.1])
    budget = fit.sigma_infinity - 0.01
    np.testing.assert_allclose(fit.delta_sigma, [budget])
    assert fit.passes_hard_gates()
    assert fit.spectral_error > 0.0
