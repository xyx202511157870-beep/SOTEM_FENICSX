import numpy as np

from atem3d.fit import (
    cole_cole_conductivity,
    fit_cole_cole_conductivity_debye,
    fit_pelton_resistivity_debye,
    pelton_resistivity_to_conductivity,
)


def test_conductivity_cole_cole_c_equals_one_is_single_debye():
    freqs = np.logspace(-2, 2, 25)
    sigma_inf = 0.1
    eta = 0.2
    tau = 0.5

    result = fit_cole_cole_conductivity_debye(
        sigma_infinity=sigma_inf,
        eta=eta,
        tau=tau,
        c=1.0,
        frequencies=freqs,
        tau_grid=np.array([tau]),
    )

    assert len(result.terms) == 1
    np.testing.assert_allclose(result.terms[0].tau, tau)
    np.testing.assert_allclose(result.terms[0].delta_sigma, np.array([sigma_inf * eta]))
    assert result.relative_l2 < 1.0e-12


def test_multi_debye_fit_approximates_fractional_conductivity_cole_cole():
    freqs = np.logspace(-1, 3, 80)
    target = cole_cole_conductivity(freqs, sigma_infinity=0.05, eta=0.25, tau=0.02, c=0.6)

    result = fit_cole_cole_conductivity_debye(
        sigma_infinity=0.05,
        eta=0.25,
        tau=0.02,
        c=0.6,
        frequencies=freqs,
        n_terms=12,
    )

    assert result.relative_l2 < 0.04
    np.testing.assert_allclose(result.target_sigma, target)
    assert all(term.delta_sigma[0] >= 0.0 for term in result.terms)


def test_pelton_resistivity_is_converted_to_conductivity_before_fitting():
    freqs = np.logspace(-1, 2, 40)
    sigma = pelton_resistivity_to_conductivity(
        freqs,
        rho0=100.0,
        chargeability=0.3,
        tau=0.1,
        c=1.0,
    )

    result = fit_pelton_resistivity_debye(
        rho0=100.0,
        chargeability=0.3,
        tau=0.1,
        c=1.0,
        frequencies=freqs,
        tau_grid=np.array([0.1]),
    )

    np.testing.assert_allclose(result.target_sigma, sigma)
    assert result.sigma_infinity > 1.0 / 100.0
    assert result.relative_l2 < 0.08


def test_pelton_fit_uses_analytic_high_frequency_conductivity():
    result = fit_pelton_resistivity_debye(
        rho0=100.0,
        chargeability=0.2,
        tau=0.1,
        c=0.7,
        frequencies=np.array([0.1, 1.0]),
        tau_grid=np.array([0.1]),
    )

    np.testing.assert_allclose(result.sigma_infinity, 1.0 / (100.0 * (1.0 - 0.2)))
