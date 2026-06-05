import numpy as np

from sotem_ip import cole_cole_conductivity, fit_cole_cole_debye
from sotem_ip.debye import debye_conductivity


def test_cole_cole_dc_limit_matches_rho0():
    sigma = cole_cole_conductivity([0.0], rho0=100.0, m=0.3, tau=1.0, c=0.3)
    assert np.isclose(sigma[0].real, 0.01)
    assert np.isclose(sigma[0].imag, 0.0)


def test_debye_fit_enforces_dc_conductivity():
    fit = fit_cole_cole_debye(rho0=100.0, m=0.3, tau=1.0, c=0.3, n_terms=8)
    sigma_dc = debye_conductivity([0.0], fit.sigma_infinity, fit.terms)[0]
    assert np.isclose(sigma_dc.real, 0.01, rtol=1.0e-5, atol=1.0e-8)
    assert fit.relative_l2 < 0.25

