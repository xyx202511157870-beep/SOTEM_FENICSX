import pytest
import numpy as np

from atem3d.fit import cole_cole_conductivity, pelton_resistivity_to_conductivity
from atem3d.materials.cole_cole import ColeColeConductivity
from atem3d.materials.cole_cole import PeltonColeColeResistivity
from atem3d.materials.prony import PronyConductivity


def test_cole_cole_conductivity_material_evaluates_complex_sigma():
    frequencies = np.logspace(-2, 2, 12)
    material = ColeColeConductivity(sigma_inf=0.02, eta=0.25, tau=1.0, c=0.5)

    sigma = material.complex_conductivity(frequencies)

    np.testing.assert_allclose(
        sigma,
        cole_cole_conductivity(
            frequencies,
            sigma_infinity=0.02,
            eta=0.25,
            tau=1.0,
            c=0.5,
        ),
    )
    np.testing.assert_allclose(material.sigma0, 0.015)


def test_cole_cole_conductivity_material_converts_to_prony():
    material = ColeColeConductivity(sigma_inf=0.02, eta=0.25, tau=1.0, c=1.0)

    prony = material.to_prony_conductivity(
        frequencies=np.logspace(-2, 2, 50),
        tau_grid=np.array([1.0]),
    )

    assert isinstance(prony, PronyConductivity)
    assert prony.sigma_inf == 0.02
    assert len(prony.terms) == 1
    np.testing.assert_allclose(prony.terms[0].delta_sigma, 0.005)
    np.testing.assert_allclose(prony.sigma0, material.sigma0)


def test_pelton_cole_cole_resistivity_material_evaluates_complex_sigma():
    frequencies = np.logspace(-2, 2, 12)
    material = PeltonColeColeResistivity(rho0=100.0, chargeability=0.3, tau=1.0, c=0.5)

    sigma = material.complex_conductivity(frequencies)

    np.testing.assert_allclose(
        sigma,
        pelton_resistivity_to_conductivity(
            frequencies,
            rho0=100.0,
            chargeability=0.3,
            tau=1.0,
            c=0.5,
        ),
    )
    assert material.sigma0 == pytest.approx(0.01)
    assert material.sigma_inf == pytest.approx(1.0 / 70.0)


def test_pelton_cole_cole_resistivity_material_converts_c1_to_single_prony():
    material = PeltonColeColeResistivity(rho0=100.0, chargeability=0.3, tau=1.0, c=1.0)

    prony = material.to_prony_conductivity(
        frequencies=np.logspace(-2, 2, 50),
        tau_grid=np.array([0.7]),
    )

    assert isinstance(prony, PronyConductivity)
    assert prony.sigma_inf == pytest.approx(material.sigma_inf)
    assert len(prony.terms) == 1
    assert prony.terms[0].tau == pytest.approx(0.7)
    np.testing.assert_allclose(prony.sigma0, material.sigma0, rtol=1.0e-12, atol=1.0e-14)
