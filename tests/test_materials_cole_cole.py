import numpy as np

from atem3d.fit import cole_cole_conductivity
from atem3d.materials.cole_cole import ColeColeConductivity
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
