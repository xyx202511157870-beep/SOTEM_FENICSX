import numpy as np

from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.solvers.dc_secondary import initialize_dc_secondary


def test_dc_secondary_zero_contrast_returns_zero_secondary_field():
    Ep0 = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 3.0]])
    material = PronyConductivity.no_ip(0.01)

    result = initialize_dc_secondary(
        Ep0=Ep0,
        sigma0=0.01,
        sigma_background=0.01,
        material=material,
    )

    np.testing.assert_allclose(result.Es0, np.zeros_like(Ep0))
    np.testing.assert_allclose(result.Etotal0, Ep0)
    assert result.phi_s is None
    assert result.contrast_is_zero is True
    np.testing.assert_allclose(result.deltaJ0, np.zeros_like(Ep0))


def test_dc_secondary_initializes_ip_memory_from_total_field():
    Ep0 = np.array([[2.0, 1.0, 0.0]])
    Es0 = np.array([[0.5, -0.25, 0.0]])
    material = PronyConductivity(
        sigma_inf=0.02,
        terms=[DebyeTerm(delta_sigma=0.005, tau=0.1)],
    )

    result = initialize_dc_secondary(
        Ep0=Ep0,
        sigma0=material.sigma0,
        sigma_background=0.01,
        material=material,
        secondary_field_solver=lambda rhs: (np.array([3.0]), Es0),
    )

    np.testing.assert_allclose(result.Etotal0, Ep0 + Es0)
    np.testing.assert_allclose(result.chi0[0], Ep0 + Es0)
    expected_delta_j = material.current_density(result.Etotal0, result.chi0) - 0.01 * Ep0
    np.testing.assert_allclose(result.deltaJ0, expected_delta_j)
    np.testing.assert_allclose(result.phi_s, np.array([3.0]))


def test_dc_secondary_passes_contrast_current_to_injected_solver():
    Ep0 = np.array([[1.0, 2.0, 3.0]])
    material = PronyConductivity.no_ip(0.02)
    captured = {}

    def fake_solver(rhs):
        captured["rhs"] = rhs.copy()
        return np.array([0.0]), np.array([[0.1, 0.2, 0.3]])

    result = initialize_dc_secondary(
        Ep0=Ep0,
        sigma0=0.02,
        sigma_background=0.01,
        material=material,
        secondary_field_solver=fake_solver,
    )

    np.testing.assert_allclose(captured["rhs"], (0.02 - 0.01) * Ep0)
    np.testing.assert_allclose(result.Es0, np.array([[0.1, 0.2, 0.3]]))
    np.testing.assert_allclose(result.deltaJ0, 0.02 * result.Etotal0 - 0.01 * Ep0)
