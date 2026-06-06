import numpy as np
import pytest

from atem3d.materials.prony import DebyeTerm, PronyConductivity


def test_debye_backward_euler_memory_update_uses_alpha_beta_coefficients():
    material = PronyConductivity(
        sigma_inf=0.1,
        terms=[DebyeTerm(delta_sigma=0.02, tau=0.5)],
    )
    chi_old = [np.array([[2.0, 0.0, -1.0]])]
    e_new = np.array([[1.0, 3.0, 0.5]])

    chi_new = material.update_memory(chi_old, e_new, dt=0.5)

    np.testing.assert_allclose(chi_new[0], 0.5 * chi_old[0] + 0.5 * e_new)


def test_debye_update_rejects_memory_shape_mismatch():
    material = PronyConductivity(
        sigma_inf=0.1,
        terms=[DebyeTerm(delta_sigma=0.02, tau=0.5)],
    )

    with pytest.raises(ValueError, match="same shape"):
        material.update_memory(
            [np.array([[1.0, 0.0, 0.0]])],
            np.array([1.0, 0.0, 0.0]),
            dt=0.5,
        )


def test_zero_delta_debye_current_density_degenerates_to_ohmic_current():
    material = PronyConductivity(
        sigma_inf=0.01,
        terms=[DebyeTerm(delta_sigma=0.0, tau=0.1)],
    )
    e = np.array([[1.0, -2.0, 0.5]])
    arbitrary_memory = [np.array([[100.0, 200.0, -300.0]])]

    current = material.current_density(e, arbitrary_memory)

    np.testing.assert_allclose(current, 0.01 * e)
