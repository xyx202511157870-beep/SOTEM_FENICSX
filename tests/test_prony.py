import numpy as np
import pytest

from atem3d.materials.prony import DebyeTerm, PronyConductivity


def test_prony_sigma0_and_effective_sigma_match_backward_euler():
    model = PronyConductivity(
        sigma_inf=0.02,
        terms=[
            DebyeTerm(delta_sigma=0.003, tau=0.1),
            DebyeTerm(delta_sigma=0.002, tau=1.0),
        ],
    )

    assert model.sigma0 == pytest.approx(0.015)
    np.testing.assert_allclose(model.alpha(0.1), np.array([0.5, 10.0 / 11.0]))
    np.testing.assert_allclose(model.beta(0.1), np.array([0.5, 1.0 / 11.0]))
    assert model.sigma_eff(0.1) == pytest.approx(
        0.02 - 0.003 * 0.5 - 0.002 / 11.0
    )


def test_prony_memory_and_current_density_are_backward_euler_consistent():
    model = PronyConductivity(
        sigma_inf=2.0,
        terms=[
            DebyeTerm(delta_sigma=0.25, tau=0.5),
            DebyeTerm(delta_sigma=0.5, tau=1.5),
        ],
    )
    chi_old = [np.array([1.0, -2.0]), np.array([0.5, 3.0])]
    e_new = np.array([4.0, -1.0])

    chi_new = model.update_memory(chi_old, e_new, dt=0.5)

    np.testing.assert_allclose(chi_new[0], 0.5 * chi_old[0] + 0.5 * e_new)
    np.testing.assert_allclose(chi_new[1], 0.75 * chi_old[1] + 0.25 * e_new)
    np.testing.assert_allclose(
        model.current_density(e_new, chi_new),
        2.0 * e_new - 0.25 * chi_new[0] - 0.5 * chi_new[1],
    )


def test_zero_delta_sigma_exactly_matches_no_ip():
    noip = PronyConductivity(sigma_inf=0.01, terms=[])
    zero_delta = PronyConductivity(
        sigma_inf=0.01,
        terms=[DebyeTerm(delta_sigma=0.0, tau=0.2)],
    )
    e = np.array([1.0, 2.0, -3.0])
    chi = [np.array([9.0, 8.0, 7.0])]

    assert noip.sigma0 == pytest.approx(zero_delta.sigma0)
    assert noip.sigma_eff(0.3) == pytest.approx(zero_delta.sigma_eff(0.3))
    np.testing.assert_allclose(noip.current_density(e, []), zero_delta.current_density(e, chi))


def test_prony_converts_to_and_from_existing_debye_ip_model():
    model = PronyConductivity(
        sigma_inf=0.02,
        terms=[DebyeTerm(delta_sigma=0.004, tau=0.25)],
    )

    legacy = model.to_debye_ip_model()
    roundtrip = PronyConductivity.from_debye_ip_model(legacy)

    assert roundtrip.sigma_inf == pytest.approx(model.sigma_inf)
    assert roundtrip.terms == model.terms


def test_prony_rejects_nonphysical_parameters():
    with pytest.raises(ValueError, match="sigma_inf"):
        PronyConductivity(sigma_inf=0.0, terms=[])
    with pytest.raises(ValueError, match="delta_sigma"):
        PronyConductivity(
            sigma_inf=1.0,
            terms=[DebyeTerm(delta_sigma=-1e-3, tau=1.0)],
        )
    with pytest.raises(ValueError, match="tau"):
        PronyConductivity(
            sigma_inf=1.0,
            terms=[DebyeTerm(delta_sigma=1e-3, tau=0.0)],
        )
    with pytest.raises(ValueError, match="sigma0"):
        PronyConductivity(
            sigma_inf=1.0,
            terms=[DebyeTerm(delta_sigma=1.0, tau=1.0)],
        )
