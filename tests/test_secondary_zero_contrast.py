import numpy as np

from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.solvers.dc_secondary import DCSecondaryInitialization
from atem3d.solvers.tdem_secondary import (
    SecondaryState,
    secondary_state_from_dc_initialization,
    secondary_step_equation_metadata,
    secondary_step_ip,
    secondary_step_noip,
)


def test_noip_secondary_zero_contrast_stays_zero_for_variable_steps():
    state = SecondaryState(
        Es=np.zeros((1, 3)),
        deltaJ=np.zeros((1, 3)),
        chi=[],
    )
    Ep_values = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        np.array([[0.25, 0.0, 0.0]]),
    ]

    for dt, Ep_old, Ep_new in zip([0.1, 0.35], Ep_values[:-1], Ep_values[1:]):
        state = secondary_step_noip(
            state,
            Ep_old=Ep_old,
            Ep_new=Ep_new,
            sigma=0.01,
            sigma_background=0.01,
            dt=dt,
        )
        np.testing.assert_allclose(state.Es, np.zeros((1, 3)))
        np.testing.assert_allclose(state.deltaJ, np.zeros((1, 3)))


def test_ip_zero_delta_secondary_matches_noip_zero_contrast():
    material = PronyConductivity(
        sigma_inf=0.01,
        terms=[DebyeTerm(delta_sigma=0.0, tau=0.2)],
    )
    state = SecondaryState(
        Es=np.zeros((1, 3)),
        deltaJ=np.zeros((1, 3)),
        chi=[np.array([[2.0, 0.0, 0.0]])],
    )

    new_state = secondary_step_ip(
        state,
        Ep_old=np.array([[2.0, 0.0, 0.0]]),
        Ep_new=np.array([[1.0, 0.0, 0.0]]),
        material=material,
        sigma_background=0.01,
        dt=0.5,
    )

    np.testing.assert_allclose(new_state.Es, np.zeros((1, 3)))
    np.testing.assert_allclose(new_state.deltaJ, np.zeros((1, 3)))
    np.testing.assert_allclose(new_state.chi[0], material.update_memory(state.chi, np.array([[1.0, 0.0, 0.0]]), 0.5)[0])


def test_noip_secondary_passes_expected_rhs_to_solver():
    old_state = SecondaryState(
        Es=np.array([[0.1, 0.0, 0.0]]),
        deltaJ=np.array([[0.21, 0.0, 0.0]]),
        chi=[],
    )
    captured = {}

    def fake_solver(rhs, sigma_eff, dt):
        captured["rhs"] = rhs.copy()
        captured["sigma_eff"] = sigma_eff
        captured["dt"] = dt
        return np.array([[0.3, 0.0, 0.0]])

    new_state = secondary_step_noip(
        old_state,
        Ep_old=np.array([[1.0, 0.0, 0.0]]),
        Ep_new=np.array([[2.0, 0.0, 0.0]]),
        sigma=0.2,
        sigma_background=0.1,
        dt=0.5,
        secondary_solver=fake_solver,
    )

    expected_rhs = (old_state.deltaJ - (0.2 - 0.1) * np.array([[2.0, 0.0, 0.0]])) / 0.5
    np.testing.assert_allclose(captured["rhs"], expected_rhs)
    assert captured["sigma_eff"] == 0.2
    assert captured["dt"] == 0.5
    np.testing.assert_allclose(new_state.Es, np.array([[0.3, 0.0, 0.0]]))
    np.testing.assert_allclose(
        new_state.deltaJ,
        0.2 * (np.array([[2.0, 0.0, 0.0]]) + new_state.Es) - 0.1 * np.array([[2.0, 0.0, 0.0]]),
    )


def test_ip_secondary_passes_expected_rhs_to_solver_and_updates_memory():
    material = PronyConductivity(
        sigma_inf=0.3,
        terms=[DebyeTerm(delta_sigma=0.06, tau=0.5)],
    )
    chi_old = [np.array([[1.0, 0.0, 0.0]])]
    old_state = SecondaryState(
        Es=np.zeros((1, 3)),
        deltaJ=np.array([[0.12, 0.0, 0.0]]),
        chi=chi_old,
    )
    captured = {}

    def fake_solver(rhs, sigma_eff, dt):
        captured["rhs"] = rhs.copy()
        captured["sigma_eff"] = sigma_eff
        return np.array([[0.2, 0.0, 0.0]])

    Ep_new = np.array([[2.0, 0.0, 0.0]])
    new_state = secondary_step_ip(
        old_state,
        Ep_old=np.array([[1.0, 0.0, 0.0]]),
        Ep_new=Ep_new,
        material=material,
        sigma_background=0.1,
        dt=0.5,
        secondary_solver=fake_solver,
    )

    alpha = 0.5 / (0.5 + 0.5)
    beta = 0.5 / (0.5 + 0.5)
    sigma_eff = 0.3 - 0.06 * beta
    c_new = (sigma_eff - 0.1) * Ep_new - 0.06 * alpha * chi_old[0]
    np.testing.assert_allclose(captured["rhs"], (old_state.deltaJ - c_new) / 0.5)
    assert captured["sigma_eff"] == sigma_eff
    np.testing.assert_allclose(
        new_state.chi[0],
        material.update_memory(chi_old, Ep_new + new_state.Es, 0.5)[0],
    )


def test_secondary_state_from_dc_initialization_preserves_secondary_fields_and_memory():
    init = DCSecondaryInitialization(
        Ep0=np.array([[1.0, 0.0, 0.0]]),
        Es0=np.array([[0.2, 0.0, 0.0]]),
        Etotal0=np.array([[1.2, 0.0, 0.0]]),
        chi0=[np.array([[1.2, 0.0, 0.0]])],
        deltaJ0=np.array([[0.03, 0.0, 0.0]]),
        phi_s=np.array([0.0]),
        contrast_is_zero=False,
    )

    state = secondary_state_from_dc_initialization(init)

    np.testing.assert_allclose(state.Es, init.Es0)
    np.testing.assert_allclose(state.deltaJ, init.deltaJ0)
    np.testing.assert_allclose(state.chi[0], init.chi0[0])


def test_secondary_state_from_dc_initialization_is_exported_from_solvers_package():
    from atem3d.solvers import secondary_state_from_dc_initialization as exported

    assert exported is secondary_state_from_dc_initialization


def test_secondary_noip_equation_metadata_records_rhs_convention():
    material = PronyConductivity(sigma_inf=0.02, terms=[])

    metadata = secondary_step_equation_metadata(
        material=material,
        sigma_background=0.01,
        dt=0.25,
    )

    assert metadata["solver_mode"] == "primary_secondary"
    assert metadata["case_type"] == "noip"
    assert metadata["total_field"] == "Etotal = Ep + Es"
    assert metadata["current_difference"] == "deltaJ = sigma(Ep + Es) - sigma_b Ep"
    assert metadata["lhs_operator"] == "K + R + M(sigma)/dt"
    assert metadata["rhs_history"] == "M(deltaJ_old - (sigma - sigma_b) Ep_new)/dt"
    assert metadata["sigma"] == 0.02
    assert metadata["sigma_background"] == 0.01
    assert metadata["dt"] == 0.25
    assert metadata["zero_contrast_condition"] == "sigma == sigma_background"


def test_secondary_ip_equation_metadata_records_prony_elimination():
    material = PronyConductivity(
        sigma_inf=0.3,
        terms=[DebyeTerm(delta_sigma=0.06, tau=0.5)],
    )

    metadata = secondary_step_equation_metadata(
        material=material,
        sigma_background=0.1,
        dt=0.5,
    )

    assert metadata["case_type"] == "ip"
    assert metadata["ip_current"] == "J = sigma_inf(Ep + Es) - sum(delta_sigma_k chi_k)"
    assert metadata["background_current"] == "Jb = sigma_b Ep"
    assert metadata["memory_update"] == "chi_k_new = alpha_k * chi_k_old + beta_k * (Ep_new + Es_new)"
    assert metadata["lhs_operator"] == "K + R + M(sigma_eff)/dt"
    assert metadata["rhs_history"] == "M(deltaJ_old - c_new)/dt"
    assert metadata["c_new"] == "(sigma_eff - sigma_b) Ep_new - sum(delta_sigma_k * alpha_k * chi_old_k)"
    assert metadata["sigma_inf"] == 0.3
    assert metadata["sigma0"] == 0.24
    assert metadata["sigma_eff"] == 0.27
    assert metadata["delta_sigma"] == [0.06]
    assert metadata["tau"] == [0.5]
    assert metadata["alpha"] == [0.5]
    assert metadata["beta"] == [0.5]
    assert metadata["delta_sigma_zero_degenerates_to_noip"] is True
