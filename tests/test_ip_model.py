import numpy as np

from atem3d.ip import DebyeTerm, DebyeIPModel


def test_debye_memory_update_matches_backward_euler_formula():
    model = DebyeIPModel(
        sigma_infinity=np.array([2.0, 3.0]),
        terms=[DebyeTerm(delta_sigma=np.array([0.5, 1.0]), tau=0.25)],
    )
    y_old = [np.array([1.0, -2.0])]
    e_new = np.array([4.0, 8.0])

    y_new = model.update_memory(y_old, e_new, dt=0.75)

    alpha = 0.25 / (0.25 + 0.75)
    beta = 0.75 / (0.25 + 0.75)
    np.testing.assert_allclose(y_new[0], alpha * y_old[0] + beta * e_new)


def test_effective_sigma_interpolates_between_high_and_low_frequency():
    model = DebyeIPModel(
        sigma_infinity=np.array([10.0, 20.0]),
        terms=[DebyeTerm(delta_sigma=np.array([3.0, 5.0]), tau=2.0)],
    )

    early = model.effective_sigma(dt=1.0e-6)
    late = model.effective_sigma(dt=1.0e6)

    np.testing.assert_allclose(early, np.array([10.0, 20.0]), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(late, np.array([7.0, 15.0]), rtol=0.0, atol=1.0e-5)


def test_no_ip_model_has_no_memory_and_ohmic_effective_sigma():
    model = DebyeIPModel.no_ip(np.array([0.01, 0.02, 0.03]))

    assert model.initial_memory(5) == []
    np.testing.assert_allclose(model.effective_sigma(dt=0.1), np.array([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(model.history_current([]), np.zeros(3))


def test_single_debye_inverse_constitutive_update_matches_closed_form():
    model = DebyeIPModel(
        sigma_infinity=np.array([2.0, 4.0]),
        terms=[DebyeTerm(delta_sigma=np.array([0.5, 1.0]), tau=0.25)],
    )
    current = np.array([3.0, -2.0])
    old_memory = np.array([0.2, -0.4])
    dt = 0.75

    electric, memories = model.inverse_constitutive_update(current, [old_memory], dt)

    rho_inf = 1.0 / model.sigma_infinity
    delta = model.terms[0].delta_sigma
    tau = model.terms[0].tau
    expected_memory = (tau * old_memory + dt * rho_inf * current) / (
        tau + dt * (1.0 - rho_inf * delta)
    )
    expected_electric = rho_inf * (current + delta * expected_memory)
    np.testing.assert_allclose(memories[0], expected_memory)
    np.testing.assert_allclose(electric, expected_electric)


def test_multi_debye_inverse_constitutive_update_satisfies_coupled_hj_law():
    model = DebyeIPModel(
        sigma_infinity=np.array([3.0, 5.0]),
        terms=[
            DebyeTerm(delta_sigma=np.array([0.3, 0.5]), tau=0.1),
            DebyeTerm(delta_sigma=np.array([0.2, 0.7]), tau=0.4),
        ],
    )
    current = np.array([1.5, -2.5])
    old_memories = [np.array([0.1, -0.2]), np.array([0.3, 0.4])]
    dt = 0.05

    electric, memories = model.inverse_constitutive_update(current, old_memories, dt)

    constitutive_current = model.sigma_infinity * electric
    for index, (term, memory) in enumerate(zip(model.terms, memories)):
        constitutive_current -= term.delta_sigma * memory
        memory_rhs = term.tau * (memory - old_memories[index]) / dt + memory
        np.testing.assert_allclose(memory_rhs, electric)

    np.testing.assert_allclose(constitutive_current, current)


def test_inverse_constitutive_coefficients_reproduce_hj_update():
    model = DebyeIPModel(
        sigma_infinity=np.array([3.0, 5.0]),
        terms=[
            DebyeTerm(delta_sigma=np.array([0.3, 0.5]), tau=0.1),
            DebyeTerm(delta_sigma=np.array([0.2, 0.7]), tau=0.4),
        ],
    )
    current = np.array([1.5, -2.5])
    old_memories = [np.array([0.1, -0.2]), np.array([0.3, 0.4])]
    dt = 0.05

    rho_eff, electric_history = model.inverse_constitutive_coefficients(old_memories, dt)
    electric, _ = model.inverse_constitutive_update(current, old_memories, dt)

    np.testing.assert_allclose(electric, rho_eff * current + electric_history)


def test_no_ip_inverse_constitutive_coefficients_are_ohmic():
    model = DebyeIPModel.no_ip(np.array([2.0, 4.0]))

    rho_eff, electric_history = model.inverse_constitutive_coefficients([], dt=0.1)

    np.testing.assert_allclose(rho_eff, np.array([0.5, 0.25]))
    np.testing.assert_allclose(electric_history, np.zeros(2))
