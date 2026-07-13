import numpy as np
import pytest

from atem3d.materials.prony import PronyConductivity
from atem3d.primary import CachedPrimaryProvider
from atem3d.solvers.dc_secondary import DCSecondaryInitialization
from atem3d.solvers.primary_secondary_forward import PrimarySecondaryForwardOperator
from atem3d.solvers.tdem_secondary import SecondaryState


def test_primary_secondary_forward_zero_contrast_returns_primary_receiver_response():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5, 2.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array(
            [
                [[10.0, 1.0, 0.0]],
                [[5.0, 0.5, 0.0]],
            ]
        ),
        receiver_dBdt=np.array(
            [
                [[0.0, 0.0, -3.0]],
                [[0.0, 0.0, -1.5]],
            ]
        ),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(
        predicted,
        [
            [10.0, 1.0, -3.0],
            [5.0, 0.5, -1.5],
        ],
    )


def test_primary_secondary_forward_adds_2d_secondary_receiver_projection():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]]]),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0]]),
    )
    seen = {}

    def dc_solver(contrast_current):
        seen["dc_contrast_current"] = contrast_current.copy()
        return None, np.array([[0.25, 0.0, 0.0]])

    def step_solver(rhs, sigma_eff, dt):
        seen["step_rhs"] = rhs.copy()
        seen["step_sigma_eff"] = sigma_eff
        seen["step_dt"] = dt
        return np.array([[0.5, 0.0, 0.0]])

    def receiver_projector(state, Ep_new, time_value, dt, components):
        seen["receiver_state_Es"] = state.Es.copy()
        seen["receiver_Ep_new"] = Ep_new.copy()
        seen["receiver_time"] = time_value
        seen["receiver_dt"] = dt
        seen["receiver_components"] = tuple(components)
        return np.array([[0.2, -0.1, 0.5]])

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.02, terms=[]),
        sigma_background=0.01,
        secondary_field_solver=dc_solver,
        secondary_step_solver=step_solver,
        secondary_receiver_projector=receiver_projector,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.2, 0.9, -2.5]])
    np.testing.assert_allclose(seen["dc_contrast_current"], [[0.02, 0.0, 0.0]])
    np.testing.assert_allclose(seen["receiver_state_Es"], [[0.5, 0.0, 0.0]])
    np.testing.assert_allclose(seen["receiver_Ep_new"], [[1.0, 0.0, 0.0]])
    assert seen["receiver_time"] == 1.0e-5
    assert seen["receiver_dt"] == 1.0e-5
    assert seen["receiver_components"] == ("Ex", "Ey", "dBzdt")


def test_primary_secondary_forward_with_state_can_resume_output_sequence():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5, 2.0e-5, 4.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]], [[40.0, 0.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -2.0]], [[0.0, 0.0, -4.0]]]),
        Ep_dc_on_V=np.array([[0.0, 0.0, 0.0]]),
    )

    def initialize(Ep0, material, sigma_background):
        zero = np.zeros_like(Ep0)
        return DCSecondaryInitialization(
            Ep0=Ep0,
            Es0=zero,
            Etotal0=Ep0,
            chi0=[],
            deltaJ0=zero,
            phi_s=None,
            contrast_is_zero=False,
        )

    def stepper(state, _Ep_old, Ep_new, _material, _sigma_background, dt, _primary_time):
        Es = state.Es + float(dt) * np.asarray(Ep_new, dtype=float)
        return SecondaryState(Es=Es, deltaJ=Es.copy(), chi=[])

    def projector(state, _Ep_new, _time_value, _dt, _components):
        return np.array([[state.Es[0, 0], 0.0, 10.0 * state.Es[0, 0]]])

    def make_operator():
        return PrimarySecondaryForwardOperator(
            primary=provider,
            fem_points=points,
            receiver_locations=receivers,
            components=("Ex", "Ey", "dBzdt"),
            material=PronyConductivity(sigma_inf=0.02, terms=[]),
            sigma_background=0.01,
            secondary_state_initializer=initialize,
            secondary_state_stepper=stepper,
            secondary_receiver_projector=projector,
        )

    full = make_operator().forward(times)
    first = make_operator().forward_with_state(times, max_new_outputs=1)
    second = make_operator().forward_with_state(
        times,
        initial_state=first.final_state,
        initial_Ep_old=first.final_Ep_old,
        previous_time=first.previous_time,
        output_index=first.output_index,
    )

    np.testing.assert_allclose(np.vstack([first.data, second.data]), full)
    assert first.output_index == 1
    assert second.output_index == 3
    assert second.previous_time == pytest.approx(4.0e-5)


def test_primary_secondary_forward_records_receiver_primary_secondary_decomposition():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5])
    diagnostics = {}
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]]]),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0]]),
    )

    def receiver_projector(state, Ep_new, time_value, dt, components):
        return np.array([[0.2, -0.1, 0.5]])

    def dc_solver(contrast_current):
        return None, np.zeros_like(contrast_current)

    def step_solver(rhs, sigma_eff, dt):
        return np.zeros_like(rhs)

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.02, terms=[]),
        sigma_background=0.01,
        secondary_field_solver=dc_solver,
        secondary_step_solver=step_solver,
        secondary_receiver_projector=receiver_projector,
        diagnostics=diagnostics,
    )

    predicted = operator.forward(times)

    rows = diagnostics["receiver_decomposition_rows"]
    assert len(rows) == 1
    assert rows[0]["time_value"] == pytest.approx(1.0e-5)
    assert rows[0]["primary_time"] == pytest.approx(1.0e-5)
    assert rows[0]["components"] == ["Ex", "Ey", "dBzdt"]
    np.testing.assert_allclose(rows[0]["primary_row"], [10.0, 1.0, -3.0])
    np.testing.assert_allclose(rows[0]["secondary_row"], [0.2, -0.1, 0.5])
    np.testing.assert_allclose(rows[0]["total_row"], predicted[0])


def test_primary_secondary_forward_adds_hz_primary_and_secondary_receiver_projection():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_H=np.array([[[0.0, 0.0, 4.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]]]),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0]]),
    )
    seen = {}

    def receiver_projector(state, Ep_new, time_value, dt, components):
        seen["receiver_components"] = tuple(components)
        return np.array([[0.2, 0.5, -0.25]])

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Hz", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_receiver_projector=receiver_projector,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.2, 4.5, -3.25]])
    assert seen["receiver_components"] == ("Ex", "Hz", "dBzdt")


def test_primary_secondary_forward_does_not_request_primary_h_without_h_components():
    class ProviderWithoutUsableH:
        def get_Ep_on_V(self, t, V):
            return np.array([[1.0, 0.0, 0.0]])

        def get_Ep_dc_on_V(self, V):
            return np.array([[1.0, 0.0, 0.0]])

        def get_receiver_E(self, t, receivers):
            return np.array([[10.0, 1.0, 0.0]])

        def get_receiver_H(self, t, receivers):
            raise AssertionError("H should not be requested without H receiver components")

        def get_receiver_dBdt(self, t, receivers):
            return np.array([[0.0, 0.0, -3.0]])

    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5])
    operator = PrimarySecondaryForwardOperator(
        primary=ProviderWithoutUsableH(),
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.0, 1.0, -3.0]])


def test_primary_secondary_forward_outputs_use_observation_time_primary_after_turnoff():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5, 2.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([1.0e-5, 2.0e-5, 3.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array(
            [
                [[10.0, 1.0, 0.0]],
                [[20.0, 2.0, 0.0]],
                [[30.0, 3.0, 0.0]],
            ]
        ),
        receiver_dBdt=np.array(
            [
                [[0.0, 0.0, -1.0]],
                [[0.0, 0.0, -2.0]],
                [[0.0, 0.0, -3.0]],
            ]
        ),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        turnoff_time=1.0e-5,
        turnoff_steps=2,
    )

    predicted = operator.forward(observation_times)

    np.testing.assert_allclose(predicted, [[10.0, 1.0, -1.0], [20.0, 2.0, -2.0]])


def test_primary_secondary_forward_samples_internal_times_after_turnoff():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5, 2.0e-5])
    turnoff_time = 1.0e-5
    output_internal_times = turnoff_time + observation_times
    internal_times = np.r_[0.5e-5, 1.0e-5, output_internal_times]
    provider = CachedPrimaryProvider(
        times=internal_times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array(
            [
                [[5.0, 0.5, 0.0]],
                [[10.0, 1.0, 0.0]],
                [[20.0, 2.0, 0.0]],
                [[30.0, 3.0, 0.0]],
            ]
        ),
        receiver_dBdt=np.array(
            [
                [[0.0, 0.0, -0.5]],
                [[0.0, 0.0, -1.0]],
                [[0.0, 0.0, -2.0]],
                [[0.0, 0.0, -3.0]],
            ]
        ),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    seen = {"step_dt": [], "receiver_time": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt):
        seen["step_dt"].append(dt)
        return state

    def receiver_projector(state, Ep_new, time_value, dt, components):
        seen["receiver_time"].append(time_value)
        return np.zeros((1, 3))

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=receiver_projector,
        turnoff_time=turnoff_time,
        turnoff_steps=2,
    )

    predicted = operator.forward(observation_times)

    np.testing.assert_allclose(predicted, [[10.0, 1.0, -1.0], [20.0, 2.0, -2.0]])
    np.testing.assert_allclose(seen["step_dt"], [0.5e-5, 0.5e-5, 1.0e-5, 1.0e-5])
    np.testing.assert_allclose(seen["receiver_time"], output_internal_times)


def test_primary_secondary_forward_uses_dc_ramp_history_during_turnoff_steps():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([1.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]]]),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    seen = {"Ep_new": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt):
        seen["Ep_new"].append(Ep_new.copy())
        return state

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=lambda state, Ep_new, time_value, dt, components: np.zeros((1, 3)),
        turnoff_time=1.0e-5,
        turnoff_steps=2,
    )

    operator.forward(observation_times)

    np.testing.assert_allclose(seen["Ep_new"][0], [[5.0, 0.0, 0.0]])
    np.testing.assert_allclose(seen["Ep_new"][1], [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(seen["Ep_new"][2], [[1.0, 0.0, 0.0]])


def test_primary_secondary_forward_passes_zero_primary_time_during_turnoff():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([1.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]]]),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    seen = {"primary_time": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt, primary_time):
        seen["primary_time"].append(primary_time)
        return state

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=lambda state, Ep_new, time_value, dt, components: np.zeros((1, 3)),
        turnoff_time=1.0e-5,
        turnoff_steps=2,
    )

    operator.forward(observation_times)

    np.testing.assert_allclose(seen["primary_time"], [0.0, 0.0, 1.0e-5])


def test_primary_secondary_forward_prefetches_receiver_reference_cache_when_available():
    class PrefetchingProvider:
        def __init__(self):
            self.prefetched = None

        def prepare_receiver_reference_cache(self, times, receivers):
            self.prefetched = (np.asarray(times, dtype=float).copy(), np.asarray(receivers, dtype=float).copy())

        def get_Ep_on_V(self, t, points):
            return np.zeros_like(np.asarray(points, dtype=float))

        def get_Ep_dc_on_V(self, points):
            return np.zeros_like(np.asarray(points, dtype=float))

        def get_receiver_E(self, t, receivers):
            return np.zeros_like(np.asarray(receivers, dtype=float))

        def get_receiver_H(self, t, receivers):
            return np.zeros_like(np.asarray(receivers, dtype=float))

        def get_receiver_dBdt(self, t, receivers):
            return np.zeros_like(np.asarray(receivers, dtype=float))

    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5, 2.0e-5])
    provider = PrefetchingProvider()
    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
    )

    operator.forward(times)

    assert provider.prefetched is not None
    np.testing.assert_allclose(provider.prefetched[0], times)
    np.testing.assert_allclose(provider.prefetched[1], receivers)


def test_primary_secondary_forward_passes_primary_time_to_extended_state_stepper():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([1.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]]]),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    seen = {"primary_time": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt, primary_time):
        seen["primary_time"].append(primary_time)
        return state

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=lambda state, Ep_new, time_value, dt, components: np.zeros((1, 3)),
        turnoff_time=1.0e-5,
        turnoff_steps=2,
    )

    operator.forward(observation_times)

    np.testing.assert_allclose(seen["primary_time"], [0.0, 0.0, 1.0e-5])


def test_primary_secondary_forward_honors_max_internal_dt():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([1.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]]]),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )
    seen = {"dt": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt):
        seen["dt"].append(dt)
        return state

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=lambda state, Ep_new, time_value, dt, components: np.zeros((1, 3)),
        turnoff_time=1.0e-5,
        turnoff_steps=1,
        max_internal_dt=2.5e-6,
    )

    operator.forward(observation_times)

    np.testing.assert_allclose(seen["dt"], np.full(8, 2.5e-6))


def test_primary_secondary_forward_uses_configured_primary_time_floor_after_turnoff():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=np.array([2.5e-6, 5.0e-6, 7.5e-6, 1.0e-5]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[2.5, 0.0, 0.0]],
                [[5.0, 0.0, 0.0]],
                [[7.5, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]] * 4),
        receiver_dBdt=np.array([[[0.0, 0.0, -1.0]]] * 4),
        Ep_dc_on_V=np.array([[100.0, 0.0, 0.0]]),
    )
    seen = {"Ep_new": []}

    def state_stepper(state, Ep_old, Ep_new, material, sigma_background, dt):
        seen["Ep_new"].append(Ep_new.copy())
        return state

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        secondary_state_stepper=state_stepper,
        secondary_receiver_projector=lambda state, Ep_new, time_value, dt, components: np.zeros((1, 3)),
        turnoff_time=0.0,
        max_internal_dt=2.5e-6,
        primary_time_floor=2.5e-6,
    )

    operator.forward(observation_times)

    np.testing.assert_allclose(
        [row[0, 0] for row in seen["Ep_new"]],
        [2.5, 5.0, 7.5, 10.0],
    )


def test_primary_secondary_forward_records_actual_internal_time_grid_diagnostics():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    observation_times = np.array([1.0e-5, 2.0e-5])
    turnoff_time = 1.0e-5
    internal_times = np.r_[0.5e-5, 1.0e-5, turnoff_time + observation_times]
    diagnostics = {}
    provider = CachedPrimaryProvider(
        times=internal_times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array(
            [
                [[5.0, 0.5, 0.0]],
                [[10.0, 1.0, 0.0]],
                [[20.0, 2.0, 0.0]],
                [[30.0, 3.0, 0.0]],
            ]
        ),
        receiver_dBdt=np.array(
            [
                [[0.0, 0.0, -0.5]],
                [[0.0, 0.0, -1.0]],
                [[0.0, 0.0, -2.0]],
                [[0.0, 0.0, -3.0]],
            ]
        ),
        Ep_dc_on_V=np.array([[10.0, 0.0, 0.0]]),
    )

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
        turnoff_time=turnoff_time,
        turnoff_steps=2,
        diagnostics=diagnostics,
    )

    operator.forward(observation_times)

    summary = diagnostics["primary_secondary_internal_time_grid"]
    assert summary["turnoff_grid_points"] == 3
    assert summary["observation_output_points"] == 2
    assert summary["total_internal_points"] == 5
    assert summary["stepped_internal_points"] == 4
    assert summary["contains_all_observation_outputs"] is True
    assert summary["first_stepped_internal_time_s"] == pytest.approx(0.5e-5)
    assert summary["last_output_internal_time_s"] == pytest.approx(3.0e-5)
    assert summary["primary_time_origin"] == "after_turnoff_observation_time"
    assert summary["primary_time_mapping"] == (
        "Ep=(1-t_internal/turnoff_time) Ep_dc for t_internal<=turnoff_time; "
        "otherwise t_primary=max(t_internal-turnoff_time, first_observation_time)"
    )
    assert summary["first_primary_time_s"] == pytest.approx(1.0e-5)
    assert summary["first_output_primary_time_s"] == pytest.approx(1.0e-5)
    assert summary["last_output_primary_time_s"] == pytest.approx(2.0e-5)
