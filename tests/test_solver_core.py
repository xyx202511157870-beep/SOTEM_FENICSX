import numpy as np
import pytest
from discretize import TensorMesh
from scipy.constants import mu_0

from atem3d.ip import DebyeTerm, DebyeIPModel
from atem3d.local_coupling import source_edge_moment_basis
from atem3d.magnetic_recovery import (
    biot_savart_h_from_cell_currents,
    biot_savart_h_from_edge_basis_cell_ip_currents,
    biot_savart_h_from_edge_basis_currents,
    biot_savart_h_from_edge_current_moments,
    cell_current_biot_matrix,
)
from atem3d.cpml import CPMLConfig, split_edge_curl
from atem3d.receivers import PointReceiver
from atem3d.source_history_runtime import (
    DrivenRecoverySourceHistoryCorrection,
    SourceHistoryCorrection,
    source_history_correction_field,
)
from atem3d.source_primary import (
    discrete_debye_history_basis,
    discrete_driven_relaxation_basis,
)
import atem3d.simulation as simulation_module
from atem3d.simulation import TDEMIPSimulation
from atem3d.sources import GroundedWireSource, StepOffWaveform


def _mesh():
    return TensorMesh([np.ones(4), np.ones(4), np.ones(3)], origin=(-2.0, -2.0, -1.5))


def test_system_matrix_uses_debye_effective_conductivity():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 2.0)
    delta_sigma = np.full(mesh.n_cells, 0.5)
    ip_model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.25)])
    sim = TDEMIPSimulation(mesh=mesh, ip_model=ip_model, time_steps=[0.75])

    matrix = sim.system_matrix(0)
    expected_sigma = ip_model.effective_sigma(0.75)
    expected = (
        mesh.edge_curl.T @ sim.face_mu_inverse_matrix @ mesh.edge_curl
        + (1.0 / 0.75) * mesh.get_edge_inner_product(expected_sigma)
    )

    np.testing.assert_allclose(matrix.toarray(), expected.toarray())


def test_zero_thickness_cpml_system_matrix_matches_plain_matrix():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 2.0)
    delta_sigma = np.full(mesh.n_cells, 0.5)
    ip_model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.25)])
    kwargs = dict(mesh=mesh, ip_model=ip_model, time_steps=[0.75])
    plain = TDEMIPSimulation(**kwargs)
    cpml = TDEMIPSimulation(**kwargs, cpml=CPMLConfig(thickness_cells=0, sigma_max=0.0))

    np.testing.assert_allclose(cpml.system_matrix(0).toarray(), plain.system_matrix(0).toarray())


def test_cpml_run_reuses_cached_curl_split(monkeypatch):
    mesh = TensorMesh([[(1.0, 3)], [(1.0, 3)], [(1.0, 3)]], origin="CCC")
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[1.0e-3, 1.0e-3],
        cpml=CPMLConfig(thickness_cells=1, sigma_max=10.0),
        linear_solver="direct",
    )
    calls = {"count": 0}

    def counted_split(mesh_arg):
        calls["count"] += 1
        return split_edge_curl(mesh_arg)

    monkeypatch.setattr(
        simulation_module,
        "split_edge_curl",
        counted_split,
        raising=False,
    )

    sim.run_data_only()

    assert calls["count"] == 1


def test_zero_thickness_cpml_run_matches_plain_run():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.02],
        sources=[source],
    )

    plain = TDEMIPSimulation(**kwargs).run()
    cpml = TDEMIPSimulation(**kwargs, cpml=CPMLConfig(thickness_cells=0, sigma_max=0.0)).run()

    np.testing.assert_allclose(cpml.e, plain.e, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(cpml.b, plain.b, rtol=1.0e-12, atol=1.0e-14)


def test_zero_source_zero_initial_fields_stays_zero():
    mesh = _mesh()
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.ones(mesh.n_cells)),
        time_steps=[0.1, 0.2],
    )

    result = sim.run()

    assert result.e.shape == (3, mesh.n_edges)
    assert result.b.shape == (3, mesh.n_faces)
    np.testing.assert_allclose(result.e, 0.0)
    np.testing.assert_allclose(result.b, 0.0)


def test_grounded_wire_step_source_produces_finite_nonzero_fields():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.02],
        sources=[source],
        receivers=[
            PointReceiver(location=(0.0, 0.5, 0.0), component="Ex"),
            PointReceiver(location=(0.0, 0.5, 0.0), component="Hz"),
        ],
    )

    result = sim.run()

    assert np.isfinite(result.e).all()
    assert np.isfinite(result.b).all()
    assert np.linalg.norm(result.e[-1]) > 0.0
    assert result.data.shape == (3, 2)


def test_run_data_only_matches_full_run_receiver_data():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.full(mesh.n_cells, 0.02), tau=0.05)],
        ),
        time_steps=[0.01, 0.02],
        sources=[source],
        receivers=[
            PointReceiver(location=(0.0, 0.5, 0.0), component="Ex"),
            PointReceiver(location=(0.0, 0.5, 0.0), component="Hz"),
        ],
        magnetic_receiver_mode="current_biot",
    )

    full = TDEMIPSimulation(**kwargs).run()
    data_only = TDEMIPSimulation(**kwargs).run_data_only()

    np.testing.assert_allclose(data_only.times, full.times)
    np.testing.assert_allclose(data_only.data, full.data)
    assert not hasattr(data_only, "e")
    assert not hasattr(data_only, "b")


def test_current_biot_magnetic_receiver_samples_recovered_conduction_field():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        initial_magnetic_mode="biot_savart_wire",
        magnetic_receiver_mode="current_biot",
    )

    result = sim.run()
    e_cc = (mesh.average_edge_to_cell_vector @ result.e[1]).reshape((mesh.n_cells, 3), order="F")
    current_density = sigma[:, None] * e_cc
    expected = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        np.array([receiver.location]),
    )[0, 2]

    np.testing.assert_allclose(result.data[1, 0], expected)


def test_source_primary_delta6_adds_debye_source_like_h_correction():
    from geoana.em.static import LineCurrentWholeSpace

    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
    ).run()

    line_current = LineCurrentWholeSpace(
        source.locations,
        current=source.current * source.waveform.initial_value(),
        mu=mu_0,
    )
    h_wire_z = line_current.magnetic_field(np.array([receiver.location]))[0, 2]
    length = np.linalg.norm(np.asarray(source.end) - np.asarray(source.start))
    expected_delta = (
        -6.0
        * mu_0
        * delta_sigma[0]
        * length**2
        * np.exp(-0.01 / (2.0 * tau))
        * h_wire_z
    )

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_source_primary_delta6_can_use_edge_source_moment_basis():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
        magnetic_recovery_source_primary_delta6_basis="edge_current",
    ).run()

    source_h_z = biot_savart_h_from_edge_current_moments(
        mesh,
        source.initial_edge_vector(mesh),
        np.array([receiver.location]),
    )[0, 2]
    length = np.linalg.norm(np.asarray(source.end) - np.asarray(source.start))
    expected_delta = (
        -6.0
        * mu_0
        * delta_sigma[0]
        * length**2
        * np.exp(-0.01 / (2.0 * tau))
        * source_h_z
    )

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_source_primary_delta6_vanishes_without_debye_terms():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.2)),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_prescribed_source_history_correction_adds_source_moment_response():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=(0.1, -0.03, 0.04, 0.02),
        receiver_matrix="current_biot",
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
        magnetic_recovery_source_history=correction,
    )

    base = TDEMIPSimulation(
        **{key: value for key, value in kwargs.items() if key != "magnetic_recovery_source_history"}
    ).run()
    corrected = TDEMIPSimulation(**kwargs).run()

    source_vector = source.initial_edge_vector(mesh)
    moments = source_edge_moment_basis(
        mesh,
        source_vector,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=[0, 2],
    )
    receiver_matrix = cell_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum(
        "lce,se->slc",
        receiver_matrix,
        moments.basis_vectors,
    )
    history = discrete_debye_history_basis([0.01], tau=tau, max_order=1)
    coefficient_matrix = np.asarray(correction.coefficients).reshape(2, 2)
    expected_delta = np.einsum(
        "ps,p,slc->lc",
        coefficient_matrix,
        history.values[1],
        static_response,
    )[0, 2]

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_prescribed_source_history_normalized_coefficients_match_absolute_scale():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    normalized = np.array([0.1, -0.03, 0.04, 0.02])
    scale = mu_0 * 0.02 * 3.0**2
    absolute = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=tuple(normalized * scale),
        receiver_matrix="current_biot",
    )
    scaled = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        normalized_coefficients=tuple(normalized),
        receiver_matrix="current_biot",
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    absolute_result = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=absolute,
    ).run()
    scaled_result = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=scaled,
    ).run()

    np.testing.assert_allclose(scaled_result.data, absolute_result.data)


def test_prescribed_source_history_correction_vanishes_without_debye_terms():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = SourceHistoryCorrection(
        tau=0.05,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=(0.1, -0.03, 0.04, 0.02),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.2)),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_prescribed_source_history_correction_vanishes_for_zero_delta_sigma():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = SourceHistoryCorrection(
        tau=0.05,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=(0.1, -0.03, 0.04, 0.02),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.zeros(mesh.n_cells), tau=0.05)],
        ),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_prescribed_source_history_corrections_sum_multiple_tau_terms():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    corrections = (
        SourceHistoryCorrection(
            tau=0.05,
            max_order=1,
            source_moment_degrees=(0,),
            coefficients=(0.1, 0.04),
            receiver_matrix="current_biot",
        ),
        SourceHistoryCorrection(
            tau=0.2,
            max_order=0,
            source_moment_degrees=(2,),
            coefficients=(-0.03,),
            receiver_matrix="current_biot",
        ),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.full(mesh.n_cells, 0.02), tau=0.05)],
        ),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=corrections,
    ).run()

    expected_delta = 0.0
    for correction in corrections:
        moments = source_edge_moment_basis(
            mesh,
            source.initial_edge_vector(mesh),
            start=source.locations[0],
            end=source.locations[-1],
            degrees=correction.source_moment_degrees,
        )
        receiver_matrix = cell_current_biot_matrix(mesh, np.array([receiver.location]))
        static_response = np.einsum(
            "lce,se->slc",
            receiver_matrix,
            moments.basis_vectors,
        )
        history = discrete_debye_history_basis(
            [0.01],
            tau=correction.tau,
            max_order=correction.max_order,
        )
        coefficient_matrix = np.asarray(correction.coefficients).reshape(
            correction.max_order + 1,
            len(correction.source_moment_degrees),
        )
        expected_delta += np.einsum(
            "ps,p,slc->lc",
            coefficient_matrix,
            history.values[1],
            static_response,
        )[0, 2]

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_driven_recovery_source_history_correction_uses_driven_basis():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = DrivenRecoverySourceHistoryCorrection(
        driver_tau=0.05,
        response_tau=0.01,
        source_moment_degrees=(0, 2),
        coefficients=(0.1, -0.03),
        receiver_matrix="current_biot",
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.full(mesh.n_cells, 0.02), tau=0.05)],
        ),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    moments = source_edge_moment_basis(
        mesh,
        source.initial_edge_vector(mesh),
        start=source.locations[0],
        end=source.locations[-1],
        degrees=correction.source_moment_degrees,
    )
    receiver_matrix = cell_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum(
        "lce,se->slc",
        receiver_matrix,
        moments.basis_vectors,
    )
    history = discrete_driven_relaxation_basis(
        [0.01],
        driver_tau=correction.driver_tau,
        response_tau=correction.response_tau,
    )
    expected_delta = history.values[1] * np.einsum(
        "s,slc->lc",
        np.asarray(correction.coefficients),
        static_response,
    )[0, 2]

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_driven_recovery_source_history_normalized_coefficients_match_absolute_scale():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    normalized = np.array([0.1, -0.03])
    scale = mu_0 * 0.02 * 3.0**2
    absolute = DrivenRecoverySourceHistoryCorrection(
        driver_tau=0.05,
        response_tau=0.01,
        source_moment_degrees=(0, 2),
        coefficients=tuple(normalized * scale),
        receiver_matrix="current_biot",
    )
    scaled = DrivenRecoverySourceHistoryCorrection(
        driver_tau=0.05,
        response_tau=0.01,
        source_moment_degrees=(0, 2),
        normalized_coefficients=tuple(normalized),
        receiver_matrix="current_biot",
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.full(mesh.n_cells, 0.02), tau=0.05)],
        ),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    absolute_result = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=absolute,
    ).run()
    scaled_result = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=scaled,
    ).run()

    np.testing.assert_allclose(scaled_result.data, absolute_result.data)


def test_driven_recovery_source_history_correction_sums_multiple_response_modes():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = DrivenRecoverySourceHistoryCorrection(
        driver_tau=0.05,
        response_tau=(0.01, 0.02),
        source_moment_degrees=(0, 2),
        coefficients=(0.1, -0.03, 0.04, 0.02),
        receiver_matrix="current_biot",
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel(
            np.full(mesh.n_cells, 0.2),
            [DebyeTerm(np.full(mesh.n_cells, 0.02), tau=0.05)],
        ),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="current_biot",
    )

    base = TDEMIPSimulation(**kwargs).run()
    corrected = TDEMIPSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    moments = source_edge_moment_basis(
        mesh,
        source.initial_edge_vector(mesh),
        start=source.locations[0],
        end=source.locations[-1],
        degrees=correction.source_moment_degrees,
    )
    receiver_matrix = cell_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum(
        "lce,se->slc",
        receiver_matrix,
        moments.basis_vectors,
    )
    coefficient_matrix = np.asarray(correction.coefficients).reshape(
        len(correction.response_taus),
        len(correction.source_moment_degrees),
    )
    expected_delta = 0.0
    for mode_index, response_tau in enumerate(correction.response_taus):
        history = discrete_driven_relaxation_basis(
            [0.01],
            driver_tau=correction.driver_tau,
            response_tau=response_tau,
        )
        expected_delta += history.values[1] * np.einsum(
            "s,slc->lc",
            coefficient_matrix[mode_index],
            static_response,
        )[0, 2]

    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_source_history_correction_field_reuses_cached_static_response(monkeypatch):
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    correction = SourceHistoryCorrection(
        tau=0.05,
        max_order=0,
        source_moment_degrees=(0,),
        coefficients=(0.1,),
        receiver_matrix="current_biot",
    )
    calls = {"count": 0}

    def receiver_matrix(mesh_arg, locations_arg, *, subdivisions):
        calls["count"] += 1
        assert mesh_arg is mesh
        assert subdivisions == 1
        return np.ones((locations_arg.shape[0], 3, mesh.n_edges))

    monkeypatch.setattr(
        "atem3d.source_history_runtime.cell_current_biot_matrix",
        receiver_matrix,
    )
    cache = {}
    kwargs = dict(
        mesh=mesh,
        sources=[source],
        time_steps=[0.01],
        time=0.01,
        locations=np.array([[0.0, 0.5, 0.0]]),
        correction=correction,
        magnetic_receiver_mode="current_biot",
        cache=cache,
    )

    first = source_history_correction_field(**kwargs)
    second = source_history_correction_field(**kwargs)

    assert calls["count"] == 1
    np.testing.assert_allclose(second, first)


def test_edge_current_biot_magnetic_receiver_samples_edge_current_moments():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="edge_current_biot",
    )

    result = sim.run()
    edge_current = mesh.get_edge_inner_product(sigma).tocsr() @ result.e[1]
    expected = biot_savart_h_from_edge_current_moments(
        mesh,
        edge_current,
        np.array([receiver.location]),
    )[0, 2]

    np.testing.assert_allclose(result.data[1, 0], expected)


def test_edge_basis_biot_magnetic_receiver_samples_edge_basis_current():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="edge_basis_biot",
        magnetic_recovery_subdivisions=2,
    )

    result = sim.run()
    edge_current = mesh.get_edge_inner_product(sigma).tocsr() @ result.e[1]
    edge_current_field = edge_current / sim.unit_edge_mass_diagonal
    expected = biot_savart_h_from_edge_basis_currents(
        mesh,
        edge_current_field,
        np.array([receiver.location]),
        subdivisions=2,
    )[0, 2]

    np.testing.assert_allclose(result.data[1, 0], expected)


def test_edge_basis_cell_biot_magnetic_receiver_samples_local_cell_current():
    mesh = _mesh()
    sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
        receivers=[receiver],
        magnetic_receiver_mode="edge_basis_cell_biot",
        magnetic_recovery_subdivisions=2,
    )

    result = sim.run()
    expected = biot_savart_h_from_edge_basis_cell_ip_currents(
        mesh,
        result.e[1],
        sim.ip_model.sigma_infinity,
        sim.ip_model.terms,
        [],
        np.array([receiver.location]),
        subdivisions=2,
    )[0, 2]

    np.testing.assert_allclose(result.data[1, 0], expected)


def test_cell_current_density_uses_mass_consistent_edge_current_projection_for_ip():
    mesh = _mesh()
    sigma_inf = np.linspace(0.1, 0.4, mesh.n_cells)
    delta_sigma = np.linspace(0.01, 0.04, mesh.n_cells)
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.02)]),
        time_steps=[0.01],
    )
    e = np.linspace(-0.2, 0.3, mesh.n_edges)
    memory = np.linspace(0.4, -0.1, mesh.n_edges)

    edge_current = (
        mesh.get_edge_inner_product(sim.ip_model.sigma_infinity) @ e
        - mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma) @ memory
    )
    unit_edge_mass = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).tocsc()
    edge_current_field = np.linalg.solve(unit_edge_mass.toarray(), edge_current)
    expected = (mesh.average_edge_to_cell_vector @ edge_current_field).reshape(
        (mesh.n_cells, 3),
        order="F",
    )

    recovered = sim._cell_current_density(e, [memory])

    np.testing.assert_allclose(recovered, expected)


def test_cell_current_density_can_scale_polarization_current_for_diagnostics():
    mesh = _mesh()
    sigma_inf = np.linspace(0.1, 0.4, mesh.n_cells)
    delta_sigma = np.linspace(0.01, 0.04, mesh.n_cells)
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.02)]),
        time_steps=[0.01],
        magnetic_recovery_polarization_scale=1.25,
    )
    e = np.linspace(-0.2, 0.3, mesh.n_edges)
    memory = np.linspace(0.4, -0.1, mesh.n_edges)

    edge_current = (
        mesh.get_edge_inner_product(sim.ip_model.sigma_infinity) @ e
        - 1.25 * (mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma) @ memory)
    )
    edge_current_field = edge_current / sim.unit_edge_mass_diagonal
    expected = (mesh.average_edge_to_cell_vector @ edge_current_field).reshape(
        (mesh.n_cells, 3),
        order="F",
    )

    recovered = sim._cell_current_density(e, [memory])

    np.testing.assert_allclose(recovered, expected)


def test_cell_current_density_can_use_low_frequency_ratio_polarization_scale():
    mesh = _mesh()
    sigma_inf = np.linspace(0.2, 0.5, mesh.n_cells)
    delta_sigma = 0.2 * sigma_inf
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.02)]),
        time_steps=[0.01],
        magnetic_recovery_polarization_scale="low_frequency_ratio",
    )
    e = np.linspace(-0.2, 0.3, mesh.n_edges)
    memory = np.linspace(0.4, -0.1, mesh.n_edges)

    ratio = sigma_inf / (sigma_inf - delta_sigma)
    edge_current = (
        mesh.get_edge_inner_product(sim.ip_model.sigma_infinity) @ e
        - mesh.get_edge_inner_product(ratio * sim.ip_model.terms[0].delta_sigma) @ memory
    )
    edge_current_field = edge_current / sim.unit_edge_mass_diagonal
    expected = (mesh.average_edge_to_cell_vector @ edge_current_field).reshape(
        (mesh.n_cells, 3),
        order="F",
    )

    recovered = sim._cell_current_density(e, [memory])

    np.testing.assert_allclose(recovered, expected)


def test_cell_current_density_can_use_component_polarization_scale():
    mesh = _mesh()
    sigma_inf = np.linspace(0.2, 0.5, mesh.n_cells)
    delta_sigma = 0.2 * sigma_inf
    scale = np.array([1.1, 0.9, 0.5])
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.02)]),
        time_steps=[0.01],
        magnetic_recovery_polarization_scale=scale,
    )
    e = np.linspace(-0.2, 0.3, mesh.n_edges)
    memory = np.linspace(0.4, -0.1, mesh.n_edges)

    polarization = mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma) @ memory
    edge_scale = np.r_[
        np.full(mesh.n_edges_x, scale[0]),
        np.full(mesh.n_edges_y, scale[1]),
        np.full(mesh.n_edges_z, scale[2]),
    ]
    edge_current = (
        mesh.get_edge_inner_product(sim.ip_model.sigma_infinity) @ e
        - edge_scale * polarization
    )
    edge_current_field = edge_current / sim.unit_edge_mass_diagonal
    expected = (mesh.average_edge_to_cell_vector @ edge_current_field).reshape(
        (mesh.n_cells, 3),
        order="F",
    )

    recovered = sim._cell_current_density(e, [memory])

    np.testing.assert_allclose(recovered, expected)


def test_edge_current_moments_can_add_initial_polarization_current_for_diagnostics():
    mesh = _mesh()
    sigma_inf = np.linspace(0.2, 0.5, mesh.n_cells)
    delta_sigma = 0.2 * sigma_inf
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.02)]),
        time_steps=[0.01],
        magnetic_recovery_initial_polarization_scale=0.25,
    )
    e = np.linspace(-0.2, 0.3, mesh.n_edges)
    memory = np.linspace(0.4, -0.1, mesh.n_edges)
    initial_memory = np.linspace(-0.1, 0.2, mesh.n_edges)

    expected = (
        mesh.get_edge_inner_product(sim.ip_model.sigma_infinity) @ e
        - mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma) @ memory
        + 0.25 * (mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma) @ initial_memory)
    )

    recovered = sim._edge_current_moments(e, [memory], [initial_memory])

    np.testing.assert_allclose(recovered, expected)


def test_repeated_time_steps_reuse_direct_factorization():
    mesh = _mesh()
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.ones(mesh.n_cells)),
        time_steps=[0.1, 0.1, 0.2],
    )
    calls = []

    def fake_factorize(matrix):
        calls.append(matrix.shape)
        return lambda rhs: np.zeros(matrix.shape[0])

    sim._factorize = fake_factorize
    sim.run()

    assert calls == [(mesh.n_edges, mesh.n_edges), (mesh.n_edges, mesh.n_edges)]


def test_cg_linear_solver_mode_matches_direct_solution_for_small_problem():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.02],
        sources=[source],
    )

    direct = TDEMIPSimulation(**kwargs).run()
    cg = TDEMIPSimulation(**kwargs, linear_solver="cg", cg_tolerance=1.0e-10).run()

    np.testing.assert_allclose(cg.e, direct.e, rtol=1.0e-8, atol=1.0e-10)


def test_pardiso_linear_solver_mode_matches_direct_solution_for_small_problem():
    pytest.importorskip("pydiso")
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.02],
        sources=[source],
    )

    direct = TDEMIPSimulation(**kwargs).run()
    pardiso = TDEMIPSimulation(**kwargs, linear_solver="pardiso").run()

    np.testing.assert_allclose(pardiso.e, direct.e, rtol=1.0e-8, atol=1.0e-10)


def test_pardiso_linear_solver_supports_active_cpml_for_small_problem():
    pytest.importorskip("pydiso")
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(4)], origin=(-2.5, -2.5, -2.0))
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.02],
        sources=[source],
        cpml=CPMLConfig(thickness_cells=1, sigma_max=2.0),
    )

    direct = TDEMIPSimulation(**kwargs, linear_solver="direct").run()
    pardiso = TDEMIPSimulation(**kwargs, linear_solver="pardiso").run()

    np.testing.assert_allclose(pardiso.e, direct.e, rtol=1.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(pardiso.b, direct.b, rtol=1.0e-8, atol=1.0e-10)


def test_cg_jacobi_preconditioner_handles_diagonal_system():
    mesh = _mesh()
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.ones(mesh.n_cells)),
        time_steps=[0.1],
        linear_solver="cg",
        cg_preconditioner="jacobi",
    )
    matrix = sim.system_matrix(0)

    preconditioner = sim._cg_preconditioner(matrix)
    ones = np.ones(matrix.shape[0])

    np.testing.assert_allclose(preconditioner @ matrix.diagonal(), ones)


def test_step_off_source_rhs_uses_initial_on_current_at_first_step():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=2.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.ones(mesh.n_cells)),
        time_steps=[0.5],
        sources=[source],
    )

    expected = source.edge_vector(mesh, current_scale=1.0) / 0.5

    np.testing.assert_allclose(sim.source_rhs(0), expected)


def test_source_rhs_uses_waveform_interval_average_didt_api():
    class IntervalOnlyWaveform:
        has_initial_fields = False

        def value(self, time):
            return 0.0

        def previous_value(self, time):
            return 0.0

        def initial_value(self):
            return 0.0

        def interval_average_didt(self, t0, t1):
            return -2.0

    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=3.0,
        waveform=IntervalOnlyWaveform(),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.ones(mesh.n_cells)),
        time_steps=[0.25],
        sources=[source],
    )

    expected = -source.edge_vector_interval_average_didt(mesh, 0.0, 0.25)

    np.testing.assert_allclose(sim.source_rhs(0), expected)


def test_step_off_initial_magnetic_flux_satisfies_static_ampere_balance():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
    )

    result = sim.run()
    e0 = result.e[0]
    b0 = result.b[0]
    lhs = mesh.edge_curl.T @ sim.face_mu_inverse_matrix @ b0
    rhs = mesh.get_edge_inner_product(sigma) @ e0 + source.initial_edge_vector(mesh)

    assert np.linalg.norm(b0) > 0.0
    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-10)


def test_biot_savart_wire_initial_magnetic_flux_matches_finite_wire_scale():
    mesh = TensorMesh(
        [
            [(80.0, 4, -1.2), (5.0, 20), (80.0, 4, 1.2)],
            [(80.0, 4, -1.2), (5.0, 16), (80.0, 4, 1.2)],
            [(80.0, 8), (5.0, 7), (1.0, 4), (0.5, 2), (80.0, 3)],
        ],
        origin=(-565.328, -565.328, -680.0),
    )
    source = GroundedWireSource(
        start=(-25.0, 0.0, -0.5),
        end=(25.0, 0.0, -0.5),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    receiver = PointReceiver(location=(0.0, 10.0, -0.5), component="Hz")
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.01)),
        time_steps=[0.0001],
        sources=[source],
        receivers=[receiver],
        initial_magnetic_mode="biot_savart_wire",
    )

    e0 = sim.initial_electric_field()
    b0 = sim.initial_magnetic_flux_density(e0)
    hz0 = receiver.sample(mesh, e0, b0, sim.mu)

    half_length = 25.0
    offset = 10.0
    expected_hz = half_length / (2.0 * np.pi * offset * np.sqrt(half_length**2 + offset**2))
    np.testing.assert_allclose(mesh.face_divergence @ b0, 0.0, atol=1.0e-20)
    np.testing.assert_allclose(hz0, expected_hz, rtol=0.25)


def test_time_step_can_start_from_biot_savart_wire_initial_flux():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01],
        sources=[source],
        initial_magnetic_mode="biot_savart_wire",
    )

    result = sim.run()
    faraday = (result.b[1] - result.b[0]) / float(sim.time_steps[0]) + mesh.edge_curl @ result.e[1]

    assert np.isfinite(result.e).all()
    assert np.isfinite(result.b).all()
    assert np.linalg.norm(result.b[0]) > 0.0
    np.testing.assert_allclose(faraday, 0.0, atol=1e-12)


def test_non_cpml_electric_solution_does_not_depend_on_initial_magnetic_mode():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.1)
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    kwargs = dict(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(sigma),
        time_steps=[0.01, 0.02],
        sources=[source],
    )

    ampere = TDEMIPSimulation(**kwargs, initial_magnetic_mode="ampere").run()
    biot = TDEMIPSimulation(**kwargs, initial_magnetic_mode="biot_savart_wire").run()

    assert np.linalg.norm(ampere.b[0] - biot.b[0]) > 0.0
    np.testing.assert_allclose(biot.e, ampere.e, rtol=1e-12, atol=1e-14)


def test_initial_magnetic_mode_zero_starts_from_zero_b_field():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01],
        sources=[source],
        initial_magnetic_mode="zero",
    )

    result = sim.run()
    expected_b1 = -float(sim.time_steps[0]) * (mesh.edge_curl @ result.e[1])

    np.testing.assert_allclose(result.b[0], 0.0)
    np.testing.assert_allclose(result.b[1], expected_b1, atol=1.0e-14)


def test_time_step_solution_satisfies_debye_ip_eliminated_equation():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.5, 0.0, 0.0),
        end=(1.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=1.0),
    )
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.05)
    sim = TDEMIPSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.03)]),
        time_steps=[0.01, 0.02],
        sources=[source],
    )

    result = sim.run()
    memories = sim.ip_model.initial_memory(mesh.n_edges, result.e[0])
    for step_index, dt in enumerate(sim.time_steps):
        alpha, beta = sim.ip_model.terms[0].coefficients(float(dt))
        me_sigma_inf = mesh.get_edge_inner_product(sim.ip_model.sigma_infinity)
        me_delta = mesh.get_edge_inner_product(sim.ip_model.terms[0].delta_sigma)
        constitutive_current = (
            me_sigma_inf @ result.e[step_index + 1]
            - me_delta @ (alpha * memories[0] + beta * result.e[step_index + 1])
        )
        lhs_ampere = mesh.edge_curl.T @ sim.face_mu_inverse_matrix @ result.b[step_index + 1]
        rhs_ampere = constitutive_current + sim.electric_source_term(result.times[step_index + 1])
        np.testing.assert_allclose(lhs_ampere, rhs_ampere, rtol=1e-8, atol=1e-10)

        faraday = (
            (result.b[step_index + 1] - result.b[step_index]) / float(dt)
            + mesh.edge_curl @ result.e[step_index + 1]
        )
        np.testing.assert_allclose(faraday, 0.0, atol=1e-12)
        memories = sim.ip_model.update_memory(memories, result.e[step_index + 1], float(dt))
