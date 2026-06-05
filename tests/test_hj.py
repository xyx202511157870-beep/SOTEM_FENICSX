import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh
from scipy.constants import mu_0
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

from atem3d.hj import (
    HJMagneticSimulation,
    face_project_debye_model,
    hj_dc_initial_current_density,
    hj_dc_initial_electric_field,
    hj_mmr_initial_magnetic_field,
    hj_magnetic_rhs,
    hj_magnetic_step,
    hj_magnetic_system_matrix,
)
from atem3d.ip import DebyeIPModel, DebyeTerm
from atem3d.local_coupling import source_face_moment_basis
from atem3d.magnetic_recovery import biot_savart_h_from_cell_currents
from atem3d.magnetic_recovery import biot_savart_h_from_face_basis_cell_ip_currents
from atem3d.magnetic_recovery import face_basis_biot_matrix
from atem3d.magnetic_recovery import face_current_biot_matrix
from atem3d.receivers import PointReceiver
from atem3d.source_history_runtime import SourceHistoryCorrection
from atem3d.source_history_runtime import InitialPolarizationSourceHistoryCorrection
from atem3d.source_history_runtime import (
    ChargeConservingInitialPolarizationSourceHistoryCorrection,
    SourceDiffusionKernelSourceHistoryCorrection,
    SourcePrimaryDelta6SourceHistoryCorrection,
    charge_conserving_face_current,
)
from atem3d.source_primary import discrete_debye_history_basis
from atem3d.sources import GroundedWireSource, StepOffWaveform


def _mesh():
    return TensorMesh([np.ones(3), np.ones(3), np.ones(3)], origin="CCC")


def test_hj_magnetic_system_matrix_matches_simpeg_no_ip():
    mesh = _mesh()
    sigma = np.linspace(0.1, 0.2, mesh.n_cells)
    dt = 0.01

    ours = hj_magnetic_system_matrix(mesh, DebyeIPModel.no_ip(sigma), [], dt)

    simpeg_sim = tdem.Simulation3DMagneticField(
        mesh,
        survey=tdem.Survey([]),
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
        time_steps=[dt],
    )
    simpeg_sim.model = sigma
    expected = simpeg_sim.getAdiag(0).tocsr()

    np.testing.assert_allclose(ours.toarray(), expected.toarray(), rtol=1.0e-12, atol=1.0e-14)


def test_face_project_debye_model_uses_face_inner_product_diagonal():
    mesh = _mesh()
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])

    projected = face_project_debye_model(mesh, model)

    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).diagonal()
    expected_sigma = mesh.get_face_inner_product(sigma_inf).diagonal() / unit_face_mass
    expected_delta = mesh.get_face_inner_product(delta_sigma).diagonal() / unit_face_mass

    assert projected.n_cells == mesh.n_faces
    np.testing.assert_allclose(projected.sigma_infinity, expected_sigma)
    np.testing.assert_allclose(projected.terms[0].delta_sigma, expected_delta)


def test_hj_magnetic_system_matrix_uses_debye_effective_resistivity_on_faces():
    mesh = _mesh()
    model = DebyeIPModel(
        sigma_infinity=np.array([2.0]),
        terms=[DebyeTerm(delta_sigma=np.array([0.4]), tau=0.2)],
    )
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    dt = 0.05

    matrix = hj_magnetic_system_matrix(mesh, model, [memory], dt)

    rho_eff, _ = model.inverse_constitutive_coefficients([memory], dt, n_dofs=mesh.n_faces)
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    face_rho = unit_face_mass @ sp.diags(rho_eff, format="csr")
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    expected = mesh.edge_curl.T @ face_rho @ mesh.edge_curl + (1.0 / dt) * edge_mu

    np.testing.assert_allclose(matrix.toarray(), expected.toarray(), rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_system_matrix_accepts_layered_debye_cell_model():
    mesh = _mesh()
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    dt = 0.05

    matrix = hj_magnetic_system_matrix(mesh, model, [memory], dt)

    _, beta = model.terms[0].coefficients(dt)
    rho_eff = 1.0 / (sigma_inf - beta * delta_sigma)
    face_rho = mesh.get_face_inner_product(rho_eff).tocsr()
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    expected = mesh.edge_curl.T @ face_rho @ mesh.edge_curl + (1.0 / dt) * edge_mu

    np.testing.assert_allclose(matrix.toarray(), expected.toarray(), rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_rhs_uses_cellwise_debye_history_resistivity_projection():
    mesh = _mesh()
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    h_old = np.linspace(-0.2, 0.3, mesh.n_edges)
    dt = 0.05

    rhs = hj_magnetic_rhs(mesh, h_old, model, [memory], dt)

    alpha, beta = model.terms[0].coefficients(dt)
    rho_eff = 1.0 / (sigma_inf - beta * delta_sigma)
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    history_mass = mesh.get_face_inner_product(rho_eff * alpha * delta_sigma).tocsr()
    electric_history = history_mass.diagonal() / unit_face_mass.diagonal() * memory
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    expected = (edge_mu @ h_old) / dt - mesh.edge_curl.T @ (
        unit_face_mass @ electric_history
    )

    np.testing.assert_allclose(rhs, expected, rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_rhs_includes_debye_history_electric_field():
    mesh = _mesh()
    model = DebyeIPModel(
        sigma_infinity=np.array([2.0]),
        terms=[DebyeTerm(delta_sigma=np.array([0.4]), tau=0.2)],
    )
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    h_old = np.linspace(-0.2, 0.3, mesh.n_edges)
    dt = 0.05

    rhs = hj_magnetic_rhs(mesh, h_old, model, [memory], dt)

    _, electric_history = model.inverse_constitutive_coefficients(
        [memory],
        dt,
        n_dofs=mesh.n_faces,
    )
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    expected = (edge_mu @ h_old) / dt - mesh.edge_curl.T @ (unit_face_mass @ electric_history)

    np.testing.assert_allclose(rhs, expected, rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_step_satisfies_debye_eliminated_faraday_equation():
    mesh = _mesh()
    model = DebyeIPModel(
        sigma_infinity=np.array([2.0]),
        terms=[DebyeTerm(delta_sigma=np.array([0.4]), tau=0.2)],
    )
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    h_old = np.linspace(-0.2, 0.3, mesh.n_edges)
    dt = 0.05

    result = hj_magnetic_step(mesh, h_old, model, [memory], dt)

    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    faraday = (edge_mu @ (result.h - h_old)) / dt + mesh.edge_curl.T @ (
        unit_face_mass @ result.e
    )
    current = mesh.edge_curl @ result.h
    constitutive_current = model.sigma_infinity[0] * result.e
    for term, updated_memory in zip(model.terms, result.memories):
        constitutive_current -= term.delta_sigma[0] * updated_memory

    np.testing.assert_allclose(faraday, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(constitutive_current, current, rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_step_uses_grounded_wire_face_source_in_ampere_current():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    model = DebyeIPModel.no_ip(np.array([2.0]))
    h_old = np.zeros(mesh.n_edges)
    dt = 0.05
    electric_source = source.face_vector_at(mesh, 0.0)

    result = hj_magnetic_step(mesh, h_old, model, [], dt, electric_source=electric_source)

    current = mesh.edge_curl @ result.h - electric_source
    constitutive_current = model.sigma_infinity[0] * result.e

    np.testing.assert_allclose(constitutive_current, current, rtol=1.0e-12, atol=1.0e-14)


def test_hj_magnetic_step_with_layered_debye_uses_cellwise_resistivity_law():
    mesh = _mesh()
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])
    memory = np.linspace(-0.1, 0.2, mesh.n_faces)
    h_old = np.linspace(-0.2, 0.3, mesh.n_edges)
    dt = 0.05

    result = hj_magnetic_step(mesh, h_old, model, [memory], dt)

    current = mesh.edge_curl @ result.h
    alpha, beta = model.terms[0].coefficients(dt)
    rho_eff = 1.0 / (sigma_inf - beta * delta_sigma)
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    face_rho = mesh.get_face_inner_product(rho_eff).tocsr()
    history_mass = mesh.get_face_inner_product(rho_eff * alpha * delta_sigma).tocsr()
    expected_electric = (face_rho @ current) / unit_face_mass.diagonal()
    expected_electric += history_mass.diagonal() / unit_face_mass.diagonal() * memory
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    faraday = (edge_mu @ (result.h - h_old)) / dt + mesh.edge_curl.T @ (
        unit_face_mass @ result.e
    )

    np.testing.assert_allclose(result.e, expected_electric, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(result.memories[0], alpha * memory + beta * result.e)
    np.testing.assert_allclose(faraday, 0.0, atol=1.0e-12)


def test_hj_magnetic_simulation_runs_multiple_no_ip_steps():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    initial_h = np.linspace(-0.2, 0.3, mesh.n_edges)
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05, 0.1],
        initial_h=initial_h,
    )

    result = sim.run()

    assert result.h.shape == (3, mesh.n_edges)
    assert result.e.shape == (3, mesh.n_faces)
    assert result.memories == []
    np.testing.assert_allclose(result.h[0], initial_h)
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    for step_index, dt in enumerate(sim.time_steps):
        faraday = (edge_mu @ (result.h[step_index + 1] - result.h[step_index])) / dt
        faraday += mesh.edge_curl.T @ (unit_face_mass @ result.e[step_index + 1])
        current = mesh.edge_curl @ result.h[step_index + 1]
        np.testing.assert_allclose(faraday, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(model.sigma_infinity[0] * result.e[step_index + 1], current)


def test_hj_magnetic_simulation_cg_solver_matches_direct_solver():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-0.5, 0.0, 0.0),
        end=(0.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    model = DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1))
    direct = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.01, 0.01],
        sources=[source],
        linear_solver="direct",
    ).run()
    cg = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.01, 0.01],
        sources=[source],
        linear_solver="cg",
        cg_tolerance=1.0e-12,
    ).run()

    np.testing.assert_allclose(cg.h, direct.h, rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(cg.e, direct.e, rtol=1.0e-9, atol=1.0e-12)


def test_hj_magnetic_simulation_samples_point_receiver_data():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    receivers = [
        PointReceiver(location=(0.0, 0.0, 0.0), component="Ex"),
        PointReceiver(location=(0.0, 0.0, 0.0), component="Hz"),
    ]
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=np.linspace(-0.2, 0.3, mesh.n_edges),
        receivers=receivers,
    )

    result = sim.run()

    expected = [
        receivers[0].sample_hj(mesh, result.e[1], result.h[1]),
        receivers[1].sample_hj(mesh, result.e[1], result.h[1]),
    ]
    assert result.data.shape == (2, 2)
    np.testing.assert_allclose(result.data[1], expected)


def test_hj_magnetic_simulation_can_recover_magnetic_receivers_from_current_biot():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    initial_h = np.linspace(-0.2, 0.3, mesh.n_edges)
    receivers = [
        PointReceiver(location=(0.0, 0.0, 0.0), component="Ex"),
        PointReceiver(location=(0.0, 0.0, 0.0), component="Hz"),
    ]
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=initial_h,
        receivers=receivers,
        magnetic_receiver_mode="current_biot",
        magnetic_recovery_subdivisions=2,
    )

    result = sim.run()

    face_current = mesh.edge_curl @ initial_h
    current_density = mesh.average_face_to_cell_vector @ face_current
    current_density = np.asarray(current_density).reshape((mesh.n_cells, 3), order="F")
    expected_h = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        np.array([receivers[1].location]),
        subdivisions=2,
    )

    np.testing.assert_allclose(result.data[0, 0], receivers[0].sample_hj(mesh, result.e[0], initial_h))
    np.testing.assert_allclose(result.data[0, 1], expected_h[0, 2], atol=1.0e-16)


def test_hj_current_biot_reuses_explicit_face_current_matrix():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    initial_h = np.linspace(-0.2, 0.3, mesh.n_edges)
    locations = np.array([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)])
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=initial_h,
        magnetic_receiver_mode="current_biot",
        magnetic_recovery_subdivisions=2,
    )

    first = sim._current_biot_h(initial_h, locations, time=0.0)
    assert len(sim._current_biot_matrix_cache) == 1
    cached_matrix = next(iter(sim._current_biot_matrix_cache.values()))
    second = sim._current_biot_h(initial_h, locations.copy(), time=0.0)

    face_current = mesh.edge_curl @ initial_h
    expected = np.einsum("kcf,f->kc", cached_matrix, face_current)
    assert next(iter(sim._current_biot_matrix_cache.values())) is cached_matrix
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)


def test_hj_prescribed_source_history_correction_adds_face_source_moment_response():
    mesh = _mesh()
    tau = 0.2
    model = DebyeIPModel(
        sigma_infinity=np.full(mesh.n_cells, 0.2),
        terms=[DebyeTerm(np.full(mesh.n_cells, 0.02), tau=tau)],
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=(0.25, -0.5, 0.125, 0.75),
        receiver_matrix="auto",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0, 2),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    history = discrete_debye_history_basis([0.05], tau=tau, max_order=1)
    coefficients = np.asarray(correction.coefficients).reshape(2, 2)
    expected_h = np.einsum("ps,p,slc->lc", coefficients, history.values[1], static_response)

    expected_delta = receiver.sample_magnetic_field_vector(expected_h[0])
    np.testing.assert_allclose(
        corrected.data[1, 0] - base.data[1, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_prescribed_source_history_normalized_coefficients_match_absolute_scale():
    mesh = _mesh()
    tau = 0.2
    model = DebyeIPModel(
        sigma_infinity=np.full(mesh.n_cells, 0.2),
        terms=[DebyeTerm(np.full(mesh.n_cells, 0.02), tau=tau)],
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    normalized = np.array([0.25, -0.5, 0.125, 0.75])
    scale = mu_0 * 0.02 * 2.0**2
    absolute = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=tuple(normalized * scale),
        receiver_matrix="auto",
    )
    scaled = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        normalized_coefficients=tuple(normalized),
        receiver_matrix="auto",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    absolute_result = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=absolute,
    ).run()
    scaled_result = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=scaled,
    ).run()

    np.testing.assert_allclose(scaled_result.data, absolute_result.data)


def test_hj_prescribed_source_history_correction_vanishes_for_zero_delta_sigma():
    mesh = _mesh()
    tau = 0.2
    model = DebyeIPModel(
        sigma_infinity=np.full(mesh.n_cells, 0.2),
        terms=[DebyeTerm(np.zeros(mesh.n_cells), tau=tau)],
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = SourceHistoryCorrection(
        tau=tau,
        max_order=1,
        source_moment_degrees=(0, 2),
        coefficients=(0.25, -0.5, 0.125, 0.75),
        receiver_matrix="auto",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_hj_source_diffusion_kernel_source_history_is_active_without_ip_terms():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.2)
    model = DebyeIPModel.no_ip(sigma)
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = SourceDiffusionKernelSourceHistoryCorrection(
        amplitude=-0.25,
        tau_multiplier=2.0,
        amplitude_time=0.0,
        source_moment_degrees=(0,),
        receiver_matrix="auto",
    )
    dt = 1.0e-7
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [dt],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0,),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    tau = correction.tau_multiplier * mu_0 * sigma[0] * length**2
    coefficient = correction.amplitude * np.exp(-(dt - correction.amplitude_time) / tau)
    expected_h = coefficient * static_response[0, 0]

    expected_delta = receiver.sample_magnetic_field_vector(expected_h)
    np.testing.assert_allclose(
        corrected.data[1, 0] - base.data[1, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_source_diffusion_kernel_normalized_amplitude_uses_source_diffusion_time():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.2)
    model = DebyeIPModel.no_ip(sigma)
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = SourceDiffusionKernelSourceHistoryCorrection(
        normalized_amplitude=-3.5,
        tau_multiplier=2.0,
        amplitude_time=0.0,
        source_moment_degrees=(0,),
        receiver_matrix="auto",
    )
    dt = 1.0e-7
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [dt],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0,),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    tau0 = mu_0 * sigma[0] * length**2
    tau = correction.tau_multiplier * tau0
    coefficient = (
        correction.normalized_amplitude
        * tau0
        * np.exp(-(dt - correction.amplitude_time) / tau)
    )
    expected_h = coefficient * static_response[0, 0]

    expected_delta = receiver.sample_magnetic_field_vector(expected_h)
    np.testing.assert_allclose(
        corrected.data[1, 0] - base.data[1, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_source_diffusion_kernel_be_decay_uses_time_step_history():
    mesh = _mesh()
    sigma = np.full(mesh.n_cells, 0.2)
    model = DebyeIPModel.no_ip(sigma)
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    dt = 1.0e-7
    correction = SourceDiffusionKernelSourceHistoryCorrection(
        amplitude=-0.25,
        tau_multiplier=2.0,
        amplitude_time=dt,
        basis_kind="be_decay",
        source_moment_degrees=(0,),
        receiver_matrix="auto",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [dt, dt],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0,),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    tau = correction.tau_multiplier * mu_0 * sigma[0] * length**2
    alpha = tau / (tau + dt)
    coefficient = correction.amplitude * alpha
    expected_h = coefficient * static_response[0, 0]

    expected_delta = receiver.sample_magnetic_field_vector(expected_h)
    np.testing.assert_allclose(
        corrected.data[2, 0] - base.data[2, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_initial_polarization_source_history_projects_initial_memory_current():
    mesh = _mesh()
    tau = 0.2
    model = DebyeIPModel(
        sigma_infinity=np.full(mesh.n_cells, 0.2),
        terms=[DebyeTerm(np.full(mesh.n_cells, 0.02), tau=tau)],
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = InitialPolarizationSourceHistoryCorrection(
        source_moment_degrees=(0, 2),
        receiver_matrix="auto",
        projection="receiver_l2",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0, 2),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    initial_memory = hj_dc_initial_electric_field(mesh, model, [source])
    unit_face = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    delta_face = mesh.get_face_inner_product(model.terms[0].delta_sigma).tocsr()
    polarization = delta_face.diagonal() / unit_face.diagonal() * initial_memory
    target_response = np.einsum("lcf,f->lc", receiver_matrix, -polarization)
    design = np.moveaxis(static_response, 0, -1).reshape(-1, 2)
    coefficients, *_ = np.linalg.lstsq(design, target_response.reshape(-1), rcond=None)
    history = discrete_debye_history_basis([0.05], tau=tau, max_order=0)
    expected_h = history.values[1, 0] * np.einsum(
        "s,slc->lc",
        coefficients,
        static_response,
    )

    expected_delta = receiver.sample_magnetic_field_vector(expected_h[0])
    np.testing.assert_allclose(
        corrected.data[1, 0] - base.data[1, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_initial_polarization_source_history_vanishes_without_ip_terms():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.2))
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=InitialPolarizationSourceHistoryCorrection(),
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_charge_conserving_face_current_removes_discrete_divergence():
    mesh = _mesh()
    sigma = np.linspace(0.2, 0.5, mesh.n_cells)
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    face_rho = mesh.get_face_inner_product(1.0 / sigma).tocsr()
    face_rho_inverse = sp.diags(1.0 / face_rho.diagonal(), format="csr")
    potential = np.linspace(-0.2, 0.3, mesh.n_cells)
    solenoidal = mesh.edge_curl @ np.linspace(-0.2, 0.1, mesh.n_edges)
    raw_current = solenoidal + face_rho_inverse @ (divergence.T @ potential)

    current = charge_conserving_face_current(mesh, sigma, raw_current)

    np.testing.assert_allclose(divergence @ current, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(current, solenoidal, atol=1.0e-10)


def test_hj_charge_conserving_initial_polarization_source_history_uses_projected_current():
    mesh = _mesh()
    tau = 0.2
    model = DebyeIPModel(
        sigma_infinity=np.full(mesh.n_cells, 0.2),
        terms=[DebyeTerm(np.full(mesh.n_cells, 0.02), tau=tau)],
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    correction = ChargeConservingInitialPolarizationSourceHistoryCorrection(
        source_moment_degrees=(0, 2),
        receiver_matrix="auto",
        projection="receiver_l2",
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=correction,
    ).run()

    face_source = -source.initial_face_vector(mesh)
    moments = source_face_moment_basis(
        mesh,
        face_source,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=(0, 2),
    )
    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    static_response = np.einsum("lcf,sf->slc", receiver_matrix, moments.basis_vectors)
    initial_memory = hj_dc_initial_electric_field(mesh, model, [source])
    unit_face = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    delta_face = mesh.get_face_inner_product(model.terms[0].delta_sigma).tocsr()
    polarization = delta_face.diagonal() / unit_face.diagonal() * initial_memory
    projected_current = charge_conserving_face_current(
        mesh,
        model.sigma_infinity,
        -polarization,
    )
    target_response = np.einsum("lcf,f->lc", receiver_matrix, projected_current)
    design = np.moveaxis(static_response, 0, -1).reshape(-1, 2)
    coefficients, *_ = np.linalg.lstsq(design, target_response.reshape(-1), rcond=None)
    history = discrete_debye_history_basis([0.05], tau=tau, max_order=0)
    expected_h = history.values[1, 0] * np.einsum(
        "s,slc->lc",
        coefficients,
        static_response,
    )

    expected_delta = receiver.sample_magnetic_field_vector(expected_h[0])
    np.testing.assert_allclose(
        corrected.data[1, 0] - base.data[1, 0],
        expected_delta,
        atol=1.0e-22,
    )


def test_hj_source_primary_delta6_adds_debye_source_like_h_correction():
    from geoana.em.static import LineCurrentWholeSpace

    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
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
        * np.exp(-0.05 / (2.0 * tau))
        * h_wire_z
    )

    np.testing.assert_allclose(corrected.data[0, 0], base.data[0, 0])
    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_hj_source_primary_delta6_can_use_face_source_current_basis():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
        magnetic_recovery_source_primary_delta6_basis="face_current",
    ).run()

    receiver_matrix = face_current_biot_matrix(mesh, np.array([receiver.location]))
    source_h_z = np.einsum(
        "kcf,f->kc",
        receiver_matrix,
        -source.initial_face_vector(mesh),
    )[0, 2]
    length = np.linalg.norm(np.asarray(source.end) - np.asarray(source.start))
    expected_delta = (
        -6.0
        * mu_0
        * delta_sigma[0]
        * length**2
        * np.exp(-0.05 / (2.0 * tau))
        * source_h_z
    )

    np.testing.assert_allclose(corrected.data[0, 0], base.data[0, 0])
    np.testing.assert_allclose(corrected.data[1, 0] - base.data[1, 0], expected_delta)


def test_hj_source_primary_delta6_source_history_matches_face_current_path():
    mesh = _mesh()
    sigma_inf = np.full(mesh.n_cells, 0.2)
    delta_sigma = np.full(mesh.n_cells, 0.02)
    tau = 0.05
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=tau)]),
        "time_steps": [0.025, 0.025],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    direct_delta6 = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
        magnetic_recovery_source_primary_delta6_basis="face_current",
    ).run()
    source_history_delta6 = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_history=SourcePrimaryDelta6SourceHistoryCorrection(
            source_moment_degrees=(0,),
            receiver_matrix="auto",
        ),
    ).run()

    np.testing.assert_allclose(
        source_history_delta6.data,
        direct_delta6.data,
        rtol=1.0e-12,
        atol=1.0e-18,
    )


def test_hj_source_primary_delta6_vanishes_without_debye_terms():
    mesh = _mesh()
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    kwargs = {
        "mesh": mesh,
        "ip_model": DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.2)),
        "time_steps": [0.05],
        "sources": [source],
        "receivers": [receiver],
        "magnetic_receiver_mode": "current_biot",
    }

    base = HJMagneticSimulation(**kwargs).run()
    corrected = HJMagneticSimulation(
        **kwargs,
        magnetic_recovery_source_primary_delta6=True,
    ).run()

    np.testing.assert_allclose(corrected.data, base.data)


def test_hj_magnetic_simulation_can_recover_magnetic_receivers_from_face_basis_cell_biot():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    initial_e = np.zeros(mesh.n_faces)
    initial_e[: mesh.n_faces_x] = 1.0
    receivers = [
        PointReceiver(location=(0.0, 0.0, 0.0), component="Ex"),
        PointReceiver(location=(0.0, 0.0, 0.0), component="Hz"),
    ]
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=np.zeros(mesh.n_edges),
        initial_e=initial_e,
        receivers=receivers,
        magnetic_receiver_mode="face_basis_cell_biot",
        magnetic_recovery_subdivisions=2,
    )

    result = sim.run()

    expected_h = biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        initial_e,
        model.sigma_infinity,
        model.terms,
        [],
        np.array([receivers[1].location]),
        subdivisions=2,
    )

    np.testing.assert_allclose(result.data[0, 0], receivers[0].sample_hj(mesh, initial_e, result.h[0]))
    np.testing.assert_allclose(result.data[0, 1], expected_h[0, 2])


def test_hj_face_basis_cell_biot_uses_polarization_scale():
    mesh = _mesh()
    model = DebyeIPModel(
        sigma_infinity=np.array([4.0]),
        terms=[DebyeTerm(np.array([1.0]), tau=0.1)],
    )
    initial_e = np.zeros(mesh.n_faces)
    initial_e[: mesh.n_faces_x] = 1.0
    receivers = [PointReceiver(location=(0.0, 0.0, 0.0), component="Hz")]
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=np.zeros(mesh.n_edges),
        initial_e=initial_e,
        receivers=receivers,
        magnetic_receiver_mode="face_basis_cell_biot",
        magnetic_recovery_subdivisions=2,
        magnetic_recovery_polarization_scale="low_frequency_ratio",
    )

    result = sim.run()

    expected_h = biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        initial_e,
        model.sigma_infinity,
        model.terms,
        [initial_e],
        np.array([receivers[0].location]),
        subdivisions=2,
        polarization_scale="low_frequency_ratio",
    )

    np.testing.assert_allclose(result.data[0, 0], expected_h[0, 2])


def test_hj_face_basis_cell_biot_can_add_initial_polarization_memory():
    mesh = _mesh()
    model = DebyeIPModel(
        sigma_infinity=np.array([4.0]),
        terms=[DebyeTerm(np.array([1.0]), tau=0.1)],
    )
    initial_e = np.zeros(mesh.n_faces)
    initial_e[: mesh.n_faces_x] = 1.0
    receivers = [PointReceiver(location=(0.0, 0.0, 0.0), component="Hz")]
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=np.zeros(mesh.n_edges),
        initial_e=initial_e,
        receivers=receivers,
        magnetic_receiver_mode="face_basis_cell_biot",
        magnetic_recovery_subdivisions=2,
        magnetic_recovery_initial_polarization_scale=0.5,
    )

    result = sim.run()

    expected_h = biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        initial_e,
        model.sigma_infinity,
        model.terms,
        [initial_e],
        np.array([receivers[0].location]),
        subdivisions=2,
        initial_polarization_scale=0.5,
        initial_memories=[initial_e],
    )

    np.testing.assert_allclose(result.data[0, 0], expected_h[0, 2])


def test_hj_magnetic_simulation_can_recover_magnetic_receivers_from_face_basis_biot():
    mesh = _mesh()
    model = DebyeIPModel.no_ip(np.array([2.0]))
    initial_h = np.linspace(-0.25, 0.35, mesh.n_edges)
    receiver = PointReceiver(location=(0.0, 0.5, 0.0), component="Hz")
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        initial_h=initial_h,
        initial_e=np.zeros(mesh.n_faces),
        receivers=[receiver],
        magnetic_receiver_mode="face_basis_biot",
        magnetic_recovery_subdivisions=2,
    )

    result = sim.run()

    face_current = mesh.edge_curl @ initial_h
    matrix = face_basis_biot_matrix(
        mesh,
        np.array([receiver.location]),
        subdivisions=2,
    )
    expected_h = np.einsum("lcf,f->lc", matrix, face_current)

    np.testing.assert_allclose(result.data[0, 0], expected_h[0, 2])


def test_hj_magnetic_simulation_run_data_only_matches_full_run():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    model = DebyeIPModel(
        sigma_infinity=np.linspace(1.0, 3.0, mesh.n_cells),
        terms=[DebyeTerm(np.linspace(0.1, 0.4, mesh.n_cells), tau=0.2)],
    )
    receivers = [
        PointReceiver(location=(0.0, 0.0, 0.0), component="Ex"),
        PointReceiver(location=(0.0, 0.0, 0.0), component="Hz"),
    ]
    kwargs = {
        "mesh": mesh,
        "ip_model": model,
        "time_steps": [0.05, 0.1],
        "sources": [source],
        "receivers": receivers,
    }

    full = HJMagneticSimulation(**kwargs).run()
    data_only = HJMagneticSimulation(**kwargs).run_data_only()

    np.testing.assert_allclose(data_only.times, full.times)
    np.testing.assert_allclose(data_only.data, full.data)
    assert len(data_only.memories) == len(full.memories)
    np.testing.assert_allclose(data_only.memories[0], full.memories[0][-1])
    assert not hasattr(data_only, "h")
    assert not hasattr(data_only, "e")


def test_hj_magnetic_simulation_runs_layered_debye_grounded_wire_steps():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=1.0),
    )
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05, 0.05],
        sources=[source],
    )

    result = sim.run()

    assert result.h.shape == (3, mesh.n_edges)
    assert result.e.shape == (3, mesh.n_faces)
    assert len(result.memories) == 1
    assert result.memories[0].shape == (3, mesh.n_faces)

    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, mu_0)).tocsr()
    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    alpha, beta = model.terms[0].coefficients(float(sim.time_steps[0]))
    rho_eff = 1.0 / (sigma_inf - beta * delta_sigma)
    face_rho = mesh.get_face_inner_product(rho_eff).tocsr()
    history_mass = mesh.get_face_inner_product(rho_eff * alpha * delta_sigma).tocsr()
    history_scale = history_mass.diagonal() / unit_face_mass.diagonal()
    for step_index, dt in enumerate(sim.time_steps):
        time = result.times[step_index + 1]
        electric_source = -source.face_vector_at(mesh, time)
        current = mesh.edge_curl @ result.h[step_index + 1] - electric_source
        old_memory = result.memories[0][step_index]
        expected_electric = (face_rho @ current) / unit_face_mass.diagonal()
        expected_electric += history_scale * old_memory
        faraday = (edge_mu @ (result.h[step_index + 1] - result.h[step_index])) / dt
        faraday += mesh.edge_curl.T @ (unit_face_mass @ result.e[step_index + 1])

        np.testing.assert_allclose(result.e[step_index + 1], expected_electric)
        np.testing.assert_allclose(
            result.memories[0][step_index + 1],
            alpha * old_memory + beta * result.e[step_index + 1],
        )
        np.testing.assert_allclose(faraday, 0.0, atol=1.0e-12)


def test_hj_dc_initial_current_density_matches_cell_centered_balance():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    sigma = np.linspace(1.0, 2.0, mesh.n_cells)
    model = DebyeIPModel.no_ip(sigma)

    current = hj_dc_initial_current_density(mesh, model, [source])

    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    source_divergence = divergence @ (-source.initial_face_vector(mesh))
    residual = divergence @ current + source_divergence
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-10)


def test_hj_dc_initial_electric_field_is_consistent_with_low_frequency_resistivity():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])

    electric = hj_dc_initial_electric_field(mesh, model, [source])
    current = hj_dc_initial_current_density(mesh, model, [source])

    unit_face_mass = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    low_frequency_rho = 1.0 / model.low_frequency_sigma()
    face_rho = mesh.get_face_inner_product(low_frequency_rho).tocsr()

    np.testing.assert_allclose(unit_face_mass @ electric, face_rho @ current)


def test_hj_magnetic_simulation_initializes_debye_memories_from_dc_electric_field():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    sigma_inf = np.linspace(1.0, 3.0, mesh.n_cells)
    delta_sigma = np.linspace(0.1, 0.4, mesh.n_cells)
    model = DebyeIPModel(sigma_inf, [DebyeTerm(delta_sigma, tau=0.2)])
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        sources=[source],
    )

    result = sim.run()

    expected_e0 = hj_dc_initial_electric_field(mesh, model, [source])
    np.testing.assert_allclose(result.e[0], expected_e0)
    np.testing.assert_allclose(result.memories[0][0], expected_e0)


def test_hj_mmr_initial_magnetic_field_satisfies_stabilized_system():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    sigma = np.linspace(1.0, 2.0, mesh.n_cells)
    model = DebyeIPModel.no_ip(sigma)

    h0 = hj_mmr_initial_magnetic_field(mesh, model, [source])

    current0 = hj_dc_initial_current_density(mesh, model, [source])
    electric_source = -source.initial_face_vector(mesh)
    ampere_residual = mesh.edge_curl @ h0 - electric_source - current0
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    continuity_residual = divergence @ (current0 + electric_source)
    wrong_sign_residual = mesh.edge_curl @ h0 + electric_source - current0

    assert h0.shape == (mesh.n_edges,)
    assert np.isfinite(h0).all()
    assert np.linalg.norm(h0) > 0.0
    np.testing.assert_allclose(continuity_residual, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(ampere_residual, 0.0, atol=1.0e-12)
    assert np.linalg.norm(wrong_sign_residual) > 0.5 * np.linalg.norm(electric_source)


def test_hj_magnetic_simulation_uses_mmr_initial_h_when_not_provided():
    mesh = _mesh()
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    model = DebyeIPModel.no_ip(np.linspace(1.0, 2.0, mesh.n_cells))
    sim = HJMagneticSimulation(
        mesh=mesh,
        ip_model=model,
        time_steps=[0.05],
        sources=[source],
    )

    result = sim.run()

    expected_h0 = hj_mmr_initial_magnetic_field(mesh, model, [source])
    np.testing.assert_allclose(result.h[0], expected_h0)
