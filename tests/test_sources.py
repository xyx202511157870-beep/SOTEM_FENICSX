import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics import time_domain as tdem

from atem3d.sources import GroundedWireSource, StepOffWaveform, _nearest_face_line_source


def test_grounded_wire_source_projects_finite_wire_to_edges_with_current_strength():
    mesh = TensorMesh([np.ones(7), np.ones(5), np.ones(3)], origin=(-3.5, -2.5, -1.5))
    source = GroundedWireSource(
        start=(-2.5, 0.0, 0.0),
        end=(2.5, 0.0, 0.0),
        current=2.0,
        waveform=StepOffWaveform(off_time=0.0),
    )

    unit = source.edge_vector(mesh, current_scale=1.0)
    doubled = source.edge_vector(mesh, current_scale=2.0)

    assert unit.shape == (mesh.n_edges,)
    assert np.count_nonzero(np.abs(unit) > 0.0) > 0
    np.testing.assert_allclose(doubled, 2.0 * unit)


def test_grounded_wire_source_projects_finite_wire_to_hj_faces_like_simpeg():
    mesh = TensorMesh([np.ones(7), np.ones(5), np.ones(3)], origin=(-3.5, -2.5, -1.5))
    source = GroundedWireSource(
        start=(-2.5, 0.0, 0.0),
        end=(2.5, 0.0, 0.0),
        current=2.0,
        waveform=StepOffWaveform(off_time=0.0),
    )

    projected = source.face_vector(mesh, current_scale=1.0)

    simpeg_source = tdem.sources.LineCurrent(
        [],
        location=source.locations,
        current=source.current,
        waveform=tdem.sources.StepOffWaveform(off_time=0.0),
    )
    simulation = type("SimulationStub", (), {"mesh": mesh})()
    expected = simpeg_source.Mfjs(simulation)

    assert projected.shape == (mesh.n_faces,)
    assert np.count_nonzero(np.abs(projected) > 0.0) > 0
    np.testing.assert_allclose(projected, expected)


def test_grounded_wire_source_can_force_axis_aligned_hj_face_projection():
    mesh = TensorMesh(
        [np.ones(4), np.ones(2), np.ones(2)],
        origin=(-2.0, -1.0, -1.0),
    )
    source = GroundedWireSource(
        start=(-1.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
        face_projection="axis_aligned",
    )

    projected = source.unit_face_vector(mesh)
    expected = _nearest_face_line_source(mesh, source.locations)

    np.testing.assert_allclose(projected, expected)
    active_x = mesh.faces_x[np.flatnonzero(np.abs(projected[: mesh.n_faces_x]) > 0.0), 0]
    np.testing.assert_allclose(np.unique(active_x), [-1.0, 0.0, 1.0])


def test_grounded_wire_face_vector_falls_back_when_simpeg_snapping_misaligns_axis_path():
    mesh = TensorMesh(
        [np.ones(3), np.ones(2), [0.5, 0.5]],
        origin=(-1.5, -1.0, -1.0),
    )
    source = GroundedWireSource(
        start=(-0.5, 0.0, -0.5),
        end=(0.5, 0.0, -0.5),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )

    projected = source.face_vector(mesh)
    divergence = mesh.cell_volumes * mesh.face_divergence * projected

    assert projected.shape == (mesh.n_faces,)
    assert np.count_nonzero(np.abs(projected) > 0.0) > 0
    assert np.count_nonzero(np.abs(divergence) > 1.0e-12) <= 2
    np.testing.assert_allclose(divergence.sum(), 0.0, atol=1.0e-12)


def test_grounded_wire_face_fallback_distributes_between_transverse_channels():
    mesh = TensorMesh(
        [np.ones(4), np.ones(2), np.ones(2)],
        origin=(-2.0, -1.0, -1.0),
    )
    locations = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    projected = _nearest_face_line_source(mesh, locations)

    fx = projected[: mesh.n_faces_x]
    active = np.flatnonzero(np.abs(fx) > 0.0)
    points = mesh.faces_x[active]
    values = fx[active]

    np.testing.assert_allclose(np.unique(points[:, 1]), [-0.5, 0.5])
    np.testing.assert_allclose(np.unique(points[:, 2]), [-0.5, 0.5])
    np.testing.assert_allclose(np.unique(values), [-0.25])
    for x_value in [-1.0, 0.0, 1.0]:
        mask = np.isclose(points[:, 0], x_value)
        np.testing.assert_allclose(values[mask].sum(), -1.0)
        np.testing.assert_allclose(np.average(points[mask, 1], weights=-values[mask]), 0.0)
        np.testing.assert_allclose(np.average(points[mask, 2], weights=-values[mask]), 0.0)


def test_grounded_wire_face_vector_follows_step_off_waveform():
    mesh = TensorMesh([np.ones(7), np.ones(5), np.ones(3)], origin=(-3.5, -2.5, -1.5))
    source = GroundedWireSource(
        start=(-2.5, 0.0, 0.0),
        end=(2.5, 0.0, 0.0),
        current=2.0,
        waveform=StepOffWaveform(off_time=0.0, on_value=3.0),
    )

    initial = source.initial_face_vector(mesh)
    at_off = source.face_vector_at(mesh, 0.0)
    after_off = source.face_vector_at(mesh, 0.1)
    previous_at_off = source.previous_face_vector_at(mesh, 0.0)

    np.testing.assert_allclose(initial, 3.0 * source.face_vector(mesh))
    np.testing.assert_allclose(at_off, initial)
    np.testing.assert_allclose(after_off, 0.0)
    np.testing.assert_allclose(previous_at_off, initial)


def test_step_off_waveform_is_on_before_off_time_and_zero_after():
    waveform = StepOffWaveform(off_time=1.5, on_value=3.0)

    assert waveform.value(1.0) == 3.0
    assert waveform.value(1.5) == 3.0
    assert waveform.value(2.0) == 0.0


def test_step_off_waveform_matches_simpeg_endpoint_convention():
    waveform = StepOffWaveform(off_time=0.0, on_value=2.0)

    assert waveform.initial_value() == 2.0
    assert waveform.value(0.0) == 2.0
    assert waveform.previous_value(0.0) == 2.0
    assert waveform.previous_value(1.0) == 0.0
