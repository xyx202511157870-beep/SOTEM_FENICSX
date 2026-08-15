import numpy as np
import pytest

from atem3d.receivers import AverageReceiver, PointReceiver, build_receiver


class LinearFakeMesh:
    def get_interpolation_matrix(self, locations, component):
        locations = np.asarray(locations, dtype=float)
        rows = []
        for x, y, z in locations:
            rows.append([1.0, x, y, z])
        return np.asarray(rows, dtype=float)


def test_disk_average_receiver_integrates_linear_field_in_component_normal_plane():
    receiver = AverageReceiver(
        location=(10.0, -2.0, 3.0),
        component="Ex",
        receiver_type="disk_average",
        radius=2.0,
    )
    e = np.array([5.0, 1.0, -2.0, 4.0])

    value = receiver.sample(LinearFakeMesh(), e, np.zeros(4))

    assert value == pytest.approx(5.0 + 10.0 + 4.0 + 12.0)
    assert receiver.sample_count == 36
    np.testing.assert_allclose(receiver.sample_weights.sum(), 1.0)
    offsets = receiver.sample_points - np.asarray(receiver.location)
    np.testing.assert_allclose(offsets[:, 0], 0.0, atol=1.0e-14)


def test_disk_average_receiver_uses_component_specific_coil_planes():
    cases = {
        "Bx": np.array([1.0, 0.0, 0.0]),
        "By": np.array([0.0, 1.0, 0.0]),
        "Bz": np.array([0.0, 0.0, 1.0]),
    }

    for component, normal in cases.items():
        receiver = AverageReceiver(
            location=(1.0, 2.0, 3.0),
            component=component,
            receiver_type="disk_average",
            radius=0.5,
        )
        offsets = receiver.sample_points - np.asarray(receiver.location)
        np.testing.assert_allclose(offsets @ normal, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(receiver.normal, normal)


def test_disk_average_receiver_accepts_rotated_coil_normal():
    normal = np.array([1.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    receiver = AverageReceiver(
        location=(0.0, 0.0, 0.0),
        component="Bz",
        receiver_type="disk_average",
        radius=1.0,
        normal=tuple(normal),
    )

    np.testing.assert_allclose(receiver.sample_points @ normal, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(receiver.sample_weights.sum(), 1.0)


def test_volume_average_receiver_averages_center_and_three_axis_offsets():
    receiver = AverageReceiver(
        location=(1.0, 2.0, 3.0),
        component="Ex",
        receiver_type="volume_average",
        radius=0.5,
    )
    e = np.array([2.0, 3.0, 5.0, 7.0])

    value = receiver.sample(LinearFakeMesh(), e, np.zeros(4))

    assert value == pytest.approx(2.0 + 3.0 + 10.0 + 21.0)
    assert receiver.sample_count == 7
    np.testing.assert_allclose(receiver.sample_weights.sum(), 1.0)


def test_average_receiver_rejects_invalid_receiver_type():
    with pytest.raises(ValueError, match="receiver_type"):
        AverageReceiver(
            location=(0.0, 0.0, 0.0),
            component="Ex",
            receiver_type="line_average",
            radius=1.0,
        )


def test_average_receiver_rejects_zero_normal():
    with pytest.raises(ValueError, match="normal"):
        AverageReceiver(
            location=(0.0, 0.0, 0.0),
            component="Bz",
            receiver_type="disk_average",
            radius=1.0,
            normal=(0.0, 0.0, 0.0),
        )


def test_build_receiver_factory_creates_point_and_average_receivers():
    point = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="point",
    )
    disk = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="disk_average",
        radius=2.0,
        normal=(0.0, 0.0, 1.0),
    )

    assert isinstance(point, PointReceiver)
    assert isinstance(disk, AverageReceiver)
    assert disk.radius == 2.0
    np.testing.assert_allclose(disk.normal, (0.0, 0.0, 1.0))


def test_average_receiver_samples_recovered_magnetic_field_vectors():
    receiver = AverageReceiver(
        location=(0.0, 0.0, 0.0),
        component="Bz",
        receiver_type="disk_average",
        radius=1.0,
    )
    h_vectors = np.tile(
        np.array([[0.0, 0.0, 3.0]]),
        (receiver.sample_count, 1),
    )

    assert receiver.uses_magnetic_field_vector is True
    assert receiver.sample_magnetic_field_vector(
        h_vectors,
        mu=2.0,
    ) == pytest.approx(6.0)
