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


def test_disk_average_receiver_averages_center_and_horizontal_offsets():
    receiver = AverageReceiver(
        location=(10.0, -2.0, 3.0),
        component="Ex",
        receiver_type="disk_average",
        radius=2.0,
    )
    e = np.array([5.0, 1.0, -2.0, 4.0])

    value = receiver.sample(LinearFakeMesh(), e, np.zeros(4))

    assert value == pytest.approx(5.0 + 10.0 + 4.0 + 12.0)
    assert receiver.sample_count == 5


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


def test_average_receiver_rejects_invalid_receiver_type():
    with pytest.raises(ValueError, match="receiver_type"):
        AverageReceiver(
            location=(0.0, 0.0, 0.0),
            component="Ex",
            receiver_type="line_average",
            radius=1.0,
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
    )

    assert isinstance(point, PointReceiver)
    assert isinstance(disk, AverageReceiver)
    assert disk.radius == 2.0


def test_average_receiver_samples_recovered_magnetic_field_vectors():
    receiver = AverageReceiver(
        location=(0.0, 0.0, 0.0),
        component="Bz",
        receiver_type="disk_average",
        radius=1.0,
    )
    h_vectors = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 5.0],
        ]
    )

    assert receiver.uses_magnetic_field_vector is True
    assert receiver.sample_magnetic_field_vector(h_vectors, mu=2.0) == pytest.approx(6.0)
