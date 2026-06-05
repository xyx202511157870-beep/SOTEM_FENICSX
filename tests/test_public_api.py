from atem3d import AverageReceiver, PointReceiver, build_receiver


def test_receiver_builders_are_available_from_public_api():
    point = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="point",
    )
    average = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="disk_average",
        radius=1.0,
    )

    assert isinstance(point, PointReceiver)
    assert isinstance(average, AverageReceiver)
