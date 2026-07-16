import numpy as np
import pytest

from atem3d.seepage_channel_model import ChannelBox, MODEL, model_for_variant


def test_approved_benchmark_contract() -> None:
    assert MODEL.coordinate_convention == "z_down"
    assert MODEL.source_endpoints == ((-50.0, 0.0, 0.1), (50.0, 0.0, 0.1))
    assert MODEL.receiver_locations == tuple(
        (0.0, y, -0.1) for y in (-20.0, -10.0, 0.0, 10.0, 20.0)
    )
    assert MODEL.components == ("Ex", "dBzdt", "Hz")
    np.testing.assert_allclose(MODEL.times, np.logspace(-5, -2, 31))

    channel = MODEL.channel
    assert channel.center == (0.0, 0.0, 20.0)
    assert channel.size == (60.0, 10.0, 10.0)
    assert channel.bounds == ((-30.0, 30.0), (-5.0, 5.0), (15.0, 25.0))
    assert channel.conductivity == pytest.approx(1.0)
    assert channel.volume_m3 == pytest.approx(6000.0)


def test_thin_variant_has_the_approved_60x1x1_contract() -> None:
    model = model_for_variant("thin_60x1x1")

    assert model.channel.center == (0.0, 0.0, 20.0)
    assert model.channel.size == (60.0, 1.0, 1.0)
    assert model.channel.bounds == ((-30.0, 30.0), (-0.5, 0.5), (19.5, 20.5))
    assert model.channel.to_z_up_bounds() == (
        (-30.0, 30.0),
        (-0.5, 0.5),
        (-20.5, -19.5),
    )
    assert model.channel.volume_m3 == pytest.approx(60.0)


def test_channel_mask_and_z_up_conversion() -> None:
    points = np.array(
        [
            [0.0, 0.0, 20.0],
            [30.0, 5.0, 25.0],
            [30.01, 0.0, 20.0],
            [0.0, 5.01, 20.0],
            [0.0, 0.0, 14.99],
        ]
    )
    np.testing.assert_array_equal(
        MODEL.channel.mask(points),
        np.array([True, True, False, False, False]),
    )
    assert MODEL.channel.to_z_up_bounds() == (
        (-30.0, 30.0),
        (-5.0, 5.0),
        (-25.0, -15.0),
    )


def test_channel_requires_underground_x_parallel_geometry() -> None:
    with pytest.raises(ValueError, match="underground"):
        ChannelBox(center=(0.0, 0.0, 4.0), size=(60.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="parallel"):
        ChannelBox(center=(0.0, 0.0, 20.0), size=(10.0, 60.0, 10.0))
