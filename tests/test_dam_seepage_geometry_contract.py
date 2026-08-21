import numpy as np

from atem3d.examples.dam_seepage_geometry import (
    CHANNEL_P3,
    COVER_TOP_Z,
    ELECTRODE_A,
    ELECTRODE_B,
    UAV_FLIGHT_HEIGHT_M,
    build_dam_seepage_geometry,
    dam_downstream_y,
    dam_upstream_y,
)


def test_html_invariants_are_frozen():
    geometry = build_dam_seepage_geometry()
    info = geometry.validate()

    assert geometry.channel_points[2] == CHANNEL_P3
    assert geometry.electrode_a == ELECTRODE_A
    assert geometry.electrode_b == ELECTRODE_B
    assert COVER_TOP_Z - CHANNEL_P3[2] == 40.0
    assert info["n_uav_stations"] == 61
    assert info["flight_height_m"] == UAV_FLIGHT_HEIGHT_M
    assert info["n_wire_vertices"] == 11
    assert info["wire_length_m"] > 1000.0


def test_uav_stations_follow_crest_planform_at_half_metre():
    stations = build_dam_seepage_geometry().uav_receiver_locations()

    np.testing.assert_allclose(stations[0, :2], [-150.0, 0.0])
    np.testing.assert_allclose(stations[-1, :2], [150.0, 0.0])
    np.testing.assert_allclose(stations[:, 2], 100.5)
    assert np.all(np.isfinite(stations))


def test_dam_faces_match_html_cross_section_formulae():
    np.testing.assert_allclose(dam_upstream_y(100.0), -6.0)
    np.testing.assert_allclose(dam_downstream_y(100.0), 6.0)
    np.testing.assert_allclose(dam_downstream_y(10.0), 276.0)
