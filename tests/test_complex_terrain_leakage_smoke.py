import numpy as np

from atem3d.examples.leakage_channel import build_leakage_channel_example
from atem3d.materials.material_map import (
    CellMaterialMap,
    apply_leakage_channel_marker,
    apply_leakage_channel_marker_with_diagnostics,
    leakage_channel_marker_diagnostics,
    mark_leakage_channel,
)
from atem3d.materials.prony import DebyeTerm, PronyConductivity


def test_cell_material_map_returns_conductivity_arrays_from_markers():
    markers = np.array([1, 1, 2, 3])
    materials = {
        1: PronyConductivity.no_ip(0.01),
        2: PronyConductivity.no_ip(0.1),
        3: PronyConductivity(
            sigma_inf=0.02,
            terms=[DebyeTerm(delta_sigma=0.005, tau=0.2)],
        ),
    }

    material_map = CellMaterialMap(markers=markers, materials=materials)

    np.testing.assert_allclose(material_map.sigma0(), [0.01, 0.01, 0.1, 0.015])
    np.testing.assert_allclose(material_map.sigma_inf(), [0.01, 0.01, 0.1, 0.02])
    assert material_map.markers_present == (1, 2, 3)


def test_mark_leakage_channel_selects_cells_near_polyline():
    centers = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [5.0, 5.0, -1.0],
            [2.0, 0.4, -1.0],
        ]
    )
    channel = np.array([[0.0, 0.0, -1.0], [2.0, 0.0, -1.0]])

    mask = mark_leakage_channel(centers, channel, radius=0.5)

    np.testing.assert_array_equal(mask, np.array([True, True, False, True]))


def test_apply_leakage_channel_marker_overwrites_selected_cells():
    markers = np.array([1, 1, 1, 1])
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    updated = apply_leakage_channel_marker(
        markers,
        centers,
        channel_points=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        radius=0.25,
        leakage_marker=7,
    )

    np.testing.assert_array_equal(updated, np.array([7, 7, 1, 7]))


def test_leakage_channel_marker_diagnostics_catches_unmarked_coarse_box():
    channel = [
        [-700.0, -500.0, -120.0],
        [-250.0, -320.0, -90.0],
        [150.0, -120.0, -140.0],
        [650.0, 80.0, -110.0],
    ]

    default = leakage_channel_marker_diagnostics(
        domain_min=[-2000.0, -2000.0, -1000.0],
        domain_max=[2000.0, 2000.0, 100.0],
        cells=[2, 2, 1],
        channel_points=channel,
        radius=900.0,
    )
    enlarged_bad = leakage_channel_marker_diagnostics(
        domain_min=[-3000.0, -3000.0, -1500.0],
        domain_max=[3000.0, 3000.0, 200.0],
        cells=[2, 2, 1],
        channel_points=channel,
        radius=900.0,
    )
    enlarged_ok = leakage_channel_marker_diagnostics(
        domain_min=[-3000.0, -3000.0, -1500.0],
        domain_max=[3000.0, 3000.0, 200.0],
        cells=[3, 3, 1],
        channel_points=channel,
        radius=900.0,
    )
    enlarged_fallback = leakage_channel_marker_diagnostics(
        domain_min=[-3000.0, -3000.0, -1500.0],
        domain_max=[3000.0, 3000.0, 200.0],
        cells=[2, 2, 1],
        channel_points=channel,
        radius=900.0,
        min_marked_cells=1,
    )

    assert default["leakage_cell_count"] > 0
    assert enlarged_bad["leakage_cell_count"] == 0
    assert enlarged_bad["nearest_channel_distance_m"] > 900.0
    assert enlarged_ok["leakage_cell_count"] > 0
    assert enlarged_ok["cell_count"] == 9
    assert enlarged_fallback["leakage_cell_count"] == 1
    assert enlarged_fallback["fallback_used"] is True


def test_apply_leakage_channel_marker_with_nearest_fallback_marks_minimum_cells():
    markers = np.array([1, 1, 1])
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )

    result = apply_leakage_channel_marker_with_diagnostics(
        markers,
        centers,
        channel_points=np.array([[7.0, 5.0, 0.0], [7.0, 6.0, 0.0]]),
        radius=1.0,
        leakage_marker=7,
        min_marked_cells=1,
    )

    np.testing.assert_array_equal(result.markers, np.array([1, 7, 1]))
    assert result.diagnostics["leakage_cell_count"] == 1
    assert result.diagnostics["fallback_used"] is True
    assert result.diagnostics["fallback_added_cell_count"] == 1
    assert result.diagnostics["nearest_channel_distance_m"] > 1.0


def test_build_leakage_channel_example_has_marked_channel_and_materials():
    example = build_leakage_channel_example(nx=5, ny=4)

    assert example.cell_centers.shape[1] == 3
    assert example.markers.shape[0] == example.cell_centers.shape[0]
    assert example.diagnostics["leakage_cell_count"] > 0
    assert example.diagnostics["terrain_elevation_min"] < example.diagnostics["terrain_elevation_max"]
    np.testing.assert_allclose(
        example.material_map.sigma0()[example.markers == example.leakage_marker],
        example.leakage_material.sigma0,
    )
