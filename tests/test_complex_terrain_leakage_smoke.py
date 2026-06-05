import numpy as np

from atem3d.examples.leakage_channel import build_leakage_channel_example
from atem3d.materials.material_map import (
    CellMaterialMap,
    apply_leakage_channel_marker,
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
