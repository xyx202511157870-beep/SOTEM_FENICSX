import numpy as np
import pytest
import yaml
from discretize import TensorMesh

from atem3d.config import _build_ip_model_properties


def test_conductivity_box_overrides_only_earth_cells() -> None:
    mesh = TensorMesh(
        [np.ones(8) * 5, np.ones(8) * 2.5, np.ones(24) * 2.5],
        x0=(-20, -10, -30),
    )
    model = {
        "coordinate_system": "z_up",
        "layers": [
            {"top": 1e9, "bottom": 0.0, "sigma_infinity": 1e-8},
            {"top": 0.0, "bottom": -1e9, "sigma_infinity": 0.01},
        ],
        "conductivity_boxes": [
            {
                "bounds": [[-15, 15], [-5, 5], [-25, -15]],
                "sigma_infinity": 1.0,
                "name": "seepage_channel",
                "minimum_cells_per_cross_section": 4,
            }
        ],
    }
    sigma, terms = _build_ip_model_properties(mesh, model)
    centers = mesh.cell_centers
    mask = np.logical_and.reduce(
        [
            (centers[:, 0] >= -15) & (centers[:, 0] <= 15),
            (centers[:, 1] >= -5) & (centers[:, 1] <= 5),
            (centers[:, 2] >= -25) & (centers[:, 2] <= -15),
        ]
    )
    np.testing.assert_allclose(sigma[mask], 1.0)
    np.testing.assert_allclose(sigma[(centers[:, 2] < 0) & ~mask], 0.01)
    assert terms == []


def test_box_resolution_fails_closed() -> None:
    mesh = TensorMesh(
        [np.ones(4) * 20, np.ones(4) * 5, np.ones(4) * 5],
        x0=(-40, -10, -30),
    )
    with pytest.raises(ValueError, match="at least 4 cells"):
        _build_ip_model_properties(
            mesh,
            {
                "coordinate_system": "z_up",
                "sigma_infinity": 0.01,
                "conductivity_boxes": [
                    {
                        "bounds": [[-30, 30], [-5, 5], [-25, -15]],
                        "sigma_infinity": 1.0,
                        "minimum_cells_per_cross_section": 4,
                    }
                ],
            },
        )


def test_thin_channel_configs_align_four_cells_across_one_metre() -> None:
    paths = (
        "examples/seepage_channel_100m_5rx_simpeg_thin_background.yaml",
        "examples/seepage_channel_100m_5rx_simpeg_thin_channel.yaml",
    )
    configs = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            configs.append(yaml.safe_load(stream))
    for config in configs:
        mesh = TensorMesh(
            [
                [tuple(item) for item in config["mesh"][axis]]
                for axis in ("hx", "hy", "hz")
            ],
            x0=config["mesh"]["origin"],
        )
        for nodes, bounds in (
            (mesh.nodes_y, (-0.5, 0.5)),
            (mesh.nodes_z, (-20.5, -19.5)),
        ):
            selected = nodes[(nodes >= bounds[0] - 1e-10) & (nodes <= bounds[1] + 1e-10)]
            np.testing.assert_allclose(selected, np.linspace(*bounds, 5), atol=1e-10)
        assert mesh.n_cells == 403_200
    box = configs[1]["model"]["conductivity_boxes"][0]
    assert box["bounds"] == [[-30.0, 30.0], [-0.5, 0.5], [-20.5, -19.5]]
    assert box["minimum_cells_per_cross_section"] == 4
