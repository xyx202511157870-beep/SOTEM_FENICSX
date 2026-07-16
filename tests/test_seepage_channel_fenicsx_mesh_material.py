import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_full_domain_module():
    path = Path("dolfinx/seepage_channel_full_domain.py")
    spec = importlib.util.spec_from_file_location("seepage_full_domain", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline_module():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location("seepage_mesh_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_receiver_configs_cover_both_sides_without_mirroring() -> None:
    module = load_full_domain_module()
    locations = (
        (0, -20, 0.1),
        (0, -10, 0.1),
        (0, 0, 0.1),
        (0, 10, 0.1),
        (0, 20, 0.1),
    )
    configs = module.receiver_configs(locations)
    assert [cfg.receiver for cfg in configs] == list(locations)
    assert all(cfg.provenance == "explicit_full_domain" for cfg in configs)


def test_box_mask_and_volume_audit() -> None:
    module = load_full_domain_module()
    centers = np.array(
        [[0, 0, -20], [29, 4, -16], [0, 6, -20], [0, 0, 1]],
        dtype=float,
    )
    volumes = np.array([1000, 1000, 1000, 1000], dtype=float)
    mask, audit = module.box_mask(
        centers,
        volumes,
        ((-30, 30), (-5, 5), (-25, -15)),
    )
    np.testing.assert_array_equal(mask, [True, True, False, False])
    assert audit["local_cell_count"] == 2
    assert audit["local_discrete_volume_m3"] == 2000.0


def test_pipeline_builds_one_mesh_config_per_explicit_receiver() -> None:
    module = load_pipeline_module()
    locations = ((0, -20, 0.1), (0, 0, 0.1), (0, 20, 0.1))
    configs = module._receiver_mesh_configs(
        module.PipelineConfig(receiver_locations=locations)
    )
    assert tuple(config.receiver for config in configs) == locations


def test_overlapping_receiver_refinement_points_share_one_geometry_tag() -> None:
    module = load_pipeline_module()
    entries, point_sizes = module._receiver_refinement_point_specs(
        module.PipelineConfig(
            receiver_locations=((0, -10, 0.1), (0, 0, 0.1), (0, 10, 0.1)),
            receiver_mesh_size=5.0,
            receiver_refinement_radius=10.0,
        )
    )
    referenced = [
        key
        for entry in entries
        for key in (entry["anchor"], *entry["cloud"], *entry["surface_cloud"])
    ]
    assert len(referenced) > len(point_sizes)
    assert set(referenced) == set(point_sizes)


class _FakeOcc:
    def __init__(self) -> None:
        self.boxes = []
        self.fragments = []

    def addBox(self, *args):
        self.boxes.append(tuple(float(value) for value in args))
        return len(self.boxes)

    def fragment(self, objects, tools):
        self.fragments.append((list(objects), list(tools)))


def test_thin_conductivity_box_uses_embedded_refinement_cloud() -> None:
    module = load_pipeline_module()
    occ = _FakeOcc()
    config = module.PipelineConfig(
        x_extent=3000.0,
        y_extent=3000.0,
        air_height=600.0,
        earth_depth=6000.0,
        conductivity_box_name="thin_channel",
        conductivity_box_bounds=((-30.0, 30.0), (-0.5, 0.5), (-20.5, -19.5)),
        conductivity_box_sigma=1.0,
        conductivity_box_mesh_size=0.25,
    )

    module._add_air_earth_domain(occ, config)
    cloud = module._conductivity_box_refinement_cloud_points(config)

    assert len(occ.boxes) == 2
    objects, tools = occ.fragments[-1]
    assert objects == [(3, 1)]
    assert tools == [(3, 2)]
    assert cloud
    assert all(-30.25 <= point[0] <= 30.25 for point in cloud)
    assert all(-0.75 <= point[1] <= 0.75 for point in cloud)
    assert all(-20.75 <= point[2] <= -19.25 for point in cloud)
    assert any(point[1] == -0.5 for point in cloud)
    assert any(point[1] == 0.5 for point in cloud)
    assert any(point[1] < -0.5 for point in cloud)
    assert any(point[2] > -19.5 for point in cloud)


def test_refinement_cloud_retains_standard_hxt_mesh_pipeline() -> None:
    module = load_pipeline_module()
    plain = module.PipelineConfig()
    imprinted = module.PipelineConfig(
        conductivity_box_name="thin_channel",
        conductivity_box_bounds=((-30.0, 30.0), (-0.5, 0.5), (-20.5, -19.5)),
        conductivity_box_sigma=1.0,
    )

    assert module._gmsh_algorithm_3d(plain) == 10
    assert module._gmsh_algorithm_3d(imprinted) == 10
    assert module._gmsh_optimize_netgen(plain) == 1
    assert module._gmsh_optimize_netgen(imprinted) == 1
