import numpy as np
from discretize import TensorMesh

from atem3d.sponge import make_sponge_sigma


def test_sponge_sigma_only_changes_outer_cells_and_is_nonnegative():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    base = np.ones(mesh.n_cells)

    sigma = make_sponge_sigma(mesh, base, thickness_cells=1, strength=9.0, power=2.0)

    assert sigma.shape == (mesh.n_cells,)
    assert np.all(sigma >= base)
    center_index = np.argmin(np.linalg.norm(mesh.cell_centers, axis=1))
    assert sigma[center_index] == base[center_index]
    assert np.max(sigma) > np.max(base)


def test_sponge_sigma_can_be_limited_to_selected_sides():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    base = np.ones(mesh.n_cells)

    sigma = make_sponge_sigma(
        mesh,
        base,
        thickness_cells=1,
        strength=9.0,
        power=2.0,
        active_sides=("x_min",),
    )

    changed = sigma > base
    assert np.any(changed)
    assert np.all(mesh.cell_centers[changed, 0] == mesh.cell_centers[:, 0].min())
    assert np.all(sigma[mesh.cell_centers[:, 0] > mesh.cell_centers[:, 0].min()] == 1.0)
