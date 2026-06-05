import numpy as np
from discretize import TensorMesh

from atem3d.boundary import BoundaryConfig, apply_boundary


def test_none_boundary_leaves_conductivity_and_ip_unchanged():
    mesh = TensorMesh([np.ones(3), np.ones(3), np.ones(3)], origin="CCC")
    sigma = np.ones(mesh.n_cells)
    delta = np.full(mesh.n_cells, 0.2)

    out_sigma, out_terms = apply_boundary(
        mesh,
        sigma,
        [(0.1, delta)],
        BoundaryConfig(kind="none"),
    )

    np.testing.assert_allclose(out_sigma, sigma)
    assert out_terms[0][0] == 0.1
    np.testing.assert_allclose(out_terms[0][1], delta)


def test_sponge_boundary_increases_outer_sigma_and_can_disable_ip_in_shell():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    sigma = np.ones(mesh.n_cells)
    delta = np.full(mesh.n_cells, 0.2)

    out_sigma, out_terms = apply_boundary(
        mesh,
        sigma,
        [(0.1, delta)],
        BoundaryConfig(kind="sponge", thickness_cells=1, strength=4.0, power=2.0),
    )

    center = np.argmin(np.linalg.norm(mesh.cell_centers, axis=1))
    shell = out_sigma > sigma
    assert out_sigma[center] == sigma[center]
    assert out_sigma.max() > sigma.max()
    assert out_terms[0][1][center] == delta[center]
    assert np.all(out_terms[0][1][shell] == 0.0)


def test_sponge_boundary_can_be_limited_to_selected_sides():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    sigma = np.ones(mesh.n_cells)
    delta = np.full(mesh.n_cells, 0.2)

    out_sigma, out_terms = apply_boundary(
        mesh,
        sigma,
        [(0.1, delta)],
        BoundaryConfig(
            kind="sponge",
            thickness_cells=1,
            strength=4.0,
            sides=("z_min",),
        ),
    )

    changed = out_sigma > sigma
    assert np.any(changed)
    assert np.all(mesh.cell_centers[changed, 2] == mesh.cell_centers[:, 2].min())
    assert np.all(out_terms[0][1][changed] == 0.0)
    assert np.all(out_terms[0][1][~changed] == delta[~changed])


def test_boundary_rejects_unknown_kind():
    mesh = TensorMesh([np.ones(1), np.ones(1), np.ones(1)])

    try:
        apply_boundary(mesh, np.ones(mesh.n_cells), [], BoundaryConfig(kind="abc"))
    except ValueError as err:
        assert "unsupported boundary kind" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_apply_boundary_rejects_cpml_as_property_only_boundary():
    mesh = TensorMesh([np.ones(1), np.ones(1), np.ones(1)])

    try:
        apply_boundary(mesh, np.ones(mesh.n_cells), [], BoundaryConfig(kind="cpml"))
    except ValueError as err:
        assert "solver-level boundary" in str(err)
    else:
        raise AssertionError("expected ValueError")
