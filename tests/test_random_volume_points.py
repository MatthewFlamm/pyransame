import numpy as np
import pyvista as pv

import pyransame


def test_square_plane_voxel():
    mesh = pv.UniformGrid(dimensions=(11, 11, 11))
    points = pyransame.random_volume_points(mesh, 200000)
    assert points.shape == (200000, 3)
    assert isinstance(points, np.ndarray)
    assert np.allclose(points.mean(axis=0), (5.0, 5.0, 5.0), rtol=5e-3, atol=5e-3)


def test_small_sample():
    mesh = pv.UniformGrid(dimensions=(11, 11, 11))
    points = pyransame.random_volume_points(mesh, 2)
    assert points.shape == (2, 3)
    assert isinstance(points, np.ndarray)


def test_nonuniformgrid_voxel():
    x = np.array([-1.0, 0.0, 2.0])
    y = np.array([-1.0, 1.0])
    z = np.array([-1.0, 1.0])
    mesh = pv.RectilinearGrid(x, y, z)
    points = pyransame.random_volume_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (0.5, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_weights_voxel():
    x = np.array([-1.0, 0.0, 2.0])
    y = np.array([-1.0, 1.0])
    z = np.array([-1.0, 1.0])
    mesh = pv.RectilinearGrid(x, y, z)
    weights = [2.0, 1.0]
    points = pyransame.random_volume_points(mesh, 200000, weights=weights)
    assert np.allclose(points.mean(axis=0), (0.25, 0.0, 0.0), rtol=5e-3, atol=5e-3)

    mesh.cell_data.clear()
    mesh.cell_data["other_str"] = [2.0, 1.0]
    points = pyransame.random_volume_points(mesh, 2000, "other_str")


def test_square_plane_tetra():
    mesh = pv.UniformGrid(dimensions=(11, 11, 11)).to_tetrahedra()
    points = pyransame.random_volume_points(mesh, 200000)
    assert points.shape == (200000, 3)
    assert isinstance(points, np.ndarray)
    assert np.allclose(points.mean(axis=0), (5.0, 5.0, 5.0), rtol=5e-3, atol=5e-3)


def test_nonuniformgrid_tetra():
    x = np.array([-1.0, 0.0, 2.0])
    y = np.array([-1.0, 1.0])
    z = np.array([-1.0, 1.0])
    mesh = pv.RectilinearGrid(x, y, z).to_tetrahedra()
    points = pyransame.random_volume_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (0.5, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_weights_tetra():
    x = np.array([-1.0, 0.0, 2.0])
    y = np.array([-1.0, 1.0])
    z = np.array([-1.0, 1.0])
    mesh = pv.RectilinearGrid(x, y, z)
    mesh["weights"] = [2.0, 1.0]
    tetra_mesh = mesh.to_tetrahedra(pass_cell_ids=True)
    tetra_mesh["weights"] = mesh["weights"][tetra_mesh.cell_data.active_scalars]
    points = pyransame.random_volume_points(tetra_mesh, 200000, weights="weights")
    assert np.allclose(points.mean(axis=0), (0.25, 0.0, 0.0), rtol=5e-3, atol=5e-3)
