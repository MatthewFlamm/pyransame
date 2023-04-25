"""Tests for random_surface_points."""
import numpy as np
import pyvista as pv

import pyransame


def test_square_plane():
    mesh = pv.Plane()
    points = pyransame.random_surface_points(mesh, 200000)
    assert points.shape == (200000, 3)
    assert isinstance(points, np.ndarray)
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_small_sample():
    mesh = pv.Plane()
    points = pyransame.random_surface_points(mesh, 2)
    assert points.shape == (2, 3)
    assert isinstance(points, np.ndarray)


def test_nonuniform_cell_size():
    mesh = pv.Plane(i_resolution=1, j_resolution=3)

    # make mesh nonuniform, but still centered at (0,0,0)
    # points not numbered the same as inside each cell
    mesh.points = np.array(
        [
            [1, -1, 0.0],
            [-1, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0.5, 0],
            [-1, 0.5, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ]
    )

    points = pyransame.random_surface_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_nonuniform_cell_size_w_precomputed_areas():
    mesh = pv.Plane(i_resolution=1, j_resolution=3)

    # make mesh nonuniform, but still centered at (0,0,0)
    # points not numbered the same as inside each cell
    mesh.points = np.array(
        [
            [1, -1, 0.0],
            [-1, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0.5, 0],
            [-1, 0.5, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ]
    )
    mesh = mesh.compute_cell_sizes(length=False, volume=False)

    points = pyransame.random_surface_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_weights():
    mesh = pv.Plane(i_resolution=1, j_resolution=2)

    # make mesh nonuniform, with 2nd cell half size
    # points not numbered the same as inside each cell
    mesh.points = np.array(
        [
            [1, -1, 0.0],
            [-1, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0.5, 0],
            [-1, 0.5, 0],
        ]
    )

    # 1/2 the number of points will have y>0 if no weighting is done
    # The expected y_mean will be -0.5 for points in cell 0 and 0.25 for piotns in cell 1
    # Therefore a weighting of 1:4 must be applied to have center be at y=0.0

    mesh.cell_data["weights"] = [1, 4]

    points = pyransame.random_surface_points(mesh, 200000, "weights")
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)

    mesh.cell_data.clear()
    mesh.cell_data["other_str"] = [1, 4]

    points = pyransame.random_surface_points(mesh, 200, "other_str")


def test_weights_array():
    mesh = pv.Plane(i_resolution=1, j_resolution=2)

    # make mesh nonuniform, with 2nd cell half size
    # points not numbered the same as inside each cell
    mesh.points = np.array(
        [
            [1, -1, 0.0],
            [-1, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0.5, 0],
            [-1, 0.5, 0],
        ]
    )

    # 1/2 the number of points will have y>0 if no weighting is done
    # The expected y_mean will be -0.5 for points in cell 0 and 0.25 for piotns in cell 1
    # Therefore a weighting of 1:4 must be applied to have center be at y=0.0
    points = pyransame.random_surface_points(mesh, 200000, [1, 4])
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)
