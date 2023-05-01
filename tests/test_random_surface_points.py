"""Tests for random_surface_points."""
import numpy as np
import pytest
import pyvista as pv

import pyransame


def test_cell_types():
    mesh = pv.Plane()
    assert mesh.get_cell(0).type == pv.CellType.QUAD
    pyransame.random_surface_points(mesh, 20)

    mesh = pv.Plane().triangulate()
    assert mesh.get_cell(0).type == pv.CellType.TRIANGLE
    pyransame.random_surface_points(mesh, 20)

    # pyvista Polygon generates lines and 2D cell
    mesh = pv.Polygon()
    mesh = pv.PolyData(mesh.points, faces=[6, 0, 1, 2, 3, 4, 5])
    assert mesh.get_cell(0).type == pv.CellType.POLYGON
    pyransame.random_surface_points(mesh, 20)

    mesh = pv.UniformGrid(dimensions=(4, 4, 1))
    assert mesh.get_cell(0).type == pv.CellType.PIXEL
    pyransame.random_surface_points(mesh, 20)

    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    strips = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
    mesh = pv.PolyData(points, strips=strips)
    assert mesh.get_cell(0).type == pv.CellType.TRIANGLE_STRIP
    pyransame.random_surface_points(mesh, 20)


def test_mixed_types():
    # as long as there are 2D cells, we should be able to sample even if there are other cell types
    # adds a 2D Quad to a Voxel only mesh
    uniform_mesh = pv.UniformGrid(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    points = uniform_mesh.points.copy()
    cells = uniform_mesh.cells.copy()
    cell_types = uniform_mesh.celltypes.copy()
    cells = np.append(cells, [4, 0, 1, 5, 4])
    cell_types = np.append(cell_types, pv.CellType.QUAD)
    mesh = pv.UnstructuredGrid(cells, cell_types, uniform_mesh.points)
    pyransame.random_surface_points(mesh, 20)


def test_unsupported_types():
    mesh = pv.UniformGrid(dimensions=(4, 4, 4))
    assert all([c.type == pv.CellType.VOXEL for c in mesh.cell])
    with pytest.raises(ValueError, match="No cells with area in DataSet"):
        pyransame.random_surface_points(mesh, 20)


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


def test_wrong_weights():
    mesh = pv.Plane()
    weights = {"not a good entry": "should raise an error"}

    with pytest.raises(ValueError, match="Invalid weights, got {'not"):
        pyransame.random_surface_points(mesh, 20, weights=weights)


def test_wrong_n():
    mesh = pv.Plane()

    with pytest.raises(ValueError, match="n must be > 0, got -20"):
        pyransame.random_surface_points(mesh, -20)
