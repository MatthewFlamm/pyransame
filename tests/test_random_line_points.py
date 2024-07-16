"""Tests for random_surface_points."""

import numpy as np
import pytest
import pyvista as pv

import pyransame


@pytest.fixture
def line():
    return pv.PolyData(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], lines=[2, 0, 1, 2, 1, 2]
    )


@pytest.fixture
def nonuniform_line():
    points = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0]])
    return pv.PolyData(points, lines=[2, 0, 1, 2, 1, 2])


@pytest.fixture
def polyline():
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    return pv.PolyData(points, lines=[3, 0, 1, 2, 3, 2, 3, 4])


def test_cell_types(line, polyline):
    assert line.get_cell(0).type == pv.CellType.LINE
    pyransame.random_line_points(line, 20)

    assert polyline.get_cell(0).type == pv.CellType.POLY_LINE
    pyransame.random_line_points(polyline, 20)


def test_mixed_types():
    # as long as there are 1D cells, we should be able to sample even if there are other cell types
    # adds a 1D LINE to an unstructured grid with VOXEL cells
    uniform_mesh = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    points = uniform_mesh.points.copy()
    cells = uniform_mesh.cells.copy()
    cell_types = uniform_mesh.celltypes.copy()
    cells = np.append(cells, [2, 0, 1])
    cell_types = np.append(cell_types, pv.CellType.LINE)
    mesh = pv.UnstructuredGrid(cells, cell_types, uniform_mesh.points)
    pyransame.random_line_points(mesh, 20)


def test_unsupported_types():
    mesh = pv.ImageData(dimensions=(4, 4, 4))
    assert all([c.type == pv.CellType.VOXEL for c in mesh.cell])
    with pytest.raises(ValueError, match="No cells with length in DataSet"):
        pyransame.random_line_points(mesh, 20)


def test_negative_number(line):
    with pytest.raises(ValueError, match="n must be > 0"):
        pyransame.random_line_points(line, -20)


def test_straight_line(line):
    points = pyransame.random_line_points(line, 200000)
    assert points.shape == (200000, 3)
    assert isinstance(points, np.ndarray)
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_small_sample(line):
    points = pyransame.random_line_points(line, 2)
    assert points.shape == (2, 3)
    assert isinstance(points, np.ndarray)


def test_nonuniform_cell_size(nonuniform_line):
    points = pyransame.random_line_points(nonuniform_line, 200000)
    assert np.allclose(points.mean(axis=0), (-0.5, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_nonuniform_cell_size_w_precomputed_areas(nonuniform_line):
    mesh = nonuniform_line.compute_cell_sizes(length=True, area=False, volume=False)

    points = pyransame.random_line_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (-0.5, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_weights(nonuniform_line):
    # 1/2 the number of points will have y>0 if no weighting is done
    # The expected y_mean will be -1 for points in cell 0 and 0.5 for piotns in cell 1
    # Therefore a weighting of 1:4 must be applied to have center be at y=0.0

    nonuniform_line.cell_data["weights"] = [1, 4]

    points = pyransame.random_line_points(nonuniform_line, 200000, "weights")
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)

    nonuniform_line.cell_data.clear()
    nonuniform_line.cell_data["other_str"] = [1, 4]

    points = pyransame.random_line_points(nonuniform_line, 200, "other_str")


def test_weights_array(nonuniform_line):
    points = pyransame.random_line_points(nonuniform_line, 200000, [1, 4])
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_wrong_weights(line):
    weights = {"not a good entry": "should raise an error"}

    with pytest.raises(TypeError):
        pyransame.random_line_points(line, 20, weights=weights)


def test_wrong_n(line):
    with pytest.raises(ValueError, match="n must be > 0, got -20"):
        pyransame.random_line_points(line, -20)
