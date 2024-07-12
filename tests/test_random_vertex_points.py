"""Tests for random_vertex_points."""

import numpy as np
import pytest
import pyvista as pv

import pyransame


@pytest.fixture
def vertex():
    return pv.PolyData(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], verts=[1, 0, 1, 1, 1, 2]
    )


@pytest.fixture
def nonuniform_vertex():
    points = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0]])
    return pv.PolyData(points, verts=[1, 0, 1, 1, 1, 2])


@pytest.fixture
def polyvertex():
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    return pv.PolyData(points, verts=[2, 0, 1, 3, 2, 3, 4])


def test_cell_types(vertex, polyvertex):
    assert vertex.get_cell(0).type == pv.CellType.VERTEX
    pyransame.random_vertex_points(vertex, 20)

    assert polyvertex.get_cell(0).type == pv.CellType.POLY_VERTEX
    pyransame.random_vertex_points(polyvertex, 20)


def test_mixed_types():
    # as long as there are 1D cells, we should be able to sample even if there are other cell types
    # adds a 1D VERTEX to an unstructured grid with VOXEL cells
    uniform_mesh = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    points = uniform_mesh.points.copy()
    cells = uniform_mesh.cells.copy()
    cell_types = uniform_mesh.celltypes.copy()
    cells = np.append(cells, [1, 0])
    cell_types = np.append(cell_types, pv.CellType.VERTEX)
    mesh = pv.UnstructuredGrid(cells, cell_types, uniform_mesh.points)
    pyransame.random_vertex_points(mesh, 20)


def test_unsupported_types():
    mesh = pv.ImageData(dimensions=(4, 4, 4))
    assert all([c.type == pv.CellType.VOXEL for c in mesh.cell])
    with pytest.raises(ValueError, match="No cells with vertices in DataSet"):
        pyransame.random_vertex_points(mesh, 20)


def test_negative_number(vertex):
    with pytest.raises(ValueError, match="n must be > 0"):
        pyransame.random_vertex_points(vertex, -20)


def test_small_sample(vertex):
    points = pyransame.random_vertex_points(vertex, 2)
    assert points.shape == (2, 3)
    assert isinstance(points, np.ndarray)


def test_nonuniform_cell_size(nonuniform_vertex):
    points = pyransame.random_vertex_points(nonuniform_vertex, 200000)
    assert np.allclose(points.mean(axis=0), (-1 / 3, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_nonuniform_cell_size_w_precomputed_areas(nonuniform_vertex):
    mesh = nonuniform_vertex.compute_cell_sizes(
        length=False, volume=False, area=False, vertex_count=True
    )

    points = pyransame.random_vertex_points(mesh, 200000)
    assert np.allclose(points.mean(axis=0), (-1 / 3, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_weights(nonuniform_vertex):
    nonuniform_vertex.cell_data["weights"] = [1, 1, 2]

    points = pyransame.random_vertex_points(nonuniform_vertex, 200000, "weights")
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)

    nonuniform_vertex.cell_data.clear()
    nonuniform_vertex.cell_data["other_str"] = [1, 1, 2]

    points = pyransame.random_vertex_points(nonuniform_vertex, 200, "other_str")


def test_weights_array(nonuniform_vertex):
    points = pyransame.random_vertex_points(nonuniform_vertex, 200000, [1, 1, 2])
    assert np.allclose(points.mean(axis=0), (0.0, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_wrong_weights(vertex):
    weights = {"not a good entry": "should raise an error"}

    with pytest.raises(TypeError):
        pyransame.random_vertex_points(vertex, 20, weights=weights)


def test_wrong_n(vertex):
    with pytest.raises(ValueError, match="n must be > 0, got -20"):
        pyransame.random_vertex_points(vertex, -20)
