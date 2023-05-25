import numpy as np
import pytest
import pyvista as pv

import pyransame


def make_wedge():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, np.sqrt(3.0) / 2.0, 1.0],
        ]
    )
    celltypes = [pv.CellType.WEDGE]
    cells = [6, 0, 1, 2, 3, 4, 5]

    return pv.UnstructuredGrid(cells, celltypes, points)


def test_cell_types():
    mesh = pv.UniformGrid(dimensions=(4, 4, 4))
    assert mesh.get_cell(0).type == pv.CellType.VOXEL
    pyransame.random_volume_points(mesh, 20)

    mesh = pv.UniformGrid(dimensions=(4, 4, 4)).to_tetrahedra()
    assert mesh.get_cell(0).type == pv.CellType.TETRA
    pyransame.random_volume_points(mesh, 20)

    mesh = pv.Pyramid()
    assert mesh.get_cell(0).type == pv.CellType.PYRAMID
    pyransame.random_volume_points(mesh, 20)

    mesh = make_wedge()
    assert mesh.get_cell(0).type == pv.CellType.WEDGE
    pyransame.random_volume_points(mesh, 20)


def test_mixed_types():
    # as long as there are 3D cells, we should be able to sample even if there are other cell types
    # adds a 2D Quad to a Voxel only mesh
    uniform_mesh = pv.UniformGrid(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    points = uniform_mesh.points.copy()
    cells = uniform_mesh.cells.copy()
    cell_types = uniform_mesh.celltypes.copy()
    cells = np.append(cells, [4, 0, 1, 5, 4])
    cell_types = np.append(cell_types, pv.CellType.QUAD)
    mesh = pv.UnstructuredGrid(cells, cell_types, uniform_mesh.points)
    pyransame.random_volume_points(mesh, 20)


def test_unsupported_types():
    mesh = pv.UniformGrid(dimensions=(4, 4, 1))
    assert all([c.type == pv.CellType.PIXEL for c in mesh.cell])
    with pytest.raises(ValueError, match="No cells with volume in DataSet"):
        pyransame.random_volume_points(mesh, 20)


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
    if "weights" not in tetra_mesh.cell_data:
        # For backwards compatability pyvista < 0.39.0
        tetra_mesh["weights"] = mesh["weights"][tetra_mesh.cell_data.active_scalars]
    points = pyransame.random_volume_points(tetra_mesh, 200000, weights="weights")
    assert np.allclose(points.mean(axis=0), (0.25, 0.0, 0.0), rtol=5e-3, atol=5e-3)


def test_wrong_weights():
    mesh = pv.UniformGrid(dimensions=(4, 4, 4))
    weights = {"not a good entry": "should raise an error"}

    with pytest.raises(TypeError):
        pyransame.random_volume_points(mesh, 20, weights=weights)


def test_wrong_n():
    mesh = pv.UniformGrid(dimensions=(4, 4, 4))

    with pytest.raises(ValueError, match="n must be > 0, got -20"):
        pyransame.random_volume_points(mesh, -20)
