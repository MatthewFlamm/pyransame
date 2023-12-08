"""Test utilities."""
from datetime import timedelta

import numpy as np
import pytest
import pyvista as pv
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from vtk import VTK_POLYHEDRON, vtkIdList, vtkPoints, vtkUnstructuredGrid

from pyransame.util import (
    _generate_points_in_hexagonal_prism,
    _generate_points_in_hexahedron,
    _generate_points_in_pentagonal_prism,
    _generate_points_in_pixel,
    _generate_points_in_polygon,
    _generate_points_in_polyhedron,
    _generate_points_in_pyramid,
    _generate_points_in_quad,
    _generate_points_in_tetra,
    _generate_points_in_tri,
    _generate_points_in_tri_strip,
    _generate_points_in_voxel,
    _generate_points_in_wedge,
)


@settings(deadline=timedelta(milliseconds=500))
# Use min_value and max_value to avoid numerical imprecision artifacts, but still span a large space
@given(
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
)
def test_generate_points_in_tri(a, b, c):
    """Use pyvista builtin find_containing_cell to test."""
    # make sure that the points are not coincident
    assume(not np.allclose(a, b, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(b, c, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(a, c, rtol=1e-3, atol=1e-3))
    tri = pv.Triangle([a, b, c])
    assume(tri.area > 1e-4)

    points = _generate_points_in_tri(np.array([a, b, c]), 1000)
    for i in range(1000):
        assert tri.find_containing_cell(points[i, :]) == 0


# equations from https://mathworld.wolfram.com/TrianglePointPicking.html
# Weisstein, Eric W. "Triangle Point Picking." From MathWorld--A Wolfram Web Resource.
def test_uniformity_tri():
    # form equilaterial triangle with one vertex at origin
    a = np.array((0.0, 0.0, 0.0))
    b = np.array((1.0, 0.0, 0.0))
    c = np.array((0.5, np.sqrt(3.0) / 2.0, 0.0))

    center = np.array((0.5, np.sqrt(3.0) / 6.0, 0.0))

    # needs a lot of points to converge
    points = _generate_points_in_tri(np.array([a, b, c]), 2000000)

    distances = np.linalg.norm(points - center, axis=-1)
    exp_distance = (
        1 / 72 * (8 * np.sqrt(3) + 3 * np.arcsinh(np.sqrt(3)) + np.log(2 + np.sqrt(3)))
    )
    assert distances.mean() == pytest.approx(exp_distance, rel=2e-3)

    distances = np.linalg.norm(points, axis=-1)
    assert distances.mean() == pytest.approx(1 / 12 * (4 + 3 * np.log(3)), rel=2e-3)


# equations from https://mathworld.wolfram.com/RegularTetrahedron.html
# Weisstein, Eric W. "Triangle Point Picking." From MathWorld--A Wolfram Web Resource.
def test_uniformity_tetra():
    # form regular tetrahedron with geometric center at origin
    a = np.array((np.sqrt(3) / 3.0, 0.0, -np.sqrt(6) / 12.0))
    b = np.array((-np.sqrt(3) / 6, 0.5, -np.sqrt(6) / 12.0))
    c = np.array((-np.sqrt(3) / 6, -0.5, -np.sqrt(6) / 12.0))
    d = np.array((0.0, 0.0, np.sqrt(6) / 4))

    center = np.array((0.0, 0.0, 0.0))

    # needs a lot of points to converge
    points = _generate_points_in_tetra(np.array([a, b, c, d]), 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


@settings(deadline=timedelta(milliseconds=500))
# Use min_value and max_value to avoid numerical imprecision artifacts, but still span a large space
@given(
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
    arrays(float, 3, elements=st.floats(min_value=-1000.0, max_value=1000.0)),
)
def test_generate_points_in_tetra(a, b, c, d):
    """Use pyvista builtin find_containing_cell to test."""
    # make sure that the points are not coincident
    assume(not np.allclose(a, b, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(b, c, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(a, c, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(a, d, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(b, d, rtol=1e-3, atol=1e-3))
    assume(not np.allclose(c, d, rtol=1e-3, atol=1e-3))

    cells = [4, 0, 1, 2, 3]
    celltypes = [pv.CellType.TETRA]
    tetra = pv.UnstructuredGrid(cells, celltypes, [a, b, c, d])
    assume(tetra.volume > 1e-4)

    points = _generate_points_in_tetra(np.array([a, b, c, d]), 1000)
    for i in range(1000):
        assert tetra.find_containing_cell(points[i, :]) == 0


def test_uniformity_quad():
    # Quad can be non-square
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    d = np.array([0.25, 0.75, 0.0])

    area_abc = 1 * 1 / 2.0
    l_side_acd = np.sqrt(0.75**2 + 0.25**2)
    l_base_acd = np.sqrt(2.0)
    area_acd = (
        1.0
        / 2.0
        * l_base_acd**2
        * np.sqrt(l_side_acd**2 / l_base_acd**2 - 1.0 / 4.0)
    )

    center_abc = b + 2 / 3 * np.array([-0.5, 0.5, 0.0])
    center_acd = d + 2 / 3 * np.array([0.25, -0.25, 0.0])

    center = (center_abc * area_abc + center_acd * area_acd) / (area_abc + area_acd)

    # needs a lot of points to converge
    points = _generate_points_in_quad(np.array([a, b, c, d]), 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_polygon():
    # Use same quad test
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    d = np.array([0.25, 0.75, 0.0])

    area_abc = 1 * 1 / 2.0
    l_side_acd = np.sqrt(0.75**2 + 0.25**2)
    l_base_acd = np.sqrt(2.0)
    area_acd = (
        1.0
        / 2.0
        * l_base_acd**2
        * np.sqrt(l_side_acd**2 / l_base_acd**2 - 1.0 / 4.0)
    )

    center_abc = b + 2 / 3 * np.array([-0.5, 0.5, 0.0])
    center_acd = d + 2 / 3 * np.array([0.25, -0.25, 0.0])

    center = (center_abc * area_abc + center_acd * area_acd) / (area_abc + area_acd)

    # needs a lot of points to converge
    points = _generate_points_in_polygon(np.array([a, b, c, d]), 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_tri_strip():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    d = np.array([0.25, 0.75, 0.0])

    area_abc = 1 * 1 / 2.0
    l_side_acd = np.sqrt(0.75**2 + 0.25**2)
    l_base_acd = np.sqrt(2.0)
    area_acd = (
        1.0
        / 2.0
        * l_base_acd**2
        * np.sqrt(l_side_acd**2 / l_base_acd**2 - 1.0 / 4.0)
    )

    center_abc = b + 2 / 3 * np.array([-0.5, 0.5, 0.0])
    center_acd = d + 2 / 3 * np.array([0.25, -0.25, 0.0])

    center = (center_abc * area_abc + center_acd * area_acd) / (area_abc + area_acd)

    # needs a lot of points to converge
    points = _generate_points_in_tri_strip(np.array([b, a, c, d]), 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_pixel():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 2.0, 0.0])
    d = np.array([0.0, 2.0, 0.0])

    # needs a lot of points to converge
    points = _generate_points_in_pixel(np.array([a, b, d, c]), 2000000)
    center = np.array([0.5, 1.0, 0.0])

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_voxel():
    mesh = pv.ImageData(dimensions=(2, 2, 2), spacing=(1.0, 1.0, 2.0))

    # needs a lot of points to converge
    points = _generate_points_in_voxel(mesh.points, 2000000)
    center = np.array([0.5, 0.5, 1.0])

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_pyramid():
    mesh = pv.Pyramid(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    center = np.array([0.0, 0.0, 0.25])

    points = _generate_points_in_pyramid(mesh.points, 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_wedge():
    a = np.array((0.0, 0.0, 0.0))
    b = np.array((1.0, 0.0, 0.0))
    c = np.array((0.5, np.sqrt(3.0) / 2.0, 0.0))
    d = np.array((0.0, 0.0, 1.0))
    e = np.array((1.0, 0.0, 1.0))
    f = np.array((0.5, np.sqrt(3.0) / 2.0, 1.0))

    center = np.array((0.5, np.sqrt(3.0) / 6.0, 0.5))

    # needs a lot of points to converge
    points = _generate_points_in_wedge(np.array([a, b, c, d, e, f]), 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_hexahedra():
    p = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    center = np.array([0.5, 0.5, 0.5])

    points = _generate_points_in_hexahedron(p, 2000000)
    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_pentagonal_prism():
    angles = np.arange(5) * 2 * np.pi / 5  # angles of regular pentagon in radians
    p = np.zeros(shape=(10, 3))
    np.sin(angles, out=p[0:5, 0])
    np.cos(angles, out=p[0:5, 1])

    np.sin(angles, out=p[5:, 0])
    np.cos(angles, out=p[5:, 1])
    p[5:, 2] = 1.0

    center = np.array([0.0, 0.0, 0.5])
    points = _generate_points_in_pentagonal_prism(p, 2000000)
    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_hexagonal_prism():
    angles = np.arange(6) * 2 * np.pi / 6  # angles of regular hexagon in radians
    p = np.zeros(shape=(12, 3))
    np.sin(angles, out=p[0:6, 0])
    np.cos(angles, out=p[0:6, 1])

    np.sin(angles, out=p[6:, 0])
    np.cos(angles, out=p[6:, 1])
    p[6:, 2] = 1.0

    center = np.array([0.0, 0.0, 0.5])
    points = _generate_points_in_hexagonal_prism(p, 2000000)
    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_polyhedron():
    dodecahedron_poly = pv.Dodecahedron()

    faces = []
    for cell in dodecahedron_poly.cell:
        faces.append(cell.point_ids)

    faces = []
    faces.append(dodecahedron_poly.n_cells)
    for cell in dodecahedron_poly.cell:
        point_ids = cell.point_ids
        faces.append(len(point_ids))
        [faces.append(id) for id in point_ids]

    faces.insert(0, len(faces))

    mesh = pv.UnstructuredGrid(
        faces, [pv.CellType.POLYHEDRON], dodecahedron_poly.points
    )

    points = _generate_points_in_polyhedron(mesh.get_cell(0), 2000000)

    assert np.allclose(points.mean(axis=0), np.array(mesh.center), rtol=1e-3, atol=1e-3)
