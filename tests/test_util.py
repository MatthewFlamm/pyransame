"""Test utilities."""
from datetime import timedelta

import numpy as np
import pytest
import pyvista as pv
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyransame.util import (
    _generate_points_in_pixel,
    _generate_points_in_quad,
    _generate_points_in_tetra,
    _generate_points_in_tri,
    _generate_points_in_voxel,
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

    points = _generate_points_in_tri(a, b, c, 1000)
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
    points = _generate_points_in_tri(a, b, c, 2000000)

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
    points = _generate_points_in_tetra(a, b, c, d, 2000000)

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

    points = _generate_points_in_tetra(a, b, c, d, 1000)
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
    points = _generate_points_in_quad(a, b, c, d, 2000000)

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_pixel():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 2.0, 0.0])
    d = np.array([0.0, 2.0, 0.0])

    # needs a lot of points to converge
    points = _generate_points_in_pixel(a, b, d, c, 2000000)
    center = np.array([0.5, 1.0, 0.0])

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)


def test_uniformity_voxel():
    mesh = pv.UniformGrid(dimensions=(2, 2, 2), spacing=(1.0, 1.0, 2.0))

    # needs a lot of points to converge
    points = _generate_points_in_voxel(*mesh.points, 2000000)
    center = np.array([0.5, 0.5, 1.0])

    assert np.allclose(points.mean(axis=0), center, rtol=1e-3, atol=1e-3)
