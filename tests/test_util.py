"""Test utilities."""
from hypothesis import assume, given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import pyvista as pv
import pytest

from pyransame.util import _generate_points_in_tri


#Use min_value and max_value to avoid numerical imprecision artifacts, but still span a large space
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
    # make sure that the two vectors are not coincident
    assume(np.linalg.norm(np.cross(b-a, c-a)) > 1e-4)

    points = _generate_points_in_tri(a, b, c, 1000)

    tri = pv.Triangle([a, b, c])

    for i in range(1000):
        assert tri.find_containing_cell(points[i,:]) == 0


# equations from https://mathworld.wolfram.com/TrianglePointPicking.html
# Weisstein, Eric W. "Triangle Point Picking." From MathWorld--A Wolfram Web Resource.
def test_uniformity():
    # form equilaterial triangle with one vertex at origin
    a = np.array((0.0, 0.0, 0.0))
    b = np.array((1.0, 0.0, 0.0))
    c = np.array((0.5, np.sqrt(3.0)/2.0, 0.0))

    center = np.array((0.5, np.sqrt(3.0)/6.0, 0.0))

    # needs a lot of points to converge
    points = _generate_points_in_tri(a, b, c, 1000000)

    distances = np.linalg.norm(points - center, axis=-1)
    exp_distance = 1/72*(8*np.sqrt(3)+ 3*np.arcsinh(np.sqrt(3))+np.log(2+np.sqrt(3)))
    assert distances.mean() == pytest.approx(exp_distance, rel=1e-3)

    distances = np.linalg.norm(points, axis=-1)
    assert distances.mean() == pytest.approx(1/12*(4+3*np.log(3)), rel=1e-3)
