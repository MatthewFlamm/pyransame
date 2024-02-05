"""Tests control and seeding of RNG."""

import numpy as np
import pyvista as pv
from numpy.random import Generator, default_rng

import pyransame


def test_rng():
    assert isinstance(pyransame.rng, Generator)

    # first show that the points are continuously randomly generated
    mesh = pv.Plane()
    points1 = pyransame.random_surface_points(mesh, 20)
    points2 = pyransame.random_surface_points(mesh, 20)
    assert not np.allclose(points1, points2)

    # finally show that the rng status can be controlled
    pyransame.rng = default_rng(seed=100)
    points1 = pyransame.random_surface_points(mesh, 20)
    pyransame.rng = default_rng(seed=100)
    points2 = pyransame.random_surface_points(mesh, 20)
    assert np.allclose(points1, points2)
