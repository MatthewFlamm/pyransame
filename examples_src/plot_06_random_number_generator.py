"""
.. _random_number_generator_example:

Controlling Random Number Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generation of random numbers is done through a :func:`numpy.random.Generator`.
The default is :func:`numpy.random.default_rng()`.
"""

import numpy as np
import pyvista as pv
from pyvista import examples

import pyransame

antarctica = examples.download_antarctica_velocity()
antarctica.points /= 1000.0  # convert to kilometers

###############################################################################
# pyransame stores the generator being used at ``pyransame.rng``

pyransame.rng

###############################################################################
# Setup common plotting routine for use later.


def plot_points(points):
    pl = pv.Plotter()
    pl.add_mesh(antarctica, color="tan")
    spheres = points.glyph(geom=pv.Sphere(radius=50), scale=False, orient=False)
    pl.add_mesh(spheres, scalars="ssavelocity", clim=[0, 750])
    pl.view_xy()
    pl.show()


###############################################################################
# Sampling twice in succession will lead to different results.

points = pyransame.random_surface_dataset(antarctica, 500)
plot_points(points)

###############################################################################
# Second sampling with slightly different results

points = pyransame.random_surface_dataset(antarctica, 500)
plot_points(points)

###############################################################################
# This time, we will control the random number generation by using a seed to
# ensure that the same result is obtained.

pyransame.rng = np.random.default_rng(seed=42)
points = pyransame.random_surface_dataset(antarctica, 500)
plot_points(points)

###############################################################################
# Second sampling with identical results.

pyransame.rng = np.random.default_rng(seed=42)
points = pyransame.random_surface_dataset(antarctica, 500)
plot_points(points)
