"""
.. _random_surface_points_example:

Random Surface Points
~~~~~~~~~~~~~~~~~~~~~~~

If only random point locations are desired, then instead of
:func:`pyransame.random_surface_dataset`, :func:`pyransame.random_surface_points`
can be used.  Compare this example to :ref:`random_surface_sampling_example`.
"""

import pyvista as pv
from pyvista import examples

import pyransame

antarctica = examples.download_antarctica_velocity()

###############################################################################
# The units of this mesh are in meters, which causes plotting issues
# over an entire continent.  So the units are first converted to kilometers.

antarctica.points /= 1000.0  # convert to kilometers

###############################################################################
# sample 500 points uniformly randomly.

points = pyransame.random_surface_points(antarctica, 500)
points

###############################################################################
# :func:`pyransame.random_surface_points` returns a `numpy.ndarray` object
# containing 500 point locations without any sampled data. To plot as spheres,
# we first create a `pyvista.PolyData` object.  Since we did not sample any
# data from ``antarctica``, we do not have any scalar data, so color the
# spheres red.

pl = pv.Plotter()
pl.add_mesh(antarctica, color="tan")
spheres = pv.PolyData(points).glyph(
    geom=pv.Sphere(radius=50), scale=False, orient=False
)
pl.add_mesh(spheres, color="red")
pl.view_xy()
pl.show()
