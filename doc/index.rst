Welcome to pyransame's documentation!
=====================================

Examples
--------

Random points on surface
~~~~~~~~~~~~~~~~~~~~~~

.. pyvista-plot::
   :include-source: True

   >>> import pyransame
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> mesh = examples.download_bunny()
   >>> points = pyransame.random_surface_points(mesh, n=500)

   Now plot result.

   >>> cpos = [
   ...     (-0.07, 0.2, 0.5),
   ...     (-0.02, 0.1, -0.0),
   ...     (0.04, 1.0, -0.2),
   ... ]
   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, color='tan')
   >>> pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
   >>> pl.show(cpos=cpos)

Weighted random sampling on surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample random points on a bunny, but weight more points towards top.


.. pyvista-plot::
   :include-source: True

   >>> import pyransame
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> mesh = examples.download_bunny()

   The random sampling occurs inside the cells of the mesh, so cell data
   is needed.  ``pyvista.cell_centers`` is used to get position of the
   cells.

   >>> weights = mesh.cell_centers().points[:, 1]**2
   >>> points = pyransame.random_surface_points(mesh, n=500, weights=weights)

   Plot result.

   >>> cpos = [
   ...     (-0.07, 0.2, 0.5),
   ...     (-0.02, 0.1, -0.0),
   ...     (0.04, 1.0, -0.2),
   ... ]
   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, color='tan')
   >>> pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
   >>> pl.show(cpos=cpos)

API documentation
-----------------
.. currentmodule:: pyransame

.. autosummary::
   :toctree: _stubs

   random_surface_points

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
