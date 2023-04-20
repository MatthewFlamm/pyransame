Welcome to pyransame's documentation!
=====================================

Examples
--------

Random points on surface
~~~~~~~~~~~~~~~~~~~~~~~~

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

Random sampling in a volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example inspired by the discussion in
https://github.com/pyvista/pyvista/discussions/2703#discussioncomment-2828140.
Using ``pyransame`` makes the solution easier and handles more complex scenarios.

.. pyvista-plot::
   :include-source: True

   >>> import numpy as np
   >>> import pyvista as pv
   >>> import pyransame

   As an example, say we have a cube with particles in it.  But we only have a description
   of the volume fraction of the particles.  Let's say that they tend to float in the domain,
   so they tend to positioned towards the top (z-position).  Also, let's say that some particle age
   is related to the y-position.

   >>> mesh = pv.UniformGrid(dimensions=(10, 10, 10))
   >>> mesh["volume_frac"] = np.exp(np.linspace(0, 5, mesh.n_cells))
   >>> mesh["volume_frac"] /= np.sum(mesh["volume_frac"])
   >>> mesh["age"] = mesh.points[:, 1]
   >>> points = pyransame.random_volume_points(mesh, 1000, weights="volume_frac")

   Sample the ``age`` data onto points.  First create a ``PointSet`` so that we can use ``pyvista.sample``

   >>> ps = pv.PointSet(points)
   >>> ps = ps.sample(mesh)

   Now plot the result

   >>> cpos = [
   ...     [32., 16., 10.],
   ...     [5.0, 3.9, 3.6],
   ...     [-0.21, -0.076, 0.97]
   ... ]
   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, style='wireframe')
   >>> pl.add_points(ps, render_points_as_spheres=True, point_size=10.0)
   >>> pl.show(cpos=cpos)

API documentation
-----------------
.. currentmodule:: pyransame

.. autosummary::
   :toctree: _stubs

   random_surface_points
   random_volume_points

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
