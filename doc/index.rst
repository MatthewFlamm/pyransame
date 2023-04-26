Welcome to pyransame's documentation!
=====================================

PYthon RAndom SAmpling for MEshes (pyransame) provides utilities
for choosing rando samples of points within cells of
`PyVista <https://docs.pyvista.org/>`_ meshes.

Examples
--------

Random points on surface
~~~~~~~~~~~~~~~~~~~~~~~~

A mesh has points that define the vertices of the cells.
Using ``pyvista`` and ``numpy`` we can quickly sample these
vertices.

.. pyvista-plot::
   :include-source: True

   >>> import numpy as np
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> mesh = examples.download_antarctica_velocity()
   >>> vertices = np.random.default_rng().choice(mesh.points, 500)
   >>> mesh = mesh.compute_cell_sizes()

   Plot the sampled vertices and the mesh colored by the local cell size.

   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, scalars="Area")
   >>> pl.add_points(vertices, render_points_as_spheres=True,
   ...               point_size=10.0, color='red')
   >>> pl.view_xy()
   >>> pl.show()

But it can be seen that these vertices are not sampled
uniformly randomly with respect to the surface area.
They are sampled with respect to the vertices of the mesh.
Areas of the mesh with more vertices, i.e. smaller cell areas
in blue, will have more sampled points.

This type of sampling will be dependent on the mesh
structure.  It also does not allow for sampling inside the
cells of the mesh.

``pyransame`` allows for sampling inside the cells of the mesh.
The sampling is now uniform within the area of the land itself rather than
depending on the mesh structure.

.. pyvista-plot::
   :include-source: True

   >>> import pyransame
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> mesh = examples.download_antarctica_velocity()
   >>> points = pyransame.random_surface_points(mesh, 500)

   Now plot result.

   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, color='tan')
   >>> pl.add_points(points, render_points_as_spheres=True,
   ...               point_size=10.0, color='red')
   >>> pl.view_xy()
   >>> pl.show()

Weighted random sampling on surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A weighted sampling can also be obtained.  For example,
if sampling is wanted to be biased roughly towards McMurdo station
at the bottom of the image.

.. pyvista-plot::
   :include-source: True

   >>> import pyransame
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> mesh = examples.download_antarctica_velocity()


   The random sampling occurs inside the cells of the mesh, so cell data
   is needed.  ``pyvista.cell_centers`` is used to get position of the
   cells relative to the top of the mesh.

   >>> weights = (mesh.bounds[3] - mesh.cell_centers().points[:, 1])**2
   >>> points = pyransame.random_surface_points(mesh, n=500, weights=weights)

   Plot result.

   >>> pl = pv.Plotter()
   >>> pl.add_mesh(mesh, color='tan')
   >>> pl.add_points(points, render_points_as_spheres=True,
   ...               point_size=10.0, color='red')
   >>> pl.view_xy()
   >>> pl.show()

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
   >>> pl.add_points(ps, render_points_as_spheres=True,
   ...               point_size=10.0)
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
