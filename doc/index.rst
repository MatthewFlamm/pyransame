Welcome to pyransame's documentation!
=====================================

PYthon RAndom SAmpling for MEshes (pyransame) chooses random samples of points within cells of
`PyVista <https://docs.pyvista.org/>`_ meshes.

.. pyvista-plot::
   :include-source: True

   A compressed example from :ref:`random_surface_sampling_example`.
   First load packages and convert data units.

   >>> import pyransame
   >>> import pyvista as pv
   >>> from pyvista import examples
   >>> antarctica = examples.download_antarctica_velocity()
   >>> antarctica.points /= 1000.  # convert to kilometers

   Sample points on mesh using ``pyransame``.

   >>> points = pyransame.random_surface_dataset(antarctica, 500)

   Plot points on mesh.

   >>> pl = pv.Plotter()
   >>> pl.add_mesh(antarctica, color='tan')
   >>> spheres = points.glyph(geom=pv.Sphere(radius=50), scale=False, orient=False)
   >>> pl.add_mesh(spheres, scalars="ssavelocity", clim=[0, 750])
   >>> pl.view_xy()
   >>> pl.show()


For more usage see :ref:`examples`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :hidden:

   examples/index
   api
