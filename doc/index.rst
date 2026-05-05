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

   >>> points = antarctica.ransam.surface_dataset(500)

   Plot points on mesh.

   >>> pl = pv.Plotter()
   >>> pl.add_mesh(antarctica, color='tan')
   >>> spheres = points.glyph(geom=pv.Sphere(radius=50), scale=False, orient=False)
   >>> pl.add_mesh(spheres, scalars="ssavelocity", clim=[0, 750])
   >>> pl.view_xy()
   >>> pl.show()


PyVista dataset accessor
========================

On PyVista 0.48 and newer, ``pip install pyransame`` registers a
``ransam`` accessor on every PyVista dataset. PyVista discovers it
through the ``pyvista.accessors`` entry point and imports the plugin
lazily the first time ``mesh.ransam`` is used.

.. code-block:: python

    import pyvista as pv
    from pyvista import examples

    bunny = examples.download_bunny()
    points = bunny.ransam.surface_points(500)
    sampled = bunny.ransam.surface_dataset(500)

The accessor exposes a method for each top-level sampling function:

================================  ==================================
Accessor method                   Equivalent function
================================  ==================================
``mesh.ransam.surface_points``    :func:`pyransame.random_surface_points`
``mesh.ransam.surface_dataset``   :func:`pyransame.random_surface_dataset`
``mesh.ransam.volume_points``     :func:`pyransame.random_volume_points`
``mesh.ransam.volume_dataset``    :func:`pyransame.random_volume_dataset`
``mesh.ransam.line_points``       :func:`pyransame.random_line_points`
``mesh.ransam.line_dataset``      :func:`pyransame.random_line_dataset`
``mesh.ransam.vertex_points``     :func:`pyransame.random_vertex_points`
``mesh.ransam.vertex_dataset``    :func:`pyransame.random_vertex_dataset`
``mesh.ransam.points``            dispatch by cell dimension
``mesh.ransam.dataset``           dispatch by cell dimension
================================  ==================================

``points`` and ``dataset`` introspect ``mesh.distinct_cell_types`` to
choose the sampler (vertex, line, surface, or volume). Meshes that
carry cells of more than one dimension raise :class:`ValueError`; pass
``kind="vertex" | "line" | "surface" | "volume"`` to force a specific
sampler.

On older PyVista releases the package still installs and imports
cleanly; only the ``mesh.ransam`` namespace is unavailable, and the
top-level ``pyransame.random_*`` functions should be used instead.

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
