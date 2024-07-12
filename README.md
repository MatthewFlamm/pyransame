# pyransame
PYthon RAndom SAmpling for MEshes

[Documentation](https://matthewflamm.github.io/pyransame/)

Utilities for choosing random samples of points within cells of [PyVista](https://github.com/pyvista/pyvista) meshes.
This package does _not_ choose random points that define the mesh itself, rather random points on 0D vertices, 1D lines, 2D surfaces or
in 3D volumes are sampled.

All linear[^1] cells from [vtk](https://gitlab.kitware.com/vtk/vtk) are supported, except for `vtkConvexPointSet`.

[^1]: Linear here means not inheriting from `vtkNonLinearCell`.

## Random sampling on a 2D surface

![Samples on a bunny](/doc/_static/surface_sampling.png)

## Random sampling in a 3D volume

![Samples inside a 3D volume](/doc/_static/volume_sampling.png)
