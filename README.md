# pyransame
PYthon RAndom SAmpling for MEshes

[Documentation](https://matthewflamm.github.io/pyransame/)

Utilities for choosing random samples of points within cells of [PyVista](https://github.com/pyvista/pyvista) meshes.
This package does _not_ choose random points that define the mesh itself, rather random points on 0D vertices, 1D lines, 2D surfaces or
in 3D volumes are sampled.

All linear[^1] cells from [vtk](https://gitlab.kitware.com/vtk/vtk) are supported, except for `vtkConvexPointSet`.

[^1]: Linear here means not inheriting from `vtkNonLinearCell`.

## PyVista dataset accessor

On PyVista 0.48+, `pip install pyransame` registers a `ransam`
accessor on every PyVista dataset. No extra import or registration step
is required. PyVista discovers the accessor through the
`pyvista.accessors` entry point and loads it lazily the first time
`mesh.ransam` is used.

```python
import pyvista as pv
from pyvista import examples

bunny = examples.download_bunny()
points = bunny.ransam.surface_points(500)        # numpy array, shape (500, 3)
sampled = bunny.ransam.surface_dataset(500)      # pyvista.PolyData with interpolated arrays
```

The accessor mirrors the top-level functions:

| Method                          | Equivalent function                  |
| ------------------------------- | ------------------------------------ |
| `mesh.ransam.surface_points(n)` | `pyransame.random_surface_points`    |
| `mesh.ransam.surface_dataset(n)`| `pyransame.random_surface_dataset`   |
| `mesh.ransam.volume_points(n)`  | `pyransame.random_volume_points`     |
| `mesh.ransam.volume_dataset(n)` | `pyransame.random_volume_dataset`    |
| `mesh.ransam.line_points(n)`    | `pyransame.random_line_points`       |
| `mesh.ransam.line_dataset(n)`   | `pyransame.random_line_dataset`      |
| `mesh.ransam.vertex_points(n)`  | `pyransame.random_vertex_points`     |
| `mesh.ransam.vertex_dataset(n)` | `pyransame.random_vertex_dataset`    |
| `mesh.ransam.points(n)`         | dispatch by cell dimension           |
| `mesh.ransam.dataset(n)`        | dispatch by cell dimension           |

`points` and `dataset` infer the sampler from the cell dimensions on
the mesh: vertex (0D), line (1D), surface (2D), or volume (3D). Mixed
dimensions raise `ValueError`. Pass `kind="vertex" | "line" | "surface"
| "volume"` to override:

```python
sampled = mixed_mesh.ransam.dataset(500, kind="surface")
```

On older PyVista releases the package still installs and imports
cleanly; only the `mesh.ransam` namespace is unavailable. Use the
top-level `pyransame.random_*` functions in that case.

## Random sampling on a 2D surface

![Samples on a bunny](/doc/_static/surface_sampling.png)

## Random sampling in a 3D volume

![Samples inside a 3D volume](/doc/_static/volume_sampling.png)
