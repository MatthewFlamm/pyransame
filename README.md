# pyransame
PYthon RAndom SAmpling for MEshes

Utilites for choosing random samples of points within cells of [PyVista](https://github.com/pyvista/pyvista) meshes.

Random samples of mesh points or cells is supported direclty in vtk, `vtkMaskPoints` for example.
Random samples of points within cells is not well supported in vtk, which is what this package provides.
