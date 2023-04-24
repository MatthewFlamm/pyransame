"""Python package for random sampling of meshes."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pyvista as pv

from .util import (
    _generate_points_in_pixel,
    _generate_points_in_quad,
    _generate_points_in_tetra,
    _generate_points_in_tri,
    _generate_points_in_voxel,
)

rng = np.random.default_rng()


def random_surface_points(
    mesh: pv.PolyData,
    n: int = 1,
    weights: Optional[Union[str, np.ndarray, Sequence]] = None,
) -> np.ndarray:
    """Generate random points on surface.

    Supported cell types:

    - Triangle
    - Pixel
    - Quad

    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh for which to generate random points.  Must have cells.

    n : int, default: 1
        Number of random points to generate

    weights : str, or sequence-like, optional
        Weights to use for probability of choosing points inside each cell.

        If a ``str`` is supplied, it will use the existing cell data on ``mesh``.
        If a ``sequence`` is supplied, it will add to the cell data using
        ``weights`` key.

    Returns
    -------
    points : np.ndarray
        ``(n, 3)`` points that exist inside cells on ``mesh``.

    Examples
    --------
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
    """
    if not isinstance(mesh, pv.PolyData):
        raise ValueError(f"mesh must by PolyData got {type(mesh)}")

    if weights is None:
        weights = np.ones(mesh.n_cells)

    if not isinstance(weights, (np.ndarray, Sequence, str)):
        raise ValueError("Invalid weights, got weights")

    if not isinstance(weights, str):
        mesh.cell_data["weights"] = weights
        weights = "weights"

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n_cells = mesh.n_cells

    if "Area" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, volume=False)

    p = mesh[weights] * mesh["Area"]

    p = p / p.sum()

    chosen = rng.choice(n_cells, n, p=p)

    points = np.empty((n, 3))
    for i in range(n):
        # TODO: assess whether gathering unique points first improves speed
        # TODO: when there are many points per cell
        c = mesh.get_cell(chosen[i])
        if c.type == pv.CellType.TRIANGLE:
            points[i, :] = _generate_points_in_tri(*c.points)
        elif c.type == pv.CellType.PIXEL:
            points[i, :] = _generate_points_in_pixel(*c.points)
        elif c.type == pv.CellType.QUAD:
            points[i, :] = _generate_points_in_quad(*c.points)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )
    return points


def random_volume_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, np.ndarray, Sequence]] = None,
) -> np.ndarray:
    """Generate random points in a volume.

    Supported cell types:

    - Tetrahedron
    - Voxel

    Parameters
    ----------
    mesh : pyvista.DataSet
        The mesh for which to generate random points.  Must have cells.

    n : int, default: 1
        Number of random points to generate

    weights : str, or sequence-like, optional
        Weights to use for probability of choosing points inside each cell.

        If a ``str`` is supplied, it will use the existing cell data on ``mesh``.
        If a ``sequence`` is supplied, it will add to the cell data using
        ``weights`` key.

    Returns
    -------
    points : np.ndarray
        ``(n, 3)`` points that exist inside cells on ``mesh``.

    Examples
    --------
    >>> import pyransame
    >>> import pyvista as pv
    >>> mesh = pv.UniformGrid(dimensions=(11, 11, 11))
    >>> points = pyransame.random_volume_points(mesh, n=500)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> pl.add_mesh(mesh, style='wireframe')
    >>> pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
    >>> pl.show(cpos=cpos)
    """
    if weights is None:
        weights = np.ones(mesh.n_cells)

    if not isinstance(weights, (np.ndarray, Sequence, str)):
        raise ValueError("Invalid weights, got weights")

    if not isinstance(weights, str):
        mesh.cell_data["weights"] = weights
        weights = "weights"

    for c in mesh.cell:
        if c.dimension != 3:
            raise ValueError("non 3D cell in DataSet")

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n_cells = mesh.n_cells

    if "Volume" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, volume=True)

    p = mesh[weights] * mesh["Volume"]

    p = p / p.sum()

    chosen = rng.choice(n_cells, n, p=p)

    points = np.empty((n, 3))
    for i in range(n):
        # TODO: assess whether gathering unique points first improves speed
        # TODO: when there are many points per cell
        c = mesh.get_cell(chosen[i])
        if c.type == pv.CellType.TETRA:
            points[i, :] = _generate_points_in_tetra(*c.points)
        elif c.type == pv.CellType.VOXEL:
            points[i, :] = _generate_points_in_voxel(*c.points)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )

    return points
