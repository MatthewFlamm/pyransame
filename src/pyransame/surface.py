"""Generating random points on a 2D surface."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyransame
import pyransame.util as util


def random_surface_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> np.ndarray:
    """
    Generate random points on surface.

    Supported cell types:

    - Triangle
    - Triangle Strip
    - Pixel
    - Polygon
    - Quad

    All cells must be convex.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The mesh for which to generate random points.  Must have cells.

    n : int, default: 1
        Number of random points to generate.

    weights : str, or array_like, optional
        Weights to use for probability of choosing points inside each cell.

        If a ``str`` is supplied, it will use the existing cell data on ``mesh``.

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
    >>> _ = pl.add_mesh(mesh, color='tan')
    >>> _ = pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
    >>> pl.show(cpos=cpos)
    """
    if weights is None:
        weights = np.ones(mesh.n_cells)

    if isinstance(weights, str):
        weights = mesh.cell_data[weights]

    weights = np.asanyarray(weights)

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n_cells = mesh.n_cells

    if "Area" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)  # type: ignore

    p = weights * mesh["Area"]

    if p.sum() == 0:
        raise ValueError("No cells with area in DataSet")

    p = p / p.sum()

    chosen_cells, unique_counts, point_indices = util._random_cells(n_cells, n, p)
    points = np.empty((n, 3))
    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        c = mesh.get_cell(chosen_cell)
        if c.type == pv.CellType.TRIANGLE:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_tri(c.points, count)
        elif c.type == pv.CellType.PIXEL:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_pixel(c.points, count)
        elif c.type == pv.CellType.QUAD:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_quad(c.points, count)
        elif c.type == pv.CellType.POLYGON:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_polygon(c.points, count)
        elif c.type == pv.CellType.TRIANGLE_STRIP:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_tri_strip(c.points, count)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )
    return points
