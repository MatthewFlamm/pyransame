"""Generating random points on a 2D surface."""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyransame
import pyransame.util as util


def random_line_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> np.ndarray:
    """
    Generate random points on lines.

    Supported cell types:

    - Line

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
    >>> p = [
    ...       [0., 0., 0.],
    ...       [1., 0., 0.],
    ...       [1., 2., 0.]
    ... ]
    >>> mesh = pv.PolyData(p, lines=[2, 0, 1, 2, 1, 2]
    >>> points = pyransame.random_surface_points(mesh, n=5)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='tan')
    >>> _ = pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
    >>> pl.view_xy()
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

    if "Length" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=True, area=False, volume=False)  # type: ignore

    p = weights * mesh["Length"]

    if p.sum() == 0:
        raise ValueError("No cells with length in DataSet")

    p = p / p.sum()

    chosen_cells, unique_counts, point_indices = util._random_cells(n_cells, n, p)
    points = np.empty((n, 3))
    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        c = mesh.get_cell(chosen_cell)
        if c.type == pv.CellType.LINE:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_line(c.points, count)
        elif c.type == pv.CellType.POLY_LINE:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_polyline(c.points, count)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )
    return points


def random_line_dataset(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> pv.PolyData:
    """
    Generate random points on surface with sampled data.

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
    points : pv.PolyData
        ``(n, 3)`` points that exist inside cells on ``mesh`` and with
        sampled data.

    Examples
    --------
    >>> import pyransame
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.download_bunny()
    >>> mesh['y'] = mesh.points[:, 1]
    >>> points = pyransame.random_surface_dataset(mesh, n=500)

    Now plot result.

    >>> cpos = [
    ...     (-0.07, 0.2, 0.5),
    ...     (-0.02, 0.1, -0.0),
    ...     (0.04, 1.0, -0.2),
    ... ]
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='tan')
    >>> _ = pl.add_points(points, scalars='y', render_points_as_spheres=True, point_size=10.0)
    >>> pl.show(cpos=cpos)
    """
    points = random_line_points(mesh, n, weights)
    return pv.PolyData(points).sample(mesh, locator="static_cell")
