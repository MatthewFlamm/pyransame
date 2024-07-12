"""Generating random points from vertices."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyransame
import pyransame.util as util


def random_vertex_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> np.ndarray:
    """
    Generate random points from vertices.

    .. note::
        This function is provided for completeness of API,
        but it is likely faster and more flexible to use
        a custom method.

    Supported cell types:

    - Vertex
    - PolyVertex

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
    Create a mesh with 1 vertex cell (1 point) and 1 polyvertex cell (5 points).

    >>> import pyransame
    >>> import pyvista as pv
    >>> p = [
    ...       [0., 0., 0.],
    ...       [1., 0., 0.],
    ...       [1., 1., 0.],
    ...       [1., 2., 0.],
    ...       [1., 3., 0.],
    ...       [1., 4., 0.],
    ... ]
    >>> mesh = pv.PolyData(p, verts=[1, 0, 5, 1, 2, 3, 4, 5])
    >>> points = pyransame.random_vertex_points(mesh, n=3)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, render_points_as_spheres=True, point_size=16.0, color='blue')
    >>> _ = pl.add_points(points, render_points_as_spheres=True, point_size=20.0, color='red')
    >>> pl.view_xy()
    >>> pl.show()
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
        mesh = mesh.compute_cell_sizes(length=False, area=False, volume=False, vertex_count=True)  # type: ignore

    p = weights * mesh["VertexCount"]

    if p.sum() == 0:
        raise ValueError("No cells with vertices in DataSet")

    p = p / p.sum()

    chosen_cells, unique_counts, point_indices = util._random_cells(n_cells, n, p)
    points = np.empty((n, 3))
    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        c = mesh.get_cell(chosen_cell)
        if c.type == pv.CellType.VERTEX:
            points[point_indices[i] : point_indices[i + 1], :] = (
                util._generate_points_in_vertex(c.points, count)
            )
        elif c.type == pv.CellType.POLY_VERTEX:
            points[point_indices[i] : point_indices[i + 1], :] = (
                util._generate_points_in_polyvertex(c.points, count)
            )
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )
    return points


def random_vertex_dataset(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> pv.PolyData:
    """
    Generate random points on vertices with sampled data.

    .. note::
        This function is provided for completeness of API,
        but it is likely faster and more flexible to use
        a custom method.

    Supported cell types:

    - Vertex
    - PolyVertex

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
    Create a mesh with 1 vertex cell (1 point) and 1 polyvertex cell (5 points).
    Add data for y position.

    >>> import pyransame
    >>> import pyvista as pv
    >>> p = [
    ...       [0., 0., 0.],
    ...       [1., 0., 0.],
    ...       [1., 1., 0.],
    ...       [1., 2., 0.],
    ...       [1., 3., 0.],
    ...       [1., 4., 0.],
    ... ]
    >>> mesh = pv.PolyData(p, verts=[1, 0, 5, 1, 2, 3, 4, 5])
    >>> mesh["y"] = mesh.points[:, 1]
    >>> dataset = pyransame.random_vertex_dataset(mesh, n=3)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, render_points_as_spheres=True, point_size=8.0, color='black')
    >>> _ = pl.add_points(dataset, render_points_as_spheres=True, point_size=20.0, scalars="y")
    >>> pl.view_xy()
    >>> pl.show()
    """
    points = random_vertex_points(mesh, n, weights)
    return pv.PolyData(points).sample(
        mesh, locator="static_cell", snap_to_closest_point=True
    )
