"""Generating random points in a 3D volume."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pyvista as pv

import pyransame
import pyransame.util as util


def random_volume_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, np.ndarray, Sequence]] = None,
) -> np.ndarray:
    """Generate random points in a volume.

    Supported cell types:

    - Tetrahedron
    - Voxel

    All cells must be convex.

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

    if isinstance(weights, str):
        weights = mesh.cell_data[weights]

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n_cells = mesh.n_cells

    if "Volume" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, volume=True)

    p = weights * mesh["Volume"]

    if p.sum() == 0:
        raise ValueError("No cells with volume in DataSet")

    p = p / p.sum()

    chosen = pyransame.rng.choice(n_cells, n, p=p)
    chosen_cells, unique_counts = np.unique(chosen, return_counts=True)
    point_indices = np.zeros(shape=chosen_cells.size + 1, dtype=int)
    point_indices[1:] = np.cumsum(unique_counts)
    points = np.empty((n, 3))
    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        c = mesh.get_cell(chosen_cell)

        if c.type == pv.CellType.TETRA:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_tetra(*c.points, count)
        elif c.type == pv.CellType.VOXEL:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_voxel(*c.points, count)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )

    return points
