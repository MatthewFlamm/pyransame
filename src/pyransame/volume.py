"""Generating random points in a 3D volume."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv
from pyvista.core.celltype import CellType

import pyransame
import pyransame.util as util


def random_volume_points(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> np.ndarray:
    """
    Generate random points in a volume.

    Supported cell types:

    - Hexagonal Prism
    - Hexahedron
    - Pentagonal Prism
    - Polyhedron
    - Pyramid
    - Tetrahedron
    - Voxel
    - Wedge

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
    >>> mesh = pv.ImageData(dimensions=(11, 11, 11))
    >>> points = pyransame.random_volume_points(mesh, n=500)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, style='wireframe')
    >>> _ = pl.add_points(points, render_points_as_spheres=True, point_size=10.0, color='red')
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

    if "Volume" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # type: ignore

    p = weights * mesh["Volume"]

    if p.sum() == 0:
        raise ValueError("No cells with volume in DataSet")

    p = p / p.sum()

    chosen_cells, unique_counts, point_indices = util._random_cells(n_cells, n, p)

    points = np.empty((n, 3))
    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        c = mesh.get_cell(chosen_cell)

        if c.type == CellType.TETRA:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_tetra(c.points, count)
        elif c.type == CellType.VOXEL:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_voxel(c.points, count)
        elif c.type == CellType.PYRAMID:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_pyramid(c.points, count)
        elif c.type == CellType.WEDGE:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_wedge(c.points, count)
        elif c.type == CellType.HEXAHEDRON:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_hexahedron(c.points, count)
        elif c.type == CellType.PENTAGONAL_PRISM:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_pentagonal_prism(c.points, count)
        elif c.type == CellType.HEXAGONAL_PRISM:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_hexagonal_prism(c.points, count)
        elif c.type == CellType.POLYHEDRON:
            points[
                point_indices[i] : point_indices[i + 1], :
            ] = util._generate_points_in_polyhedron(c, count)
        else:
            raise NotImplementedError(
                f"Random generation for {c.type.name} not yet supported"
            )
    return points


def random_volume_dataset(
    mesh: pv.DataSet,
    n: int = 1,
    weights: Optional[Union[str, npt.ArrayLike]] = None,
) -> pv.PolyData:
    """
    Generate random points in a volume with sampled data.

    Supported cell types:

    - Hexagonal Prism
    - Hexahedron
    - Pentagonal Prism
    - Polyhedron
    - Pyramid
    - Tetrahedron
    - Voxel
    - Wedge

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
    mesh : pv.PolyData
        ``(n, 3)`` points that exist inside cells on ``mesh`` and with
        sampled data.

    Examples
    --------
    >>> import pyransame
    >>> import pyvista as pv
    >>> mesh = pv.ImageData(dimensions=(11, 11, 11))
    >>> mesh['y'] = mesh.points[:, 1]
    >>> points = pyransame.random_volume_dataset(mesh, n=500)

    Now plot result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, style='wireframe')
    >>> _ = pl.add_points(points, scalars='y', render_points_as_spheres=True, point_size=10.0, color='red')
    >>> pl.show()
    """
    points = random_volume_points(mesh, n, weights)
    return pv.PolyData(points).sample(mesh)
