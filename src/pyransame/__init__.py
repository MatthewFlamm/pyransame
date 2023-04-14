"""Python package for random sampling of meshes."""
from collections.abc import Sequence

import numpy as np
import pyvista as pv

from .util import _generate_points_in_tri


def random_surface_points(mesh, n, weights=None):
    if not isinstance(mesh, pv.PolyData):
        raise ValueError(f"mesh must by PolyData got {type(mesh)}")

    if weights is None:
        weights = np.ones(mesh.n_cells)

    if not isinstance(weights, (np.ndarray, Sequence, str)):
        raise ValueError("Invalid weights, got weights")
    
    if not isinstance(weights, str):
        mesh["weights"] = weights
        weights = "weights"

    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n_cells = mesh.n_cells

    if "Area" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes(length=False, volume=False)

    p = mesh["weights"] * mesh["Area"]

    p = p / p.sum()

    # TODO: Use modern numpy rng
    chosen = np.random.choice(n_cells, n, p=p)

    points = np.empty((n, 3))
    for i in range(n):
        # TODO: assess whether gathering unique points first improves speed 
        # TODO: when there are many points per cell
        points[i, :] = _generate_points_in_tri(*mesh.get_cell(chosen[i]).points)

    return points
