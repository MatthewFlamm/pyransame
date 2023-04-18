"""Utilities."""

import numpy as np

rng = np.random.default_rng()


def _generate_points_in_tri(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int = 1
) -> np.ndarray:
    v1 = b - a
    v2 = c - a

    r = rng.random(size=(n, 2))
    r = np.apply_along_axis(lambda ir: ir if ir.sum() <= 1.0 else 1.0 - ir, -1, r)

    points = a + np.atleast_2d(r[:, 0]).T * v1 + np.atleast_2d(r[:, 1]).T * v2
    return points


def _tetra_random_coordinates(r: np.ndarray):
    if r[0:2].sum() > 1.0:
        r[0:2] = 1.0 - r[0:2]
    if r[1:].sum() > 1.0:
        tmp = r[2]
        r[2] = 1.0 - r[0:2].sum()
        r[0] = tmp
    elif r.sum() > 1.0:
        tmp = r[2]
        r[2] = r.sum() - 1.0
        r[0] = 1.0 - r[1] - tmp
    return r
        


def _generate_points_in_tetra(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int = 1
) -> np.ndarray:

    r = rng.random(size=(n, 3))
    r = np.apply_along_axis(_tetra_random_coordinates, -1, r)
    r0 = 1 - np.sum(r, axis=-1)

    points = np.atleast_2d(r0).T * a + np.atleast_2d(r[:, 0]).T * b + np.atleast_2d(r[:, 1]).T * c + np.atleast_2d(r[:, 2]).T * d
    return points
