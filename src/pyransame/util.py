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


def _area_tri(pa, pb, pc):
    a = np.linalg.norm(pb - pa)
    b = np.linalg.norm(pc - pb)
    c = np.linalg.norm(pc - pa)
    return 1.0 / 4.0 * np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))


def _generate_points_in_quad(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int = 1
) -> np.ndarray:
    area1 = _area_tri(a, b, c)
    area2 = _area_tri(a, c, d)

    points = np.empty((n, 3))

    p = np.array([area1, area2])
    p = p / p.sum()
    r = rng.choice(np.array([0, 1], dtype=int), size=n, p=p)

    for i in range(n):
        if r[i] == 0:
            points[i, :] = _generate_points_in_tri(a, b, c)
        else:
            points[i, :] = _generate_points_in_tri(a, c, d)
    return points


def _tetra_random_coordinates(r: np.ndarray):
    if r[0:2].sum() > 1.0:
        r[0:2] = 1.0 - r[0:2]

    if r.sum() > 1.0:
        if r[1:].sum() > 1.0:
            tmp = r[2].copy()
            r[2] = 1 - r[0] - r[1]
            r[1] = 1 - tmp
        else:
            tmp = r[2].copy()
            r[2] = r.sum() - 1.0
            r[0] = 1.0 - r[1] - tmp
    return r


def _generate_points_in_tetra(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int = 1
) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2 = d - a

    r = rng.random(size=(n, 3))
    r = np.apply_along_axis(_tetra_random_coordinates, -1, r)

    points = (
        a
        + np.atleast_2d(r[:, 0]).T * v0
        + np.atleast_2d(r[:, 1]).T * v1
        + np.atleast_2d(r[:, 2]).T * v2
    )
    return points


def _generate_points_in_voxel(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2 = e - a

    r = rng.random(size=(n, 3))

    points = (
        a
        + np.atleast_2d(r[:, 0]).T * v0
        + np.atleast_2d(r[:, 1]).T * v1
        + np.atleast_2d(r[:, 2]).T * v2
    )
    return points


def _generate_points_in_pixel(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    v0 = b - a
    v1 = c - a

    r = rng.random(size=(n, 2))

    points = a + np.atleast_2d(r[:, 0]).T * v0 + np.atleast_2d(r[:, 1]).T * v1
    return points
