"""Utilities."""

import numpy as np
import numpy.typing as npt

import pyransame


def _generate_points_in_tri(points: npt.NDArray, n: int = 1) -> npt.NDArray[np.float_]:
    a, b, c = points
    v1 = b - a
    v2 = c - a

    r = pyransame.rng.random(size=(n, 2))
    r = np.apply_along_axis(lambda ir: ir if ir.sum() <= 1.0 else 1.0 - ir, -1, r)

    return a + np.atleast_2d(r[:, 0]).T * v1 + np.atleast_2d(r[:, 1]).T * v2


def _area_tri(points: npt.NDArray) -> float:
    pa, pb, pc = points
    a = np.linalg.norm(pb - pa)
    b = np.linalg.norm(pc - pb)
    c = np.linalg.norm(pc - pa)
    return 1.0 / 4.0 * np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))


def _generate_points_in_tri_strip(
    points: npt.NDArray, n: int = 1
) -> npt.NDArray[np.float_]:
    ntri = points.shape[0] - 2

    areas = np.empty(shape=ntri, dtype=float)
    for i in range(ntri):
        areas[i] = _area_tri(points[i : i + 3, :])

    out = np.empty((n, 3))

    p = areas / areas.sum()
    r = pyransame.rng.choice(ntri, size=n, p=p)

    for i in range(n):
        out[i, :] = _generate_points_in_tri(points[r[i] : r[i] + 3, :])

    return out


def _generate_points_in_quad(points: npt.NDArray, n: int = 1) -> npt.NDArray[np.float_]:
    tri1 = [0, 1, 2]
    tri2 = [0, 2, 3]
    area1 = _area_tri(points[tri1, :])
    area2 = _area_tri(points[tri2, :])

    out = np.empty((n, 3))

    p = np.array([area1, area2])
    p = p / p.sum()
    r = pyransame.rng.choice(np.array([0, 1], dtype=int), size=n, p=p)

    for i in range(n):
        if r[i] == 0:
            out[i, :] = _generate_points_in_tri(points[tri1, :])
        else:
            out[i, :] = _generate_points_in_tri(points[tri2, :])
    return out


def _generate_points_in_polygon(points: npt.NDArray, n: int = 1) -> npt.NDArray:
    ntri = points.shape[0] - 2

    areas = np.empty(shape=ntri, dtype=float)
    for i in range(ntri):
        areas[i] = _area_tri(points[[0, i + 1, i + 2], :])

    out = np.empty((n, 3))

    p = areas / areas.sum()
    r = pyransame.rng.choice(ntri, size=n, p=p)

    for i in range(n):
        out[i, :] = _generate_points_in_tri(points[[0, r[i] + 1, r[i] + 2], :])

    return out


def _tetra_random_coordinates(r: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
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
    points: npt.NDArray, n: int = 1
) -> npt.NDArray[np.float_]:
    a, b, c, d = points
    v0 = b - a
    v1 = c - a
    v2 = d - a

    r = pyransame.rng.random(size=(n, 3))
    r = np.apply_along_axis(_tetra_random_coordinates, -1, r)

    return (
        a
        + np.atleast_2d(r[:, 0]).T * v0
        + np.atleast_2d(r[:, 1]).T * v1
        + np.atleast_2d(r[:, 2]).T * v2
    )


def _area_tetra(points: npt.NDArray) -> float:
    a = np.linalg.norm(points[0, :] - points[1, :])
    b = np.linalg.norm(points[0, :] - points[2, :])
    c = np.linalg.norm(points[0, :] - points[3, :])
    x = np.linalg.norm(points[2, :] - points[3, :])
    y = np.linalg.norm(points[1, :] - points[3, :])
    z = np.linalg.norm(points[1, :] - points[2, :])

    X = b**2 + c**2 - x**2
    Y = a**2 + c**2 - y**2
    Z = a**2 + b**2 - z**2

    return (
        np.sqrt(
            4 * a**2 * b**2 * c**2
            - a**2 * X**2
            - b**2 * Y**2
            - c**2 * Z**2
            + X * Y * Z
        )
        / 12.0
    )


def _generate_points_in_pyramid(
    points: npt.NDArray, n: int = 1
) -> npt.NDArray[np.float_]:
    tetra0 = [0, 1, 2, 4]
    tetra1 = [0, 2, 3, 4]

    area0 = _area_tetra(points[tetra0, :])
    area1 = _area_tetra(points[tetra1, :])

    areas = np.array([area0, area1])

    p = areas / areas.sum()
    r = pyransame.rng.choice(2, size=n, p=p)

    out = np.empty((n, 3))
    for i in range(n):
        if r[i] == 0:
            out[i, :] = _generate_points_in_tetra(points[tetra0, :])
        else:
            out[i, :] = _generate_points_in_tetra(points[tetra1, :])

    return out


def _generate_points_in_voxel(
    points: npt.NDArray,
    n: int = 1,
) -> npt.NDArray[np.float_]:
    a, b, c, _, e, _, _, _ = points
    v0 = b - a
    v1 = c - a
    v2 = e - a

    r = pyransame.rng.random(size=(n, 3))

    return (
        a
        + np.atleast_2d(r[:, 0]).T * v0
        + np.atleast_2d(r[:, 1]).T * v1
        + np.atleast_2d(r[:, 2]).T * v2
    )


def _generate_points_in_pixel(
    points: npt.NDArray,
    n: int = 1,
) -> npt.NDArray[np.float_]:
    a, b, c, _ = points
    v0 = b - a
    v1 = c - a

    r = pyransame.rng.random(size=(n, 2))

    return a + np.atleast_2d(r[:, 0]).T * v0 + np.atleast_2d(r[:, 1]).T * v1
