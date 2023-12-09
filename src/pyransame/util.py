"""Utilities."""

from typing import Tuple

import numpy as np
import pyvista as pv
from vtk import mutable  # type: ignore

import pyransame


def _random_cells(
    n_cells: int, n: int, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = pyransame.rng.choice(n_cells, n, p=p)
    chosen_cells, unique_counts = np.unique(chosen, return_counts=True)
    point_indices = np.zeros(shape=chosen_cells.size + 1, dtype=int)
    point_indices[1:] = np.cumsum(unique_counts)
    return chosen_cells, unique_counts, point_indices


def _generate_points_in_tri(points: np.ndarray, n: int = 1) -> np.ndarray:
    a, b, c = points
    v1 = b - a
    v2 = c - a

    r = pyransame.rng.random(size=(n, 2))
    r = np.apply_along_axis(lambda ir: ir if ir.sum() <= 1.0 else 1.0 - ir, -1, r)

    return a + np.atleast_2d(r[:, 0]).T * v1 + np.atleast_2d(r[:, 1]).T * v2


def _area_tri(points: np.ndarray) -> float:
    pa, pb, pc = points
    a = np.linalg.norm(pb - pa)
    b = np.linalg.norm(pc - pb)
    c = np.linalg.norm(pc - pa)
    return 1.0 / 4.0 * np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))


def _generate_points_in_tri_strip(points: np.ndarray, n: int = 1) -> np.ndarray:
    ntri = points.shape[0] - 2

    areas = np.empty(shape=ntri, dtype=float)
    for i in range(ntri):
        areas[i] = _area_tri(points[i : i + 3, :])

    out = np.empty((n, 3))

    p = areas / areas.sum()

    chosen_cells, unique_counts, point_indices = _random_cells(ntri, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tri(
            points[chosen_cell : chosen_cell + 3, :], n=count
        )

    return out


def _generate_points_in_quad(points: np.ndarray, n: int = 1) -> np.ndarray:
    tri1 = [0, 1, 2]
    tri2 = [0, 2, 3]
    area1 = _area_tri(points[tri1, :])
    area2 = _area_tri(points[tri2, :])

    out = np.empty((n, 3))

    p = np.array([area1, area2])
    p = p / p.sum()

    chosen_cells, unique_counts, point_indices = _random_cells(2, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        if chosen_cell == 0:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tri(
                points[tri1, :], n=count
            )
        else:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tri(
                points[tri2, :], n=count
            )
    return out


def _generate_points_in_polygon(points: np.ndarray, n: int = 1) -> np.ndarray:
    ntri = points.shape[0] - 2

    areas = np.empty(shape=ntri, dtype=float)
    for i in range(ntri):
        areas[i] = _area_tri(points[[0, i + 1, i + 2], :])

    out = np.empty((n, 3))

    p = areas / areas.sum()

    chosen_cells, unique_counts, point_indices = _random_cells(ntri, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tri(
            points[[0, chosen_cell + 1, chosen_cell + 2], :], n=count
        )

    return out


def _tetra_random_coordinates(r: np.ndarray) -> np.ndarray:
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


def _generate_points_in_tetra(points: np.ndarray, n: int = 1) -> np.ndarray:
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


def _area_tetra(points: np.ndarray) -> float:
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


def _area_pyramid(points: np.ndarray) -> float:
    tetra0 = [0, 1, 2, 4]
    tetra1 = [0, 2, 3, 4]

    area0 = _area_tetra(points[tetra0, :])
    area1 = _area_tetra(points[tetra1, :])

    return area0 + area1


def _generate_points_in_pyramid(points: np.ndarray, n: int = 1) -> np.ndarray:
    tetra0 = [0, 1, 2, 4]
    tetra1 = [0, 2, 3, 4]

    area0 = _area_tetra(points[tetra0, :])
    area1 = _area_tetra(points[tetra1, :])

    areas = np.array([area0, area1])

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(2, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        if chosen_cell == 0:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tetra(
                points[tetra0, :], n=count
            )
        else:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tetra(
                points[tetra1, :], n=count
            )

    return out


def _generate_points_in_voxel(
    points: np.ndarray,
    n: int = 1,
) -> np.ndarray:
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
    points: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    a, b, c, _ = points
    v0 = b - a
    v1 = c - a

    r = pyransame.rng.random(size=(n, 2))

    return a + np.atleast_2d(r[:, 0]).T * v0 + np.atleast_2d(r[:, 1]).T * v1


def _area_wedge(points: np.ndarray) -> float:
    tetra = [0, 2, 1, 4]
    pyramid = [0, 2, 5, 3, 4]

    return _area_tetra(points[tetra, :]) + _area_pyramid(points[pyramid, :])


def _generate_points_in_wedge(points: np.ndarray, n: int = 1) -> np.ndarray:
    tetra = [0, 2, 1, 4]
    pyramid = [0, 2, 5, 3, 4]

    area0 = _area_tetra(points[tetra, :])
    area1 = _area_pyramid(points[pyramid, :])

    areas = np.array([area0, area1])

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(2, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        if chosen_cell == 0:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tetra(
                points[tetra, :], n=count
            )
        else:
            out[
                point_indices[i] : point_indices[i + 1], :
            ] = _generate_points_in_pyramid(points[pyramid, :], n=count)

    return out


def _area_hexahedron(points: np.ndarray) -> float:
    tetras = [
        [0, 1, 4, 3],
        [3, 7, 6, 4],
        [1, 5, 4, 6],
        [2, 3, 6, 1],
        [3, 1, 6, 4],
    ]
    return np.array([_area_tetra(points[tetra, :]) for tetra in tetras]).sum()


def _generate_points_in_hexahedron(points: np.ndarray, n: int = 1) -> np.ndarray:
    tetras = [
        [0, 1, 4, 3],
        [3, 7, 6, 4],
        [1, 5, 4, 6],
        [2, 3, 6, 1],
        [3, 1, 6, 4],
    ]

    areas = np.array([_area_tetra(points[tetra, :]) for tetra in tetras])

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(5, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tetra(
            points[tetras[chosen_cell], :], n=count
        )

    return out


def _generate_points_in_pentagonal_prism(points: np.ndarray, n: int = 1) -> np.ndarray:
    wedge = [2, 4, 3, 7, 9, 8]
    hexahedron = [0, 1, 2, 4, 5, 6, 7, 9]

    areas = np.array(
        [_area_wedge(points[wedge, :]), _area_hexahedron(points[hexahedron, :])]
    )

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(2, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        if chosen_cell == 0:
            out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_wedge(
                points[wedge, :], n=count
            )
        else:
            out[
                point_indices[i] : point_indices[i + 1], :
            ] = _generate_points_in_hexahedron(points[hexahedron, :], n=count)

    return out


def _generate_points_in_hexagonal_prism(points: np.ndarray, n: int = 1) -> np.ndarray:
    hexahedron0 = [0, 1, 2, 5, 6, 7, 8, 11]
    hexahedron1 = [5, 2, 3, 4, 11, 8, 9, 10]

    areas = np.array(
        [
            _area_hexahedron(points[hexahedron0, :]),
            _area_hexahedron(points[hexahedron1, :]),
        ]
    )

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(2, n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        if chosen_cell == 0:
            out[
                point_indices[i] : point_indices[i + 1], :
            ] = _generate_points_in_hexahedron(points[hexahedron0, :], n=count)
        else:
            out[
                point_indices[i] : point_indices[i + 1], :
            ] = _generate_points_in_hexahedron(points[hexahedron1, :], n=count)

    return out


def _generate_points_in_polyhedron(cell: pv.Cell, n: int = 1) -> np.ndarray:
    faces = cell.faces

    # version >0.39 pyvista could be used in the future
    para_center = [0.0, 0.0, 0.0]
    sub_id = cell.GetParametricCenter(para_center)
    # EvaluateLocation requires mutable sub_id
    sub_id = mutable(sub_id)
    # center and weights are returned from EvaluateLocation
    cell_center = [0.0, 0.0, 0.0]
    weights = [0.0] * cell.n_points
    cell.EvaluateLocation(sub_id, para_center, cell_center, weights)

    tetras = []

    for face in faces:
        face_points = face.points
        ntri = face_points.shape[0] - 2
        for i in range(ntri):
            tetra = face_points[[0, i + 1, i + 2], :]
            tetra = np.append(tetra, [cell_center], axis=0)
            tetras.append(tetra)

    areas = np.array([_area_tetra(tetra) for tetra in tetras])

    p = areas / areas.sum()
    out = np.empty((n, 3))

    chosen_cells, unique_counts, point_indices = _random_cells(len(tetras), n, p)

    for i, (chosen_cell, count) in enumerate(zip(chosen_cells, unique_counts)):
        out[point_indices[i] : point_indices[i + 1], :] = _generate_points_in_tetra(
            tetras[chosen_cell], n=count
        )

    return out


def _generate_points_in_line(points: np.ndarray, n: int = 1):
    a, b = points
    r = pyransame.rng.random(size=(1, n))
    return a + (b - a) * np.atleast_2d(r).T
