"""Utilities."""

import numpy as np


def _generate_points_in_tri(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int=1) -> np.ndarray:
    v1 = b - a
    v2 = c - a
    
    points = np.empty((n, 3), dtype=float)

    for i in range(n):
        r1, r2 = 100.0, 100.0
        while r1 + r2 > 1.0:
            # TODO: Use modern numpy rng
            r1, r2 = np.random.random(), np.random.random()
        points[i, :] = a + r1 * v1 + r2 * v2
    return points

def _is_point_inside_triangle(v1: np.ndarray, v2: np.ndarray, p: np.ndarray) -> bool:

    denom = np.cross(v1, v2)

    a = np.cross(p, v1) / denom
    b = np.cross(p, v2) / denom

    if a < 0 or b < 0 or a + b > 1:
        return False
    return True