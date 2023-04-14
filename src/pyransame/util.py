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
