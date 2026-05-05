"""PyVista dataset accessor for ``pyransame``.

Registers a ``ransam`` accessor on :class:`pyvista.DataSet` so the
sampling routines can be invoked as methods on any PyVista mesh::

    >>> import pyvista as pv
    >>> import pyransame  # noqa: F401  (ensures accessor is registered)
    >>> mesh = pv.Plane()
    >>> points = mesh.ransam.surface_points(10)

Importing this module is a no-op on PyVista versions that predate the
dataset accessor API (added in PyVista 0.48). The package therefore
remains importable on older PyVista releases; the ``ransam`` namespace
will not be attached to datasets.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyransame.line import random_line_dataset, random_line_points
from pyransame.surface import random_surface_dataset, random_surface_points
from pyransame.vertex import random_vertex_dataset, random_vertex_points
from pyransame.volume import random_volume_dataset, random_volume_points

ACCESSOR_NAME = "ransam"


class RansameAccessor:
    """Accessor exposing :mod:`pyransame` sampling routines on a dataset.

    Available as ``dataset.ransam`` once :mod:`pyransame` is imported on
    PyVista >= 0.48.
    """

    def __init__(self, mesh: pv.DataSet) -> None:
        self._mesh = mesh

    def surface_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """Random points on 2D surface cells. See :func:`pyransame.random_surface_points`."""
        return random_surface_points(self._mesh, n=n, weights=weights)

    def surface_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """Random sampled :class:`pyvista.PolyData` on surface cells. See :func:`pyransame.random_surface_dataset`."""
        return random_surface_dataset(self._mesh, n=n, weights=weights)

    def volume_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """Random points in 3D volume cells. See :func:`pyransame.random_volume_points`."""
        return random_volume_points(self._mesh, n=n, weights=weights)

    def volume_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """Random sampled :class:`pyvista.PolyData` in volume cells. See :func:`pyransame.random_volume_dataset`."""
        return random_volume_dataset(self._mesh, n=n, weights=weights)

    def line_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """Random points on 1D line cells. See :func:`pyransame.random_line_points`."""
        return random_line_points(self._mesh, n=n, weights=weights)

    def line_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """Random sampled :class:`pyvista.PolyData` on line cells. See :func:`pyransame.random_line_dataset`."""
        return random_line_dataset(self._mesh, n=n, weights=weights)

    def vertex_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """Random points sampled from 0D vertex cells. See :func:`pyransame.random_vertex_points`."""
        return random_vertex_points(self._mesh, n=n, weights=weights)

    def vertex_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """Random sampled :class:`pyvista.PolyData` from vertex cells. See :func:`pyransame.random_vertex_dataset`."""
        return random_vertex_dataset(self._mesh, n=n, weights=weights)


def _register() -> bool:
    """Attach the accessor to :class:`pyvista.DataSet`.

    Returns ``True`` if registration succeeded, ``False`` if the running
    PyVista does not support dataset accessors (<0.48) or if registration
    raised because the accessor was already attached (e.g. during a
    re-import in tests).
    """
    register = getattr(pv, "register_dataset_accessor", None)
    if register is None:
        return False
    try:
        register(ACCESSOR_NAME, pv.DataSet)(RansameAccessor)
    except ValueError:
        # Already registered (e.g. plugin imported twice).
        return False
    return True


_register()
