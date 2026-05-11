"""PyVista dataset accessor for ``pyransame``.

Registers a ``ransame`` accessor on :class:`pyvista.DataSet` so the
sampling routines can be invoked as methods on any PyVista mesh::

    >>> import pyvista as pv
    >>> import pyransame  # noqa: F401  (ensures accessor is registered)
    >>> mesh = pv.Plane()
    >>> points = mesh.ransame.surface_points(10)

Importing this module is a no-op on PyVista versions that predate the
dataset accessor API (added in PyVista 0.48). The package therefore
remains importable on older PyVista releases; the ``ransame`` namespace
will not be attached to datasets.
"""

from typing import Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyransame.line import random_line_dataset, random_line_points
from pyransame.surface import random_surface_dataset, random_surface_points
from pyransame.vertex import random_vertex_dataset, random_vertex_points
from pyransame.volume import random_volume_dataset, random_volume_points

ACCESSOR_NAME = "ransame"

Kind = Literal["vertex", "line", "surface", "volume"]
_DIM_TO_KIND: dict[int, Kind] = {
    0: "vertex",
    1: "line",
    2: "surface",
    3: "volume",
}


class RansameAccessor:
    """
    Accessor exposing :mod:`pyransame` sampling routines on a dataset.

    Available as ``dataset.ransame`` once :mod:`pyransame` is imported on
    PyVista >= 0.48. Each method forwards to the corresponding
    top-level ``random_*`` function in :mod:`pyransame`.
    """

    def __init__(self, mesh: pv.DataSet) -> None:
        self._mesh = mesh

    def surface_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """
        Random points on 2D surface cells.

        Forwards to :func:`pyransame.random_surface_points`. See that
        function for the full parameter description and supported cell
        types.
        """
        return random_surface_points(self._mesh, n=n, weights=weights)

    def surface_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """
        Random sampled :class:`pyvista.PolyData` on surface cells.

        Forwards to :func:`pyransame.random_surface_dataset`, which
        returns a :class:`pyvista.PolyData` with point data interpolated
        from the input mesh.
        """
        return random_surface_dataset(self._mesh, n=n, weights=weights)

    def volume_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """
        Random points in 3D volume cells.

        Forwards to :func:`pyransame.random_volume_points`. See that
        function for the full parameter description and supported cell
        types.
        """
        return random_volume_points(self._mesh, n=n, weights=weights)

    def volume_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """
        Random sampled :class:`pyvista.PolyData` in volume cells.

        Forwards to :func:`pyransame.random_volume_dataset`, which
        returns a :class:`pyvista.PolyData` with point data interpolated
        from the input mesh.
        """
        return random_volume_dataset(self._mesh, n=n, weights=weights)

    def line_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """
        Random points on 1D line cells.

        Forwards to :func:`pyransame.random_line_points`. See that
        function for the full parameter description and supported cell
        types.
        """
        return random_line_points(self._mesh, n=n, weights=weights)

    def line_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """
        Random sampled :class:`pyvista.PolyData` on line cells.

        Forwards to :func:`pyransame.random_line_dataset`, which returns
        a :class:`pyvista.PolyData` with point data interpolated from
        the input mesh.
        """
        return random_line_dataset(self._mesh, n=n, weights=weights)

    def vertex_points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> np.ndarray:
        """
        Random points sampled from 0D vertex cells.

        Forwards to :func:`pyransame.random_vertex_points`. See that
        function for the full parameter description and supported cell
        types.
        """
        return random_vertex_points(self._mesh, n=n, weights=weights)

    def vertex_dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
    ) -> pv.PolyData:
        """
        Random sampled :class:`pyvista.PolyData` from vertex cells.

        Forwards to :func:`pyransame.random_vertex_dataset`, which
        returns a :class:`pyvista.PolyData` with point data interpolated
        from the input mesh.
        """
        return random_vertex_dataset(self._mesh, n=n, weights=weights)

    def _infer_kind(self) -> Kind:
        mesh = self._mesh
        if mesh.n_cells == 0:
            msg = "Mesh has no cells; cannot infer sampling kind."
            raise ValueError(msg)
        lo = mesh.min_cell_dimensionality
        hi = mesh.max_cell_dimensionality
        if lo != hi:
            dims = sorted({ct.dimension for ct in mesh.distinct_cell_types})
            kinds = sorted(_DIM_TO_KIND[d] for d in dims if d in _DIM_TO_KIND)
            msg = (
                f"Mesh has cells of multiple dimensions ({dims}, "
                f"kinds {kinds}); pass ``kind=`` to disambiguate."
            )
            raise ValueError(msg)
        if lo not in _DIM_TO_KIND:  # pragma: no cover - defensive guard
            msg = f"Unsupported cell dimension {lo}; pass ``kind=`` explicitly."
            raise ValueError(msg)
        return _DIM_TO_KIND[lo]

    def points(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
        *,
        kind: Optional[Kind] = None,
    ) -> np.ndarray:
        """
        Random points, dispatched by cell dimension.

        Inspects the mesh's distinct cell types and forwards to the
        matching ``random_*_points`` function. Pass ``kind`` explicitly
        to override the inferred dimension on mixed-dimension meshes.

        Parameters
        ----------
        n : int, default: 1
            Number of random points to generate.

        weights : str or array_like, optional
            Per-cell sampling weights, forwarded to the underlying
            ``random_*_points`` routine.

        kind : {"vertex", "line", "surface", "volume"}, optional
            Force a specific sampler. When ``None`` (the default), the
            kind is inferred from the cell dimensions present on the
            mesh and a :class:`ValueError` is raised if more than one
            dimension is found.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)`` array of sampled points.

        Examples
        --------
        >>> import pyransame  # noqa: F401  (registers ``ransame``)
        >>> import pyvista as pv
        >>> pts = pv.Plane().ransame.points(5)
        >>> pts.shape
        (5, 3)
        """
        resolved = kind if kind is not None else self._infer_kind()
        return _POINTS_DISPATCH[resolved](self._mesh, n=n, weights=weights)

    def dataset(
        self,
        n: int = 1,
        weights: Optional[Union[str, npt.ArrayLike]] = None,
        *,
        kind: Optional[Kind] = None,
    ) -> pv.PolyData:
        """
        Random sampled :class:`pyvista.PolyData`, dispatched by cell dimension.

        Inspects the mesh's distinct cell types and forwards to the
        matching ``random_*_dataset`` function. Pass ``kind`` explicitly
        to override the inferred dimension on mixed-dimension meshes.

        Parameters
        ----------
        n : int, default: 1
            Number of random points to generate.

        weights : str or array_like, optional
            Per-cell sampling weights, forwarded to the underlying
            ``random_*_dataset`` routine.

        kind : {"vertex", "line", "surface", "volume"}, optional
            Force a specific sampler. When ``None`` (the default), the
            kind is inferred from the cell dimensions present on the
            mesh and a :class:`ValueError` is raised if more than one
            dimension is found.

        Returns
        -------
        pyvista.PolyData
            Sampled mesh with point data interpolated from the input.

        Examples
        --------
        >>> import pyransame  # noqa: F401  (registers ``ransame``)
        >>> import pyvista as pv
        >>> sampled = pv.Plane().ransame.dataset(5)
        >>> sampled.n_points
        5
        """
        resolved = kind if kind is not None else self._infer_kind()
        return _DATASET_DISPATCH[resolved](self._mesh, n=n, weights=weights)


_POINTS_DISPATCH: dict[Kind, Callable[..., np.ndarray]] = {
    "vertex": random_vertex_points,
    "line": random_line_points,
    "surface": random_surface_points,
    "volume": random_volume_points,
}

_DATASET_DISPATCH: dict[Kind, Callable[..., pv.PolyData]] = {
    "vertex": random_vertex_dataset,
    "line": random_line_dataset,
    "surface": random_surface_dataset,
    "volume": random_volume_dataset,
}


def _register() -> bool:
    """Attach the accessor to :class:`pyvista.DataSet`.

    Returns ``True`` if registration succeeded, ``False`` if the running
    PyVista does not support dataset accessors (<0.48) or if registration
    raised because the accessor was already attached (e.g. during a
    re-import in tests).
    """
    register = getattr(pv, "register_dataset_accessor", None)
    if register is None:  # pragma: no cover - exercised only on PyVista < 0.48
        return False
    try:
        register(ACCESSOR_NAME, pv.DataSet)(RansameAccessor)
    except ValueError:  # pragma: no cover - double-registration edge case
        # Already registered (e.g. plugin imported twice).
        return False
    return True


_register()
