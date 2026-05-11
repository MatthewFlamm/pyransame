"""Tests for the ``ransame`` PyVista dataset accessor."""

import numpy as np
import pytest
import pyvista as pv

import pyransame
from pyransame import RansameAccessor

ACCESSOR_NAME = "ransame"

pytestmark = pytest.mark.skipif(
    not hasattr(pv, "register_dataset_accessor"),
    reason="requires PyVista >= 0.48 dataset accessor API",
)


def test_accessor_registered():
    mesh = pv.Plane()
    assert hasattr(mesh, ACCESSOR_NAME)
    assert isinstance(mesh.ransame, RansameAccessor)


def test_accessor_cached_per_instance():
    mesh = pv.Plane()
    assert mesh.ransame is mesh.ransame


def test_surface_points():
    mesh = pv.Plane()
    points = mesh.ransame.surface_points(25)
    assert points.shape == (25, 3)


def test_surface_dataset_matches_function():
    mesh = pv.Plane()
    mesh["x"] = mesh.points[:, 0]

    pyransame.rng = np.random.default_rng(0)
    sampled_method = mesh.ransame.surface_dataset(50)

    pyransame.rng = np.random.default_rng(0)
    sampled_func = pyransame.random_surface_dataset(mesh, 50)

    assert np.allclose(sampled_method.points, sampled_func.points)
    assert np.allclose(sampled_method["x"], sampled_func["x"])


def test_volume_points_and_dataset():
    mesh = pv.ImageData(dimensions=(10, 10, 10))
    points = mesh.ransame.volume_points(20)
    assert points.shape == (20, 3)

    sampled = mesh.ransame.volume_dataset(20)
    assert sampled.n_points == 20


def test_line_points_and_dataset():
    mesh = pv.Line()
    points = mesh.ransame.line_points(15)
    assert points.shape == (15, 3)

    sampled = mesh.ransame.line_dataset(15)
    assert sampled.n_points == 15


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 3), reason="requires vtk version 9.3 or higher"
)
def test_vertex_points_and_dataset():
    mesh = pv.PolyData(pv.ImageData(dimensions=(5, 5, 5)).points)
    points = mesh.ransame.vertex_points(10)
    assert points.shape == (10, 3)

    sampled = mesh.ransame.vertex_dataset(10)
    assert sampled.n_points == 10


def test_dispatch_points_surface():
    mesh = pv.Plane()
    pts = mesh.ransame.points(20)
    assert pts.shape == (20, 3)


def test_dispatch_dataset_volume():
    mesh = pv.ImageData(dimensions=(8, 8, 8))
    sampled = mesh.ransame.dataset(15)
    assert sampled.n_points == 15


def test_dispatch_kind_override():
    mesh = pv.Plane()
    pts = mesh.ransame.points(10, kind="surface")
    assert pts.shape == (10, 3)


def test_dispatch_line_inferred():
    mesh = pv.Line()
    pts = mesh.ransame.points(7)
    assert pts.shape == (7, 3)


def test_dispatch_raises_on_mixed_dim():
    # PolyData carrying both line and triangle cells.
    plane = pv.Plane().triangulate()
    line = pv.Line()
    mixed = plane + line
    with pytest.raises(ValueError, match="multiple dimensions"):
        mixed.ransame.points(5)
    # Override is honored.
    pts = mixed.ransame.points(5, kind="surface")
    assert pts.shape == (5, 3)


def test_dispatch_raises_on_empty_mesh():
    mesh = pv.PolyData()
    with pytest.raises(ValueError, match="no cells"):
        mesh.ransame.points(1)


def test_accessor_available_on_subclasses():
    # Registering against ``DataSet`` should expose accessor on every
    # concrete subclass.
    for mesh in (
        pv.Plane(),
        pv.ImageData(dimensions=(4, 4, 4)),
        pv.Line(),
    ):
        assert isinstance(mesh.ransame, RansameAccessor)
