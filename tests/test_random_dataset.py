"""Tests for random_*_dataset."""

import numpy as np
import pytest
import pyvista as pv

import pyransame


def test_random_surface_dataset():
    mesh = pv.Plane()
    mesh["x"] = mesh.points[:, 0]
    mesh["y"] = mesh.points[:, 1]
    mesh["z"] = mesh.points[:, 2]

    sampled = pyransame.random_surface_dataset(mesh, 50)
    assert np.allclose(sampled.points[:, 0], sampled["x"])
    assert np.allclose(sampled.points[:, 1], sampled["y"])
    assert np.allclose(sampled.points[:, 2], sampled["z"])


def test_random_volume_dataset():
    mesh = pv.ImageData(dimensions=(10, 10, 10))
    mesh["x"] = mesh.points[:, 0]
    mesh["y"] = mesh.points[:, 1]
    mesh["z"] = mesh.points[:, 2]

    sampled = pyransame.random_volume_dataset(mesh, 50)
    assert np.allclose(sampled.points[:, 0], sampled["x"])
    assert np.allclose(sampled.points[:, 1], sampled["y"])
    assert np.allclose(sampled.points[:, 2], sampled["z"])


def test_random_line_dataset():
    mesh = pv.Line()
    mesh["x"] = mesh.points[:, 0]
    mesh["y"] = mesh.points[:, 1]
    mesh["z"] = mesh.points[:, 2]

    sampled = pyransame.random_line_dataset(mesh, 50)
    assert np.allclose(sampled.points[:, 0], sampled["x"])
    assert np.allclose(sampled.points[:, 1], sampled["y"])
    assert np.allclose(sampled.points[:, 2], sampled["z"])


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 3), reason="requires vtk ve4sion 9.3 or higher"
)
def test_random_vertex_dataset():
    mesh = pv.ImageData(dimensions=(10, 10, 10))
    mesh = pv.PolyData(mesh.points)  # make all vertex cells
    mesh.point_data["x"] = mesh.points[:, 0]
    mesh.point_data["y"] = mesh.points[:, 1]
    mesh.point_data["z"] = mesh.points[:, 2]

    sampled = pyransame.random_vertex_dataset(mesh, 50)
    assert np.allclose(sampled.points[:, 0], sampled["x"])
    assert np.allclose(sampled.points[:, 1], sampled["y"])
    assert np.allclose(sampled.points[:, 2], sampled["z"])
