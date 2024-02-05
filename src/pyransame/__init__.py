"""Python package for random sampling of meshes."""

from numpy.random import default_rng

rng = default_rng()

from .line import random_line_dataset, random_line_points
from .surface import random_surface_dataset, random_surface_points
from .volume import random_volume_dataset, random_volume_points
