"""Python package for random sampling of meshes."""

from numpy.random import default_rng

rng = default_rng()

# Register the ``ransam`` dataset accessor when the running PyVista
# supports it (>= 0.48). Importing the module is a no-op otherwise so
# the package stays importable on older PyVista releases.
from .accessor import RansameAccessor  # noqa: E402
from .line import random_line_dataset, random_line_points
from .surface import random_surface_dataset, random_surface_points
from .vertex import random_vertex_dataset, random_vertex_points
from .volume import random_volume_dataset, random_volume_points
