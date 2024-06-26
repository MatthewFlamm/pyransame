# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = "pyransame"
copyright = "2023, Matthew Flamm"
author = "Matthew Flamm"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "jupyter_sphinx",
    "numpydoc",
    "sphinx.ext.autosummary",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {"github_url": "https://github.com/MatthewFlamm/pyransame"}

import pyvista

pyvista.OFF_SCREEN = True

pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False

numpydoc_validation_checks = {
    "all",
    "RT02",  # name for single return
    "SA01",  # do not require see also
}

import re

# from pyvista
# -- .. pyvista-plot:: directive ----------------------------------------------
from numpydoc.docscrape_sphinx import SphinxDocString

IMPORT_PYVISTA_RE = r"\b(import +pyvista|from +pyvista +import)\b"
IMPORT_MATPLOTLIB_RE = r"\b(import +matplotlib|from +matplotlib +import)\b"

plot_setup = """
from pyvista import set_plot_theme as __s_p_t
__s_p_t('document')
del __s_p_t
"""
plot_cleanup = plot_setup


def _str_examples(self):
    examples_str = "\n".join(self["Examples"])

    if (
        self.use_plots
        and re.search(IMPORT_MATPLOTLIB_RE, examples_str)
        and "plot::" not in examples_str
    ):
        out = []
        out += self._str_header("Examples")
        out += [".. plot::", ""]
        out += self._str_indent(self["Examples"])
        out += [""]
        return out
    elif (
        re.search(IMPORT_PYVISTA_RE, examples_str)
        and "plot-pyvista::" not in examples_str
    ):
        out = []
        out += self._str_header("Examples")
        out += [".. pyvista-plot::", ""]
        out += self._str_indent(self["Examples"])
        out += [""]
        return out
    else:
        return self._str_section("Examples")


SphinxDocString._str_examples = _str_examples

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"

sys.path.insert(0, os.path.dirname(__file__))

# print(sys.path)

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["../examples_src/"],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Remove sphinx configuration comments from code blocks
    "remove_config_comments": True,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": "FileNameSortKey",
    "image_scrapers": "sphinxext.dynamic_scraper",
}
