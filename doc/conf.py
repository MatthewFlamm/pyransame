# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "pyransame"
copyright = "2023, Matthew Flamm"
author = "Matthew Flamm"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "pyvista.ext.plot_directive",
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
