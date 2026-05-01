# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import pathlib
import sys

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# see https://pypi.org/project/setuptools-scm/ for details
from importlib.metadata import version as _get_version, PackageNotFoundError


print("python exec:", sys.executable)
print("sys.path:", sys.path)
root = pathlib.Path(__file__).parent.parent.absolute()
os.environ["PYTHONPATH"] = str(root)
sys.path.insert(0, str(root))

import ocetrac  # isort:skip

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ocetrac"
copyright = "2025, ocetrac"
author = "ocetrac"
try:
    release = _get_version("ocetrac")
except PackageNotFoundError:
    release = "unknown"
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

autosummary_generate = True

autodoc_default_flags = ["members", "inherited-members"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Changed themes
html_theme = "pydata_sphinx_theme"  # "pangeo", "alabaster"
html_static_path = ["_static"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo": {
        "image_light": "img/tranparent_logo.png",
        "image_dark":  "img/tranparent_logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url":  "https://github.com/ocetrac/ocetrac",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url":  "https://pypi.org/project/ocetrac",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "footer_start": ["copyright"],
    "footer_end":   ["sphinx-version"],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "img/tranparent_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Output file base name for HTML help builder.
htmlhelp_basename = "ReadtheDocsTemplatedoc"

# -- nbsphinx specific options ----------------------------------------------
# this allows notebooks to be run even if they produce errors.
nbsphinx_allow_errors = True
