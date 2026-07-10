# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# Pyccolo is pip-installed in the Read the Docs build (see .readthedocs.yml), so
# no sys.path manipulation is needed for autodoc to import it. For a local
# ``make html`` from a source checkout, an editable install (``pip install -e
# '.[docs]'``) makes ``import pyccolo`` resolve just the same.

# The bundled ``_ext`` directory holds the ``pyccolo-events`` extension, which
# renders the event taxonomy straight from ``pyccolo.trace_events``.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext"))

# -- Project information -----------------------------------------------------

project = "pyccolo"
copyright = "2020, Stephen Macke"
author = "Stephen Macke"

# The full version, including alpha/beta/rc tags, and the short X.Y version.
try:
    from pyccolo import __version__ as release
except Exception:
    release = ""
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# ref: https://stackoverflow.com/questions/56336234/build-fail-sphinx-error-contents-rst-not-found
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "sphinx_copybutton",
    "pyccolo_events",
]

# Every worked example in the guides/tutorials is a runnable ``.. testcode::``
# block executed by ``make -C docs doctest`` (wired into CI), so the docs cannot
# silently drift from the library. These two imports are in scope for every
# snippet, matching how a reader would ``import`` at the top of a file.
doctest_global_setup = "import ast\nimport pyccolo as pyc"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Link out to the CPython docs (pyccolo references the ``ast`` module heavily).
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Keep autodoc output in source order rather than alphabetized.
autodoc_member_order = "bysource"

# sphinx-argparse's ``ArgParseDomain`` (>=0.5) does not implement
# ``merge_domaindata``, so Sphinx's parallel read path (Read the Docs builds
# with ``-j auto``) aborts with ``NotImplementedError``. Its domain data is just
# a list of command tuples plus a group->tuples dict, both keyed on docname at
# index 3, so a correct merge is a couple of lines. Patch it in if absent.
try:
    from sphinxarg.ext import ArgParseDomain as _ArgParseDomain
except Exception:  # pragma: no cover - sphinxarg always present in docs builds
    _ArgParseDomain = None

if _ArgParseDomain is not None and "merge_domaindata" not in vars(_ArgParseDomain):

    def _merge_domaindata(self, docnames, otherdata):
        wanted = set(docnames)
        for entry in otherdata.get("commands", []):
            if entry[3] in wanted:
                self.data["commands"].append(entry)
        for group, entries in otherdata.get("commands-by-group", {}).items():
            bucket = self.data["commands-by-group"].setdefault(group, [])
            bucket.extend(entry for entry in entries if entry[3] in wanted)

    _ArgParseDomain.merge_domaindata = _merge_domaindata


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_title = f"Pyccolo {version}" if version else "Pyccolo"

# Read the Docs theme options: keep the full nav tree expanded and deep enough
# to reach the Diátaxis subsections. See
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
