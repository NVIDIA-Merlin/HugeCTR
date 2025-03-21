"""
 Copyright (c) 2023, NVIDIA CORPORATION.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import shutil
import sys
import zlib

from natsort import natsorted
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.abspath("../.."))

repodir = os.path.abspath(os.path.join(__file__, r"../../.."))
gitdir = os.path.join(repodir, r".git")

# -- Project information -----------------------------------------------------

project = "Merlin HugeCTR"
copyright = "2022, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_external_toc",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
    "sphinxcontrib.copydirs",
]

# MyST configuration settings
external_toc_path = "toc.yaml"
myst_enable_extensions = [
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "tasklist",
    "dollarmath",
]
myst_linkify_fuzzy_links = False
myst_heading_anchors = 4
nb_execution_mode = "off"

# Some documents are RST and include `.. toctree::` directives.
suppress_warnings = ["etoc.toctree", "myst.header", "misc.highlighting_failure"]

# Stopgap solution for the moment. Ignore warnings about links to directories.
# In README.md files, the following links make sense while browsing GitHub.
# In HTML, less so.
nitpicky = True
nitpick_ignore = [
    (r"myst", r"./multi-modal-data/"),
]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "notebooks/prototype_indices.ipynb",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "titles_only": True,
    "analytics_id": "G-NVJ1Y1YJHK",
}
html_copy_source = False
html_show_sourcelink = False

# Whitelist pattern for tags (set to None to ignore all tags)
# Determine if Sphinx is reading conf.py from the checked out
# repo (a Git repo) vs SMV reading conf.py from an archive of the repo
# at a commit (not a Git repo).
if os.path.exists(gitdir):
    tag_refs = (
        subprocess.check_output(["git", "tag", "-l", "v*"]).decode("utf-8").split()
    )
    tag_refs = natsorted(tag_refs)[-6:]
    smv_tag_whitelist = r"^(" + r"|".join(tag_refs) + r")$"
else:
    # SMV is reading conf.py from a Git archive of the repo at a specific commit.
    smv_tag_whitelist = r"^v.*$"

# Only include main branch for now
smv_branch_whitelist = "^main$"

smv_refs_override_suffix = "-docs"

# Sidestep the build warning for Keras
keras_inv = b'''\
# Sphinx inventory version 2
# Project: Keras
# Version: 2.9.0
# The remainder of this file is compressed with zlib.
''' + zlib.compress(b'''\
keras.engine.base_layer.Layer py:class 1 layers/base_layer/index.html#layer-class -
''')

inv_dir = TemporaryDirectory()
inv_fname = os.path.join(inv_dir.name, "objects.inv")

with open(inv_fname, "wb") as f:
    f.write(keras_inv)
    f.close()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "merlin-core": ("https://nvidia-merlin.github.io/core/main/", None),
    "keras": ("https://keras.io/api", (inv_fname, None)),
}

html_sidebars = {"**": ["versions.html"]}
html_baseurl = "https://nvidia-merlin.github.io/HugeCTR/main"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

source_suffix = [".rst", ".md"]

autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "bysource",
}

autosummary_generate = True

copydirs_additional_dirs = [
    "../../notebooks",
    "../../release_notes.md",
]
copydirs_file_rename = {
    "README.md": "index.md",
}

def cleanup_tempdir(app, exception):
    if os.path.exists(inv_dir.name):
        shutil.rmtree(inv_dir.name)

def setup(app):
    app.connect('build-finished', cleanup_tempdir)
