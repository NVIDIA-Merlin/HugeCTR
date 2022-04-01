"""
 Copyright (c) 2021, NVIDIA CORPORATION.

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

import errno

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import subprocess
import sys
from re import I

from natsort import natsorted

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
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_markdown_tables",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

myst_enable_extensions = ["html_image"]

myst_heading_anchors = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "notebooks/multi-modal-data/02-Data-Enrichment.ipynb",  # Shuts down the kernel and breaks build
    "notebooks/news-example.ipynb",  # Triggers NVMLError_Unknown
    "notebooks/README.md",
    "notebooks/prototype_indices.ipynb",
    "notebooks/training_with_hdfs.ipynb",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
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
smv_branch_whitelist = "^master$"

html_sidebars = {"**": ["versions.html"]}

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


def copy_files(src_dir: str):
    """
    src_dir: A path, specified as relative to the
             docs/source directory in the repository.
             Sphinx considers all directories as relative
             to the docs/source directory.
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src_dir)
    sys.path.insert(0, src_dir)
    out_dir_name = os.path.basename(src_dir)
    out_dir = os.path.abspath("{}/".format(out_dir_name))

    if os.path.isdir(os.path.join(src_dir, out_dir_name)):
        sys.path.insert(0, os.path.join(src_dir, out_dir_name))

    print(r"Copying source documentation from directory: {}".format(src_dir), file=sys.stderr)
    print(r"  ...to destination directory: {}".format(out_dir), file=sys.stderr)

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(src_dir, out_dir)


copy_files(r"../../notebooks")
