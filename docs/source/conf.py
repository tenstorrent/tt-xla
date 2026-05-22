# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "TT-XLA"
copyright = "Tenstorrent"
author = "Tenstorrent"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.email",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST: generate heading anchor IDs up to H3 so in-page `#some-slug` links resolve.
myst_heading_anchors = 3

# Napoleon settings (NumPy-style, matching tt-metal)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True

# Email settings
email_automode = True

# Intersphinx — cross-refs to sibling Tenstorrent docs.
# tt-mlir is still on mdBook (no objects.inv) — add a mapping once it migrates to Sphinx.
intersphinx_mapping = {
    "tt-metalium": ("https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/", None),
    "ttnn": ("https://docs.tenstorrent.com/tt-metal/latest/ttnn/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "_static/tt_logo.svg"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
# Pull the canonical Tenstorrent theme + fonts from the root docs site so they
# don't drift per-project. The CSS resolves its `./fonts/...` URLs relative to
# itself, so fonts come from the same root _static/ automatically.
html_css_files = ["https://docs.tenstorrent.com/_static/tt_theme.css"]
html_baseurl = "/tt-xla/"
html_context = {"logo_link_url": "https://docs.tenstorrent.com/"}
html_last_updated_fmt = "%b %d, %Y"
