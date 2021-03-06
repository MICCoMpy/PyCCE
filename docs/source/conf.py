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
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('Z:/codes_development/pyCCE'))
sys.path.insert(0, os.path.abspath('/home/onizhuk/midway/codes_development/pyCCE'))

# -- Project information -----------------------------------------------------

project = 'PyCCE'
copyright = '2021, Mykyta Onizhuk'
author = 'Mykyta Onizhuk'

# The full version, including alpha/beta/rc tags
release = '1.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx-mathjax-offline',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_favicon = 'favicon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

nbsphinx_input_prompt = '%.0s'
nbsphinx_output_prompt = '%.0s'
nbsphinx_prompt_width = '0'

autodoc_member_order = 'bysource'
add_module_names = False

html_logo = 'logo_white.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
#    'style_nav_header_background': '#800000'
}
