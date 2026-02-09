# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'One Click AI Spark'
copyright = '2026, Mohammad Asif Khan'
author = 'Mohammad Asif Khan'
release = '3.0.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Custom CSS and JavaScript
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
html_static_path = ['_static']
html_logo = None
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
}
