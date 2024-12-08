# Configuration file for the Sphinx documentation builder.
#
import sys, os
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../notebooks'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '2D reconstruction'
copyright = '2024, Michael Levinov'
author = 'Michael Levinov'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.imgmath',
    'nbsphinx',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = []

# Enable MyST in notebooks
nb_custom_formats = {
    ".ipynb": ["nbsphinx", {"mdformat": "myst"}],
}

# Enable MyST extensions
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_static_path = ['_static']
