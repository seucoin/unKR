# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
import sphinx_rtd_theme
import doctest
import unKR

project = 'unKR'
copyright = '2024, SEUKGE'
author = 'SEU'

release = '1.0.1'
version = '1.0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    # 'sphinx_copybutton',
    'recommonmark',
    'sphinx_markdown_tables',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "seu2024",  # Username
    "github_repo": "test_doc",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}
