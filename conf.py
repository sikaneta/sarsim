# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sarsim'
copyright = '2022, Ishuwa Sikaneta'
author = 'Ishuwa Sikaneta'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

latex_elements = {
# Additional stuff for the LaTeX preamble.
'preamble': r'''
\usepackage{amsmath}
\newcommand{\vct}[1]{\ensuremath{\mathbf{#1}}} 
\newcommand{\ux}{\ensuremath{\hat{\vct{i}}}}
\newcommand{\uy}{\ensuremath{\hat{\vct{j}}}}
\newcommand{\uz}{\ensuremath{\hat{\vct{k}}}}
'''
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
