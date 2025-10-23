# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect  # Used in linkcode_resolve
import os
import sys
from datetime import date
from importlib import metadata

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SMS Fusion"
copyright = f"{date.today().year}, 4Subsea"
author = "4Subsea"
release = metadata.version("smsfusion")
github_repo = "https://github.com/4Subsea/smsfusion-python/"
pypi_url = "https://pypi.org/project/smsfusion/"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Intershpinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates", "_templates/autosummary"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Logo and favicon
html_favicon = "_static/favicon.png"
html_logo = "_static/4insight-logo.svg"

html_theme_options = {
    "logo": {"link": "http://docs.4insight.io", "text": "DOCUMENTATION"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_repo,
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": pypi_url,
            "icon": "fas fa-box",
        },
    ],
    "navbar_end": [
        "search-button",  # Includes the search icon
        "navbar-icon-links",  # Disables the light/dark mode toggle
    ],
    "navbar_persistent": [],  # Removes the search field
    "secondary_sidebar_items": [
        "page-toc",  # Only show "On this page", hides the "Show sources" link
    ],
    "article_header_start": [],  # Hides default breadcrumb
}

html_context = {
    "default_mode": "light",  # Sets the default theme to light
}


def linkcode_resolve(domain, info):

    if domain != "py":
        return None
    if not info["module"]:
        return None

    obj = sys.modules[info["module"]]

    for part in info["fullname"].split("."):
        obj = getattr(obj, part)

    obj = inspect.unwrap(obj)

    # Inspect cannot find source file for properties
    if isinstance(obj, property):
        return None

    path = os.path.relpath(inspect.getfile(obj))
    src, lineno = inspect.getsourcelines(obj)

    path = f"{github_repo}blob/main/{path}#L{lineno}-L{lineno + len(src) - 1}"

    return path
