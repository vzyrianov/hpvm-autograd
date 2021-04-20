from datetime import date

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path
this_folder = Path(__file__).parent
sys.path.insert(0, (this_folder / "..").absolute().as_posix())

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"

# generate autosummary pages
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# General substitutions.
project = "PredTuner"
copyright = f"2020-{date.today().year}, University of Illinois"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'friendly'
pygments_style = "sphinx"

# A list of prefixs that are ignored when creating the module index. (new in Sphinx 0.6)
# modindex_common_prefix = []

# Options for HTML output
# -----------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # "github_url": "https://gitlab.engr.illinois.edu/llvm/hpvm-beta",
    "show_prev_next": False,
    "search_bar_position": "sidebar",
}

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

# Options for LaTeX output
# ------------------------

# Use a latex engine that allows for unicode characters in docstrings
latex_engine = "xelatex"
# The paper size ('letter' or 'a4').
latex_paper_size = "letter"

latex_appendices = ["tutorial"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pytorch": ("https://pytorch.org/docs/stable", None),
}

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "obj"
