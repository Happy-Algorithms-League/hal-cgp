# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import inspect
import os
import sys

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "../")
sys.path.insert(0, rootdir)

import cgp  # noqa: E402 isort:skip

project = "hal-cgp"
copyright = "2021, Happy Algorithms League"
author = cgp.__author__

# The full version, including alpha/beta/rc tags
release = cgp.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "recommonmark",
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",
    "sphinx_multiversion",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Configuration for multiversion builds
smv_branch_whitelist = "master"  # Only build master branch
smv_remote_whitelist = None
smv_tag_whitelist = "0.2.0"  # Only release 0.2.0 has a sphinx documentation
smv_released_pattern = r".*"  # Tags only
smv_outputdir_format = "{ref.name}"  # Use the branch/tag name

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for Sphinx Gallery ---------------------------------------------


class ExampleCLIArgs:
    def __repr__(self):
        return "ExampleCLIArgs"

    def __call__(self, sphinx_gallery_conf, script_vars):
        if "example_caching.py" in script_vars["src_file"]:
            return []
        else:
            return ["--max-generations", "10"]


sphinx_gallery_conf = {
    "filename_pattern": "/*.py",
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "matplotlib_animations": True,
    "image_scrapers": ("matplotlib",),
}

for arg in sys.argv:
    if "reset_argv" in arg:
        sphinx_gallery_conf["reset_argv"] = ExampleCLIArgs()
        break


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object.

    Code from the Numpy repository:
    https://github.com/numpy/numpy/blob/a0028bca0117874606bce99261d978df8d3f6610/doc/source/conf.py#L332
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(cgp.__file__))
    return f"https://github.com/Happy-Algorithms-League/hal-cgp/blob/master/cgp/{fn}{linespec}"
