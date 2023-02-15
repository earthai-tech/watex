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
from sphinx.ext.apidoc import main
import shutil
import re 
import time 

import sphinx_gallery
from sphinx_gallery.sorting import ExampleTitleSortKey


for p in ('.', '../' ): #../../'
    sys.path.insert(0, os.path.abspath(p))
    
sys.path.insert(0, os.path.abspath("sphinxext"))

import watex 
from watex.utils._packaging_version import parse


try:
    # Configure plotly to integrate its output into the HTML pages generated by
    # sphinx-gallery.
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"
except ImportError:
    pass


# -- Project information -----------------------------------------------------

project = 'watex'
copyright = f"2022-{time.strftime('%Y')}"
author = 'K. Laurent Kouadio'
# The full version, including alpha/beta/rc tags
version = release = watex.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx-mathjax-offline', 
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages', 
    "sphinx_gallery.gen_gallery",
    "sphinx-prompt",
    "doi_role", 
    "add_toctree_functions",
    'numpydoc',
    'sphinx_copybutton', 
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_issues',

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Sphinx-issues configuration
issues_github_path = 'WEgeophysics/watex'

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# Don't add a source link in the sidebar
html_show_sourcelink =  True

# Control the appearance of type hints
autodoc_typehints = "none"
autodoc_typehints_format = "short"

# Allow shorthand references for main function interface
rst_prolog = """
.. currentmodule:: watex
"""

# Define replacements (used in whatsnew bullets)

rst_epilog = """

.. role:: raw-html(raw)
   :format: html
   
.. |ohmS| replace:: Pseudo-area of the fractured zone 
.. |sfi| replace:: Pseudo-fracturing index 
.. |VES| replace:: Vertical Electrical Sounding 
.. |ERP| replace:: Electrical Resistivity Profiling 
.. |MT| replace:: Magnetotelluric 
.. |AMT| replace:: Audio-Magnetotellurics 
.. |CSAMT| replace:: Controlled Source |AMT| 
.. |NSAMT| replace:: Natural Source |AMT| 
.. |EM| replace:: electromagnetic
.. |EMAP| replace:: |EM| array profiling

""" 
# noqa

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# lists=( 'nature' , 'classic', 'alabaster', 'bizstyle', 'traditional', 'haiku', 'sphinx_rtd_theme') 

html_theme =  "pydata_sphinx_theme" 

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.


html_static_path = ['_static', 'example_thumbs']
html_css_files =  [f'css/custom.css?v={watex.__version__}']

# todo_include_todos = True
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WEgeophysics/watex",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "StackOverflow",
            "url": "https://stackoverflow.com/tags/watex",
            "icon": "fab fa-stack-overflow",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/k_kouao",
            "icon": "fab fa-twitter",
            "type": "fontawesome",
        },
            # {
            #     "name": "PyPI",
            #     "url": "https://pypi.org/project/watex", 
            #     "icon": "fa-solid fa-box",
            # },
    ],
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_end":  ["navbar-icon-links"], # [ "theme-switcher"]
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 7,
    #"primary_sidebar_end": ["sidebar-ethical-ads"]
}

html_context = {
    "default_mode": "light",
    # "github_user": "WEgeophysics",
    # "github_repo": "watex",
    # "github_version": "master",
    # "doc_path": "docs",
}


# html_sourcelink_suffix = ""
# rediraffe_redirects = {
#     "contributing.rst": "community/index.rst",
# }


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
    "index": [], 
    "**":[]
    # "glr_examples/index":[],
    #"**": ["sidebar-nav-bs.html"] 
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'WATexdoc'
    
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'scikit-learn': ('http://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "pycsamt": ("https://pycsamt.readthedocs.io/en/latest/", None),
    "mtpy": ("https://mtpy.readthedocs.io/en/master/", None)
}

# -- Options for LaTeX output -------------------------------------------------

# latex_engine = 'xelatex'
# latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
#     "preamble": r"""
#         \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
#         \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
#         \let\oldhref\href
#         \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
#         """
# }
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#      (master_doc, 'watex.tex', u'WATex Documentation',
#       u'L. Kouadio', 'manual'),
# ]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
# man_pages = [
#     (master_doc, 'watex', u'WATex Documentation',
#       [author], 1)
# ]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)

# texinfo_documents = [
#     (master_doc, 'WATex', u'WATex Documentation',
#       author, 'watex', 'Machine learning in water exploration',
#       'Miscellaneous'),
# ]


def setup(app):
    # run  apidoc 
    app.connect('builder-inited', make_wx_apidoc
    )
    
    # do not run the examples when using linkcheck by using a small priority
    # (default priority is 500 and sphinx-gallery using builder-inited event too)
    # app.connect("builder-inited", disable_plot_gallery_for_linkcheck, priority=50)
    #app.add_js_file('copybutton.js')
    app.add_css_file('css/custom.css')
    # to hide/show the prompt in code examples:
    app.connect("build-finished", filter_search_index)

# -- Options for Epub output -------------------------------------------------
# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
epub_identifier = ''

# A unique identification for the text.
#
epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration --------------------------------------------------
MOCK_MODULES = [
    'osgeo',
    'osgeo.ogr',
    'osgeo.gdal',
    'osgeo.osr',
    'tests',
]
import mock

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
# xxxxxxxxxxxxxxxxxxxxxxx   Element functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def make_wx_apidoc(_):
    """ Generate watex API doc and create a subsequent list of  modules. 
    Refer to https://github.com/sphinx-contrib/apidoc
    """
    # get the current directory 
    c_dir = os.path.dirname(__file__)
    # specify the module directory 
    m_dir = os.path.join(c_dir, '../watex')
    # cur_dir = os.path.dirname(__file__)
    # generate the apidoc dir. *.rst files will be stored in 
    # the output 'wx_apidoc'
    output_path = os.path.join(c_dir, 'modules', 'wx_apidoc')
    
    # ignore tree removal 
    shutil.rmtree(output_path, ignore_errors=True)
    
    # run the subprocess command with main 
    main(
        ['--separate','--module-first','--no-headings', '--no-toc',
        '--force','-o', output_path, m_dir, 'tests/'
        ]
    )

# -config sphinx gallery ---
# originally taken from scikit-learn 

class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.
    Assumes README.txt exists for all subsections and uses the subsection with
    dashes, '---', as the adornment.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "new_features":
            return "0"

        readme = os.path.join(src_path, "README.txt")

        try:
            with open(readme, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return directory

        title_match = self.regex.search(content)
        if title_match is not None:
            return title_match.group(1)
        return directory
    
def filter_search_index(app, exception):
    if exception is not None:
        return

    # searchindex only exist when generating html
    if app.builder.name != "html":
        return

    print("Removing methods from search index")

    searchindex_path = os.path.join(app.builder.outdir, "searchindex.js")
    with open(searchindex_path, "r") as f:
        searchindex_text = f.read()

    searchindex_text = re.sub(r"{__init__.+?}", "{}", searchindex_text)
    searchindex_text = re.sub(r"{__call__.+?}", "{}", searchindex_text)

    with open(searchindex_path, "w") as f:
        f.write(searchindex_text)
        
def disable_plot_gallery_for_linkcheck(app):
    if app.builder.name == "linkcheck":
        sphinx_gallery_conf["plot_gallery"] = "False"
        
class WXExampleTitleSortKey(ExampleTitleSortKey):
    """Sorts release highlights based on version number."""

    def __call__(self, filename):
        title = super().__call__(filename)
        prefix = "plot_new_features_"

        # Use title to sort if not a release highlight
        if not filename.startswith(prefix):
            return title

        major_minor = filename[len(prefix) :].split("_")[:2]
        version_float = float(".".join(major_minor))

        # negate to place the newest version highlights first
        return -version_float
    
v = parse(release)
if v.release is None:
    raise ValueError(
        "Ill-formed version: {!r}. Version should follow PEP440".format(version)
    )

if v.is_devrelease:
    binder_branch = "master"
else:
    major, minor = v.release[:2]
    binder_branch = "{}.{}.X".format(major, minor)

sphinx_gallery_conf = {
    "doc_module": "watex",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "reference_url": {"watex": None},
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["glr_examples"],
    "subsection_order": SubSectionTitleOrder("../examples"),
    "within_subsection_order": WXExampleTitleSortKey,
    "binder": {
        "org": "watex",
        "repo": "watex",
        "binderhub_url": "https://mybinder.org",
        "branch": binder_branch,
        "dependencies": "./binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
}


