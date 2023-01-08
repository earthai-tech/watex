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
# import sphinx_rtd_theme
# import sphinx_bootstrap_theme
#import watex 

for p in ('.', '../' ): #../../'
    sys.path.insert(0, os.path.abspath(p))
    
sys.path.insert(0, os.path.abspath("sphinxext"))

import watex 
import sphinx_gallery

try:
    # Configure plotly to integrate its output into the HTML pages generated by
    # sphinx-gallery.
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"
except ImportError:
    # Make it possible to render the doc when not running the examples
    # that need plotly.
    pass


# print(sys.path )
# -- Element functions ------------------------------------------------

def run_apidoc(_):

    cur_dir = os.path.dirname(__file__)
    module = os.path.join(cur_dir, '../watex') #'../../watex'
    output_path = os.path.join(cur_dir, 'api')
    shutil.rmtree(output_path, ignore_errors=True)
    main(['--separate',
        '--module-first',
        '--no-toc',
        '--force',
        '-o', output_path, module, 'tests/'
    ])


# -- Project information -----------------------------------------------------

project = 'watex'
copyright = '2022, L. Kouadio'
author = 'L. Kouadio'

# The full version, including alpha/beta/rc tags
version = release = watex.__version__

#release = "0.1.3"


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
    #"sphinxext.opengraph",
    #'numpydoc',
    'sphinx_copybutton', 
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_issues',
    # "nbsphinx",
    'sphinx_panels', 
    #'autoapi.sphinx',
     # "myst_nb",
]
# nbsphinx_custom_formats = {
#       ".md": ["jupytext.reads", {"fmt": "mystnb"}],
# }

# nb_custom_formats = {
#     ".md": ["jupytext.reads", {"fmt": "mystnb"}],
# }
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

# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#'nature' #'classic'#'alabaster', 'bizstyle'#'traditional'#'haiku' # 'sphinx_rtd_theme' 
html_theme = "pydata_sphinx_theme" #'classic'"bootstrap" 

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.


html_theme_options = {
        "logo": {
            # "text": "watex",
            "image_dark":"logo.svg", #logo0.svg",
            "alt_text": "watex",
    }, 
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
                #     "url": "https://pypi.org/project/watex", # next 
                #     "icon": "fa-solid fa-box",
                # },
    ],
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher","navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    # 'navbar_links': [
    #     ("About", "about"),
    #     ("Installing", "installation"), 
    #     ("User guide", "user_guide"), 
    #     ("API", "api_references"),
    #     ("Examples","examples"), 
    #     ("Citing", "citing"), 
    # ], 
}

html_context = {
    #"default_mode": "light",
    "github_user": "WEgeophysics",
    "github_repo": "watex",
    "github_version": "master",
    "doc_path": "docs",
}

html_static_path = ['_static', 'example_thumbs']

html_css_files =  [f'custom.css?v={watex.__version__}']

html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
#html_sourcelink_suffix = ""
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


rediraffe_redirects = {
    "contributing.rst": "community/index.rst",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_css_files = ["custom.css"]
# todo_include_todos = True

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
    "examples/index": [],
    "**": ["sidebar-nav-bs.html"],
}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'WATexdoc'

# Add the 'copybutton' javascript, to hide/show the prompt in code
# examples, originally taken from scikit-learn's doc/conf.py
def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.add_js_file('copybutton.js')
    app.add_css_file('custom.css')
    
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'scikit-learn': ('http://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    
}
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# -- Options for LaTeX output ------------------------------------------------
# latex_engine = 'xelatex'
latex_elements = {
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
    "preamble": r"""
        \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
        \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
        \let\oldhref\href
        \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
        """
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'WATex.tex', u'WATex Documentation',
     u'Kouadio K. Laurent', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'watex', u'WATex Documentation',
     [author], 1)
]

# ------------config sphinx gallery ---------------------------------------
# customize watex using class made 
# by from scikit-learn developers 

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
    
binder_branch="master"
sphinx_gallery_conf = {
    "doc_module": "watex",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "reference_url": {"watex": None},
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["glr_examples"],
    "subsection_order": SubSectionTitleOrder("../examples"),
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

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'WATex', u'WATex Documentation',
     author, 'watex', 'A machine learning research package for hydrogeophysics',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']



# -- Extension configuration -------------------------------------------------
MOCK_MODULES = [
    'osgeo',
    'osgeo.ogr',
    'osgeo.gdal',
    'osgeo.osr',
    'test',

]

import mock

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

#############################################
# reimport utils to fix circular import
from watex import utils 