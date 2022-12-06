**Command line interface**
============================

:code:`watex` commands (`in progress- not available yet`)
------------------------------------------------------------------

watex provides some command line tools under the :code:`watex` command.
Typing the command shows the help message with the list of available commands :

.. code-block:: bash

  $ watex
  Usage: watex [OPTIONS] COMMAND [ARGS]...
  
    The watex command line interface.
  
  Options:
    -h, --help  Show this message and exit.
  
  Commands:
    path       Manipulate data path.
    version    watex installed version.

Getting the version number with :code:`watex version`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the number of the installed version :

.. code-block:: bash

  $ watex version
  watex 0.4.0

Manipulating the data path with :code:`watex path`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`watex path` provides several subcommands for manipulating the data path.
Just typing :code:`watex path` performs no action but shows the help and the list of available commands :

.. code-block:: bash

  $ watex path
  Usage: watex path [OPTIONS] COMMAND [ARGS]...
  
    Manipulate data path.
  
    Data path is either global or local. If the local path is not available,
    the global path is used instead.
  
    The path commands depend on the current directory where they are executed.
  
  Options:
    -c, --create  Create the local path if missing.
    -h, --help    Show this message and exit.
  
  Commands:
    base      Current base data path.
    metronix  Current path for Metronix calibration files.


Without options, :code:`watex path base` and :code:`watex path metronix` just show the path where data, like calibration files, will be searched. This path depends on the working directory.

The option :code:`--create` will create the corresponding local path.


Extending the command line interface
--------------------------------------

You can add commands to the :code:`watex` command line interface by using `Click <https://click.palletsprojects.com/>`_ and `setuptools entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`_.

Let us look at an example.
We have a simple `Click <https://click.palletsprojects.com/>`_ program in our package:

.. code-block:: python

  # mypkg/watex_cli.py
  import click

  @click.command('say-hello')
  def cli():
      click.echo('Hello world !')

We also have a `setup.py` for installing our package.
To extend the :code:`watex` command, we need to informs the `setup()` function in the following way:

.. code-block:: python

  # setup.py
  setup(
    # ...
    entry_points={
        'watex.commands': [
            'say-hello=mypkg.watex_cli:cli',
        ]
    },
  )

Once `mypkg` is installed (:code:`python setup.py install` or :code:`pip install .`), the :code:`watex` command can now expose our new subcommand:

.. code-block:: bash

  $ watex say-hello
  Hello world !