# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:59:14 2022

@author: Daniel
"""

import pathlib
import os
from pkg_resources import iter_entry_points
import click
import watex 


class PluginGroup(click.Group):

    def __init__(self, *args, **kwds):
        self.extra_commands = {
            e.name: e.load() for e in iter_entry_points('watex.commands')
        }
        super().__init__(*args, **kwds)

    def list_commands(self, ctx):
        return sorted(super().list_commands(ctx) + list(self.extra_commands))

    def get_command(self, ctx, name):
        return self.extra_commands.get(name) or super().get_command(ctx, name)


@click.group(cls=PluginGroup, context_settings={'help_option_names': ('-h', '--help')})
def cli():
    """ The watex command line interface.
    """
    pass


@cli.command()
def version():
    """ watex installed version.
    """
    click.echo(f"watex{'+' if watex.plus else ''} {watex.__version__}")


@cli.group()
@click.option('-c', '--create', is_flag=True,
    help="Create the local path if missing."
)
@click.pass_context
def path(ctx, create):
    """ Manipulate data path.
    Data path is either global or local.
    If the local path is not available, the global path is used instead.
    The path commands depend on the current directory where they are executed.
    """
    ctx.obj = {'create': create}


# def path_action(ctx, target):
#     if not ctx.obj['create']:
#         click.echo(watex.data.data_path(target))
#     else:
#         p = watex.data.local_data_path(target)
#         if p.exists():
#             click.echo(f"{p} already exists")
#         else:
#             p.mkdir(parents=True)
#             click.echo(f"{p} created")


# @path.command('base')
# @click.pass_context
# def path_base(ctx):
#     """ Current base data path.
#     """
#     path_action(ctx, '')


# @path.command('metronix')
# @click.pass_context
# def path_metronix(ctx):
#     """ Current path for Metronix calibration files.
#     """
#     path_action(ctx, watex.calibrations.METRONIX_DATA_PATH)

# references 
# https://click.palletsprojects.com/en/8.1.x/
# https://setuptools.pypa.io/en/latest/userguide/entry_point.html
