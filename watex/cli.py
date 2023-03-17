# -*- coding: utf-8 -*-

from pkg_resources import iter_entry_points
import watex as wx  
import click
 
class PluginGroup(click.Group):

    def __init__(self, *args, **kwds):
        self.extra_commands = {
            e.name: e.load() for e in iter_entry_points('wx.commands')
        }
        super().__init__(*args, **kwds)

    def list_commands(self, ctx):
        return sorted(super().list_commands(ctx) + list(self.extra_commands))

    def get_command(self, ctx, name):
        return self.extra_commands.get(name) or super().get_command(ctx, name)


@click.group(cls=PluginGroup, 
             context_settings={'help_option_names': ('-h', '--help')})
def cli():
    """ The watex command line interface.
    """
    pass

@cli.command()
@click.option ('-v','--version', 'version',   help ='show watex version')
@click.option ('--show',  default=False, show_default= True,
               help ='show watex version and dependencies')
def version(version, show):
    """ watex installed version.
    """
    if show: 
        click.echo(f"watex {wx.show_versions ()}")
    else: 
        click.echo(f"watex {wx._version.version}")
    
# XXXTODO : write consistent CLI 

# references 
# https://click.palletsprojects.com/en/8.1.x/
# https://setuptools.pypa.io/en/latest/userguide/entry_point.html
