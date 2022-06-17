# -*- coding: utf-8 -*-
"""
===============================================================================
Copyright (c) 2021 Kouadio K. Laurent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================

.. synopsis_:: watex._watexlog
            Module to deal with exceptions from all the whole scripts 
            
Created on Tue May 18 12:53:48 2021

@author: @Daniel03

"""

import os 
import yaml
import logging 
import logging.config
import inspect 

class watexlog:
    """
    Field to logs watex module Files  in order to tracks all 
    exceptions.
    
    """
    
    @staticmethod
    def load_configure (path2configure =None, OwnloggerBaseConf=False) :
        """
        configure/setup the logging according to the input configfile

        :param configfile: .yml, .ini, .conf, .json, .yaml.
        Its default is the logging.yml located in the same dir as this module.
        It can be modified to use env variables to search for a log config file.
        """
        
        configfile=path2configure
        
        if configfile is None or configfile == "":
            if OwnloggerBaseConf ==False :
                logging.basicConfig()
            else :
                watexlog().set_logger_output()
            
        elif configfile.endswith(".yaml") or configfile.endswith(".yml") :
            this_module_path=os.path.abspath(__file__)
    
            print('module path', this_module_path)
            
            logging.info ("this module is : %s", this_module_path)
            print('os.path.dirname(this_module_path)=', os.path.dirname(this_module_path))

            yaml_path=os.path.join(os.path.dirname(this_module_path),
                                   configfile)
            print("yaml_path", yaml_path)
            
            logging.info('Effective yaml configuration file %s', yaml_path)

            if os.path.exists(yaml_path) :
                with open (yaml_path,"rt") as f :
                    config=yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            else :
                logging.exception(
                    "the config yaml file %s does not exist?", yaml_path)
                
        elif configfile.endswith(".conf") or configfile.endswith(".ini") :
            logging.config.fileConfig(configfile,
                                     disable_existing_loggers=False)
            
        elif configfile.endswith(".json") :
            pass 
        else :
            logging.exception("logging configuration file %s is not supported" %
                configfile)
            
    
    @staticmethod        
    def get_watex_logger(loggername=''):
        """
        create a named logger (try different)
        :param loggername: the name (key) of the logger object in this Python interpreter.
        :return:
        """
        logger =logging.getLogger(loggername)
        watexlog.load_configure() #set configuration 

        return logger

    @staticmethod
    def load_configure_set_logfile (path2configfile=None): # loggername =None, setLevel=Name
        """
        configure/setup the logging according to the input configure .yaml file.

        :param configfile: .yml, or add ownUsersConfigYamlfile (*.yml) 
        Its default is the logging.yml located in logfiles folder 
        It can be modified to use env variables to search for a log config file.
        
        """
        
        ownUserLogger="wlog.yml"
        if path2configfile is None :
            env_var=os.environ['watex']
            path2configfile =os.path.join( env_var, 'watex','utils',
                ownUserLogger)
            

        elif path2configfile is not None :
            if os.path.isdir(os.path.dirname(path2configfile)):
                if path2configfile.endswith('.yml') or path2configfile.endswith('.yaml'):
                    
                    logging.info('Effective yaml configuration file :%s', path2configfile)
                else :
                    logging.exception('File provided {%s}, is not a .yaml config file !'%os.path.basename(path2configfile))
            else :
                
                logging.exception ('Wrong path to .yaml config file.')
        
        yaml_path=path2configfile
        os.chdir(os.path.dirname(yaml_path))
        if os.path.exists(yaml_path) :
            with open (yaml_path,"rt") as f :
                config=yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else :
            logging.exception(
                "the config yaml file %s does not exist?", yaml_path) 
            
def test_yaml_configfile(yamlfile="wlog.yml"):
    
    this_func_name = inspect.getframeinfo(inspect.currentframe())[2]

    UsersOwnConfigFile = yamlfile
    watexlog.load_configure(UsersOwnConfigFile)
    logger = watexlog.get_csamtpy_logger(__name__)
    
    print((logger, id(logger), logger.name, logger.level, logger.handlers))
    
    # 4 use the named-logger
    logger.debug(this_func_name + ' __debug message')
    logger.info(this_func_name + ' __info message')
    logger.warn(this_func_name + ' __warn message')
    logger.error(this_func_name + ' __error message')
    logger.critical(this_func_name + ' __critical message')

    print(("End of: ", this_func_name))
    
if __name__=='__main__':
    # ownerlogfile = '/utils/wlog.yml'
    watexlog().load_configure(path2configure='wlog.yml')
    
    watexlog().get_watex_logger().error('First pseudo test error')
    watexlog().get_watex_logger().info('Just a logging test')
    

    
    
    
    
    