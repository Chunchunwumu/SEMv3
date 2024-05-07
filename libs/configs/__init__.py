from . import default
import importlib
import os

class CFG:
    def __init__(self):
        self.__dict__['cfg'] = None
    
    def __getattr__(self, name):
        return getattr(self.__dict__['cfg'], name)
    
    def __setattr__(self, name, val):
        setattr(self.__dict__['cfg'], name, val)


cfg = CFG()
cfg.__dict__['cfg'] = default


def setup_config(cfg_name):
    global cfg
    if os.path.isfile(cfg_name):
        module_name = cfg_name[:-3].split("/")[-4:]
        module_name = ".".join(module_name)
        print(module_name)
    else:
        module_name = 'libs.configs.' + cfg_name
    cfg_module = importlib.import_module(module_name)
    cfg.__dict__['cfg'] = cfg_module
    
