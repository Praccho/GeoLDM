import importlib

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Need 'target' to instantiate")
    
    module, cls = config["target"].rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)(**config.get("params", dict()))