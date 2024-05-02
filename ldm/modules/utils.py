import importlib

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Need 'target' to instantiate")
    
    module, cls = config["target"].rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)(**config.get("params", dict()))

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))