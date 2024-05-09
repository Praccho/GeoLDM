# adapted from https://github.com/CompVis/latent-diffusion

import importlib
import os
import math
import types
import torch
import torch.nn as nn
from abc import ABC
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Need 'target' to instantiate")
    
    module, cls = config["target"].rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)(**config.get("params", dict()))

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def disabled_train(self, mode=True):
    return self