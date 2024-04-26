# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from modules.encoder import VAE_Encoder
from modules.decoder import VAE_Decoder

class VAE(pl.LightningModule):
    def __init__():
        super().__init__()