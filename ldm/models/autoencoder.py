# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from modules.encoder import VAE_Encoder
from modules.decoder import VAE_Decoder

class VAE(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, emb_dim, ckpt_path=None):
        super().__init__()
        self.encoder = VAE_Encoder(**ddconfig)
        self.decoder = VAE_Decoder(**ddconfig)
        self.loss = None # TODO CREATE LOSS MODULE
        self.emb_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*emb_dim, kernel_size=1)
        self.post_emb_conv = nn.Conv2d(2*emb_dim, 2*ddconfig["z_channels"], kernel_size=1)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]