from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl

from ldm.modules.utils import instantiate_from_config, extract_into_tensor
from einops import rearrange

def disabled_train(self, mode=True):
    return self

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 backbone_config,
                 timesteps = 1000,
                 loss_type = "?",
                 monitor = "val/loss",
                 image_size = 8,
                 channels = 2,
                 log_every_t = 100,
                 clip_denoised = True,
                 original_elbo_weight = 0.,
                 v_posterior = 0.,
                 l_simple_weight=1.,
                 learn_logvar=False,
                 logvar_init=0.,
                 betas = None,
                 alphas = None,
                 device = None,
                 scale_factor = 0.18215, # the holy number
                 scale_by_std = False,
            ):
        super().__init__()
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)
        self.backbone = instantiate_from_config(backbone_config)
        
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size  
        self.channels = channels
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(timesteps)

    def register_schedule(self, timesteps):
        self.betas = cosine_beta_schedule(timesteps)
        self.alphas = 1.-self.betas

        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]))

        post_var = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(self.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_sub_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_sub_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_inv_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        self.register_buffer('post_var', to_torch(post_var))
        self.register_buffer('post_log_var', to_torch(np.log(np.maximum(post_var, 1e-20))))
        self.register_buffer('post_mean_x0_coef', to_torch(np.sqrt(alphas_cumprod_prev) * self.betas / (1. - alphas_cumprod)))
        self.register_buffer('post_mean_xt_coef', to_torch(np.sqrt(self.alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)))

        lvlb_weights = self.betas ** 2 / (2 * self.post_var * to_torch(self.alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    def instantiate_first_stage(self, cfg):
        self.first_stage_model = instantiate_from_config(cfg)
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def q_sample(self, x0, t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def step_loss(self, x0, ctx, t):
        loss_dict = {}
        pref = 'train' if self.training else 'val'
 
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.backbone(xt, ctx, t)
        
        loss_simple = torch.nn.functional.mse_loss(noise, noise_pred, reduction='none').mean()
        loss_dict.update({f'{pref}/loss': loss_simple})
        
        return loss_simple, loss_dict

    def forward(self, x, ctx):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
        return self.step_loss(x, ctx, t)
    
    def get_input_key(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float().to(self.device)

        return x

    def get_input(self, batch):
        x = self.get_input_key("street_image")
        z_posterior = self.first_stage_model(x)
        z = z_posterior.sample().detach()

        ctx = self.get_input_key(batch, "satellite_emb")
        ctx = self.cond_stage_model(ctx)

        return x, ctx

    def shared_step(self, batch):
        x, ctx = self.get_input(batch)
        loss, loss_dict = self(x, ctx)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)   

    
def cosine_beta_schedule(timesteps, s=0.008):

    """

    Generates a cosine beta schedule for the given number of timesteps.

    Parameters:

    - timesteps (int): The number of timesteps for the schedule.

    - s (float): A small constant used in the calculation. Default: 0.008.

    Returns:

    - betas (torch.Tensor): The computed beta values for each timestep.

    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.9999)