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
                 use_ema = True, #what is ema
                 image_size = 8,
                 channels = 2,
                 log_every_t = 100,
                 clip_denoised = True,
                 #not writing given betas,, adjust init and write a beta schedule function
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
        # self.use_ema = use_ema
        # if self.use_ema:
        #     self.model_ema = LitEma(self.model)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        #call schedule method

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

    def instantiate_first_stage(self, cfg):
        self.first_stage_model = instantiate_from_config(cfg)
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def q_sample(self, x_0, t, noise):
        

    def step_loss(self, x_0, ctx, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)

    def forward(self, x, ctx):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
        return self.step_loss(x, t, ctx)
    
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


    #keep, but instead of overriding register_schedule for latentdiffusion, use this instead
    def register_schedule(self, betas, timesteps):
        to_torch = partial(torch.Tensor, dtype = torch.float32)
        #casting/creating partial
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])


        betas = to_torch(betas).to(self.device)
        alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)
    
        #calculation for diffusion:
        sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)


        #calculation for posterior
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                        1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)


        posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    
        posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)


        posterior_mean_coef1 = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)


        posterior_mean_coef2 = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)
    
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights = to_torch(lvlb_weights).to(self.device)
   

    
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