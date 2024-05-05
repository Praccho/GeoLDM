from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl

from ldm.modules.utils import instantiate_from_config, extract_into_tensor, disabled_train
from ldm.modules.components import ResBlock, AttnBlock
from einops import rearrange, repeat
from tqdm import tqdm

class SatelliteHead(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.inblock = ResBlock(self.in_channels, self.out_channels)
        self.outblock = AttnBlock(self.out_channels)

    def forward(self, sat_emb, lat_emb, lng_emb):
        bs, c, h, w = sat_emb.shape

        se = self.inblock(sat_emb)
        se = self.outblock(se)
        se = se.reshape(bs, c, h * w)
        se = torch.cat([se, lat_emb.unsqueeze(-1), lng_emb.unsqueeze(-1)], dim=-1)
        se = se.permute(0,2,1).contiguous()

        return se
    
class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 backbone_config,
                 timesteps = 1000,
                 monitor = "val/loss",
                 embed_size = 4,
                 image_size = 16,
                 log_every_t = 100,
                 clip_denoised = True,
                 scale_factor = 0.18215, # the holy number
                 p_uncond = 0.
            ):
        super().__init__()
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_model = instantiate_from_config(cond_stage_config).half()
        self.backbone = instantiate_from_config(backbone_config)
        
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.embed_size = embed_size
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.p_uncond = p_uncond

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(timesteps)

    def register_schedule(self, timesteps):
        betas = make_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float16)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_sub_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_sub_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_inv_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('post_var', to_torch(post_var))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('post_log_var', to_torch(np.log(np.maximum(post_var, 1e-20))))
        self.register_buffer('post_mean_x0_coef', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('post_mean_xt_coef', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        lvlb_weights = self.betas ** 2 / (
                    2 * self.post_var * to_torch(alphas) * (1 - self.alphas_cumprod))
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def instantiate_first_stage(self, cfg):
        self.first_stage_model = instantiate_from_config(cfg).half()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def pred_prev(self, xt , t, noise):
        return (
                extract_into_tensor(self.sqrt_inv_alphas_cumprod, t, xt.shape) * xt -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
        )

    def q_sample(self, x0, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract_into_tensor(self.sqrt_sub_alphas_cumprod, t, x0.shape) * noise
        )
        
    def p_sample(self, x, ctx, t):
        eps_pred = self.backbone(x, t, ctx)
        x_rec = self.pred_prev(x, t, eps_pred).clamp_(-1.,1.)
        mean = (
            extract_into_tensor(self.post_mean_x0_coef, t, x.shape) * x_rec +
            extract_into_tensor(self.post_mean_xt_coef, t, x.shape) * x
        )
        post_log_var = extract_into_tensor(self.post_log_var, t, x.shape) 

        noise = torch.randn(x.shape, device=self.device)
        mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        ret = mean + mask * (0.5 * post_log_var).exp() * noise

        return ret

    def step_loss(self, x0, ctx, t):
        loss_dict = {}
        pref = 'train' if self.training else 'val'
 
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.backbone(xt, t, ctx)
        
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
        x = x.to(memory_format=torch.contiguous_format).to(device=self.device, dtype=torch.float16)

        return x

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False, bs=None):
        bs = bs if bs else len(batch)
        x = self.get_input_key(batch, "street_image")[:bs]
        z_posterior = self.first_stage_model.encode(x)
        z = z_posterior.sample().half()
        z = self.scale_factor * z
        z = z.detach()

        sat_emb = batch["satellite_emb"][:bs].to(device=self.device, dtype=torch.float16)
        lat_emb = batch["lat_emb"][:bs].to(device=self.device, dtype=torch.float16)
        lng_emb = batch["lng_emb"][:bs].to(device=self.device, dtype=torch.float16)

        if self.training:
            mask = torch.rand(len(batch)) < self.p_uncond
            mask = mask.expand(*sat_emb.shape[:])
            print(mask.shape)
            sat_emb[mask] = 0
            lat_emb[mask] = 0
            lng_emb[mask] = 0

        ctx = self.cond_stage_model(sat_emb, lat_emb, lng_emb)

        ret = [z, ctx]

        if return_first_stage_outputs:
            z = (1. / self.scale_factor) * z
            xrec = self.first_stage_model.decode(z)
            ret.extend([x, xrec])
        if return_original_cond:
            xc = self.get_input_key(batch, "satellite_image")[:bs]
            ret.append(xc)

        return ret

    def shared_step(self, batch):
        x, ctx = self.get_input(batch)
        loss, loss_dict = self(x, ctx)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)   

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss
    
    @torch.no_grad()
    def sample(self, ctx, batch_sz):
        shape = (batch_sz, self.embed_size, self.image_size, self.image_size)
        # sample noise N(0, I):
        imgs = torch.randn(shape, dtype=torch.float16, device=self.device)
        for t in tqdm(reversed(range(0, self.timesteps)), desc='Sampling t', total=self.timesteps):
            ts = torch.full((batch_sz,), t, device=self.device, dtype=torch.long)
            imgs = self.p_sample(imgs, ctx, ts)
            # print("max min:", torch.max(imgs), torch.min(imgs))
            
        return imgs.half()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch,
                                           return_first_stage_outputs=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["original_conditioning"] = xc

        if sample:
            samples = self.sample(c, N)
            samples = (1. / self.scale_factor) * samples
            x_samples = self.first_stage_model.decode(samples)
            log["samples"] = x_samples

        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.backbone.parameters()) + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        return opt

    
def make_beta_schedule(n_timestep, cosine_s=8e-3):

    timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
    )
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)

    return betas.numpy()