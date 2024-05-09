import torch
import numpy as np

from tqdm import tqdm

class DDIMWrapper(object):
    def __init__(self, model):
        self.model = model
        self.ddpm_num_timesteps = model.timesteps
        
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def register_schedule(self, ddim_num_steps):
        self.ddim_timesteps = np.asarray(list(range(0, self.ddpm_num_timesteps, self.ddpm_num_timesteps // ddim_num_steps))) + 1
        alphas_cumprod = self.model.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_sub_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_sub_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_inv_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1.)))

        ddim_alphas = alphas_cumprod.cpu()[self.ddim_timesteps]
        ddim_alphas_prev = np.asarray([alphas_cumprod.cpu()[0]] + alphas_cumprod.cpu()[self.ddim_timesteps[:-1]].tolist())
        
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_sub_alphas_cumprod', to_torch(np.sqrt(1. - ddim_alphas)))

    @torch.no_grad()
    def p_sample(self, x, ctx, t, idx, cfg_scale = None):
        bs, device = x.shape[0], self.model.device

        eps_pred = self.model.backbone(x, t, ctx)

        if cfg_scale:
            eps_pred_cond, eps_pred_uncond = torch.chunk(eps_pred, 2)
            eps_pred = (cfg_scale + 1) * eps_pred_cond - cfg_scale * eps_pred_uncond
            x, _ = torch.chunk(x, 2)
            t, _ = torch.chunk(t, 2)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_sub_alphas = self.ddim_sqrt_sub_alphas_cumprod
        
        a_t = torch.full((x.shape[0],1,1,1), alphas[idx], device=device)
        a_prev = torch.full((x.shape[0],1,1,1), alphas_prev[idx], device=device)
        sqrt_sub_at = torch.full((x.shape[0],1,1,1), sqrt_sub_alphas[idx], device=device)

        pred_x0 = (x - sqrt_sub_at * eps_pred) / a_t.sqrt()
        dir_xt = (1. - a_prev).sqrt() * eps_pred
        xrec = a_prev.sqrt() * pred_x0 + dir_xt
        return xrec

        

    @torch.no_grad()
    def sample(self, ddim_num_steps, ctx, batch_sz, shape, cfg_scale = None):
        _, C, H, W = shape
        shape = (batch_sz, C, H, W)
        self.register_schedule(ddim_num_steps)

        bs = batch_sz * 2 if cfg_scale else batch_sz

        if cfg_scale:
            uncond_ctx = torch.zeros_like(ctx, dtype=torch.float32, device = self.model.device)
            ctx = torch.cat([ctx, uncond_ctx])
        

        imgs = torch.randn(shape, dtype=torch.float32, device=self.model.device)
        time_range = tqdm(np.flip(self.ddim_timesteps), desc='DDIM sampling', total=ddim_num_steps)

        for i, t in enumerate(time_range):
            if cfg_scale:
                imgs = imgs.repeat(2,1,1,1)
            idx = len(self.ddim_timesteps) - i - 1
            ts = torch.full((bs,), int(t), device=self.model.device, dtype=torch.long)

            imgs = self.p_sample(imgs, ctx, ts, idx, cfg_scale = cfg_scale)
            
        return imgs

