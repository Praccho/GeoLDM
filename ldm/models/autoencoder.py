# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from ldm.modules.encoder import VAE_Encoder
from ldm.modules.decoder import VAE_Decoder

from ldm.modules.utils import instantiate_from_config

class DiagonalGaussianDistribution():
    def __init__(self, mean_logvar):
        self.mean_logvar = mean_logvar
        self.mean, self.logvar = torch.chunk(mean_logvar, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        eps = torch.randn(self.mean.shape).to(device=self.mean_logvar.device, dtype=torch.float32)
        z = self.mean + self.std * eps
        return z
    
    def kl(self):
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1,2,3])


class VAE(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, emb_dim, ckpt_path=None, monitor=None):
        super().__init__()
        self.encoder = VAE_Encoder(**ddconfig)
        self.decoder = VAE_Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.emb_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*emb_dim, kernel_size=1)
        self.post_emb_conv = nn.Conv2d(emb_dim, ddconfig["z_channels"], kernel_size=1)
        self.emb_dim = emb_dim
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.emb_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_emb_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        posterior = self.encode(input)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior
    
    def get_input(self, batch):
        x = batch["street_image"]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float().to(self.device)
        return x

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch)
        rec, post = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, rec, post, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, rec, post, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
        
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.emb_conv.parameters())+
                                  list(self.post_emb_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
