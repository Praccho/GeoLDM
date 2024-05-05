# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat
import math

from abc import ABCMeta, abstractmethod

   
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q0 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.k0 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.v0 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        r = x
        r = self.norm(x)

        q = self.q0(r)
        k = self.k0(r)
        v = self.v0(r)

        b, c, h, w = x.shape
        
        # NOTE: sequence here is all pixels, dk = num_channels = c
        # we permute because q as it's currently [b, c, h*w]
        q = q.reshape(b, c, h * w)
        q = q.permute(0,2,1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b,c, h * w)
        

        wts = F.softmax(torch.bmm(q, k) * (int(c) ** -0.5), dim=-1)
        wts = wts.permute(0,2,1)

        # NOTE: we do V @ W = (W @ V).T here because we want to tpose 
        # back to [c, h*w] after
        r = torch.bmm(v, wts)
        r = r.reshape(b,c,h,w)
        
        r = self.proj(r)

        return x + r
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        out_ch = out_ch if out_ch else in_ch
        self.out_ch = out_ch
        
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if self.in_ch != self.out_ch:
            self.shortcut = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, padding=0)
    
    def forward(self, x):
        r = x

        r = self.groupnorm_1(r)
        r = F.silu(r)
        r = self.conv_1(r)

        r = self.groupnorm_2(r)
        r = F.silu(r)
        r = self.conv_2(r)

        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        
        return x + r
 
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0,1,0,1), mode="constant", value=0)
        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x