# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        out_ch = out_ch if out_ch else None
        self.out_ch = out_ch
        
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if self.in_ch != self.out_ch:
            self.shortcut = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, padding=1)
    
    def forward(self, x):
        r = x

        r = self.groupnorm_1(r)
        r = self.conv_1(r)
        r = self.groupnorm_2(r)
        r = self.conv_2(r)

        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        
        return x + r
    
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
        v = self.k0(r)

        b, c, h, w = x.shape
        
        # NOTE: sequence here is all pixels, dk = num_channels = c
        # we permute because q as it's currently [b, c, h*w]
        q = q.reshape(b, c, h * w)
        q = q.permute(0,2,1)
        k = k.reshape(b, c, h * w)

        w = F.softmax(torch.bmm(q, k) ** (int(c) ** -0.5), dim=-1)
        w = w.permute(0,2,1)

        # NOTE: we do V @ W = (W @ V).T here because we want to tpose 
        # back to [c, h*w] after
        r = torch.bmm(v, w)
        r = r.reshape(b,c,h,w)
        
        r = self.proj(r)

        return x + r

class VAE_Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, in_ch, resolution, z_channels):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_ch

        self.conv_in = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()

        cur_res = resolution

        in_ch_mult = (1,) + tuple(ch_mult)

        for lvl in range(self.num_resolutions):
            res_block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[lvl]
            block_out = ch * ch_mult[lvl]

            for _ in range(num_res_blocks):
                res_block.append(ResBlock(block_in, block_out))
                block_in = block_out
                if lvl in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.res_block = res_block
            down.attn = attn

            if lvl < self.num_resolutions - 1:
                # getting mixed opinions on this padding here, pretty sure should be 1
                down.downsample = nn.Conv2d(block_in, block_in, kernel_size=3, stride=2, padding=1)
                cur_res = cur_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_ch=block_in, out_ch=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_ch=block_in, out_ch=block_in)
    
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, z_channels * 2, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for lvl in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[lvl].res_block[i_block](h)
                if len(self.down[lvl].attn) > 0:
                    h = self.down[lvl].attn[i_block](h)
            if lvl < self.num_resolutions-1:
                h = self.down[lvl].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = self.conv_out(h)

        return h