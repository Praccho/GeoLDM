# a wild mix of orginal stable diffusion https://github.com/CompVis/latent-diffusion
# and Umar Jamil's https://github.com/hkproj/pytorch-stable-diffusion

import torch
from torch import nn
from torch.nn import functional as F
from ldm.modules.components import ResBlock, AttnBlock, Downsample

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
                if cur_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.res_block = res_block
            down.attn = attn

            if lvl < self.num_resolutions - 1:
                # getting mixed opinions on this padding here, pretty sure should be 1
                down.downsample = Downsample(block_in)
                cur_res = cur_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResBlock(block_in, block_in)
    
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
        h = F.silu(h)
        h = self.conv_out(h)

        return h