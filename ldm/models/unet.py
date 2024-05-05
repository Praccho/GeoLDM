from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from ldm.modules.utils import SiLU,linear,ResBlock,AttentionBlock,SpatialTransformer,TimestepEmbedSequential,Downsample,Upsample,conv_nd,zero_module


# adapted from https://github.com/CompVis/latent-diffusion

class UNetModel(nn.Module):
    def __init__(self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            context_dim = None,
            legacy=True,
            ):

            super().__init__()

            if num_heads_upsample == -1:
                num_heads_upsample = num_heads

            if num_heads == -1:
                assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

            if num_head_channels == -1:
                assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

            self.image_size = image_size
            self.in_channels = in_channels
            self.model_channels = model_channels
            self.out_channels = out_channels
            self.num_res_blocks = num_res_blocks
            self.attention_resolutions = attention_resolutions
            self.dropout = dropout
            self.channel_mult = channel_mult
            self.conv_resample = conv_resample
            self.num_classes = num_classes
            self.use_checkpoint = use_checkpoint
            self.dtype = torch.float32
            self.num_heads = num_heads
            self.num_head_channels = num_head_channels
            self.num_heads_upsample = num_heads_upsample

            time_embed_dim = model_channels * 4
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

            if self.num_classes is not None:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)

            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            self._feature_size = model_channels
            input_block_chans = [model_channels]
            ch = model_channels
            ds = 1
            for level, mult in enumerate(channel_mult):
                for _ in range(num_res_blocks):
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            #num_heads = 1
                            dim_head = num_head_channels
                        layers.append(
                            # AttentionBlock(
                            #     ch,
                            #     use_checkpoint=use_checkpoint,
                            #     num_heads=num_heads,
                            #     num_head_channels=dim_head,
                            #     use_new_attention_order=use_new_attention_order,
                            # ),
                              SpatialTransformer(
                                ch, num_heads, dim_head, depth=1, context_dim=context_dim
                            )
                        )
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch
                    input_block_chans.append(ch)
                if level != len(channel_mult) - 1:
                    out_ch = ch
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                            if resblock_updown
                            else Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                    )
                    ch = out_ch
                    input_block_chans.append(ch)
                    ds *= 2
                    self._feature_size += ch

            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels
            if legacy:
                #num_heads = 1
                dim_head = num_head_channels
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                SpatialTransformer(
                    ch, num_heads, dim_head, depth=1, context_dim=context_dim
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
            self._feature_size += ch

            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            #num_heads = 1
                            dim_head = ch // num_heads 
                        layers.append(
                            # AttentionBlock(
                            #     ch,
                            #     use_checkpoint=use_checkpoint,
                            #     num_heads=num_heads_upsample,
                            #     num_head_channels=dim_head,
                            #     use_new_attention_order=use_new_attention_order,
                            # ) 
                            # ,
                              SpatialTransformer(
                                ch, num_heads, dim_head, depth=1, context_dim=context_dim
                            )
                        )
                    if level and i == num_res_blocks:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch

            self.out = nn.Sequential(
                GroupNorm32(32, ch),
                SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=torch.float16)),
            )
            

    def forward(self, x, t, ctx):
        acts = []
        temb = self.time_embed(timestep_embedding(t, self.model_channels))
        
        h = x.type(self.dtype)
        ctx = ctx.type(self.dtype)

        for block in self.input_blocks:
            h = block(h, temb, ctx)
            acts.append(h)

        h = self.middle_block(h, temb, ctx)

        for block in self.output_blocks:
            h = torch.cat([h, acts.pop()], dim=1)
            h = block(h, temb, ctx)

        h = h.type(x.dtype)

        return self.out(h)
    
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def timestep_embedding(timesteps, dim, max_period=10000):
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim // 2, dtype=torch.float32) / (dim // 2)
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding