# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Callable, List, Optional, Tuple, Union
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp


class ApproxGELU(nn.GELU):
    def __init__(self):
        super().__init__(approximate="tanh")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=ApproxGELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # print(x.shape, gate_msa.shape, shift_msa.shape, scale_msa.shape)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EEGDiffuser(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=32,
        time_points=2000,
        embed_dim=512,
        patch_size=5,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=9,
        learn_sigma=True,

    ):
        super().__init__()
        if time_points % patch_size:
            raise ValueError("time_points must be divisible by patch_size")
        self.self_condition = True
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.time_points = time_points
        self.sequence_len = time_points // patch_size

        self.x_embedder = PatchEmbed(
            time_points = time_points,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.x_self_cond_embedder = (
            PatchEmbed(time_points=time_points, patch_size=patch_size,
                       in_chans=in_channels, embed_dim=embed_dim)
            if self.self_condition else None
        )
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.empty(1, self.sequence_len, embed_dim).normal_(std=0.02))

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(22 ** 0.5))
        # print(pos_embed.shape, self.pos_embed.data.shape)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)


        if self.x_self_cond_embedder is not None:
            nn.init.zeros_(self.x_self_cond_embedder.proj.weight)
            if self.x_self_cond_embedder.proj.bias is not None:
                nn.init.zeros_(self.x_self_cond_embedder.proj.bias)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        for layer in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            assert isinstance(layer, nn.Linear)
            nn.init.normal_(layer.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            assert isinstance(block, DiTBlock)
            self._zero_linear(block.adaLN_modulation[-1])

        # Zero-out output layers:
        self._zero_linear(self.final_layer.adaLN_modulation[-1])
        self._zero_linear(self.final_layer.linear)

    @staticmethod
    def _zero_linear(layer):
        assert isinstance(layer, nn.Linear)
        nn.init.constant_(layer.weight, 0)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


    def forward(self, x, t, y, x_self_cond=None, force_drop_ids=None):
        """
        Forward pass of DiT.
        x: (B, channels, time_points) tensor of noisy EEG.
        t: (B,) tensor of diffusion timesteps.
        y: (B,) tensor of class labels.
        """
        B, C, P = x.shape
        if x_self_cond is not None and not self.self_condition:
            raise ValueError("Self-conditioning is disabled for this model")
        if self.x_self_cond_embedder is not None:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            if x_self_cond.shape != x.shape:
                raise ValueError("x_self_cond must match the noisy EEG shape")
            feedback = self.x_self_cond_embedder(x_self_cond)
            x = self.x_embedder(x) + feedback + self.pos_embed
        else:
            x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        x = x.contiguous().view(B, self.time_points, self.out_channels)
        x = x.transpose(1, 2)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, x_self_cond=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if x_self_cond is not None:
            if x_self_cond.shape != x.shape:
                raise ValueError("x_self_cond must match the noisy EEG shape")
            half_cond = x_self_cond[:len(half)]
            x_self_cond = torch.cat([half_cond, half_cond], dim=0)
        model_out = self.forward(combined, t, y, x_self_cond=x_self_cond)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class PatchEmbed(nn.Module):

    def __init__(
            self,
            time_points: Optional[int] = 800,
            patch_size: int = 5,
            in_chans: int = 22,
            embed_dim: int = 256,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, P = x.shape
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    # CPU shape check for a 4-second, 22-channel, 250-Hz motor imagery trial.
    model = EEGDiffuser(
        in_channels=22, time_points=1000, patch_size=10,
        embed_dim=192, num_heads=4, num_classes=2,
    )
    x = torch.randn(2, 22, 1000)
    t = torch.tensor([1, 500])
    y = torch.tensor([0, 1])
    print(model(x, t, y).shape)  # torch.Size([2, 44, 1000])
