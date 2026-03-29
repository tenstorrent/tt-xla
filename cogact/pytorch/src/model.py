# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CogACT model architecture: DiT (Diffusion Transformer) action model and VLM+DiT wrapper.

Based on the CogACT paper (https://github.com/microsoft/CogACT) which extends
a Prismatic VLM with a DiT-based diffusion action head for robotic manipulation.
"""
import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds conditioning labels (VLM cognition features) with CFG support."""

    def __init__(self, conditioning_dim, hidden_size):
        super().__init__()
        self.projection = nn.Linear(conditioning_dim, hidden_size)
        self.uncondition = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, z, force_uncond=False):
        if force_uncond:
            return self.uncondition.unsqueeze(0).expand(z.shape[0], -1)
        return self.projection(z)


class ActionEmbedder(nn.Module):
    """Embeds action vectors into hidden representations."""

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.projection = nn.Linear(action_dim, hidden_size)

    def forward(self, x):
        return self.projection(x)


class DiTBlock(nn.Module):
    """Transformer block with self-attention and MLP (simple residual)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Final layer: LayerNorm + Linear to action dimension."""

    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class DiT(nn.Module):
    """Diffusion Transformer for action prediction.

    DiT-Base configuration: depth=12, hidden_size=768, num_heads=12.
    Takes noisy actions, diffusion timestep, and VLM cognition features as input,
    and predicts denoised actions.
    """

    def __init__(
        self,
        action_dim=7,
        hidden_size=768,
        depth=12,
        num_heads=12,
        conditioning_dim=4096,
        num_action_tokens=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_tokens = num_action_tokens

        self.x_embedder = ActionEmbedder(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(conditioning_dim, hidden_size)

        # 1 conditioning token + num_action_tokens
        num_tokens = 1 + num_action_tokens
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_tokens, hidden_size)
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )

        self.final_layer = FinalLayer(hidden_size, action_dim)

        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, x, t, z):
        """
        Args:
            x: Noisy actions (B, num_action_tokens, action_dim)
            t: Diffusion timesteps (B,)
            z: Conditioning features from VLM (B, conditioning_dim)

        Returns:
            Predicted noise/actions (B, num_action_tokens, action_dim)
        """
        x_emb = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        z_emb = self.z_embedder(z)

        cond_token = (t_emb + z_emb).unsqueeze(1)
        tokens = torch.cat([cond_token, x_emb], dim=1)
        tokens = tokens + self.positional_embedding

        for block in self.blocks:
            tokens = block(tokens)

        action_tokens = tokens[:, 1:, :]
        return self.final_layer(action_tokens)


class CogACTWrapper(nn.Module):
    """Wraps PrismaticVLM + DiT action model for inference.

    Runs the VLM forward pass to extract cognition features (last hidden state),
    then runs a single DiT forward pass to produce action predictions.
    """

    def __init__(self, vlm, action_model):
        super().__init__()
        self.vlm = vlm
        self.action_model = action_model

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = vlm_outputs.hidden_states[-1]
        cognition_features = hidden_states[:, -1, :]

        B = cognition_features.shape[0]
        device = cognition_features.device
        dtype = cognition_features.dtype
        timesteps = torch.zeros(B, dtype=torch.long, device=device)
        noisy_actions = torch.randn(
            B,
            self.action_model.num_action_tokens,
            self.action_model.action_dim,
            device=device,
            dtype=dtype,
        )

        return self.action_model(noisy_actions, timesteps, cognition_features)
