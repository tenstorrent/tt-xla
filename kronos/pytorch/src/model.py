# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/shiyu-coder/Kronos
# License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .module import (
    HierarchicalEmbedding,
    TemporalEmbedding,
    TransformerBlock,
    RMSNorm,
    DependencyAwareLayer,
    DualHead,
)


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos foundation model for financial time series forecasting.

    A decoder-only Transformer that operates on hierarchical discrete tokens
    (s1/s2 pairs) produced by the Kronos tokenizer from OHLCV candlestick data.

    Args:
        s1_bits: Number of bits for pre tokens.
        s2_bits: Number of bits for post tokens.
        n_layers: Number of Transformer blocks.
        d_model: Dimension of the model's embeddings and hidden states.
        n_heads: Number of attention heads.
        ff_dim: Dimension of the feedforward network.
        ffn_dropout_p: Dropout probability for the feedforward network.
        attn_dropout_p: Dropout probability for the attention layers.
        resid_dropout_p: Dropout probability for residual connections.
        token_dropout_p: Dropout probability for token embeddings.
        learn_te: Whether to use learnable temporal embeddings.
    """

    def __init__(
        self,
        s1_bits,
        s2_bits,
        n_layers,
        d_model,
        n_heads,
        ff_dim,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        token_dropout_p,
        learn_te,
    ):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2**self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model**-0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Forward pass of the Kronos model.

        Args:
            s1_ids: Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids: Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp: Optional temporal stamp tensor. Shape: [batch_size, seq_len, 5]
            padding_mask: Optional mask for padding tokens. Shape: [batch_size, seq_len]

        Returns:
            Tuple of (s1_logits, s2_logits) with shapes
            [batch_size, seq_len, s1_vocab_size] and [batch_size, seq_len, s2_vocab_size].
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        s1_probs = F.softmax(s1_logits.detach(), dim=-1)
        sample_s1_ids = torch.multinomial(
            s1_probs.view(-1, self.s1_vocab_size), 1
        ).view(s1_ids.shape)
        sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits
