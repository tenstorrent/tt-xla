# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistral inference implemented in Flax NNX.

Can load weights from huggingface model.

Conventions used for axis labeling:

- B: batch
- S: seqlen
- V: vocab
- E: embed
- D: head_dim
- H: num_heads
- HQ: num_q_heads
- K: num_kv_heads

Note: The rotary embedding implementation used here is from flaxformers, which
is same as hugginface transformers'. This is not compatible with
mistral-inference's implementation. The order of the features must be swapped if
using mistral weights. More details in class `RotaryEmbedding`.

Not supported:
- Sliding window

"""

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import flax.struct
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer, LogicalRules
from jax import Array, ShapeDtypeStruct
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding
from jaxtyping import Float, Integer
from transformers import MistralConfig

from .embedding import apply_rotary_embedding, generate_fixed_pos_embedding


class Axis(str, Enum):
    EMBED = "embed"
    MLP = "mlp"
    HEAD = "head"
    QHEAD = "qhead"
    KVHEAD = "kvhead"
    VOCAB = "vocab"

    def __str__(self) -> str:
        return self.value


@flax.struct.dataclass
class KVCacheLayer:
    cache_k: Float[Array, "B S H D"]
    cache_v: Float[Array, "B S H D"]
    index: Integer[Array, ""]

    @property
    def max_seqlen(self) -> int:
        return self.cache_k.shape[1]

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCacheLayer":
        assert len(shape) == 4, f"shape should be (B,S,H,D), got: {shape}"

        if sharding_rules is None and mesh is None:
            sharding = SingleDeviceSharding(jax.devices("cpu")[0])
        elif sharding_rules is not None and mesh is not None:
            rules_dict = {k: v for k, v in sharding_rules}
            sharding = NamedSharding(
                mesh,
                PartitionSpec(
                    None, None, rules_dict[Axis.KVHEAD], rules_dict[Axis.HEAD]
                ),
            )
        else:
            raise ValueError("mesh and sharding_rules must be both None or both set")

        return cls(
            cache_k=jnp.zeros(shape, dtype=dtype, device=sharding),
            cache_v=jnp.zeros(shape, dtype=dtype, device=sharding),
            index=jnp.array(0, dtype="int32"),
        )

    def update(
        self, k: Float[Array, "B S H D"], v: Float[Array, "B S H D"]
    ) -> "KVCacheLayer":
        KB, KS, _KH, _KD = k.shape
        VB, VS, _VH, _VD = v.shape
        assert (KB, KS) == (
            VB,
            VS,
        ), f"k and v should have same batch,seqlen: {(KB, KS)} != {(VB,VS)}"
        Z = jnp.array(0, dtype=self.index.dtype)
        return KVCacheLayer(
            cache_k=jax.lax.dynamic_update_slice(
                self.cache_k, k, (Z, self.index, Z, Z)
            ),
            cache_v=jax.lax.dynamic_update_slice(
                self.cache_v, v, (Z, self.index, Z, Z)
            ),
            index=self.index + KS,
        )


@flax.struct.dataclass
class KVCache:
    layers: list[KVCacheLayer]

    @classmethod
    def create(
        cls,
        num_layers: int,
        batch_size: int,
        max_seqlen: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCache":
        shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        return cls(
            layers=[
                KVCacheLayer.create(shape, dtype, mesh, sharding_rules)
                for _ in range(num_layers)
            ]
        )


class RotaryEmbedding(nnx.Module):
    def __init__(self, features: int, length: int, theta: float):
        sin, cos = generate_fixed_pos_embedding(features, length, max_timescale=theta)
        self.sin = nnx.Variable(sin)
        self.cos = nnx.Variable(cos)

    def __call__(
        self,
        q: Float[Array, "B S HQ D"],
        k: Float[Array, "B S K D"],
        index: Optional[Integer[Array, "B"]] = None,
    ) -> tuple[Float[Array, "B S HQ D"], Float[Array, "B S K D"]]:
        assert index is None or q.shape[1] == 1, "seqlen==1 required when index is set"
        out_q, out_k = apply_rotary_embedding(
            q,
            k,
            self.cos.value,
            self.sin.value,
            decode=index is not None,
            rotary_index=index,
        )
        return out_q.astype(q.dtype), out_k.astype(k.dtype)


def _init_with_sharding(
    init_fn: Initializer,
) -> Callable[[tuple[Axis, ...]], Initializer]:
    def init(sharding):
        return nnx.with_partitioning(init_fn, sharding=sharding)

    return init


class FeedForward(nnx.Module):
    def __init__(
        self, dim: int, hidden_dim: int, dtype: Any, param_dtype: Dtype, rngs: nnx.Rngs
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.param_dtype = param_dtype

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.w1 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w2 = nnx.LinearGeneral(
            self.hidden_dim,
            self.dim,
            kernel_init=init((Axis.MLP, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w3 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Float[Array, "B S E"]) -> Float[Array, "B S E"]:
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class Attention(nnx.Module):
    """Mistral attention supports different number of Q heads vs KV heads."""

    dim: int
    n_q_heads: int
    head_dim: int
    n_kv_heads: int
    dtype: Any

    wq: nnx.LinearGeneral
    wk: nnx.LinearGeneral
    wv: nnx.LinearGeneral
    wo: nnx.LinearGeneral

    def __init__(
        self,
        dim: int,
        n_q_heads: int,
        head_dim: int,
        n_kv_heads: int,
        rope: RotaryEmbedding,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.n_q_heads = n_q_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.rope = rope
        self.dtype = dtype

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.wq = nnx.LinearGeneral(
            self.dim,
            (self.n_q_heads, self.head_dim),
            use_bias=False,
            kernel_init=init((Axis.EMBED, Axis.QHEAD, Axis.HEAD)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wk = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wv = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wo = nnx.LinearGeneral(
            (self.n_q_heads, self.head_dim),
            self.dim,
            axis=(-2, -1),
            kernel_init=init((Axis.QHEAD, Axis.HEAD, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @property
    def _queries_per_head(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    def __call__(self, x: Float[Array, "B S E"]) -> Array:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.rope(xq, xk)

        out = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True)
        out = self.wo(out)
        return out

    def decode(
        self,
        x: Float[Array, "B S E"],
        cache: KVCacheLayer,
    ) -> tuple[Float[Array, "B S E"], KVCacheLayer]:
        B, T, _E = x.shape
        assert B == 1, "only batch size 1 supported for now"
        assert T == 1, "decode takes one token at a time"

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        cache = cache.update(xk, xv)
        xk = cache.cache_k
        xv = cache.cache_v

        index = jnp.array([cache.index - T], dtype="int32")
        xq, xk = self.rope(xq, xk, index=index)

        out = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            query_seq_lengths=jnp.array([T], dtype="int32"),
            key_value_seq_lengths=cache.index.reshape(B),
        )
        out = self.wo(out)
        return out, cache


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        rope: RotaryEmbedding,
        dtype: Any,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        init = _init_with_sharding(nnx.initializers.ones_init())

        self.n_q_heads = n_q_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope=rope,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ffn_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "B S E"]) -> Float[Array, "B S E"]:
        r = self.attention(self.attention_norm(x))
        h = x + r
        r = self.mlp(self.ffn_norm(h))
        return h + r

    def decode(
        self,
        x: Float[Array, "B S E"],
        cache: KVCacheLayer,
    ) -> tuple[Float[Array, "B S E"], KVCacheLayer]:
        r, cache = self.attention.decode(self.attention_norm(x), cache)
        h = x + r
        r = self.mlp(self.ffn_norm(h))
        return h + r, cache


class MistralModel(nnx.Module):
    layers: list[TransformerBlock]
    config: MistralConfig
    sharding_rules: LogicalRules | None

    def __init__(
        self,
        config: MistralConfig,
        *,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
        sharding_rules: LogicalRules | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.sharding_rules = sharding_rules

        embed_init = _init_with_sharding(nnx.initializers.lecun_normal())
        norm_init = _init_with_sharding(nnx.initializers.zeros_init())
        linear_init = _init_with_sharding(nnx.initializers.lecun_normal())

        head_dim = config.head_dim
        assert type(head_dim) is int

        rope = RotaryEmbedding(
            features=head_dim,
            length=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.embed = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=embed_init((Axis.VOCAB, Axis.EMBED)),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.norm = nnx.RMSNorm(
            config.hidden_size,
            scale_init=norm_init((Axis.EMBED,)),
            epsilon=config.rms_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=linear_init((Axis.EMBED, Axis.VOCAB)),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.layers = [
            TransformerBlock(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                n_q_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                norm_eps=config.rms_norm_eps,
                rope=rope,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in range(0, config.num_hidden_layers)
        ]

    def __call__(self, input_ids: Integer[Array, "B S"]) -> Float[Array, "B S V"]:
        """
        Args:
            input_ids: Array of shape (batch, seqlen)
        Returns:
            array of shape (batch, seqlen, vocab_size)
        """
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits

    def decode(
        self,
        input_ids: Integer[Array, "B S"],
        cache: KVCache,
    ) -> tuple[Float[Array, "B S V"], KVCache]:
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h, cache.layers[i] = layer.decode(h, cache.layers[i])
        h = self.norm(h)
        logits = self.output(h)
        return logits, cache

    def create_cache(
        self, batch_size: int, max_seqlen: int, mesh: Mesh | None = None
    ) -> KVCache:
        head_dim = self.config.head_dim
        assert type(head_dim) is int
        return KVCache.create(
            len(self.layers),
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            mesh=mesh,
            sharding_rules=self.sharding_rules,
        )
