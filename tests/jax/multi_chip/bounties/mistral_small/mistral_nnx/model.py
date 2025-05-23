"""Mistral inference implemented in Flax NNX.

Can load weights from huggingface model

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

from contextlib import ExitStack
from enum import Enum
import functools
from flax import nnx
from flax.typing import Dtype, Initializer
from jax import Array, ShapeDtypeStruct
from jaxtyping import Float, Integer
from safetensors import safe_open
from transformers import MistralConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from typing import Any, Callable, Literal, Optional, Sequence
import flax.core.spmd
import flax.struct
import jax
import jax.numpy as jnp
import os
import safetensors
import safetensors.flax

from .embedding import apply_rotary_embedding, generate_fixed_pos_embedding
from .util import keystr_simple


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
    def create(cls, shape: tuple[int, ...], *, dtype: Dtype) -> "KVCacheLayer":
        assert len(shape) == 4, f"shape should be (B,S,H,D), got: {shape}"
        return cls(
            cache_k=jnp.zeros(shape, dtype=dtype),
            cache_v=jnp.zeros(shape, dtype=dtype),
            index=jnp.array(0, dtype="int32"),
        )

    def update(
        self, k: Float[Array, "B S H D"], v: Float[Array, "B S H D"]
    ) -> "KVCacheLayer":
        """Update the cache at the given index.

        Can be used for prefill by passing array with seqlen > 1.

        Args:
            k: key array of shape (seqlen, num_kv_heads, head_dim)
            v: value array of same shape

        Returns:
            updated cache layer with seqlen incremented by the amount from input.
        """
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
        layers: int,
        batch_size: int,
        max_seqlen: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: Dtype,
    ) -> "KVCache":
        shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        return cls(
            layers=[KVCacheLayer.create(shape, dtype=dtype) for _ in range(layers)]
        )


class RotaryEmbedding(nnx.Module):
    def __init__(self, features: int, length: int, theta: float):
        """
        Args:
            features: Number of features
            length: Max context length
            theta: RoPE theta term
        """
        sin, cos = generate_fixed_pos_embedding(features, length, max_timescale=theta)
        self.sin = nnx.Variable(sin)
        self.cos = nnx.Variable(cos)

    def __call__(
        self,
        q: Float[Array, "B S HQ D"],
        k: Float[Array, "B S K D"],
        index: Optional[Integer[Array, "B"]] = None,
    ) -> tuple[Float[Array, "B S HQ D"], Float[Array, "B S K D"]]:
        """Apply rotary embedding to query and key arrays.

        Args:
            q: query of shape (batch, seqlen, heads, head_dim)
            k: key of shape (batch, seqlen, heads, head_dim)
            index: position offset of query shape (batch,).
                Used for incremental decoding w/ kv cache. seqlen==1 required
                for q if this is set.

        Returns:
            (q, k) with embedding applied.

        Note: This expects features to be ordered in odds and evens, i.e.
        `x1, x3, x5 ... x2, x4, x6 ...`.

        The weights for the whole model should be in this order. If using
        weights that expect features to be (i.e. the mistral weights as used by
        mistral-inference lib), it has to be converted into this order.
        """
        # Limitation of flaxformers apply_rotary_embedding
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
    """Returns a function that when invoked, returns an annotated initializer
    when given logical axis annotations.
    """

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

    def __call__(self, x: Array) -> Array:
        """
        Args:
          x: Array of shape (batch, seqlen, dim)
        """
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
        """Decode using KV cache.

        Args:
            x: The next items in the sequence. Shape (batch, seqlen, dim)

        Returns:
            (output, cache_layer) the output and updated cache layer.
        """
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

    def __init__(
        self,
        config: MistralConfig,
        *,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        embed_init = _init_with_sharding(nnx.initializers.lecun_normal())
        norm_init = _init_with_sharding(nnx.initializers.zeros_init())
        linear_init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.config = config
        self.dtype = dtype

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
        """Inference mode using kvcache.

        Args:
            input_ids: Array of shape (batch, seqlen)
            cache: the KV Cache, for incremental decoding.
        Returns:
            array of shape (batch, seqlen, vocab_size)
        """
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h, cache.layers[i] = layer.decode(h, cache.layers[i])
        h = self.norm(h)
        logits = self.output(h)
        return logits, cache

    def create_cache(self, batch_size: int, max_seqlen: int) -> KVCache:
        head_dim = self.config.head_dim
        assert type(head_dim) is int
        return KVCache.create(
            len(self.layers),
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
        )

    def save(self, path: str):
        """Save model parameters into a safetensor file."""
        state = nnx.state(self, nnx.OfType(nnx.Param))
        tensors = {}
        for k, v in nnx.to_flat_state(state):
            tensors[keystr_simple(k, separator="/")] = v.value
        safetensors.flax.save_file(tensors, path)

    @classmethod
    def _load_with(
        cls,
        loader: Callable[[nnx.State], nnx.State],
        model_name: str,
        dtype: Dtype,
        param_dtype: Dtype,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ):
        """Create a model instance using the specific weight loading function.

        Args:
            loader: weight loading function that takes abstract Param state and
                returns loaded Params.
        """
        config = MistralConfig.from_pretrained(model_name)
        assert isinstance(config, MistralConfig)
        abs_model = nnx.eval_shape(
            lambda: cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(0),
            )
        )
        graphdef = nnx.graphdef(abs_model)
        abs_params = nnx.state(abs_model, nnx.OfType(nnx.Param))

        # Initialize non-param states from an actual new instance.
        # The non-returned arrays should be eliminated by jit.
        # This gets things like precomputed rope-embedding constants.
        @jax.jit
        def non_param():
            model = cls(config, dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(0))
            return nnx.state(model, nnx.Not(nnx.OfType(nnx.Param)))

        non_params = jax.block_until_ready(non_param())
        params = loader(abs_params)

        if sharding_rules:
            with flax.core.spmd.logical_axis_rules(sharding_rules):
                pspecs = nnx.get_partition_spec(abs_params)
            params = jax.lax.with_sharding_constraint(params, pspecs)
        return nnx.merge(graphdef, non_params, params)

    @classmethod
    def load(
        cls,
        model_name: str,
        param_path: str,
        dtype="float32",
        param_dtype="bfloat16",
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ) -> "MistralModel":
        """Load model from safetensor file saved with `save`.

        Args:
            model_name: HF model name
            param_path: path to the previously created safetensor file
            dtype: computation dtype
            param_dtype: dtype of the parameters
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).
        """
        return cls._load_with(
            functools.partial(_load_saved_params, param_path),
            model_name,
            dtype,
            param_dtype,
            sharding_rules,
        )

    @classmethod
    def load_from_hf_pt_model(
        cls,
        model_name: str,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.bfloat16,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ) -> "MistralModel":
        """Load model from HF pytorch model.

        Renames, transposes, and reshapes tensors as necessary.

        Args:
            model_name: HF model name
            dtype: computation dtype
            param_dtype: dtype of the parameters
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).

        Returns:
            MistralModel with loaded weights.
        """
        return cls._load_with(
            functools.partial(_load_hf_pt_params, model_name),
            model_name,
            dtype,
            param_dtype,
            sharding_rules,
        )


def _load_saved_params(path: str, abs_state: nnx.State) -> nnx.State:
    """Load model from safetensors file.

    Args:
       path: path to safetensors file.
        abs_state: abstract state of nnx.Params of the model.

    Returns:
        nnx.State with actual param arrays that can be `nnx.merge`d into the
        model.
    """
    with safe_open(path, framework="np") as f:

        def load_one(path, abs_param: ShapeDtypeStruct):
            k = keystr_simple(path[:-1], separator="/")
            param = jnp.array(f.get_tensor(k), dtype=abs_param.dtype)
            assert (
                param.shape == abs_param.shape
            ), f"Wrong shape for {keystr_simple(path, separator='/')}. Expected: {abs_param.shape}, actual: {param.shape}"
            return param

        return jax.tree_util.tree_map_with_path(load_one, abs_state)


def _load_hf_pt_params(hf_model: str, abs_state: nnx.State) -> nnx.State:
    """Load model from HF pytorch model.

    Renames, transposes, and reshapes tensors as necessary.

    Args:
        hf_model: HF model name or path.
        abs_state: abstract state of nnx.Params of the model.

    Returns:
        nnx.State with actual param arrays that can be `nnx.merge`d into the
        model.
    """
    HF_MODEL_SHARD_INDEX = "model.safetensors.index.json"
    index = cached_file(hf_model, HF_MODEL_SHARD_INDEX)
    shard_paths, meta = get_checkpoint_shard_files(hf_model, index)
    assert isinstance(shard_paths, list)

    with ExitStack() as stack:
        shards = {
            os.path.basename(s): stack.enter_context(safe_open(s, "flax"))
            for s in shard_paths
        }

        def transpose_only(param: jax.Array, _abs_param):
            return param.transpose()

        def transpose_reshape(param: jax.Array, abs_param):
            return param.transpose().reshape(abs_param.shape)

        def identity(param: jax.Array, _abs_param):
            return param

        def load_one(path, abs_param):
            # Map to hf pt model (name, need_transpose)
            name_map = {
                "embed/embedding/value": ("model.embed_tokens.weight", identity),
                "output/kernel/value": ("lm_head.weight", transpose_only),
                "norm/scale/value": ("model.norm.weight", identity),
            }
            layer_name_map = {
                "attention/wq/kernel/value": (
                    "self_attn.q_proj.weight",
                    transpose_reshape,
                ),
                "attention/wk/kernel/value": (
                    "self_attn.k_proj.weight",
                    transpose_reshape,
                ),
                "attention/wv/kernel/value": (
                    "self_attn.v_proj.weight",
                    transpose_reshape,
                ),
                "attention/wo/kernel/value": (
                    "self_attn.o_proj.weight",
                    transpose_reshape,
                ),
                "attention_norm/scale/value": ("input_layernorm.weight", identity),
                "mlp/w1/kernel/value": ("mlp.gate_proj.weight", transpose_only),
                "mlp/w2/kernel/value": ("mlp.down_proj.weight", transpose_only),
                "mlp/w3/kernel/value": ("mlp.up_proj.weight", transpose_only),
                "ffn_norm/scale/value": (
                    "post_attention_layernorm.weight",
                    identity,
                ),
            }
            if path[0].key == "layers":
                idx = path[1].key
                layer_name, postprocess = layer_name_map[
                    keystr_simple(path[2:], separator="/")
                ]
                name = f"model.layers.{idx}.{layer_name}"
            else:
                name, postprocess = name_map[keystr_simple(path, separator="/")]

            param = shards[meta["weight_map"][name]].get_tensor(name)
            param = postprocess(param, abs_param)
            assert (
                param.shape == abs_param.shape
            ), f"Wrong shape for {keystr_simple(path, separator='/')}. Expected: {abs_param.shape}, actual: {param.shape}"
            return param.astype(abs_param.dtype)

        return jax.tree_util.tree_map_with_path(load_one, abs_state)


def convert_and_save_if_not_exist(
    weights_path: str,
    model_name: str,
    param_dtype: Dtype,
):
    if os.path.exists(weights_path):
        return
    model = MistralModel.load_from_hf_pt_model(model_name, param_dtype=param_dtype)
    model.save(weights_path)
