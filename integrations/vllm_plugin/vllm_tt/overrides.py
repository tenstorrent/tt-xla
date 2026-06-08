# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import OrderedDict

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


def tt_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Fused TT SDPA attention for an HF vision tower.

    vLLM has no API to route a HF-instantiated vision tower's attention (e.g.
    Gemma-4's Gemma4VisionAttention) to a device backend, so we register this
    into HF's ALL_ATTENTION_FUNCTIONS and point the tower's _attn_implementation
    at it.

    Q/K/V: [batch, heads, seq, head_dim]; returns (attn_output, None) with
    attn_output [batch, seq, heads, head_dim] (the sdpa_attention_forward
    contract).
    """
    n_rep = getattr(module, "num_key_value_groups", 1) or 1
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", False)
    )
    if scaling is None:
        scaling = float(query.shape[-1]) ** -0.5

    # The TT SDPA op asserts query/key seq length is tile-aligned (multiple of
    # 32). Vision encoders typically produce non-aligned patch counts (e.g.
    # Gemma-4 vision uses 630). Pad Q/K/V along the seq dim with zeros and
    # add an additive mask to zero out softmax contributions from padded K
    # positions. Trim padded Q rows from the output afterwards.
    _TILE = 32
    seq_q = query.shape[2]
    seq_k = key.shape[2]
    pad_q = (-seq_q) % _TILE
    pad_k = (-seq_k) % _TILE

    if pad_q > 0:
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_q))
    if pad_k > 0:
        key = torch.nn.functional.pad(key, (0, 0, 0, pad_k))
        value = torch.nn.functional.pad(value, (0, 0, 0, pad_k))
        masked_value = torch.finfo(query.dtype).min
        # TTIR SDPA op requires the mask's dim 2 to equal the (padded) query
        # sequence length — broadcast-shaped [B, 1, 1, K] is rejected. Use the
        # explicit [B, 1, Q_pad, K_pad] layout. Since masking only depends on
        # the key position, every query row is identical.
        if attention_mask is None:
            attention_mask = torch.zeros(
                (query.shape[0], 1, seq_q + pad_q, seq_k + pad_k),
                dtype=query.dtype,
                device=query.device,
            )
            attention_mask[..., seq_k:] = masked_value
        else:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_k), value=masked_value
            )
        # The TT SDPA op disallows is_causal=True when attn_mask is set.
        is_causal = False

    attn_output = torch.ops.tt.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
        attn_mask=attention_mask,
        scale=scaling,
    )

    if pad_q > 0:
        attn_output = attn_output[:, :, :seq_q, :]

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# Idempotently register so reloads (e.g. pytest reruns) don't raise.
if "tt" not in ALL_ATTENTION_FUNCTIONS:
    ALL_ATTENTION_FUNCTIONS.register("tt", tt_sdpa_attention_forward)


class TTRMSNorm(nn.Module):
    """TT-compatible RMSNorm replacement for vLLM's RMSNorm.

    vLLM's RMSNorm.forward_native accesses `self.weight.data`, which causes an
    AssertionError during torch.compile/torch.export tracing with FakeTensors.
    Accessing `.data` on a FakeTensor lifts it out of the fake tensor context,
    resulting in: "cannot call `.data` on a Tensor, the Tensor is a FakeTensor".

    This class reimplements the RMSNorm forward pass using `self.weight` directly
    (without `.data`), making it compatible with TT tracing and compilation.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        assert isinstance(layer, RMSNorm)
        self.hidden_size = layer.hidden_size
        self.variance_epsilon = layer.variance_epsilon
        self.variance_size_override = layer.variance_size_override
        self.has_weight = layer.has_weight
        self.weight = layer.weight

        if hasattr(layer, "rocm_norm_func") and hasattr(
            layer, "rocm_norm_func_with_add"
        ):
            self.rocm_norm_func = layer.rocm_norm_func
            self.rocm_norm_func_with_add = layer.rocm_norm_func_with_add

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {self.hidden_size}, but found: {x.shape[-1]}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if self.hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {self.hidden_size}"
                )

            x_var = x[:, :, : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight and self.weight is not None:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual


class TTRotaryEmbedding(nn.Module):
    """TT-compatible RotaryEmbedding that computes cos/sin on-the-fly.

    vLLM's RotaryEmbedding pre-builds a cos_sin_cache and uses index_select
    (gather) with position_ids at runtime. This lowers to ttir.embedding which
    requires indices on host via from_device, breaking metal trace mode.

    This replacement computes cos/sin from inv_freq and positions using math
    ops (outer product + cos/sin) that stay entirely on device.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        assert isinstance(layer, RotaryEmbedding)
        self.head_size = layer.head_size
        self.rotary_dim = layer.rotary_dim
        self.is_neox_style = layer.is_neox_style
        # Delegates to the subclass implementation for correct frequency scaling
        # (e.g. Llama3RotaryEmbedding applies frequency-dependent scaling)
        inv_freq = layer._compute_inv_freq(layer.base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(num_tokens, -1, self.head_size)
        x_rot = ApplyRotaryEmb.forward_static(
            x[..., : self.rotary_dim], cos, sin, self.is_neox_style
        )
        if self.rotary_dim == self.head_size:
            return x_rot.reshape(orig_shape)
        return torch.cat((x_rot, x[..., self.rotary_dim :]), dim=-1).reshape(orig_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions_flat = positions.flatten().to(torch.float32)
        num_tokens = positions_flat.shape[0]

        freqs = torch.outer(positions_flat, self.inv_freq)
        cos = freqs.cos().to(query.dtype)
        sin = freqs.sin().to(query.dtype)

        query = self._apply_rotary(query, cos, sin, num_tokens)
        if key is not None:
            key = self._apply_rotary(key, cos, sin, num_tokens)
        return query, key


def get_fqn(module):
    return module.__class__.__qualname__


def tt_rmsnorm_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RMSNorm)
    return TTRMSNorm(layer)


def tt_rotary_embedding_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RotaryEmbedding)
    return TTRotaryEmbedding(layer)


MODULE_TYPE_TO_TT_OVERRIDE = OrderedDict(
    [
        ("RMSNorm", tt_rmsnorm_module),
    ]
)

# isinstance-based overrides for classes where subclasses need the same treatment
ISINSTANCE_OVERRIDES = [
    (RotaryEmbedding, tt_rotary_embedding_module),
]


def _route_vision_attention_to_tt(model: torch.nn.Module) -> None:
    """Point the HF vision/audio tower's _attn_implementation at the registered
    "tt" attention so its forward dispatches to tt_sdpa_attention_forward.
    No-op on towers without an HF config (getattr-guarded)."""
    for tower_attr in ("vision_tower", "audio_tower"):
        tower = getattr(model, tower_attr, None)
        if tower is None:
            continue
        cfg = getattr(tower, "config", None)
        if cfg is None:
            continue
        prev = getattr(cfg, "_attn_implementation", None)
        cfg._attn_implementation = "tt"
        logger.info("Routed %s attention through TT SDPA (was %r)", tower_attr, prev)


def _install_static_shape_merge_multimodal_embeddings() -> None:
    """Replace vLLM's masked_scatter_-based _merge_multimodal_embeddings: the
    boolean-mask scatter lowers to a dynamic-dim (set_dimension_size) tensor
    that tt-mlir's Shardy pass rejects. cumsum+gather+where gives the same
    result with static shapes. Idempotent (attribute-flag guarded)."""
    import vllm.model_executor.models.utils as _vllm_utils

    if getattr(_vllm_utils, "_tt_static_shape_merge_installed", False):
        return

    def _tt_merge_multimodal_embeddings(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        if len(multimodal_embeddings) == 0:
            return inputs_embeds
        mm_embeds_flat = _vllm_utils._flatten_embeddings(multimodal_embeddings).to(
            dtype=inputs_embeds.dtype
        )
        # Zero-based positional index of each mm token among the mm tokens
        # (0, 0, 1, 1, 2, ... where ascents happen at mm positions). Subtract 1
        # so non-mm positions point to index -1 (clamped to 0 below for safety).
        mm_indices = is_multimodal.to(torch.int64).cumsum(dim=0) - 1
        mm_indices = mm_indices.clamp(min=0)
        # Gather mm embeddings at those indices for every position. Non-mm
        # positions read garbage which torch.where then discards.
        mm_padded = mm_embeds_flat[mm_indices]
        merged = torch.where(is_multimodal.unsqueeze(-1), mm_padded, inputs_embeds)
        return merged

    _vllm_utils._merge_multimodal_embeddings = _tt_merge_multimodal_embeddings
    _vllm_utils._tt_static_shape_merge_installed = True
    logger.info(
        "Installed static-shape _merge_multimodal_embeddings (replaces "
        "masked_scatter_-based default that emits dynamic shapes)."
    )


_install_static_shape_merge_multimodal_embeddings()


def _promote_pre_allocated_attrs_to_buffers(model: torch.nn.Module) -> None:
    """Re-register plain torch.Tensor attributes as buffers so .to() moves them.

    vLLM's Gemma-4 multimodal model pre-allocates ``self.per_layer_embeddings``
    as a plain attribute (not a registered buffer) on CPU. ``model.to(device)``
    only relocates parameters and registered buffers, leaving it stranded on
    CPU; a later add against an XLA tensor then trips dynamo's mixed-device
    check. Promote such attributes to non-persistent buffers so .to() follows
    them.
    """
    pre_allocated_attrs = ("per_layer_embeddings",)
    for attr in pre_allocated_attrs:
        if not hasattr(model, attr):
            continue
        t = getattr(model, attr)
        if not isinstance(t, torch.Tensor):
            continue
        if attr in dict(model.named_buffers(recurse=False)):
            continue
        delattr(model, attr)
        model.register_buffer(attr, t, persistent=False)


def replace_modules(model: torch.nn.Module) -> None:
    logger.info(
        "Replacing vLLM modules with TT-compatible overrides where necessary..."
    )

    def _find_override(module):
        fqn = get_fqn(module)
        if fqn in MODULE_TYPE_TO_TT_OVERRIDE:
            return MODULE_TYPE_TO_TT_OVERRIDE[fqn](module)
        for base_cls, override_fn in ISINSTANCE_OVERRIDES:
            if isinstance(module, base_cls):
                return override_fn(module)
        return None

    def _process_module(module, name=None, parent=None):
        replacement = _find_override(module)
        if replacement is not None:
            assert (
                parent is not None and name is not None
            ), "Top Level module is not expected to be wrapped."
            logger.debug("replace %s with %s", module, replacement)
            setattr(parent, name, replacement)
            module = replacement

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
    _route_vision_attention_to_tt(model)
    _promote_pre_allocated_attrs_to_buffers(model)
