# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import OrderedDict

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


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


# Alignment requirement for TT hardware matmuls with BFP8 weight dtype.
# BFP8 packs elements in groups of 8; TT tiles are 32×32.
# We use 32 so it satisfies both constraints.
_TT_MATMUL_ALIGN = 32


def _pad_to_align(x: torch.Tensor, align: int) -> tuple[torch.Tensor, int]:
    """Pad dim-0 of x to the next multiple of align. Returns (padded_x, original_N)."""
    N = x.shape[0]
    pad_to = (N + align - 1) // align * align
    if pad_to > N:
        padding = x.new_zeros(pad_to - N, *x.shape[1:])
        x = torch.cat([x, padding], dim=0)
    return x, N


class TTPatchMerger(nn.Module):
    """Wraps pixtral PatchMerger to pad merged token count to a TT-aligned multiple.

    The BFP8 weight dtype constraint requires the activation (M) dimension of
    every matmul to be a multiple of 8 (and of the 32-element tile size).
    For a 1540×1540 Pixtral image the 2×2 spatial merge produces 55×55=3025
    tokens; 3025 % 8 = 1, violating the constraint. This wrapper pads to the
    next multiple of 32 before the Linear, then strips the padding rows after.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.spatial_merge_size = layer.spatial_merge_size
        self.mlp_input_dim = layer.mlp_input_dim
        self.merging_layer = layer.merging_layer

    def _permute(self, x: torch.Tensor, image_sizes: list) -> torch.Tensor:
        from vllm.model_executor.models.pixtral import get_sub_grids

        sub_grids = get_sub_grids(x, image_sizes, self.spatial_merge_size)
        permuted = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted.append(grid.view(-1, n_patches).t())
        return torch.cat(permuted, dim=0)

    def forward(self, x: torch.Tensor, image_sizes: list) -> torch.Tensor:
        x = self._permute(x, image_sizes)  # (N_merged, D * spatial_merge_size^2)
        x, N = _pad_to_align(x, _TT_MATMUL_ALIGN)
        x = self.merging_layer(x)  # (pad_to, D)
        return x[:N]


class TTVisionLanguageAdapter(nn.Module):
    """Wraps pixtral VisionLanguageAdapter to pad token count to a TT-aligned multiple.

    Same BFP8 constraint as TTPatchMerger — pads around both Linear projections
    and strips padding before returning so downstream split sizes remain correct.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.w_in = layer.w_in
        self.gelu = layer.gelu
        self.w_out = layer.w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, N = _pad_to_align(x, _TT_MATMUL_ALIGN)
        x = self.w_out(self.gelu(self.w_in(x)))  # (pad_to, out_dim)
        return x[:N]


def get_fqn(module):
    return module.__class__.__qualname__


def tt_rmsnorm_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RMSNorm)
    return TTRMSNorm(layer)


def tt_rotary_embedding_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RotaryEmbedding)
    return TTRotaryEmbedding(layer)


def tt_patch_merger_module(layer: torch.nn.Module) -> torch.nn.Module:
    try:
        from vllm.model_executor.models.pixtral import PatchMerger
    except ImportError:
        return layer
    if not isinstance(layer, PatchMerger):
        logger.warning("PatchMerger override skipped: unexpected type %s", type(layer).__name__)
        return layer
    return TTPatchMerger(layer)


def tt_vision_language_adapter_module(layer: torch.nn.Module) -> torch.nn.Module:
    try:
        from vllm.model_executor.models.pixtral import VisionLanguageAdapter
    except ImportError:
        return layer
    if not isinstance(layer, VisionLanguageAdapter):
        logger.warning(
            "VisionLanguageAdapter override skipped: unexpected type %s",
            type(layer).__name__,
        )
        return layer
    return TTVisionLanguageAdapter(layer)


MODULE_TYPE_TO_TT_OVERRIDE = OrderedDict(
    [
        ("RMSNorm", tt_rmsnorm_module),
        ("PatchMerger", tt_patch_merger_module),
        ("VisionLanguageAdapter", tt_vision_language_adapter_module),
    ]
)

# isinstance-based overrides for classes where subclasses need the same treatment
ISINSTANCE_OVERRIDES = [
    (RotaryEmbedding, tt_rotary_embedding_module),
]


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
