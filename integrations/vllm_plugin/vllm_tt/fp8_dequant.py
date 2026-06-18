# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FP8 -> bf16 dequantizing linear method for the TT (OOT) platform.

vLLM's ``Fp8LinearMethod`` selects a hardware fp8 GEMM kernel in its
``__init__`` via ``init_fp8_linear_kernel`` ->
``choose_scaled_mm_linear_kernel``, which indexes a kernel table by
``current_platform._enum``. The TT platform reports ``PlatformEnum.OOT``,
which is not a key in that table, so constructing the method raises
``KeyError: <PlatformEnum.OOT: 6>`` before the model can even load.

The TT backend has no fp8 GEMM kernel; instead it runs everything in
bf16. So rather than wiring up an fp8 kernel, we DEQUANTIZE fp8 weights
to plain bf16 at load time. After dequant the linear layer holds a normal
bf16 ``weight`` parameter (no ``weight_scale`` / ``input_scale``), and the
existing TT path (``shard_model`` + ``Xla*ParallelLinear``) works
unchanged because it just reads ``layer.weight`` and assumes bf16.

Supported (the Devstral / mistralai per-tensor static fp8 case):
  - per-tensor static fp8 weights (``float8_e4m3fn`` + per-(logical-)shard
    ``weight_scale``), with ``activation_scheme`` static or dynamic.
  - fused modules (QKV / MergedColumn) that carry one scale per logical
    shard; each shard is dequantized with its own scale.
  - ``modules_to_not_convert`` / ``ignored_layers`` (e.g. ``lm_head``,
    vision tower) are left to vLLM's ``UnquantizedLinearMethod`` and are
    never touched here.

NOT supported (guarded with a clear error):
  - block-quantized weights (``weight_block_size`` set / ``weight_scale_inv``).
  - per-channel weight scales (scale not reducible to one value per shard).

The activation ``input_scale`` (static activation scheme) is simply dropped:
once weights are bf16 there is no fp8 activation quantization on TT.
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.utils import replace_parameter

from .logger import tt_init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = tt_init_logger(__name__)


@register_weight_loader_v2_supported_method
class TTFp8DequantLinearMethod(Fp8LinearMethod):
    """Fp8 linear method that dequantizes to bf16 instead of running an
    fp8 GEMM kernel.

    We subclass ``Fp8LinearMethod`` so that ``create_weights`` (which
    registers the fp8 ``weight`` + ``weight_scale`` [+ ``input_scale``]
    parameters the checkpoint loader fills in) is reused verbatim. We
    override ``__init__`` to SKIP ``init_fp8_linear_kernel`` (the source of
    the ``KeyError: OOT``) and override ``process_weights_after_loading`` to
    dequantize the loaded fp8 weight into a plain bf16 ``weight``.
    """

    def __init__(self, quant_config: Fp8Config):
        # NOTE: intentionally do NOT call super().__init__(); that calls
        # init_fp8_linear_kernel(...) which raises KeyError on the OOT
        # platform. We only need the few attributes that create_weights and
        # our dequant path read.
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        self.marlin_input_dtype = None

        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = quant_config.activation_scheme == "static"

        # Attributes referenced by the (unused-on-TT) base apply()/process
        # branches. Set to safe values so any stray access doesn't explode.
        self.use_marlin = False
        self.use_deep_gemm = False
        self.use_aiter_and_is_supported = False
        self.fp8_linear = None

        if self.block_quant:
            raise NotImplementedError(
                "TT fp8 dequant does not support block-quantized fp8 "
                "checkpoints (weight_block_size set / weight_scale_inv). "
                "Only per-tensor static/dynamic fp8 is supported."
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        """Dequantize the fp8 weight to a plain bf16 weight.

        Per-tensor (static) fp8: ``weight`` is ``[N, K]`` float8_e4m3fn and
        ``weight_scale`` is ``[num_logical_widths]`` (one scale per fused
        shard). We dequantize each logical shard with its own scale and
        replace ``layer.weight`` with a contiguous bf16 ``[N, K]`` tensor in
        the SAME (non-transposed) orientation the original bf16 checkpoint
        would have — this is what ``Xla*ParallelLinear`` and the plain
        Column/Row partitioners expect.
        """
        weight = layer.weight
        weight_scale = layer.weight_scale

        if weight.dtype != torch.float8_e4m3fn:
            # Already a plain weight (nothing to do). Be permissive.
            logger.warning(
                "TTFp8DequantLinearMethod: weight dtype is %s, expected "
                "float8_e4m3fn; skipping dequant.",
                weight.dtype,
            )
            return

        target_dtype = getattr(layer, "orig_dtype", torch.bfloat16)
        if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            target_dtype = torch.bfloat16

        logical_widths = getattr(layer, "logical_widths", None)
        scale = weight_scale.data.to(torch.float32).reshape(-1)

        out_dim = weight.shape[0]

        if logical_widths and len(logical_widths) == scale.numel():
            # Per-fused-shard scales (QKV / MergedColumn). Build a per-row scale
            # vector once, then scale in fp32 and round ONCE to the target dtype.
            # This is a single fused pass (no torch.empty prealloc of the output,
            # no per-shard temporaries, no redundant .contiguous() copy). We
            # scale in fp32 rather than bf16: truncating the fp32 weight_scale to
            # bf16 before the multiply adds a systematic per-shard bias (~2x the
            # worst-case dequant error) on top of the unavoidable final bf16
            # rounding. fp8->fp32 is exact, the fp32 multiply is exact, and we
            # round once at the end. The transient fp32 buffer is freed right
            # after; host dequant is <0.1% of load time, so this costs nothing
            # that matters.
            rows = torch.empty((out_dim, 1), dtype=torch.float32, device=weight.device)
            start = 0
            for idx, width in enumerate(logical_widths):
                if width:
                    rows[start : start + width, 0] = scale[idx]  # scale is fp32
                start += width
            assert start == out_dim, (
                f"logical_widths {logical_widths} sum {start} != weight "
                f"out dim {out_dim}"
            )
            deq = (weight.to(torch.float32) * rows).to(target_dtype)
        elif scale.numel() == 1:
            deq = (weight.to(torch.float32) * scale[0]).to(target_dtype)
        else:
            raise NotImplementedError(
                "TT fp8 dequant only supports per-tensor (or per-fused-shard) "
                f"weight scales. Got weight_scale with {scale.numel()} "
                f"elements for weight of shape {tuple(weight.shape)} "
                f"(logical_widths={logical_widths}). Per-channel fp8 is not "
                "supported."
            )

        # deq is already contiguous (fresh .to() result); no extra copy needed.
        replace_parameter(layer, "weight", deq)

        # Drop fp8 quant state: after dequant the layer is plain bf16. Remove
        # the scale parameters so downstream code (and shard_model) see a
        # vanilla Linear-like module.
        for attr in ("weight_scale", "weight_scale_inv", "input_scale"):
            if hasattr(layer, attr):
                try:
                    delattr(layer, attr)
                except (AttributeError, KeyError):
                    setattr(layer, attr, None)

        # Make sure the bf16 weight is also reflected as orig_dtype so any
        # later dtype assumptions hold.
        layer.orig_dtype = target_dtype

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # After process_weights_after_loading, layer.weight is plain bf16 in
        # [N, K] orientation, so a normal linear works. In the TP path the
        # module is replaced by Xla*ParallelLinear and this apply() is never
        # called; this is only a fallback for the non-TP / unwrapped case.
        return torch.nn.functional.linear(x, layer.weight, bias)


_ORIG_GET_QUANT_METHOD = Fp8Config.get_quant_method


def _tt_get_quant_method(
    self: Fp8Config, layer: torch.nn.Module, prefix: str
) -> "QuantizeMethodBase | None":
    """Monkeypatched ``Fp8Config.get_quant_method``.

    On the OOT (TT) platform, return our dequantizing linear method instead
    of vLLM's kernel-based ``Fp8LinearMethod`` for fp8-serialized linear
    layers that are NOT in ``ignored_layers``. Everything else (ignored
    layers, MoE, attention/kv-cache, non-OOT platforms) falls through to the
    original implementation.
    """
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.interface import PlatformEnum

    if (
        current_platform._enum == PlatformEnum.OOT
        and isinstance(layer, LinearBase)
        and self.is_checkpoint_fp8_serialized
    ):
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()
        logger.debug("TT fp8: using dequantizing linear method for layer %s", prefix)
        return TTFp8DequantLinearMethod(self)

    return _ORIG_GET_QUANT_METHOD(self, layer, prefix)


def install_fp8_dequant_hook() -> None:
    """Install the fp8->bf16 dequant hook by monkeypatching
    ``Fp8Config.get_quant_method``. Idempotent."""
    if getattr(Fp8Config.get_quant_method, "_tt_patched", False):
        return
    _tt_get_quant_method._tt_patched = True
    Fp8Config.get_quant_method = _tt_get_quant_method
    logger.info("Installed TT fp8 dequantization hook for OOT platform.")
