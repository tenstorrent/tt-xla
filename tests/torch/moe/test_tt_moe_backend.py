# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity tests for the `tt_moe` HuggingFace ExpertsInterface backend.

Each test instantiates a single HF MoE *decoder layer* (never the whole
`*ForCausalLM` model — see below), patches its self-attention to a no-op,
and compares its forward with the `tt_moe` backend against HF's stock
`eager` backend. Zero'd attention makes the decoder's self-attn residual an
identity, so the output delta from the input comes entirely from the MoE
MLP path — this isolates MoE correctness from attention/cache plumbing.

Why build the DecoderLayer directly and not `AutoModelForCausalLM.from_config`:
full-model construction allocates `embed_tokens` and `lm_head` (both
vocab × hidden). For DeepSeek-V3 that's 129280 × 7168 × 2 ≈ 3.7 GB of
weights the test never touches — enough to DRAM-OOM on llmbox for the
MoE-layer tests. We only need one DecoderLayer.

The `@use_experts_implementation` decorator reads
`config._experts_implementation` at forward-dispatch time, so setting it on
the config before constructing the layer is sufficient for tt_moe routing;
no post-hoc rebinding is needed.

DP path only. EP coverage requires a multi-device mesh and lives in
model-specific layer tests.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable, Type

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from transformers import AutoConfig
from tt_torch.moe_backend import TT_MOE_BACKEND_NAME, register_tt_moe_backend

# ---------------------------------------------------------------------------
# Attention no-op helper
# ---------------------------------------------------------------------------


def _attention_noop(self, hidden_states, *args, **kwargs):
    """Replacement for `self_attn.forward` that returns zeros.

    HF decoder layers do `hidden_states, _ = self.self_attn(...)` then
    `hidden_states = residual + hidden_states`. Returning zeros makes the
    attention path a no-op while preserving tuple shape.
    """
    return torch.zeros_like(hidden_states), None


def _patch_layer_attn_to_noop(layer: nn.Module) -> nn.Module:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise RuntimeError(
            f"Expected `self_attn` on decoder layer {type(layer).__name__}; "
            f"update the noop helper for this architecture."
        )
    attn.forward = types.MethodType(_attention_noop, attn)
    return layer


# ---------------------------------------------------------------------------
# Model table
# ---------------------------------------------------------------------------


def _gpt_oss_layer_cls() -> Type[nn.Module]:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    return GptOssDecoderLayer


def _olmoe_layer_cls() -> Type[nn.Module]:
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    return OlmoeDecoderLayer


def _qwen3_moe_layer_cls() -> Type[nn.Module]:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

    return Qwen3MoeDecoderLayer


@dataclass
class MoEModelCase:
    """One entry in the parametrized test table."""

    name: str  # human-readable pytest id
    pretrained: str  # HF hub id used only for AutoConfig
    layer_cls: Callable[[], Type[nn.Module]]  # lazy import of DecoderLayer
    layer_idx: int = 0  # MoE layer idx to instantiate (> first_k_dense_replace)
    # Override the chosen seq_len when the default 128 does not divide
    # evenly into the model's tile shape. None keeps 128.
    seq_len_override: int | None = None


MODEL_TABLE = [
    MoEModelCase(
        name="gpt_oss_20b",
        pretrained="openai/gpt-oss-20b",
        layer_cls=_gpt_oss_layer_cls,
    ),
    MoEModelCase(
        name="olmoe_1b_7b",
        pretrained="allenai/OLMoE-1B-7B-0924",
        layer_cls=_olmoe_layer_cls,
    ),
    MoEModelCase(
        name="qwen3_moe_30b",
        pretrained="Qwen/Qwen3-30B-A3B",
        layer_cls=_qwen3_moe_layer_cls,
    ),
    # Non-canonical Experts (ModuleList-based) — DeepSeek V3 / DeepSeek V3-2 /
    # old vendored models — are deliberately out of scope: a stacked-weight
    # adapter for ModuleList Experts is follow-up work.
]


# ---------------------------------------------------------------------------
# Builder + tests
# ---------------------------------------------------------------------------


def _build_decoder_layer(
    case: MoEModelCase, dtype: torch.dtype
) -> tuple[nn.Module, AutoConfig]:
    """Instantiate a single MoE DecoderLayer with the tt_moe backend selected.

    Avoids full-model construction (embed_tokens + lm_head + extra dense
    layers) which would balloon device memory for large-vocab / large-hidden
    models like DeepSeek-V3.
    """
    config = AutoConfig.from_pretrained(case.pretrained, trust_remote_code=True)
    config.num_hidden_layers = max(config.num_hidden_layers, case.layer_idx + 1)
    config.use_cache = False
    config._attn_implementation = "eager"
    config._experts_implementation = TT_MOE_BACKEND_NAME

    layer_cls = case.layer_cls()
    layer = layer_cls(config, layer_idx=case.layer_idx).eval().to(dtype)
    return layer, config


def _case_ids(cases):
    return [c.name for c in cases]


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("case", MODEL_TABLE, ids=_case_ids(MODEL_TABLE))
def test_tt_moe_full_decoder_dp(case: MoEModelCase):
    """Single MoE decoder layer forward: tt_moe DP must match HF eager."""
    xr.set_device_type("TT")

    register_tt_moe_backend(cluster_axis=0)
    torch.manual_seed(0)
    layer, config = _build_decoder_layer(case, torch.bfloat16)
    _patch_layer_attn_to_noop(layer)

    seq_len = case.seq_len_override or 128
    hidden_states = torch.randn((1, seq_len, config.hidden_size), dtype=torch.bfloat16)

    run_graph_test(
        layer,
        [hidden_states],
        framework=Framework.TORCH,
    )
