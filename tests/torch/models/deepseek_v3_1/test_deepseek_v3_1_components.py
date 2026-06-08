# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component-level PCC tests for DeepSeek V3.1 on TT (Galaxy 4x8).

These tests isolate the two suspected sources of the ``test_deepseek_v3_1_tp_galaxy_4_layers``
(``tests/benchmark/test_llms.py``) PCC regression — the MLA attention block
(prefill + decode) and the (sparse) MoE block — and run each one standalone on
TT vs CPU so the regression can be localized to a single component.

To stay faithful to the benchmark, everything that affects numerics is shared
with it:

- Same model: built through the ``tt_forge_models`` ``ModelLoader`` for
  ``ModelVariant.DEEPSEEK_V3_1_MODIFIED`` with ``num_layers=4``, BF16 weights,
  ``enable_sparse_mlp`` applied (so MoE layers are ``A2aSparseMLPWithSharedExperts``).
- Same sharding: the per-tensor specs come straight from
  ``ModelLoader.load_shard_spec`` (filtered to the module under test) on the
  same ``(4, 8)`` ``("batch", "model")`` mesh.
- Same compile options: ``optimization_level=0``, ``experimental_weight_dtype="bfp_bf8"``,
  ``experimental_kv_cache_dtype=None``, trace disabled, permute-matmul fusion off —
  matching the values the benchmark passes for this test.
- Same MLA static cache (``MLACache`` / ``MLAStaticLayer``) and additive causal
  mask construction.

Each test reports the measured PCC (via a custom comparator) regardless of
pass/fail, so a sweep across the three tests pinpoints which block degrades.

Galaxy-only: model construction calls ``enable_sparse_mlp`` which requires the
32-device Galaxy mesh.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.testers.compiler_config import CompilerConfig
from infra.utilities import MLACache
from infra.utilities.torch_multichip_utils import enable_spmd
from torch import nn
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

# ---------------------------------------------------------------------------
# Test configuration (mirrors test_deepseek_v3_1_tp_galaxy_4_layers)
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
BATCH_SIZE = 64
# The MoE block allocates a per-token x 256-expert dispatch buffer; BATCH_SIZE=64
# OOMs (3.7 GB DRAM buffer). 8 fits and stays divisible by the mesh "batch" dim (4).
MOE_BATCH_SIZE = 8
PREFILL_LEN = 128
MAX_CACHE_LEN = 256
DTYPE = torch.bfloat16

# Layer used for the attention tests. The MLA block is structurally identical
# in every layer, so layer 0 is representative.
ATTN_LAYER_IDX = 0
# first_k_dense_replace == 3, so with 4 layers exactly one MoE layer exists
# (index 3); layers 0..2 are dense DeepseekV3MLP.
MOE_LAYER_IDX = 3

# Matches the PCC bar the full-model benchmark asserts for this test.
REQUIRED_PCC = 0.90

# Fixed prompt used to derive realistic-magnitude hidden states.
PROMPT = "Tell me a short story about a robot learning to paint."


def _compile_config() -> CompilerConfig:
    """Compile options identical to what the benchmark passes for this test."""
    return CompilerConfig(
        optimization_level=0,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype=None,
        experimental_enable_permute_matmul_fusion=False,
        enable_trace=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, computed exactly like the benchmark's compute_pcc."""
    x = a.detach().to(torch.float32).flatten()
    y = b.detach().to(torch.float32).flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm()).item()
    if denom == 0:
        return 1.0 if torch.allclose(x, y, rtol=1e-2, atol=1e-2) else float("nan")
    return max(-1.0, min(1.0, ((vx @ vy) / denom).item()))


def _pcc_comparator(label: str, required_pcc: float = REQUIRED_PCC) -> Callable:
    """Build a custom comparator that prints the PCC then asserts the threshold.

    Always logging the value (not only on failure) is what makes a sweep across
    these tests useful for localizing the regression.
    """

    def _compare(tt_res, cpu_res, args, kwargs):
        tt = tt_res.to("cpu")
        cpu = cpu_res.to("cpu")
        pcc = _pcc(tt, cpu)
        print(
            f"\n[pcc] {label}: pcc={pcc:.6f} (required >= {required_pcc})",
            flush=True,
        )
        assert (
            not np.isnan(pcc) and pcc >= required_pcc
        ), f"{label}: PCC {pcc:.6f} below required {required_pcc}"

    return _compare


def _mesh(loader: ModelLoader) -> Mesh:
    """The Galaxy (4, 8) ("batch", "model") mesh used by the benchmark."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_name)


def _submodule_weight_specs(loader: ModelLoader, model, submodule: nn.Module) -> dict:
    """Per-tensor shard specs for ``submodule``, taken from the full-model spec.

    ``load_shard_spec`` keys by tensor identity; ``nn.Module.to(device)`` moves
    parameters in place (same Python objects), so the precomputed dict stays
    valid after the device move. Filtering to the submodule's own parameters
    guarantees the isolated block is sharded byte-for-byte like the benchmark.
    """
    full = loader.load_shard_spec(model)
    owned = {id(p) for p in submodule.parameters()}
    return {t: s for t, s in full.items() if id(t) in owned}


def _build_causal_mask(cache, layer_idx, cache_position, batch_size):
    """Additive (B, 1, q_len, max_cache_len) mask, built like the model does."""
    return cache.layers[layer_idx].build_causal_mask(
        cache_position, batch_size, DTYPE, torch.device("cpu")
    )


def _init_mla_cache(config) -> MLACache:
    """Pre-allocate an MLACache on CPU (mirrors benchmark's init_mla_cache)."""
    cache = MLACache(config=config, max_cache_len=MAX_CACHE_LEN)
    text_config = config.get_text_config(decoder=True)
    dummy_kv = torch.zeros((BATCH_SIZE, 1, 1, text_config.kv_lora_rank), dtype=DTYPE)
    dummy_pe = torch.zeros(
        (BATCH_SIZE, 1, 1, text_config.qk_rope_head_dim), dtype=DTYPE
    )
    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)
    return cache


# ---------------------------------------------------------------------------
# Adapters: expose the HF-style submodule forwards as positional-tensor graphs
# ---------------------------------------------------------------------------


class _AttnRunner(nn.Module):
    """Drives a single MLA attention block with explicit positional tensors.

    The MLA ``past_key_value`` cache is passed as an argument (not held as an
    attribute) so the device runner moves it to the device alongside the other
    inputs — ``nn.Module.to`` would not relocate a plain attribute.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(
        self, hidden_states, attention_mask, position_ids, cache_position, cache
    ):
        attn_output, _, _ = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        return attn_output


# ---------------------------------------------------------------------------
# Shared model fixture (loaded once for all component tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model():
    """Load the 4-layer DeepSeek V3.1 model once, exactly as the benchmark does."""
    enable_spmd()
    xr.set_device_type("TT")

    loader = ModelLoader(
        variant=ModelVariant.DEEPSEEK_V3_1_MODIFIED, num_layers=NUM_LAYERS
    )
    model = loader.load_model(dtype_override=DTYPE)
    model.eval()
    return loader, model


def _realistic_hidden_states(
    model, tokenizer, norm: nn.Module, batch_size: int = BATCH_SIZE
) -> torch.Tensor:
    """(batch_size, PREFILL_LEN, hidden) post-RMSNorm hidden states from a prompt.

    Embeds a real prompt and applies ``norm`` (the block's pre-block RMSNorm), so
    the magnitudes match what the block sees in the full model. The PCC compares
    identical math on TT vs CPU, so this only needs to be representative in scale.
    ``batch_size`` is overridable for memory-heavy blocks (the MoE block OOMs at
    BATCH_SIZE=64); per-token numerics are independent so PCC stays representative.
    """
    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        max_length=PREFILL_LEN,
        truncation=True,
        padding="max_length",
    )
    input_ids = encoded["input_ids"][:, :PREFILL_LEN].repeat(batch_size, 1)
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
        hidden = norm(embeds)
    return hidden.to(DTYPE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.galaxy
def test_deepseek_v3_1_attention_prefill(loaded_model):
    """MLA attention prefill (q_len=PREFILL_LEN) on TT vs CPU."""
    loader, model = loaded_model

    layer = model.model.layers[ATTN_LAYER_IDX]
    attn = layer.self_attn
    runner = _AttnRunner(attn)

    hidden_states = _realistic_hidden_states(
        model, loader.tokenizer, layer.input_layernorm
    )
    cache = _init_mla_cache(model.config)
    cache_position = torch.arange(PREFILL_LEN, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    attention_mask = _build_causal_mask(
        cache, ATTN_LAYER_IDX, cache_position, BATCH_SIZE
    )

    mesh = _mesh(loader)

    def shard_spec_fn(module, args, kwargs):
        specs = _submodule_weight_specs(loader, model, module.attn)
        specs[args[0]] = ("batch", None, None)  # hidden_states
        cache_arg = args[4]
        specs[cache_arg.layers[ATTN_LAYER_IDX].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        specs[cache_arg.layers[ATTN_LAYER_IDX].k_pe] = ("batch", None, None, None)
        return specs

    import chisel

    with chisel.session(
        results_path="deepseek_v3_1_attention_prefill_report.jsonl",
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:

        run_graph_test(
            runner,
            [hidden_states, attention_mask, position_ids, cache_position, cache],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
            compiler_config=_compile_config(),
            comparison_config=ComparisonConfig(
                pcc=PccConfig(enabled=True, required_pcc=REQUIRED_PCC)
            ),
            custom_comparator=_pcc_comparator("attention_prefill"),
        )


@pytest.mark.galaxy
def test_deepseek_v3_1_attention_decode(loaded_model):
    """MLA attention single-step decode (q_len=1) on TT vs CPU.

    The cache is pre-populated (positions 0..PREFILL_LEN-1) so the decode step
    attends to a full KV history, exactly as it would mid-generation. The cache
    is filled directly with deterministic values rather than via a model forward
    so this test does not depend on the device state left by other tests sharing
    the module-scoped model.
    """
    loader, model = loaded_model

    layer = model.model.layers[ATTN_LAYER_IDX]
    attn = layer.self_attn
    runner = _AttnRunner(attn)

    # ---- Pre-populate the cache (positions 0..PREFILL_LEN-1) ----
    cache = _init_mla_cache(model.config)
    gen = torch.Generator().manual_seed(0)
    cache_layer = cache.layers[ATTN_LAYER_IDX]
    cache_layer.compressed_kv[:, :, :PREFILL_LEN, :] = torch.randn(
        cache_layer.compressed_kv[:, :, :PREFILL_LEN, :].shape, generator=gen
    ).to(DTYPE)
    cache_layer.k_pe[:, :, :PREFILL_LEN, :] = torch.randn(
        cache_layer.k_pe[:, :, :PREFILL_LEN, :].shape, generator=gen
    ).to(DTYPE)

    # ---- Single decode step at position PREFILL_LEN ----
    decode_hidden = _realistic_hidden_states(
        model, loader.tokenizer, layer.input_layernorm
    )[:, :1, :]
    decode_pos = torch.tensor([PREFILL_LEN], dtype=torch.long)
    position_ids = decode_pos.unsqueeze(0)
    attention_mask = _build_causal_mask(cache, ATTN_LAYER_IDX, decode_pos, BATCH_SIZE)

    mesh = _mesh(loader)

    def shard_spec_fn(module, args, kwargs):
        specs = _submodule_weight_specs(loader, model, module.attn)
        specs[args[0]] = ("batch", None, None)  # hidden_states (q_len == 1)
        cache_arg = args[4]
        specs[cache_arg.layers[ATTN_LAYER_IDX].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        specs[cache_arg.layers[ATTN_LAYER_IDX].k_pe] = ("batch", None, None, None)
        return specs

    import chisel

    with chisel.session(
        results_path="deepseek_v3_1_attention_decode_report.jsonl",
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:

        run_graph_test(
            runner,
            [decode_hidden, attention_mask, position_ids, decode_pos, cache],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
            compiler_config=_compile_config(),
            comparison_config=ComparisonConfig(
                pcc=PccConfig(enabled=True, required_pcc=REQUIRED_PCC)
            ),
            custom_comparator=_pcc_comparator("attention_decode"),
        )


@pytest.mark.galaxy
def test_deepseek_v3_1_moe(loaded_model):
    """Sparse MoE block (A2aSparseMLPWithSharedExperts) on TT vs CPU."""
    loader, model = loaded_model

    layer = model.model.layers[MOE_LAYER_IDX]
    moe = layer.mlp  # A2aSparseMLPWithSharedExperts after enable_sparse_mlp

    hidden_states = _realistic_hidden_states(
        model,
        loader.tokenizer,
        layer.post_attention_layernorm,
        batch_size=MOE_BATCH_SIZE,
    )

    mesh = _mesh(loader)

    def shard_spec_fn(module, args, kwargs):
        specs = _submodule_weight_specs(loader, model, module)
        specs[args[0]] = ("batch", None, None)  # hidden_states
        return specs

    import chisel

    with chisel.session(
        results_path="deepseek_v3_1_moe_report.jsonl",
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:

        run_graph_test(
            moe,
            [hidden_states],
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
            compiler_config=_compile_config(),
            comparison_config=ComparisonConfig(
                pcc=PccConfig(enabled=True, required_pcc=REQUIRED_PCC)
            ),
            custom_comparator=_pcc_comparator("moe"),
        )
